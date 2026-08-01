"""Phase 3 — fetch a DEM, sample elevation profiles along every edge, and price them.

Source is USGS 3DEP 1 m lidar, which fully covers the study area
(``VA_FEMA-NRCS_SouthCentral_2017_D17``, tile ``x54y413``, UTM 17N — hence EPSG:6346).

Two details that matter more than they look:

* **Smoothing is mandatory.** Raw 1 m lidar has centimetre-scale noise. Differencing it
  over a 10 m sample step yields spurious grades of several percent, and Tobler's
  exponential turns those into large, entirely fictional time penalties. A Gaussian
  pre-filter is not cosmetic — it is the difference between a usable model and garbage.
* **Validation is free.** The site publishes forward *and* reverse elevation gain for
  all 40 trails. That is 80 independent ground-truth values to regress against.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter

from .config import (
    CONFIG,
    CRS_PROJECTED,
    DEM_PATH,
    DEM_TILE_PATH,
    DEM_TILE_URL,
    EDGES,
    EDGES_TIMED,
    ElevationParams,
    FT_PER_M,
    NODES,
    ToblerParams,
    TRAILS_RAW,
)
from .tobler import profile_gain_m, profile_travel_time

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# DEM acquisition
# --------------------------------------------------------------------------------------


def _download_tile(url: str, dest, chunk: int = 1 << 20):
    """Stream a large GeoTIFF to disk, writing to a temp file so a partial download is
    never mistaken for a complete cache entry."""
    import requests

    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        written = 0
        with open(tmp, "wb") as fh:
            for block in resp.iter_content(chunk_size=chunk):
                fh.write(block)
                written += len(block)
                if total and written % (32 << 20) < chunk:
                    log.info("  %.0f%% (%.0f MB)", 100 * written / total, written / 1e6)
    tmp.replace(dest)
    log.info("downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def _clip_tile(tile_path, edges: gpd.GeoDataFrame, params: ElevationParams, out_path):
    """Clip the source tile to the padded network bbox and align it to CRS_PROJECTED."""
    from rasterio.warp import calculate_default_transform, reproject, transform_bounds
    from rasterio.windows import from_bounds

    minx, miny, maxx, maxy = edges.to_crs(CRS_PROJECTED).total_bounds
    pad = params.dem_buffer_m
    target_bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    with rasterio.open(tile_path) as src:
        src_bounds = transform_bounds(CRS_PROJECTED, src.crs, *target_bounds)
        window = from_bounds(*src_bounds, transform=src.transform).round_offsets().round_lengths()
        window = window.intersection(
            rasterio.windows.Window(0, 0, src.width, src.height)
        )
        data = src.read(1, window=window, masked=True)
        win_transform = src.window_transform(window)
        profile = src.profile.copy()

        if src.crs.to_epsg() == rasterio.crs.CRS.from_string(CRS_PROJECTED).to_epsg():
            out_arr, out_transform, out_crs = data.filled(np.nan), win_transform, src.crs
        else:
            dst_transform, width, height = calculate_default_transform(
                src.crs,
                CRS_PROJECTED,
                window.width,
                window.height,
                *rasterio.windows.bounds(window, src.transform),
                resolution=params.dem_resolution_m,
            )
            out_arr = np.full((height, width), np.nan, dtype="float32")
            reproject(
                source=data.filled(np.nan),
                destination=out_arr,
                src_transform=win_transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=CRS_PROJECTED,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
            out_transform, out_crs = dst_transform, CRS_PROJECTED

    profile.update(
        driver="GTiff",
        height=out_arr.shape[0],
        width=out_arr.shape[1],
        transform=out_transform,
        crs=out_crs,
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="deflate",
        tiled=True,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_arr.astype("float32"), 1)
    log.info("clipped DEM -> %s shape=%s", out_path.name, out_arr.shape)
    return out_path


def fetch_dem(
    edges: gpd.GeoDataFrame,
    params: ElevationParams | None = None,
    path=DEM_PATH,
    *,
    force: bool = False,
):
    """Obtain the 3DEP DEM clipped to the network bbox, and cache it to disk.

    Primary route is the 1 m lidar source tile from the USGS S3 bucket. The dynamic
    elevation service (via ``py3dep``) is only a fallback: its async client fails
    intermittently on Windows with spurious DNS errors, and silently degrading from 1 m
    lidar to a 10 m DEM without noticing would corrupt every slope in the model.

    Cached aggressively — marimo re-runs downstream cells whenever a Tobler slider moves,
    and re-fetching 200 MB of lidar each time would make the notebook unusable.
    """
    if path.exists() and not force:
        log.info("using cached DEM at %s", path)
        return path

    params = params or CONFIG.elevation

    try:
        if not DEM_TILE_PATH.exists() or DEM_TILE_PATH.stat().st_size < 1_000_000:
            log.info("downloading 3DEP 1 m tile (~206 MB) from %s", DEM_TILE_URL)
            _download_tile(DEM_TILE_URL, DEM_TILE_PATH)
        return _clip_tile(DEM_TILE_PATH, edges, params, path)
    except Exception as exc:  # noqa: BLE001
        log.warning("1 m tile route failed (%s); falling back to py3dep 1/3 arc-second", exc)

    import py3dep

    bounds = edges.to_crs("EPSG:4326").total_bounds
    pad_deg = params.dem_buffer_m / 111_000.0
    bbox = (
        bounds[0] - pad_deg / np.cos(np.radians(bounds[1])),
        bounds[1] - pad_deg,
        bounds[2] + pad_deg / np.cos(np.radians(bounds[3])),
        bounds[3] + pad_deg,
    )
    dem = py3dep.get_dem(bbox, resolution=10, crs=4326)
    dem = dem.rio.reproject(CRS_PROJECTED, resampling=Resampling.bilinear)
    dem.rio.to_raster(path, compress="deflate", tiled=True)
    log.warning("USING 10 m FALLBACK DEM — slopes will be under-resolved")
    return path


def load_dem(path=DEM_PATH, params: ElevationParams | None = None):
    """Read the DEM and return ``(smoothed_array, rasterio_transform, crs, nodata_mask)``.

    Smoothing is specified in metres and converted to pixels here, so the physical amount
    of smoothing is identical whichever DEM resolution we ended up with.
    """
    params = params or CONFIG.elevation
    with rasterio.open(path) as src:
        array = src.read(1, masked=True).astype("float32")
        transform = src.transform
        crs = src.crs

    cell_m = abs(transform.a)
    sigma_px = max(params.smooth_sigma_m / cell_m, 0.0)

    invalid = np.ma.getmaskarray(array) | ~np.isfinite(array.filled(np.nan))
    filled = array.filled(np.nan)
    if invalid.any():
        # Fill holes with the scene mean so the Gaussian filter doesn't smear NaNs.
        filled = np.where(invalid, np.nanmean(filled), filled)

    smoothed = (
        gaussian_filter(filled, sigma=sigma_px, mode="nearest") if sigma_px > 0.1 else filled
    )
    log.info(
        "DEM %s at %.2f m/cell, smoothing sigma %.1f m = %.2f px",
        array.shape,
        cell_m,
        params.smooth_sigma_m,
        sigma_px,
    )
    return smoothed, transform, crs, invalid


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------


def sample_raster(array: np.ndarray, transform, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear sample of a raster at projected coordinates."""
    inv = ~transform
    cols, rows = inv * (np.asarray(xs), np.asarray(ys))
    cols = np.clip(cols - 0.5, 0, array.shape[1] - 1.001)
    rows = np.clip(rows - 0.5, 0, array.shape[0] - 1.001)

    c0, r0 = np.floor(cols).astype(int), np.floor(rows).astype(int)
    c1, r1 = c0 + 1, r0 + 1
    fc, fr = cols - c0, rows - r0

    return (
        array[r0, c0] * (1 - fc) * (1 - fr)
        + array[r0, c1] * fc * (1 - fr)
        + array[r1, c0] * (1 - fc) * fr
        + array[r1, c1] * fc * fr
    )


def densify(geom, step_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a LineString to a fixed step, returning ``(distances, xs, ys)``.

    Original vertices are kept as well as the regular stations, so sharp direction or
    grade changes are never smoothed away by the resampling itself.
    """
    length = geom.length
    n = max(int(np.ceil(length / step_m)), 1)
    stations = np.linspace(0.0, length, n + 1)
    vertex_d = np.array([geom.project(type(geom.interpolate(0))(c)) for c in geom.coords])
    distances = np.unique(np.concatenate([stations, vertex_d]))
    points = [geom.interpolate(d) for d in distances]
    return distances, np.array([p.x for p in points]), np.array([p.y for p in points])


def edge_profiles(
    edges: gpd.GeoDataFrame,
    array: np.ndarray,
    transform,
    params: ElevationParams | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Elevation profile ``(distances, elevations)`` for every edge."""
    params = params or CONFIG.elevation
    profiles = []
    for geom in edges.geometry:
        distances, xs, ys = densify(geom, params.sample_step_m)
        profiles.append((distances, sample_raster(array, transform, xs, ys)))
    return profiles


# --------------------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------------------


def price_edges(
    edges: gpd.GeoDataFrame,
    profiles: list[tuple[np.ndarray, np.ndarray]],
    tobler: ToblerParams | None = None,
) -> gpd.GeoDataFrame:
    """Attach directional travel times and elevation statistics to every edge."""
    tobler = tobler or CONFIG.tobler
    out = edges.copy()

    fwd_s, rev_s, gain_f, gain_r, lo, hi = [], [], [], [], [], []
    for (distances, elevations), off_trail in zip(profiles, edges["off_trail"]):
        fwd_s.append(profile_travel_time(distances, elevations, tobler, off_trail=off_trail))
        rev_s.append(
            profile_travel_time(distances, elevations, tobler, off_trail=off_trail, reverse=True)
        )
        gain_f.append(profile_gain_m(elevations))
        gain_r.append(profile_gain_m(elevations, reverse=True))
        lo.append(float(np.min(elevations)))
        hi.append(float(np.max(elevations)))

    out["time_fwd_s"] = fwd_s
    out["time_rev_s"] = rev_s
    out["gain_fwd_m"] = gain_f
    out["gain_rev_m"] = gain_r
    out["min_ele_m"] = lo
    out["max_ele_m"] = hi
    out["length_mi"] = out["length_m"] / 1609.344
    # Points are only earned on real trail, never on off-trail connectors.
    out["score_mi"] = np.where(out["off_trail"], 0.0, out["length_mi"])
    return out


# --------------------------------------------------------------------------------------
# Validation against the site's published gains
# --------------------------------------------------------------------------------------


def validate_against_official(edges: gpd.GeoDataFrame, trails: gpd.GeoDataFrame) -> pd.DataFrame:
    """Regress DEM-derived elevation gain against the 40x2 official values.

    Trails are edge-split, so per-trail gain is summed over that trail's edges. Summing
    per-edge gain slightly overestimates versus a single continuous profile (each edge
    boundary can add a fractional up-tick), but the bias is small and consistent.
    """
    on_trail = edges[~edges["off_trail"]]
    agg = (
        on_trail.groupby("trail_id")
        .agg(dem_gain_m=("gain_fwd_m", "sum"), dem_rev_gain_m=("gain_rev_m", "sum"))
        .reset_index()
    )
    merged = agg.merge(
        trails[["trail_id", "name", "official_gain_ft", "official_rev_gain_ft"]],
        on="trail_id",
        how="left",
    )
    merged["dem_gain_ft"] = (merged["dem_gain_m"] * FT_PER_M).round(0)
    merged["dem_rev_gain_ft"] = (merged["dem_rev_gain_m"] * FT_PER_M).round(0)
    merged["fwd_pct_err"] = (
        100
        * (merged["dem_gain_ft"] - merged["official_gain_ft"])
        / merged["official_gain_ft"].clip(lower=10)
    ).round(1)
    merged["rev_pct_err"] = (
        100
        * (merged["dem_rev_gain_ft"] - merged["official_rev_gain_ft"])
        / merged["official_rev_gain_ft"].clip(lower=10)
    ).round(1)
    return merged[
        [
            "trail_id",
            "name",
            "official_gain_ft",
            "dem_gain_ft",
            "fwd_pct_err",
            "official_rev_gain_ft",
            "dem_rev_gain_ft",
            "rev_pct_err",
        ]
    ]


def gain_correlation(report: pd.DataFrame) -> dict[str, float]:
    """R^2 of DEM gain vs official gain, pooling both directions."""
    official = pd.concat([report["official_gain_ft"], report["official_rev_gain_ft"]])
    dem = pd.concat([report["dem_gain_ft"], report["dem_rev_gain_ft"]])
    mask = official.notna() & dem.notna()
    official, dem = official[mask], dem[mask]
    r = float(np.corrcoef(official, dem)[0, 1])
    return {
        "n": int(mask.sum()),
        "r2": round(r**2, 4),
        "bias_ft": round(float((dem - official).mean()), 1),
        "mae_ft": round(float((dem - official).abs().mean()), 1),
    }


def smoothing_sensitivity(
    edges: gpd.GeoDataFrame,
    trails: gpd.GeoDataFrame,
    sigmas=(0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0),
    tobler: ToblerParams | None = None,
) -> pd.DataFrame:
    """How much does the DEM smoothing choice actually change the answer?

    Two things are tracked: agreement with the published elevation gains, and the total
    time to traverse the network once. The second is what the optimizer consumes, so
    stability there matters more than a marginally better R^2.
    """
    import dataclasses

    rows = []
    for sigma in sigmas:
        params = dataclasses.replace(CONFIG.elevation, smooth_sigma_m=sigma)
        array, transform, _, _ = load_dem(params=params)
        profiles = edge_profiles(edges, array, transform, params)
        timed = price_edges(edges, profiles, tobler)
        stats = gain_correlation(validate_against_official(timed, trails))
        rows.append(
            {
                "sigma_m": sigma,
                "r2": stats["r2"],
                "bias_ft": stats["bias_ft"],
                "mae_ft": stats["mae_ft"],
                "network_hours": round(float(timed["time_fwd_s"].sum()) / 3600, 2),
                "total_gain_ft": round(float(timed["gain_fwd_m"].sum()) * FT_PER_M),
            }
        )
    return pd.DataFrame(rows)


def run(
    tobler: ToblerParams | None = None, params: ElevationParams | None = None
) -> gpd.GeoDataFrame:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    edges = gpd.read_parquet(EDGES)
    trails = gpd.read_parquet(TRAILS_RAW)

    fetch_dem(edges, params)
    array, transform, _, _ = load_dem(params=params)
    profiles = edge_profiles(edges, array, transform, params)
    timed = price_edges(edges, profiles, tobler)
    timed.to_parquet(EDGES_TIMED)

    report = validate_against_official(timed, trails)
    log.info("DEM vs official gain: %s", gain_correlation(report))
    worst = report.reindex(report["fwd_pct_err"].abs().sort_values(ascending=False).index).head(6)
    log.info("largest gain discrepancies:\n%s", worst.to_string(index=False))
    log.info(
        "network traverse-once time: %.2f h", (timed["time_fwd_s"].sum()) / 3600.0
    )
    return timed


if __name__ == "__main__":
    run()
