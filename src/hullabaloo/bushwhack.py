"""Phase 4 — off-trail connectors via least-cost paths over the terrain.

This phase is load-bearing, not decorative. The 40 trails form **three disconnected
components** that no snapping tolerance will join, so without off-trail links there is no
feasible tour that visits more than one component.

Method: build a raster whose cell values are "seconds to cross this cell" from the
DEM-derived slope, run at 60% of on-trail Tobler speed, then run Dijkstra over the grid
(``skimage.graph.MCP_Geometric``) between candidate node pairs.

An honest caveat, stated here and in the notebook: ``MCP_Geometric`` is **isotropic** —
it prices a cell by local slope *magnitude*, so uphill and downhill cost the same during
path *selection*. We compensate by re-integrating the true directional Tobler time along
the returned polyline with the same routine used for on-trail edges, so the connector's
cost in the routing graph is properly asymmetric even though its shape was chosen
isotropically. ``MCP_Flexible`` is the fully anisotropic upgrade if this ever matters.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from shapely.geometry import LineString, Point

from .config import (
    CONFIG,
    CONNECTORS,
    CRS_PROJECTED,
    BushwhackParams,
    EDGES_TIMED,
    ElevationParams,
    NODES,
    ToblerParams,
)
from .elevation import load_dem, sample_raster
from .tobler import profile_gain_m, profile_travel_time, tobler_speed_ms
from .topology import connected_components

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Cost surface
# --------------------------------------------------------------------------------------


def elevation_surface(
    array: np.ndarray, transform, params: BushwhackParams | None = None
) -> tuple[np.ndarray, object]:
    """Coarsen the DEM to the connector-routing resolution: ``(elevation, transform)``.

    Factored out of :func:`build_cost_surface` so that re-pricing a connector later
    samples exactly the same surface it was originally routed over. Sampling the raw 1 m
    DEM instead would give subtly different elevations and therefore different times.
    """
    params = params or CONFIG.bushwhack
    px = abs(transform.a)
    factor = max(int(round(params.cost_surface_res_m / px)), 1)
    if factor > 1:
        h = (array.shape[0] // factor) * factor
        w = (array.shape[1] // factor) * factor
        array = array[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
        transform = transform * transform.identity().scale(factor, factor)
    return array, transform


def build_cost_surface(
    array: np.ndarray,
    transform,
    tobler: ToblerParams | None = None,
    params: BushwhackParams | None = None,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Seconds-per-cell cost raster for off-trail travel.

    Returns ``(cost, elevation, transform)`` at the (possibly coarsened) resolution.
    1 m cells over the whole study area is a needlessly large grid for routing 900 m
    connectors; coarsening to ~3 m is ~9x cheaper and visually identical.
    """
    tobler = tobler or CONFIG.tobler
    params = params or CONFIG.bushwhack

    array, transform = elevation_surface(array, transform, params)
    cell = abs(transform.a)
    dz_dy, dz_dx = np.gradient(array, cell)
    slope = np.hypot(dz_dx, dz_dy)

    speed = tobler_speed_ms(slope, tobler, off_trail=True)
    cost = cell / speed

    impassable = slope > params.max_offtrail_slope
    cost[impassable] = np.inf
    log.info(
        "cost surface %s at %.1f m/cell | %.2f%% impassable by slope",
        cost.shape,
        cell,
        100 * impassable.mean(),
    )
    return cost, array, transform


#: NHD "Waterbody - Large Scale" layer on the USGS hydro MapServer. Queried over plain
#: synchronous HTTP on purpose: ``pynhd``'s async client fails intermittently on Windows
#: with spurious DNS errors, and this mask is too important to leave to a flaky call.
NHD_WATERBODY_URL = (
    "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/12/query"
)

#: If a fallback heuristic ever flags more than this fraction of the study area as water,
#: it is wrong and gets discarded. An early flatness-based heuristic cheerfully flagged
#: 20% of the map as lake, which would have quietly warped every connector.
MAX_PLAUSIBLE_WATER_FRACTION = 0.02


def fetch_waterbodies(bounds_wgs84: tuple[float, float, float, float]):
    """NHD waterbody polygons intersecting the bbox, in ``CRS_PROJECTED``."""
    import requests

    resp = requests.get(
        NHD_WATERBODY_URL,
        params={
            "geometry": ",".join(str(v) for v in bounds_wgs84),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "outSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "GNIS_NAME,AREASQKM,FTYPE",
            "returnGeometry": "true",
            "f": "geojson",
        },
        timeout=120,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("features"):
        return gpd.GeoDataFrame(geometry=[], crs=CRS_PROJECTED)
    gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
    return gdf.to_crs(CRS_PROJECTED)


#: NLCD land cover, rendered by the MRLC GeoServer. Requested as a rendered PNG and
#: classified back via the standard NLCD palette, since the WMS does not serve raw class
#: values. Rendering shifts colours by a unit or two, so matching is nearest-colour.
NLCD_WMS = (
    "https://www.mrlc.gov/geoserver/mrlc_display/NLCD_2021_Land_Cover_L48/wms"
)

#: Standard NLCD legend. Only the classes that occur in this study area are listed.
NLCD_PALETTE: dict[tuple[int, int, int], int] = {
    (70, 107, 159): 11,  # open water
    (222, 197, 197): 21,  # developed, open space
    (217, 146, 130): 22,  # developed, low intensity
    (235, 0, 0): 23,  # developed, medium intensity
    (171, 0, 0): 24,  # developed, high intensity
    (179, 172, 159): 31,  # barren
    (104, 171, 95): 41,  # deciduous forest
    (28, 95, 44): 42,  # evergreen forest
    (181, 197, 143): 43,  # mixed forest
    (223, 223, 194): 52,  # shrub/scrub
    (196, 212, 0): 71,  # herbaceous
    (220, 217, 57): 81,  # pasture / hay
    (171, 108, 40): 82,  # cultivated crops
    (184, 217, 235): 90,  # woody wetlands
    (108, 159, 184): 95,  # emergent wetlands
}

#: Default built-up classes: low/medium/high-intensity development — houses, driveways,
#: parking, commercial. A bare-earth DEM has no idea these exist, and without masking them
#: the router will happily send you through somebody's back garden.
#:
#: Note 21 ("Developed, Open Space") is *not* included: in a forested area it is mostly
#: road right-of-way, and blocking it forbids crossing roads. See
#: ``BushwhackParams.developed_classes``.
DEVELOPED_CLASSES = (22, 23, 24)


def fetch_landcover(bounds_wgs84, width: int = 1800, height: int = 1800):
    """NLCD class raster over a lon/lat bbox, as ``(classes, bounds)``."""
    from io import BytesIO

    import requests
    from PIL import Image

    resp = requests.get(
        NLCD_WMS,
        params={
            "service": "WMS",
            "version": "1.1.1",
            "request": "GetMap",
            "layers": "NLCD_2021_Land_Cover_L48",
            "bbox": ",".join(str(v) for v in bounds_wgs84),
            "width": width,
            "height": height,
            "srs": "EPSG:4326",
            "format": "image/png",
        },
        timeout=180,
    )
    resp.raise_for_status()
    if "image" not in resp.headers.get("Content-Type", ""):
        raise RuntimeError(f"NLCD WMS returned {resp.headers.get('Content-Type')}")

    rgb = np.asarray(Image.open(BytesIO(resp.content)).convert("RGB")).astype(np.int16)

    palette = np.array(list(NLCD_PALETTE.keys()), dtype=np.int16)
    codes = np.array(list(NLCD_PALETTE.values()), dtype=np.uint8)
    # Nearest palette colour per pixel.
    diff = rgb.reshape(-1, 1, 3) - palette.reshape(1, -1, 3)
    nearest = np.argmin((diff.astype(np.int32) ** 2).sum(axis=2), axis=1)
    return codes[nearest].reshape(rgb.shape[:2]), bounds_wgs84


def mask_developed(
    cost: np.ndarray,
    transform,
    bounds_wgs84,
    *,
    classes=DEVELOPED_CLASSES,
    buffer_cells: int = 1,
) -> tuple[np.ndarray, float]:
    """Make developed land impassable to off-trail routing.

    Discovered the hard way: the optimal route's longest bushwhack ran 848 m straight
    through a residential neighbourhood — houses, driveways, mown lawns and a swimming
    pool — because the cost surface is derived from a bare-earth DEM and slope alone said
    the going was easy. Land cover is the missing input.
    """
    from pyproj import Transformer

    cost = cost.copy()
    classes_arr, bnds = fetch_landcover(bounds_wgs84)

    rows, cols = np.indices(cost.shape)
    xs, ys = transform * (cols + 0.5, rows + 0.5)
    to_wgs = Transformer.from_crs(CRS_PROJECTED, "EPSG:4326", always_xy=True)
    lons, lats = to_wgs.transform(xs, ys)

    minx, miny, maxx, maxy = bnds
    h, w = classes_arr.shape
    px = np.clip(((lons - minx) / (maxx - minx) * w).astype(int), 0, w - 1)
    py = np.clip(((maxy - lats) / (maxy - miny) * h).astype(int), 0, h - 1)
    sampled = classes_arr[py, px]

    developed = np.isin(sampled, classes)
    if buffer_cells:
        developed = binary_dilation(developed, iterations=buffer_cells)
    fraction = float(developed.mean())
    cost[developed] = np.inf
    log.info(
        "masked developed land: %.1f%% of the study area (NLCD classes %s)",
        100 * fraction,
        ",".join(str(c) for c in classes),
    )
    return cost, fraction


def mask_water(
    cost: np.ndarray,
    elevation: np.ndarray,
    transform,
    bounds_wgs84,
    *,
    buffer_cells: int = 2,
) -> tuple[np.ndarray, gpd.GeoDataFrame | None]:
    """Make water bodies impassable.

    Pandapas Pond sits in the middle of the study area. A cost surface that ignores it
    will happily route a bushwhack connector straight across open water and silently
    invalidate the whole answer, so this is a correctness requirement rather than a
    refinement.
    """
    from rasterio.features import rasterize

    cost = cost.copy()
    water = np.zeros(cost.shape, dtype=bool)
    gdf = None

    try:
        gdf = fetch_waterbodies(bounds_wgs84)
        if len(gdf):
            water = rasterize(
                [(g, 1) for g in gdf.geometry],
                out_shape=cost.shape,
                transform=transform,
                fill=0,
                dtype="uint8",
            ).astype(bool)
            log.info(
                "masked %d NHD waterbodies, %.2f ha, %d cells",
                len(gdf),
                gdf.area.sum() / 10_000,
                int(water.sum()),
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("NHD waterbody query failed (%s)", exc)

    if not water.any():
        # Last resort: still water gives near-perfectly flat lidar returns. Deliberately
        # strict, and sanity-checked against the area budget below.
        from scipy.ndimage import label

        cell = abs(transform.a)
        dz_dy, dz_dx = np.gradient(elevation, cell)
        flat = np.hypot(dz_dx, dz_dy) < 0.0015
        labels, n = label(flat)
        if n:
            sizes = np.bincount(labels.ravel())
            min_cells = int(5000 / (cell**2))
            for lbl in np.where(sizes > min_cells)[0]:
                if lbl:
                    water |= labels == lbl
        fraction = water.mean()
        if fraction > MAX_PLAUSIBLE_WATER_FRACTION:
            log.error(
                "flatness heuristic flagged %.1f%% of the area as water — implausible, "
                "discarding it and masking NO water. Connectors may cross ponds; fix the "
                "NHD query before trusting this run.",
                100 * fraction,
            )
            water[:] = False
        else:
            log.warning("using flatness heuristic: %d cells (%.2f%%)", int(water.sum()), 100 * fraction)

    if water.any():
        water = binary_dilation(water, iterations=buffer_cells)
        cost[water] = np.inf
    return cost, gdf


# --------------------------------------------------------------------------------------
# Candidate selection
# --------------------------------------------------------------------------------------


def candidate_pairs(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    params: BushwhackParams | None = None,
) -> list[tuple[int, int, str]]:
    """Node pairs worth attempting an off-trail connector between.

    Two kinds are admitted:

    * **cross-component** — mandatory, these are what make the tour feasible at all;
    * **within-component shortcuts** — pairs that are close as the crow flies but far
      apart on the network (high detour ratio), which is where a bushwhack can actually
      pay for itself.
    """
    import networkx as nx

    params = params or CONFIG.bushwhack

    comp = connected_components(edges, len(nodes))
    coords = np.array([[g.x, g.y] for g in nodes.geometry])

    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    for row in edges.itertuples(index=False):
        t = min(row.time_fwd_s, row.time_rev_s)
        u, v = int(row.u), int(row.v)
        if not graph.has_edge(u, v) or graph[u][v]["t"] > t:
            graph.add_edge(u, v, t=t)

    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    pairs: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()

    for i, j in tree.query_pairs(params.max_connector_dist_m):
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        straight = float(np.linalg.norm(coords[i] - coords[j]))
        if comp[i] != comp[j]:
            pairs.append((i, j, "cross-component"))
            continue
        try:
            on_net = nx.shortest_path_length(graph, i, j, weight="t")
        except nx.NetworkXNoPath:
            pairs.append((i, j, "cross-component"))
            continue
        # Compare like with like: convert the network time to an equivalent distance at a
        # nominal 1 m/s so the ratio is dimensionless and interpretable.
        if on_net > params.min_detour_ratio * (straight / 1.0):
            pairs.append((i, j, "shortcut"))

    log.info(
        "%d candidate pairs (%d cross-component, %d shortcut)",
        len(pairs),
        sum(1 for p in pairs if p[2] == "cross-component"),
        sum(1 for p in pairs if p[2] == "shortcut"),
    )
    return pairs


# --------------------------------------------------------------------------------------
# Least-cost routing
# --------------------------------------------------------------------------------------


def _to_rc(transform, x: float, y: float, shape) -> tuple[int, int]:
    inv = ~transform
    col, row = inv * (x, y)
    return (
        int(np.clip(round(row), 0, shape[0] - 1)),
        int(np.clip(round(col), 0, shape[1] - 1)),
    )


def _to_xy(transform, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    xs, ys = transform * (cols + 0.5, rows + 0.5)
    return np.column_stack([xs, ys])


def least_cost_connectors(
    pairs: list[tuple[int, int, str]],
    nodes: gpd.GeoDataFrame,
    cost: np.ndarray,
    elevation: np.ndarray,
    transform,
    tobler: ToblerParams | None = None,
    params: BushwhackParams | None = None,
    elev_params: ElevationParams | None = None,
) -> gpd.GeoDataFrame:
    """Route each candidate pair over the cost surface and price the resulting polyline."""
    from skimage.graph import MCP_Geometric

    tobler = tobler or CONFIG.tobler
    params = params or CONFIG.bushwhack
    elev_params = elev_params or CONFIG.elevation

    coords = np.array([[g.x, g.y] for g in nodes.geometry])
    mcp = MCP_Geometric(cost, fully_connected=True)

    # Group by source so each source's Dijkstra sweep is reused across all its targets.
    by_source: dict[int, list[tuple[int, str]]] = {}
    for i, j, kind in pairs:
        by_source.setdefault(i, []).append((j, kind))

    records = []
    for src, targets in by_source.items():
        start = _to_rc(transform, *coords[src], cost.shape)
        if not np.isfinite(cost[start]):
            continue
        try:
            mcp.find_costs([start])
        except Exception as exc:  # noqa: BLE001
            log.warning("MCP sweep failed from node %d: %s", src, exc)
            continue

        for dst, kind in targets:
            end = _to_rc(transform, *coords[dst], cost.shape)
            if not np.isfinite(cost[end]):
                continue
            try:
                path = np.array(mcp.traceback(end))
            except Exception:  # noqa: BLE001 - unreachable target
                continue
            if len(path) < 2:
                continue

            xy = _to_xy(transform, path[:, 0], path[:, 1])
            # Anchor the ends exactly on the nodes so the graph stays properly noded.
            xy[0], xy[-1] = coords[src], coords[dst]
            line = LineString(xy)
            if line.length < 1.0:
                continue

            distances = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            )
            elevations = sample_raster(elevation, transform, xy[:, 0], xy[:, 1])

            records.append(
                {
                    "u": src,
                    "v": dst,
                    "kind": kind,
                    "length_m": line.length,
                    "straight_m": float(np.linalg.norm(coords[src] - coords[dst])),
                    "time_fwd_s": profile_travel_time(
                        distances, elevations, tobler, off_trail=True
                    ),
                    "time_rev_s": profile_travel_time(
                        distances, elevations, tobler, off_trail=True, reverse=True
                    ),
                    "gain_fwd_m": profile_gain_m(elevations),
                    "gain_rev_m": profile_gain_m(elevations, reverse=True),
                    "geometry": line,
                }
            )

    if not records:
        return gpd.GeoDataFrame(
            columns=["u", "v", "kind", "length_m", "time_fwd_s", "time_rev_s", "geometry"],
            geometry="geometry",
            crs=CRS_PROJECTED,
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_PROJECTED)
    # Prefer the cheapest connector per node pair, then keep the most useful ones.
    gdf["key"] = [tuple(sorted((int(a), int(b)))) for a, b in zip(gdf["u"], gdf["v"])]
    gdf = gdf.sort_values("time_fwd_s").drop_duplicates("key").drop(columns="key")
    gdf["sinuosity"] = (gdf["length_m"] / gdf["straight_m"].clip(lower=1)).round(2)

    cross = gdf[gdf["kind"] == "cross-component"]
    shortcuts = gdf[gdf["kind"] == "shortcut"].nsmallest(
        max(params.max_connectors - len(cross), 0), "time_fwd_s"
    )
    gdf = pd.concat([cross, shortcuts]).reset_index(drop=True)

    gdf["trail_id"] = pd.NA
    gdf["name"] = "bushwhack " + gdf["u"].astype(str) + "-" + gdf["v"].astype(str)
    gdf["off_trail"] = True
    gdf["length_mi"] = gdf["length_m"] / 1609.344
    gdf["score_mi"] = 0.0  # off-trail earns no points
    log.info("kept %d connectors (%d cross-component)", len(gdf), len(cross))
    return gdf


def reprice(
    connectors: gpd.GeoDataFrame,
    elevation: np.ndarray,
    transform,
    tobler: ToblerParams | None = None,
) -> gpd.GeoDataFrame:
    """Re-time existing connectors under different Tobler parameters.

    Where an off-trail connector *goes* does not depend on pace: multiplying every speed
    by the same factor leaves the cost surface's relative costs untouched, so the
    least-cost path between two nodes is unchanged and only its duration moves. That
    makes a pace sweep cheap — no cost surface, no land-cover masks, no MCP sweeps, just
    a re-integration along geometry we already have.

    Timing is reconstructed from each stored line's own vertices, which are precisely the
    cost-surface cells :func:`least_cost_connectors` walked. Re-densifying at a fixed
    step instead would resample the profile and quietly disagree with the connector times
    the optimizer was originally priced against.

    Skipping this step is the subtle way to get a pace sweep wrong: :func:`graph.
    build_network` reads ``time_fwd_s`` / ``time_rev_s`` straight off the connector
    frame, so bushwhacks would stay frozen at whatever pace built the file while every
    trail scaled around them — a mixed-pace model that still solves cleanly.
    """
    tobler = tobler or CONFIG.tobler
    out = connectors.copy()

    fwd, rev, gain_f, gain_r = [], [], [], []
    for geom in out.geometry:
        xy = np.asarray(geom.coords, dtype=float)
        distances = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        )
        elevations = sample_raster(elevation, transform, xy[:, 0], xy[:, 1])
        fwd.append(profile_travel_time(distances, elevations, tobler, off_trail=True))
        rev.append(
            profile_travel_time(distances, elevations, tobler, off_trail=True, reverse=True)
        )
        gain_f.append(profile_gain_m(elevations))
        gain_r.append(profile_gain_m(elevations, reverse=True))

    out["time_fwd_s"] = fwd
    out["time_rev_s"] = rev
    out["gain_fwd_m"] = gain_f
    out["gain_rev_m"] = gain_r
    return out


def run(
    tobler: ToblerParams | None = None, params: BushwhackParams | None = None
) -> gpd.GeoDataFrame:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    edges = gpd.read_parquet(EDGES_TIMED)
    nodes = gpd.read_parquet(NODES)

    array, transform, _, _ = load_dem()
    cost, elevation, transform = build_cost_surface(array, transform, tobler, params)
    bounds_wgs84 = tuple(edges.to_crs("EPSG:4326").total_bounds)
    cost, water = mask_water(cost, elevation, transform, bounds_wgs84)
    bp = params or CONFIG.bushwhack
    if bp.avoid_developed:
        try:
            cost, _ = mask_developed(
                cost, transform, bounds_wgs84, classes=bp.developed_classes
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "land-cover mask FAILED (%s) — connectors may cross private property; "
                "review outputs/connector_review.png before trusting this run",
                exc,
            )

    pairs = candidate_pairs(edges, nodes, params)
    connectors = least_cost_connectors(
        pairs, nodes, cost, elevation, transform, tobler, params
    )
    connectors.to_parquet(CONNECTORS)
    return connectors


if __name__ == "__main__":
    run()
