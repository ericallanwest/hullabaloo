"""Phase 1 — ingest the official trail inventory and GPX tracks.

The Trail Not Taken is a small Express app behind a form login. The flow is:

    POST /login  (username, password)      -> connect.sid session cookie
    GET  /displayTrailList                 -> HTML table of every trail
    GET  /downloadTrailGPX?trailid=<id>    -> GPX for one trail

GPX quirk worth remembering: ``<ele>`` values are **metres**, while the HTML table
reports elevation gain in **feet**. Mixing them silently produces a 3.28x error.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import geopandas as gpd
import pandas as pd
import requests
from dotenv import load_dotenv
from shapely.geometry import LineString

from .config import (
    CONFIG,
    CRS_GEOGRAPHIC,
    GPX_DIR,
    GPX_URL,
    LOGIN_URL,
    M_PER_MILE,
    TRAIL_TABLE,
    TRAILS_RAW,
    TRAIL_LIST_URL,
)

log = logging.getLogger(__name__)

GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/1"}

#: One row of the trail table. Matched against the live HTML as of 2026-07.
_ROW_RE = re.compile(
    r"<tr>\s*"
    r"<td align=left>(?P<system>.*?)</td>\s*"
    r"<td align=left><b><a href='/trailInfo\?trailid=(?P<trail_id>\d+)'>(?P<name>.*?)</a></b></td>\s*"
    r"<td align=center>(?P<dist_mi>[\d.]+)</td>\s*"
    r"<td align=center>(?P<gain_ft>-?\d+)</td>\s*"
    r"<td align=center>(?P<rev_gain_ft>-?\d+)</td>\s*"
    r"<td align=center>(?P<dist_th_mi>-?[\d.]+)</td>",
    re.S,
)

EXPECTED_TRAIL_COUNT = 40


class IngestError(RuntimeError):
    pass


def login(user: str | None = None, password: str | None = None) -> requests.Session:
    """Authenticate and return a session carrying the ``connect.sid`` cookie."""
    load_dotenv()
    user = user or os.environ.get("TNT_USER")
    password = password or os.environ.get("TNT_PASS")
    if not user or not password:
        raise IngestError(
            "Set TNT_USER and TNT_PASS in .env (copy .env.example) or pass them explicitly."
        )

    session = requests.Session()
    session.headers["User-Agent"] = "hullabaloo-route-optimizer/0.1 (personal event planning)"
    resp = session.post(LOGIN_URL, data={"username": user, "password": password}, timeout=30)
    resp.raise_for_status()

    if "connect.sid" not in session.cookies:
        raise IngestError("Login did not return a session cookie — check credentials.")
    log.info("Authenticated as %s", user)
    return session


def fetch_trail_table(session: requests.Session) -> pd.DataFrame:
    """Scrape ``/displayTrailList`` into a tidy frame of trail metadata."""
    resp = session.get(TRAIL_LIST_URL, timeout=60)
    resp.raise_for_status()
    rows = [m.groupdict() for m in _ROW_RE.finditer(resp.text)]
    if not rows:
        raise IngestError(
            "Parsed zero trails — the page markup has probably changed, or the session "
            "was rejected and we were served the login page."
        )

    df = pd.DataFrame(rows)
    df["trail_id"] = df["trail_id"].astype(int)
    for col in ("dist_mi", "gain_ft", "rev_gain_ft", "dist_th_mi"):
        df[col] = pd.to_numeric(df[col])
    df["name"] = df["name"].str.strip()
    df["system"] = df["system"].str.strip()

    if len(df) != EXPECTED_TRAIL_COUNT:
        log.warning("Expected %d trails, found %d", EXPECTED_TRAIL_COUNT, len(df))
    if df["trail_id"].duplicated().any():
        raise IngestError("Duplicate trail_id values in the trail table.")

    return df.sort_values("trail_id").reset_index(drop=True)


def download_gpx(
    session: requests.Session,
    trail_ids: list[int],
    dest: Path = GPX_DIR,
    *,
    force: bool = False,
    delay_s: float = 0.3,
) -> dict[int, Path]:
    """Download each trail's GPX. Idempotent — existing non-trivial files are kept."""
    dest.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for trail_id in trail_ids:
        path = dest / f"{trail_id}.gpx"
        if path.exists() and path.stat().st_size > 200 and not force:
            paths[trail_id] = path
            continue
        resp = session.get(GPX_URL.format(trail_id=trail_id), timeout=120)
        resp.raise_for_status()
        if b"<trkpt" not in resp.content:
            raise IngestError(f"trail_id={trail_id} returned no track points.")
        path.write_bytes(resp.content)
        paths[trail_id] = path
        log.info("downloaded trail %s (%d bytes)", trail_id, len(resp.content))
        time.sleep(delay_s)  # be polite to a small hobby server
    return paths


def parse_gpx(path: Path) -> tuple[LineString, list[float]]:
    """Return the track geometry (lon/lat) and its per-vertex elevations in metres.

    Multiple ``<trkseg>`` elements are concatenated; the current dataset has exactly one
    per file, but concatenating keeps us safe if that changes.
    """
    root = ET.parse(path).getroot()
    coords: list[tuple[float, float]] = []
    elevations: list[float] = []
    for pt in root.findall(".//gpx:trkpt", GPX_NS):
        lon, lat = float(pt.attrib["lon"]), float(pt.attrib["lat"])
        ele_el = pt.find("gpx:ele", GPX_NS)
        # Drop consecutive duplicate positions — they add nothing and break simplify().
        if coords and coords[-1] == (lon, lat):
            continue
        coords.append((lon, lat))
        elevations.append(float(ele_el.text) if ele_el is not None else float("nan"))

    if len(coords) < 2:
        raise IngestError(f"{path.name} has fewer than 2 distinct track points.")
    return LineString(coords), elevations


def build_trails_gdf(table: pd.DataFrame, gpx_paths: dict[int, Path]) -> gpd.GeoDataFrame:
    """Join the scraped metadata to the parsed GPX geometry."""
    records = []
    for row in table.itertuples(index=False):
        geom, elevations = parse_gpx(gpx_paths[row.trail_id])
        records.append(
            {
                "trail_id": row.trail_id,
                "name": row.name,
                "system": row.system,
                "official_dist_mi": row.dist_mi,
                "official_gain_ft": row.gain_ft,
                "official_rev_gain_ft": row.rev_gain_ft,
                "dist_to_th_mi": row.dist_th_mi,
                "n_points": len(geom.coords),
                # Retained purely for QA against the DEM in Phase 3.
                "gpx_ele_m": elevations,
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_GEOGRAPHIC)


def validate_trails(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compare computed geodesic length against the site's published distance."""
    geod = gdf.geometry.to_crs("EPSG:6346").length
    out = pd.DataFrame(
        {
            "trail_id": gdf["trail_id"],
            "name": gdf["name"],
            "computed_mi": (geod / M_PER_MILE).round(3),
            "official_mi": gdf["official_dist_mi"],
        }
    )
    out["delta_mi"] = (out["computed_mi"] - out["official_mi"]).round(3)
    out["pct_diff"] = (100 * out["delta_mi"] / out["official_mi"].clip(lower=0.05)).round(1)
    return out


def run(force: bool = False) -> gpd.GeoDataFrame:
    """Execute the full ingest phase and persist the result."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    session = login()
    table = fetch_trail_table(session)
    table.to_csv(TRAIL_TABLE, index=False)
    paths = download_gpx(session, table["trail_id"].tolist(), force=force)
    gdf = build_trails_gdf(table, paths)

    # gpx_ele_m is a variable-length list per row; parquet handles it, but keep it out of
    # any downstream geospatial format that would choke on a nested type.
    gdf.to_parquet(TRAILS_RAW)

    report = validate_trails(gdf)
    log.info(
        "ingested %d trails | computed %.2f mi vs official %.2f mi",
        len(gdf),
        report["computed_mi"].sum(),
        report["official_mi"].sum(),
    )
    worst = report.reindex(report["pct_diff"].abs().sort_values(ascending=False).index).head(5)
    log.info("largest length discrepancies:\n%s", worst.to_string(index=False))
    return gdf


if __name__ == "__main__":
    run()
