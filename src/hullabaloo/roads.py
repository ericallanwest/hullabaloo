"""Forest roads — the legal, full-speed way between trails.

The 40 scored trails form three disconnected components, and the first version of this
project bridged them with off-trail bushwhack connectors. That was wrong in an important
way: the gaps are actually spanned by **forest service roads**, which you can walk at full
speed and which are unambiguously legal.

Two sources, and the difference between them matters:

* **USFS EDW road layers** are authoritative for *National Forest System* roads, but they
  are a systems inventory — they omit non-system roads entirely. Queried here, they place
  ``BRUSH MOUNTAIN`` 2.6 km from the nearest trail and leave the whole northern component
  887 m from any road.
* **OpenStreetMap** ``highway=track`` carries the roads people actually walk, including
  the unnamed and non-system ones. It has the road linking Beauty, Crosscut and Highway
  that the USFS layer does not.

So OSM is the primary source and USFS is a cross-check. Roads earn **no points** — they
are not among the 40 scored trails — but they cost full-speed time rather than the 60%
bushwhack penalty.
"""

from __future__ import annotations

import json
import logging

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from .config import CRS_GEOGRAPHIC, CRS_PROJECTED, RAW

log = logging.getLogger(__name__)

OSM_ROADS_PATH = RAW / "osm_roads.geojson"
USFS_ROADS_PATH = RAW / "usfs_roads.geojson"

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

#: Overpass rejects the default ``python-requests/x.y`` user agent outright with a bare
#: ``406 Not Acceptable`` — no explanation, and it looks exactly like a malformed query.
#: Their usage policy asks clients to identify themselves, so this both complies and is
#: the difference between the fetch working and not.
USER_AGENT = "hullabaloo-route-optimizer/0.1 (+https://github.com/ericallanwest/hullabaloo)"

#: Overpass is a free, heavily-shared service: 429 (rate limited) and 504 (gateway timeout)
#: are routine rather than exceptional, so a single attempt per mirror gives up far too
#: easily on a query that works perfectly well a few seconds later.
OVERPASS_ATTEMPTS = 3
OVERPASS_BACKOFF_S = 20


def _overpass(query: str, *, timeout: int = 180) -> dict:
    """POST a query to Overpass, trying each mirror in turn and retrying the flaky ones."""
    import time

    import requests

    last = None
    for attempt in range(OVERPASS_ATTEMPTS):
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                resp = requests.post(
                    endpoint,
                    data=query.encode("utf-8"),
                    headers={"User-Agent": USER_AGENT},
                    timeout=timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                last = exc
                log.warning("Overpass %s failed (attempt %d): %s", endpoint, attempt + 1, exc)
        if attempt < OVERPASS_ATTEMPTS - 1:
            time.sleep(OVERPASS_BACKOFF_S)
    raise RuntimeError(f"all Overpass endpoints failed after {OVERPASS_ATTEMPTS} rounds: {last}")

#: Named ways that are walkable and useful but carry a tag ``highway=track`` misses.
#:
#: The tag is a poor guide to whether a way can be walked here, and every one of these was
#: found by asking why the optimizer still wanted to go off-trail somewhere:
#:
#: * **Meadowbrook Drive** — ``highway=tertiary``, paved, passes within 10 m of Highway.
#: * **Stone Cutter's Hollow Access Road** — ``highway=path``, gravel, reaches Mineral Way
#:   and Wavelength. It sits 9 m from Meadowbrook, so the pair link Highway to that whole
#:   cluster, which is exactly where the two longest bushwhack connectors ran.
#: * **Forest Service Road** — mostly ``highway=track`` and caught by the clause below, but
#:   the stretch through Queen Anne (way 416627180, ``alt_name=Road 708``) is tagged
#:   ``highway=service``. It is unpaved ``tracktype=grade2`` forest road, a sibling of the
#:   FSR 808 already in the network, and Queen Anne meets it in two places. Missing it
#:   accounted for 7 of the 13 remaining bushwhack candidates.
#:
#: Matched on name rather than way id so the query survives OSM re-splitting a way; the
#: ``.`` in "Cutter.s" tolerates either a straight or a typographic apostrophe. Matching by
#: name also means a road keeps arriving whole when only some of its pieces are tagged
#: ``track`` — which is the situation for Forest Service Road.
NAMED_WAYS_PATTERN = (
    "Meadowbrook Drive|Stone ?Cutter.s Hollow Access|Forest Service Road"
)

#: ``highway=track`` is the OSM tag for forest/agricultural roads. Deliberately narrow: a
#: broad name regex times out on the public Overpass instances, but an anchored two-name
#: alternation inside the bounding box is cheap.
OVERPASS_QUERY = """[out:json][timeout:120];
(
  way["highway"="track"]({miny},{minx},{maxy},{maxx});
  way["name"~"{named}",i]({miny},{minx},{maxy},{maxx});
);
out geom;
"""

USFS_ROAD_SERVICE = (
    "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_RoadBasic_01/MapServer/{layer}/query"
)


def fetch_osm_roads(bounds_wgs84, path=OSM_ROADS_PATH, *, force: bool = False):
    """Forest tracks from OpenStreetMap, cached to disk."""
    if path.exists() and not force:
        log.info("using cached OSM roads at %s", path.name)
        return gpd.read_file(path).to_crs(CRS_PROJECTED)

    minx, miny, maxx, maxy = bounds_wgs84
    payload = _overpass(
        OVERPASS_QUERY.format(
            minx=minx, miny=miny, maxx=maxx, maxy=maxy, named=NAMED_WAYS_PATTERN
        )
    )

    records = []
    for element in payload.get("elements", []):
        geometry = element.get("geometry") or []
        if len(geometry) < 2:
            continue
        tags = element.get("tags", {})
        records.append(
            {
                "osm_id": element["id"],
                "name": tags.get("name") or tags.get("ref") or "Forest road",
                "surface": tags.get("surface"),
                "access": tags.get("access"),
                "geometry": LineString([(p["lon"], p["lat"]) for p in geometry]),
            }
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_GEOGRAPHIC)
    gdf.to_file(path, driver="GeoJSON")
    log.info("fetched %d OSM track ways -> %s", len(gdf), path.name)
    return gdf.to_crs(CRS_PROJECTED)


def fetch_usfs_roads(bounds_wgs84, path=USFS_ROADS_PATH, *, force: bool = False):
    """National Forest System roads, as an independent cross-check on OSM."""
    if path.exists() and not force:
        return gpd.read_file(path).to_crs(CRS_PROJECTED)

    import requests

    minx, miny, maxx, maxy = bounds_wgs84
    frames = []
    for layer, motorized in ((0, "open"), (1, "closed-to-motorized")):
        resp = requests.get(
            USFS_ROAD_SERVICE.format(layer=layer),
            params={
                "geometry": json.dumps(
                    {
                        "xmin": minx,
                        "ymin": miny,
                        "xmax": maxx,
                        "ymax": maxy,
                        "spatialReference": {"wkid": 4326},
                    }
                ),
                "geometryType": "esriGeometryEnvelope",
                "inSR": 4326,
                "outSR": 4326,
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "name,id,jurisdiction,seg_length",
                "returnGeometry": "true",
                "f": "geojson",
            },
            timeout=180,
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if features:
            frame = gpd.GeoDataFrame.from_features(features, crs=CRS_GEOGRAPHIC)
            frame["motorized"] = motorized
            frames.append(frame)

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=CRS_PROJECTED)

    import pandas as pd

    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=CRS_GEOGRAPHIC)
    gdf.to_file(path, driver="GeoJSON")
    log.info("fetched %d USFS road segments -> %s", len(gdf), path.name)
    return gdf.to_crs(CRS_PROJECTED)


#: A road may wander further than ``buffer_m`` from the trails in the middle and still be
#: the thing that joins two parts of the network. Interior gaps up to this length are kept
#: rather than cut, which is the difference between a through-route and two stubs.
MAX_ROAD_GAP_M = 800.0


def roads_near_network(
    roads: gpd.GeoDataFrame,
    trails: gpd.GeoDataFrame,
    buffer_m: float = 250.0,
    max_gap_m: float = MAX_ROAD_GAP_M,
) -> gpd.GeoDataFrame:
    """Trim road to the neighbourhood of the trail network without severing it.

    Without any clip the graph acquires miles of road heading off into the forest that no
    route could ever use, which only slows the optimizer down. But clipping each way to
    the buffer and keeping whatever survives is too blunt, and it produced a genuinely
    misleading result: Meadowbrook Drive runs more than 250 m from any trail through its
    middle, so the clip removed that section and left **three disconnected pieces**. The
    optimizer then rediscovered the road's own alignment by bushwhacking 0.38 mi across
    the gap we had just created — an off-trail leg that existed purely as an artefact.

    So the ends are trimmed but the span between retained sections is kept, provided the
    gap is under ``max_gap_m``. Dead-end spurs still get cut, because a spur is retained
    at one end only and has no interior gap to preserve.
    """
    from shapely.geometry import Point
    from shapely.ops import substring

    envelope = trails.to_crs(CRS_PROJECTED).union_all().buffer(buffer_m)
    projected = roads.to_crs(CRS_PROJECTED)

    records, filled_m = [], 0.0
    for _, row in projected.iterrows():
        line = row.geometry
        inside = line.intersection(envelope)
        if inside.is_empty:
            continue
        pieces = list(inside.geoms) if inside.geom_type == "MultiLineString" else [inside]
        pieces = [p for p in pieces if p.geom_type == "LineString" and p.length >= 20]
        if not pieces:
            continue

        # Where each retained piece sits along the original way, as a distance span.
        spans = []
        for piece in pieces:
            ends = (line.project(Point(piece.coords[0])), line.project(Point(piece.coords[-1])))
            spans.append([min(ends), max(ends)])

        merged: list[list[float]] = []
        for lo, hi in sorted(spans):
            if merged and lo - merged[-1][1] <= max_gap_m:
                filled_m += max(0.0, lo - merged[-1][1])
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])

        for lo, hi in merged:
            segment = substring(line, lo, hi)
            if segment.geom_type != "LineString" or segment.length < 20:
                continue
            records.append({**{k: row[k] for k in ("name",) if k in row}, "geometry": segment})

    out = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_PROJECTED)
    if len(out):
        log.info(
            "kept %d road segments, %.2f mi within %.0f m of the trail network "
            "(%.0f m of that bridging interior gaps)",
            len(out),
            out.length.sum() / 1609.344,
            buffer_m,
            filled_m,
        )
    return out


OSM_PATHS_QUERY = """[out:json][timeout:150];
way["highway"~"^(path|footway|track|cycleway|bridleway)$"]({miny},{minx},{maxy},{maxx});
out body geom;
"""


OSM_PATHS_PATH = RAW / "osm_paths.json"


def fetch_osm_junctions(bounds_wgs84, path=OSM_PATHS_PATH, *, force: bool = False):
    """Points where two or more OSM ways share a node.

    OSM is *node-based*: editors connect ways by reusing a node id, so junctions are
    explicit rather than inferred. That makes them an excellent independent check on the
    junctions this project derives by snapping GPX tracks together — a completely
    different method, on completely different source data.

    Note the shared-node set is not a clean junction list: OSM also splits ways where
    tags change (surface, access), so consecutive pieces of one logical trail share a node
    too. Use it to confirm junctions, not to count them.
    """
    from collections import Counter

    from shapely.geometry import Point

    payload = None
    if path.exists() and not force:
        payload = json.loads(path.read_text(encoding="utf-8"))
        log.info("using cached OSM paths at %s", path.name)
    else:
        minx, miny, maxx, maxy = bounds_wgs84
        payload = _overpass(
            OSM_PATHS_QUERY.format(minx=minx, miny=miny, maxx=maxx, maxy=maxy), timeout=200
        )
        path.write_text(json.dumps(payload), encoding="utf-8")

    counts: Counter = Counter()
    positions: dict[int, tuple[float, float]] = {}
    for element in payload.get("elements", []):
        if element.get("type") != "way":
            continue
        geometry = element.get("geometry") or []
        node_ids = element.get("nodes") or []
        if len(geometry) < 2 or len(node_ids) != len(geometry):
            continue
        for node_id, point in zip(node_ids, geometry):
            counts[node_id] += 1
            positions[node_id] = (point["lon"], point["lat"])

    shared = [nid for nid, c in counts.items() if c > 1]
    return gpd.GeoSeries(
        [Point(positions[n]) for n in shared], crs=CRS_GEOGRAPHIC
    ).to_crs(CRS_PROJECTED)


def validate_junctions_against_osm(
    edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, tolerances=(10, 20, 30, 50)
):
    """What fraction of our inferred junctions does OSM independently confirm?"""
    import pandas as pd
    from collections import Counter
    from scipy.spatial import cKDTree

    bounds = tuple(edges.to_crs(CRS_GEOGRAPHIC).total_bounds)
    osm = fetch_osm_junctions(bounds)

    degree = Counter(list(edges["u"]) + list(edges["v"]))
    mine = nodes[nodes["node_id"].map(lambda n: degree[n] >= 3)]

    a = np.array([[g.x, g.y] for g in osm])
    b = np.array([[g.x, g.y] for g in mine.geometry])
    if not len(a) or not len(b):
        return pd.DataFrame()

    dist_to_mine, _ = cKDTree(b).query(a)
    dist_to_osm, _ = cKDTree(a).query(b)

    return pd.DataFrame(
        [
            {
                "tol_m": tol,
                "our_junctions_confirmed_by_osm_pct": round(
                    100 * float((dist_to_osm <= tol).mean()), 1
                ),
                "osm_nodes_matched_by_ours_pct": round(
                    100 * float((dist_to_mine <= tol).mean()), 1
                ),
            }
            for tol in tolerances
        ]
    )


def load_roads(trails: gpd.GeoDataFrame, *, force: bool = False) -> gpd.GeoDataFrame:
    """Roads ready to merge into the network: clipped, projected, named."""
    bounds = tuple(trails.to_crs(CRS_GEOGRAPHIC).total_bounds)
    pad = 0.01
    padded = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)

    osm = fetch_osm_roads(padded, force=force)
    roads = roads_near_network(osm, trails)
    roads["road"] = True
    return roads
