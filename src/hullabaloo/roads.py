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

#: ``highway=track`` is the OSM tag for forest/agricultural roads. Deliberately a narrow
#: query: a broader one with name regexes times out on the public Overpass instances.
OVERPASS_QUERY = """[out:json][timeout:120];
way["highway"="track"]({miny},{minx},{maxy},{maxx});
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

    import requests

    minx, miny, maxx, maxy = bounds_wgs84
    query = OVERPASS_QUERY.format(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    payload = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            resp = requests.post(endpoint, data=query.encode("utf-8"), timeout=180)
            resp.raise_for_status()
            payload = resp.json()
            break
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            log.warning("Overpass endpoint %s failed: %s", endpoint, exc)
    if payload is None:
        raise RuntimeError("all Overpass endpoints failed")

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


def roads_near_network(
    roads: gpd.GeoDataFrame, trails: gpd.GeoDataFrame, buffer_m: float = 250.0
) -> gpd.GeoDataFrame:
    """Keep only road within ``buffer_m`` of the trail network, clipped to that envelope.

    Without this the graph acquires miles of road heading off into the forest that no
    route could ever use, which only slows the optimizer down.
    """
    envelope = trails.to_crs(CRS_PROJECTED).union_all().buffer(buffer_m)
    clipped = roads.to_crs(CRS_PROJECTED).intersection(envelope)

    records = []
    for (_, row), geom in zip(roads.iterrows(), clipped):
        if geom.is_empty:
            continue
        pieces = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for piece in pieces:
            if piece.geom_type != "LineString" or piece.length < 20:
                continue
            records.append({**{k: row[k] for k in ("name",) if k in row}, "geometry": piece})

    out = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_PROJECTED)
    if len(out):
        log.info(
            "kept %d road segments, %.2f mi within %.0f m of the trail network",
            len(out),
            out.length.sum() / 1609.344,
            buffer_m,
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

    import requests
    from shapely.geometry import Point

    payload = None
    if path.exists() and not force:
        payload = json.loads(path.read_text(encoding="utf-8"))
        log.info("using cached OSM paths at %s", path.name)
    else:
        minx, miny, maxx, maxy = bounds_wgs84
        query = OSM_PATHS_QUERY.format(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                resp = requests.post(endpoint, data=query.encode("utf-8"), timeout=200)
                resp.raise_for_status()
                payload = resp.json()
                path.write_text(json.dumps(payload), encoding="utf-8")
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("Overpass endpoint %s failed: %s", endpoint, exc)
    if payload is None:
        raise RuntimeError("all Overpass endpoints failed and no cache is present")

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
