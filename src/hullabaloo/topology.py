"""Phase 2 — turn 40 independently-recorded GPX tracks into a routable network.

The input is *not* a network. It is 40 separate polylines that happen to overlap. Recon
on the raw data showed:

  * only 20 of 80 endpoints meet another trail's **endpoint** within 10 m, but
    55 of 80 fall within 10 m of another trail's **geometry** — so the network is
    dominated by T-junctions into line interiors. Endpoint snapping alone cannot node it.
  * 10 genuine interior X-crossings need planarizing.
  * the result is 3 disconnected components no matter how far you snap; real off-trail
    connectors (Phase 4) are required to join them.

So the algorithm is: collect split positions along each trail from three sources
(true intersections, foreign-endpoint projections, own endpoints), cut each trail at
those positions, then cluster the resulting edge endpoints into shared nodes.

Every output edge keeps its parent ``trail_id``. That is not cosmetic — "full trail
completed" is defined over the complete edge set of a trail, so the optimizer needs it.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import substring

from .config import (
    CONFIG,
    CRS_GEOGRAPHIC,
    CRS_PROJECTED,
    EDGES,
    M_PER_MILE,
    NODES,
    START_LAT,
    START_LON,
    TRAILS_RAW,
    TopologyParams,
)

log = logging.getLogger(__name__)

DEPOT_NODE_NAME = "START_FINISH"


# --------------------------------------------------------------------------------------
# Union-find, for clustering coincident endpoints into nodes
# --------------------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# --------------------------------------------------------------------------------------
# Split-point collection
# --------------------------------------------------------------------------------------


def _collect_split_distances(
    lines: dict[int, LineString], tol: float, min_edge: float
) -> dict[int, list[float]]:
    """For each trail, the along-line distances at which it must be cut.

    Three sources, all expressed as distances along the trail being cut:

    1. **True intersections** — where two trails actually cross (the 10 X-crossings).
    2. **Foreign endpoint projections** — where another trail's endpoint lands on this
       trail's interior within ``tol``. This is the dominant junction type here.
    3. **Own endpoints** — 0 and length, so every trail is bounded.
    """
    trail_ids = list(lines)
    splits: dict[int, set[float]] = {t: {0.0, lines[t].length} for t in trail_ids}

    # Spatial index so we only test plausibly-nearby pairs.
    index = gpd.GeoSeries(list(lines.values()), index=trail_ids).sindex
    id_by_pos = {i: t for i, t in enumerate(trail_ids)}

    for tid in trail_ids:
        line = lines[tid]
        for pos in index.query(line.buffer(tol), predicate="intersects"):
            other_id = id_by_pos[int(pos)]
            if other_id == tid:
                continue
            other = lines[other_id]

            # (1) true crossings
            if line.intersects(other):
                inter = line.intersection(other)
                for geom in getattr(inter, "geoms", [inter]):
                    if geom.is_empty:
                        continue
                    # A shared overlapping run yields a LineString; cut at both ends.
                    pts = (
                        [Point(geom.coords[0]), Point(geom.coords[-1])]
                        if geom.geom_type == "LineString"
                        else [geom]
                        if geom.geom_type == "Point"
                        else []
                    )
                    for pt in pts:
                        splits[tid].add(line.project(pt))
                        splits[other_id].add(other.project(pt))

            # (2) other trail's endpoints landing on this trail's interior
            for endpoint in (Point(other.coords[0]), Point(other.coords[-1])):
                if line.distance(endpoint) <= tol:
                    splits[tid].add(line.project(endpoint))

    # Collapse split positions that sit closer together than the snap tolerance.
    #
    # This spacing must be >= tol, not merely >= min_edge_len. Two cuts closer together
    # than the snap tolerance produce an edge whose two endpoints then cluster into the
    # *same* node during noding — i.e. a zero-length self-loop. Refusing to create such
    # edges in the first place is cleaner than deleting them afterwards, and it is also
    # self-consistent: two junctions closer than tol are, by our own definition, one
    # junction.
    spacing = max(min_edge, tol)
    cleaned: dict[int, list[float]] = {}
    for tid, values in splits.items():
        length = lines[tid].length
        ordered = sorted(v for v in values if -1e-9 <= v <= length + 1e-9)
        kept: list[float] = []
        for v in ordered:
            v = min(max(v, 0.0), length)
            if not kept or v - kept[-1] >= spacing:
                kept.append(v)
        # Ensure the far end is always present and never leaves a sliver behind.
        if kept[-1] < length:
            if length - kept[-1] < spacing and len(kept) > 1:
                kept[-1] = length
            else:
                kept.append(length)
        cleaned[tid] = kept
    return cleaned


def _split_trails(
    lines: dict[int, LineString],
    meta: dict[int, dict],
    split_distances: dict[int, list[float]],
) -> gpd.GeoDataFrame:
    """Cut each trail at its split distances into ordered edges."""
    records = []
    for tid, cuts in split_distances.items():
        line = lines[tid]
        for seq, (start, end) in enumerate(zip(cuts[:-1], cuts[1:])):
            piece = substring(line, start, end)
            if piece.is_empty or piece.length <= 0:
                continue
            if piece.geom_type != "LineString":
                continue
            records.append(
                {
                    "trail_id": tid,
                    "name": meta[tid]["name"],
                    "seq": seq,
                    "length_m": piece.length,
                    "off_trail": False,
                    "geometry": piece,
                }
            )
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_PROJECTED)
    gdf.insert(0, "edge_id", range(len(gdf)))
    return gdf


# --------------------------------------------------------------------------------------
# Noding
# --------------------------------------------------------------------------------------


def _assign_nodes(edges: gpd.GeoDataFrame, tol: float) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Cluster edge endpoints within ``tol`` into shared nodes, and snap geometry to them.

    Snapping matters: leaving endpoints "close but not equal" produces a network that
    looks connected on a map and silently isn't.
    """
    terminals = np.array(
        [c for geom in edges.geometry for c in (geom.coords[0], geom.coords[-1])]
    )
    tree = cKDTree(terminals)
    uf = _UnionFind(len(terminals))
    for a, b in tree.query_pairs(tol):
        uf.union(a, b)

    labels = np.array([uf.find(i) for i in range(len(terminals))])
    _, node_index = np.unique(labels, return_inverse=True)

    centroids = np.zeros((node_index.max() + 1, 2))
    for node_id in range(node_index.max() + 1):
        centroids[node_id] = terminals[node_index == node_id].mean(axis=0)

    starts = node_index[0::2]
    ends = node_index[1::2]

    snapped = []
    for geom, u, v in zip(edges.geometry, starts, ends):
        coords = list(geom.coords)
        coords[0] = tuple(centroids[u])
        coords[-1] = tuple(centroids[v])
        snapped.append(LineString(coords))

    out = edges.copy()
    out["geometry"] = snapped
    out["u"] = starts
    out["v"] = ends
    out["length_m"] = out.geometry.length

    nodes = gpd.GeoDataFrame(
        {"node_id": range(len(centroids))},
        geometry=[Point(xy) for xy in centroids],
        crs=CRS_PROJECTED,
    )
    return out, nodes


def _drop_degenerate(edges: gpd.GeoDataFrame, tol: float) -> gpd.GeoDataFrame:
    """Remove collapsed edges left over from noding.

    A self-loop shorter than the snap tolerance is an artifact. A self-loop *longer* than
    the tolerance is a genuine lollipop — a trail that returns to its own start — and must
    be kept, because dropping it would silently delete scoreable trail mileage.
    """
    degenerate = (edges["u"] == edges["v"]) & (edges["length_m"] < tol)
    degenerate |= edges["length_m"] < 1e-6
    if degenerate.any():
        lost_m = float(edges.loc[degenerate, "length_m"].sum())
        log.info(
            "dropped %d degenerate edge(s) totalling %.2f m (%.4f%% of network)",
            int(degenerate.sum()),
            lost_m,
            100 * lost_m / max(float(edges["length_m"].sum()), 1e-9),
        )
    return edges.loc[~degenerate].reset_index(drop=True)


def junctions_in_band(
    trails: gpd.GeoDataFrame, low: float, high: float
) -> pd.DataFrame:
    """Endpoint-to-trail contacts whose gap falls in ``[low, high)`` metres.

    These are the junctions that only exist because the snap tolerance is generous.
    Every one deserves a look on satellite imagery before being trusted, since this is
    exactly where a too-loose tolerance invents connections that aren't there.
    """
    projected = trails.to_crs(CRS_PROJECTED)
    lines = {int(r.trail_id): r.geometry for r in projected.itertuples(index=False)}
    names = {int(r.trail_id): r.name for r in projected.itertuples(index=False)}

    rows = []
    for tid, line in lines.items():
        for label, coord in (("start", line.coords[0]), ("end", line.coords[-1])):
            pt = Point(coord)
            for other_id, other in lines.items():
                if other_id == tid:
                    continue
                gap = other.distance(pt)
                if low <= gap < high:
                    rows.append(
                        {
                            "trail": names[tid],
                            "endpoint": label,
                            "gap_m": round(gap, 1),
                            "connects_to": names[other_id],
                            "lat_lon": _to_wgs84(pt),
                        }
                    )
    return pd.DataFrame(rows).sort_values("gap_m").reset_index(drop=True)


def _to_wgs84(point: Point) -> str:
    wgs = gpd.GeoSeries([point], crs=CRS_PROJECTED).to_crs(CRS_GEOGRAPHIC).iloc[0]
    return f"{wgs.y:.5f}, {wgs.x:.5f}"


def _add_depot(
    edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, tol: float
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, int]:
    """Insert the start/finish point, splitting the nearest edge if it lands mid-edge.

    The trailhead sits ~58 m off the Gateway trail, so it needs its own off-trail access
    edge rather than being snapped onto the network and quietly gaining free distance.
    """
    depot_pt = (
        gpd.GeoSeries([Point(START_LON, START_LAT)], crs=CRS_GEOGRAPHIC)
        .to_crs(CRS_PROJECTED)
        .iloc[0]
    )

    distances = edges.geometry.distance(depot_pt)
    nearest_idx = distances.idxmin()
    nearest = edges.loc[nearest_idx]
    along = nearest.geometry.project(depot_pt)
    anchor_pt = nearest.geometry.interpolate(along)

    edges = edges.copy()
    nodes = nodes.copy()

    # If the projection falls at an existing node, reuse it; otherwise split the edge.
    if along <= tol:
        anchor_node = int(nearest["u"])
        anchor_pt = Point(nodes.loc[anchor_node, "geometry"].coords[0])
    elif along >= nearest.geometry.length - tol:
        anchor_node = int(nearest["v"])
        anchor_pt = Point(nodes.loc[anchor_node, "geometry"].coords[0])
    else:
        anchor_node = int(nodes["node_id"].max()) + 1
        first = substring(nearest.geometry, 0, along)
        second = substring(nearest.geometry, along, nearest.geometry.length)
        next_edge_id = int(edges["edge_id"].max()) + 1

        edges = edges.drop(index=nearest_idx)
        for geom, u, v, eid in (
            (first, int(nearest["u"]), anchor_node, next_edge_id),
            (second, anchor_node, int(nearest["v"]), next_edge_id + 1),
        ):
            edges.loc[len(edges)] = {
                "edge_id": eid,
                "trail_id": nearest["trail_id"],
                "name": nearest["name"],
                "seq": nearest["seq"],
                "length_m": geom.length,
                "off_trail": False,
                "geometry": geom,
                "u": u,
                "v": v,
            }
        nodes.loc[len(nodes)] = {"node_id": anchor_node, "geometry": anchor_pt}

    depot_node = int(nodes["node_id"].max()) + 1
    nodes.loc[len(nodes)] = {"node_id": depot_node, "geometry": depot_pt}

    access = LineString([depot_pt, anchor_pt])
    edges.loc[len(edges)] = {
        "edge_id": int(edges["edge_id"].max()) + 1,
        "trail_id": pd.NA,
        "name": "depot access",
        "seq": 0,
        "length_m": access.length,
        "off_trail": True,  # priced at the off-trail speed factor
        "geometry": access,
        "u": depot_node,
        "v": anchor_node,
    }

    # Appending rows via .loc degrades the geometry column to plain object dtype and
    # silently drops the CRS, so rebuild both frames explicitly.
    edges = gpd.GeoDataFrame(
        edges.reset_index(drop=True), geometry="geometry", crs=CRS_PROJECTED
    )
    nodes = gpd.GeoDataFrame(
        nodes.reset_index(drop=True), geometry="geometry", crs=CRS_PROJECTED
    )
    edges["u"] = edges["u"].astype(int)
    edges["v"] = edges["v"].astype(int)
    edges["edge_id"] = edges["edge_id"].astype(int)
    nodes["node_id"] = nodes["node_id"].astype(int)
    log.info(
        "depot node %d added, %.1f m off-trail to node %d (%s)",
        depot_node,
        access.length,
        anchor_node,
        nearest["name"],
    )
    return edges, nodes, depot_node


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def connected_components(edges: pd.DataFrame, n_nodes: int) -> np.ndarray:
    """Component label per node id, using union-find over the edge list."""
    uf = _UnionFind(n_nodes)
    for u, v in zip(edges["u"], edges["v"]):
        uf.union(int(u), int(v))
    labels = np.array([uf.find(i) for i in range(n_nodes)])
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse


def validate(
    edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, raw_length_m: float
) -> pd.DataFrame:
    """Structural QA. Every one of these has caught a real bug during development."""
    degree = pd.concat([edges["u"], edges["v"]]).value_counts()
    comp = connected_components(edges, len(nodes))
    on_trail = edges[~edges["off_trail"]]

    # Each trail's edges should form one contiguous chain.
    broken = []
    for tid, grp in on_trail.groupby("trail_id"):
        sub_nodes = pd.unique(pd.concat([grp["u"], grp["v"]]))
        sub_comp = connected_components(grp, int(max(sub_nodes)) + 1)
        if len({sub_comp[int(n)] for n in sub_nodes}) != 1:
            broken.append(tid)

    short_loops = ((edges["u"] == edges["v"]) & (edges["length_m"] < 50)).sum()
    checks = [
        ("edges", len(edges), ""),
        ("nodes", len(nodes), ""),
        ("trails", on_trail["trail_id"].nunique(), "expect 40"),
        (
            "network components",
            int(comp.max()) + 1,
            "tolerance-dependent; see tolerance_sweep()",
        ),
        ("dangle nodes (degree 1)", int((degree == 1).sum()), ""),
        ("isolated nodes (degree 0)", int(len(nodes) - degree.index.nunique()), "expect 0"),
        ("degenerate self-loops (<50 m)", int(short_loops), "expect 0"),
        ("zero-length edges", int((edges["length_m"] < 1e-6).sum()), "expect 0"),
        ("total on-trail miles", round(on_trail["length_m"].sum() / M_PER_MILE, 3), "expect ~40.13"),
        (
            "length preserved vs raw",
            f"{100 * on_trail['length_m'].sum() / raw_length_m:.3f}%",
            "expect ~100%",
        ),
        ("trails split across components", len(broken), "expect 0"),
        ("median edge length (m)", round(float(edges["length_m"].median()), 1), ""),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "expected"])


def component_summary(edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Which trails sit in which component — the key input to bushwhack planning."""
    comp = connected_components(edges, len(nodes))
    on_trail = edges[~edges["off_trail"]].copy()
    on_trail["component"] = [comp[int(u)] for u in on_trail["u"]]
    return (
        on_trail.groupby("component")
        .agg(
            n_trails=("trail_id", "nunique"),
            miles=("length_m", lambda s: round(s.sum() / M_PER_MILE, 2)),
            trails=("name", lambda s: ", ".join(sorted(set(s)))),
        )
        .sort_values("n_trails", ascending=False)
    )


def tolerance_sweep(
    trails: gpd.GeoDataFrame, tolerances=(2, 5, 10, 15, 20, 30), params: TopologyParams | None = None
) -> pd.DataFrame:
    """Component count vs snap tolerance — justifies the chosen tolerance."""
    params = params or CONFIG.topology
    rows = []
    for tol in tolerances:
        edges, nodes, _ = build_network(trails, replace_tol(params, tol), add_depot=False)
        comp = connected_components(edges, len(nodes))
        rows.append(
            {
                "tol_m": tol,
                "components": int(comp.max()) + 1,
                "edges": len(edges),
                "nodes": len(nodes),
                "dangles": int((pd.concat([edges["u"], edges["v"]]).value_counts() == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def replace_tol(params: TopologyParams, tol: float) -> TopologyParams:
    from dataclasses import replace as _replace

    return _replace(params, snap_tol_m=tol)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def build_network(
    trails: gpd.GeoDataFrame,
    params: TopologyParams | None = None,
    *,
    add_depot: bool = True,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, int | None]:
    params = params or CONFIG.topology
    projected = trails.to_crs(CRS_PROJECTED)

    lines = {
        int(r.trail_id): r.geometry.simplify(params.simplify_tol_m)
        for r in projected.itertuples(index=False)
    }
    meta = {int(r.trail_id): {"name": r.name} for r in projected.itertuples(index=False)}

    splits = _collect_split_distances(lines, params.snap_tol_m, params.min_edge_len_m)
    edges = _split_trails(lines, meta, splits)
    edges, nodes = _assign_nodes(edges, params.snap_tol_m)
    edges = _drop_degenerate(edges, params.snap_tol_m)

    depot_node = None
    if add_depot:
        edges, nodes, depot_node = _add_depot(edges, nodes, params.snap_tol_m)

    return edges, nodes, depot_node


def run(params: TopologyParams | None = None) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    trails = gpd.read_parquet(TRAILS_RAW)
    raw_length_m = float(trails.to_crs(CRS_PROJECTED).length.sum())

    edges, nodes, depot_node = build_network(trails, params)
    nodes["is_depot"] = nodes["node_id"] == depot_node

    edges.to_parquet(EDGES)
    nodes.to_parquet(NODES)

    log.info("validation:\n%s", validate(edges, nodes, raw_length_m).to_string(index=False))
    log.info("components:\n%s", component_summary(edges, nodes).to_string())
    return edges, nodes


if __name__ == "__main__":
    run()
