"""Phase 7 — turn the optimized route into things you can actually use.

Outputs
-------
``hullabaloo.gpkg``   one multi-layer GeoPackage: edges, nodes, trails, connectors,
                      route, depot. This is the "single topologically sound geospatial
                      file" the project set out to produce.
``route.gpx``         the tour as a GPX track, loadable onto a watch or phone.
``route_cues.csv``    turn-by-turn cue sheet with running time and running score.
``route_map.png``     static overview for the README.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point

from .config import CONFIG, CRS_GEOGRAPHIC, CRS_PROJECTED, OUTPUTS, TRAILS_RAW
from .graph import Network, Route

log = logging.getLogger(__name__)

GPKG_PATH = OUTPUTS / "hullabaloo.gpkg"
GPX_PATH = OUTPUTS / "route.gpx"
CUES_PATH = OUTPUTS / "route_cues.csv"
MAP_PATH = OUTPUTS / "route_map.png"


# --------------------------------------------------------------------------------------
# Route geometry
# --------------------------------------------------------------------------------------


def arc_geometry(net: Network, arc) -> LineString:
    """Geometry of one arc, oriented in the direction of travel."""
    row = net.edges.loc[net.edges["edge_id"] == arc.edge_id].iloc[0]
    geom = row.geometry
    # Edge geometry runs from row.u to row.v; flip it when the arc goes the other way.
    if int(row["u"]) != arc.u:
        geom = LineString(list(geom.coords)[::-1])
    return geom


def route_geometry(route: Route) -> LineString:
    """The whole tour as one continuous LineString, in traversal order."""
    coords: list[tuple[float, float]] = []
    for arc in route.arcs:
        piece = list(arc_geometry(route.net, arc).coords)
        if coords and coords[-1] == piece[0]:
            piece = piece[1:]
        coords.extend(piece)
    return LineString(coords) if len(coords) > 1 else LineString()


def route_gdf(route: Route) -> gpd.GeoDataFrame:
    """One row per arc, with running time and running score — the analytical view."""
    net = route.net
    rows = []
    elapsed = 0.0
    seen: set[int] = set()
    from .graph import score_edges

    for step, arc in enumerate(route.arcs):
        elapsed += arc.time_s
        seen.add(arc.edge_id)
        score, miles, trails = score_edges(seen, net, route.race)
        rows.append(
            {
                "step": step,
                "edge_id": arc.edge_id,
                "trail_id": arc.trail_id,
                "name": arc.name,
                "off_trail": arc.off_trail,
                "from_node": arc.u,
                "to_node": arc.v,
                "time_s": round(arc.time_s, 1),
                "elapsed_s": round(elapsed, 1),
                "elapsed_h": round(elapsed / 3600, 3),
                "running_miles": round(miles, 3),
                "running_trails": trails,
                "running_score": round(score, 3),
                "geometry": arc_geometry(net, arc),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=net.edges.crs)


# --------------------------------------------------------------------------------------
# Cue sheet
# --------------------------------------------------------------------------------------


def cue_sheet(route: Route) -> pd.DataFrame:
    """Collapse consecutive arcs on the same trail into one human-readable instruction."""
    detail = route_gdf(route)
    if detail.empty:
        return pd.DataFrame()

    groups: list[dict] = []
    for row in detail.itertuples(index=False):
        label = "BUSHWHACK" if row.off_trail else row.name
        if groups and groups[-1]["segment"] == label:
            leg = groups[-1]
            leg["time_s"] += row.time_s
            leg["elapsed_s"] = row.elapsed_s
            leg["to_node"] = row.to_node
            leg["running_score"] = row.running_score
            leg["running_miles"] = row.running_miles
            leg["running_trails"] = row.running_trails
            leg["length_m"] += route.net.edges.set_index("edge_id")["length_m"].get(
                row.edge_id, 0.0
            )
        else:
            groups.append(
                {
                    "segment": label,
                    "off_trail": row.off_trail,
                    "from_node": row.from_node,
                    "to_node": row.to_node,
                    "length_m": route.net.edges.set_index("edge_id")["length_m"].get(
                        row.edge_id, 0.0
                    ),
                    "time_s": row.time_s,
                    "elapsed_s": row.elapsed_s,
                    "running_miles": row.running_miles,
                    "running_trails": row.running_trails,
                    "running_score": row.running_score,
                }
            )

    out = pd.DataFrame(groups)
    out.insert(0, "leg", range(1, len(out) + 1))
    out["miles"] = (out["length_m"] / 1609.344).round(2)
    out["leg_min"] = (out["time_s"] / 60).round(1)
    out["elapsed"] = out["elapsed_s"].apply(
        lambda s: f"{int(s // 3600)}:{int((s % 3600) // 60):02d}"
    )
    return out[
        [
            "leg",
            "segment",
            "off_trail",
            "miles",
            "leg_min",
            "elapsed",
            "running_miles",
            "running_trails",
            "running_score",
        ]
    ]


# --------------------------------------------------------------------------------------
# GPX
# --------------------------------------------------------------------------------------


def write_gpx(route: Route, path: Path = GPX_PATH, name: str = "Hullabaloo optimized route"):
    """Write the tour as a GPX 1.1 track.

    Timestamps are synthesized from each arc's modelled duration, so a GPS app will show
    the predicted schedule — which is far more useful in the field than bare geometry.
    """
    geom_wgs = (
        gpd.GeoSeries([route_geometry(route)], crs=CRS_PROJECTED).to_crs(CRS_GEOGRAPHIC).iloc[0]
    )

    gpx = ET.Element(
        "gpx",
        {
            "version": "1.1",
            "creator": "hullabaloo-route-optimizer",
            "xmlns": "http://www.topografix.com/GPX/1/1",
        },
    )
    meta = ET.SubElement(gpx, "metadata")
    ET.SubElement(meta, "name").text = name
    summary = route.evaluate()
    ET.SubElement(meta, "desc").text = (
        f"score {summary['score']} = {summary['trails_completed']} trails "
        f"+ {summary['unique_miles']} unique miles, {summary['time_h']} h"
    )

    trk = ET.SubElement(gpx, "trk")
    ET.SubElement(trk, "name").text = name
    seg = ET.SubElement(trk, "trkseg")

    coords = list(geom_wgs.coords)
    start = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    total_s = route.time_s
    for i, (lon, lat) in enumerate(coords):
        pt = ET.SubElement(seg, "trkpt", {"lat": f"{lat:.7f}", "lon": f"{lon:.7f}"})
        fraction = i / max(len(coords) - 1, 1)
        stamp = start + timedelta(seconds=fraction * total_s)
        ET.SubElement(pt, "time").text = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    ET.ElementTree(gpx).write(path, encoding="UTF-8", xml_declaration=True)
    log.info("wrote %s (%d points)", path.name, len(coords))
    return path


# --------------------------------------------------------------------------------------
# GeoPackage
# --------------------------------------------------------------------------------------


def write_geopackage(
    net: Network,
    route: Route | None = None,
    path: Path = GPKG_PATH,
    trails: gpd.GeoDataFrame | None = None,
):
    """Write every layer into one OGC GeoPackage."""
    if path.exists():
        path.unlink()

    edges = net.edges.copy()
    # GeoPackage has no nested types, and pandas NA in an int column upsets the driver.
    edges["trail_id"] = pd.to_numeric(edges["trail_id"], errors="coerce")
    edges.to_file(path, layer="edges", driver="GPKG")

    net.nodes.to_file(path, layer="nodes", driver="GPKG")

    connectors = edges[edges["off_trail"]]
    if len(connectors):
        connectors.to_file(path, layer="connectors", driver="GPKG")

    if trails is None and TRAILS_RAW.exists():
        trails = gpd.read_parquet(TRAILS_RAW)
    if trails is not None:
        # gpx_ele_m is a variable-length list per row — not representable in GPKG.
        trails.drop(columns=[c for c in ("gpx_ele_m",) if c in trails.columns]).to_crs(
            CRS_PROJECTED
        ).to_file(path, layer="trails", driver="GPKG")

    depot = net.nodes[net.nodes.get("is_depot", False) == True]  # noqa: E712
    if len(depot):
        depot.to_file(path, layer="depot", driver="GPKG")

    if route is not None and route.arcs:
        route_gdf(route).to_file(path, layer="route", driver="GPKG")
        gpd.GeoDataFrame(
            [route.evaluate()], geometry=[route_geometry(route)], crs=CRS_PROJECTED
        ).to_file(path, layer="route_line", driver="GPKG")

    log.info("wrote %s", path.name)
    return path


# --------------------------------------------------------------------------------------
# Map
# --------------------------------------------------------------------------------------


def plot_route(net: Network, route: Route | None, path: Path = MAP_PATH, dpi: int = 150):
    """Static overview map: whole network, the route, connectors used, and the start."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 9))
    on_trail = net.edges[~net.edges["off_trail"]]
    on_trail.plot(ax=ax, color="#c9ccd1", linewidth=1.1, zorder=1)

    if route is not None and route.arcs:
        used = route_gdf(route)
        used[~used["off_trail"]].plot(ax=ax, color="#1f77b4", linewidth=2.6, zorder=3)
        bush = used[used["off_trail"]]
        if len(bush):
            bush.plot(ax=ax, color="#d62728", linewidth=2.2, linestyle="--", zorder=4)

    depot = net.nodes[net.nodes.get("is_depot", False) == True]  # noqa: E712
    if len(depot):
        depot.plot(ax=ax, color="#2ca02c", markersize=140, marker="*", zorder=5)

    summary = route.evaluate() if route is not None else {}
    title = "Hullabaloo — optimized 7-hour route"
    if summary:
        title += (
            f"\nscore {summary['score']}  =  {summary['trails_completed']} trails"
            f"  +  {summary['unique_miles']} unique mi     ({summary['time_h']} h)"
        )
    ax.set_title(title, fontsize=13)
    ax.set_axis_off()

    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([], [], color="#c9ccd1", lw=2, label="trail network (unused)"),
            Line2D([], [], color="#1f77b4", lw=3, label="route on trail"),
            Line2D([], [], color="#d62728", lw=3, ls="--", label="bushwhack connector"),
            Line2D([], [], color="#2ca02c", marker="*", ls="", ms=14, label="start / finish"),
        ],
        loc="lower right",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path.name)
    return path


def export_all(net: Network, route: Route | None = None) -> dict[str, Path]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    written = {"geopackage": write_geopackage(net, route)}
    if route is not None and route.arcs:
        written["gpx"] = write_gpx(route)
        cues = cue_sheet(route)
        cues.to_csv(CUES_PATH, index=False)
        written["cues"] = CUES_PATH
        log.info("cue sheet:\n%s", cues.to_string(index=False))
    written["map"] = plot_route(net, route)
    return written
