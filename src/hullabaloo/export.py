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
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, NamedTuple
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
MAP_HTML_PATH = OUTPUTS / "route_map.html"


# --------------------------------------------------------------------------------------
# Route geometry
# --------------------------------------------------------------------------------------


class _EdgeInfo(NamedTuple):
    """Everything the route exporters need to know about one undirected edge."""

    geometry: LineString
    u: int
    length_m: float
    gain_fwd_m: float
    gain_rev_m: float


def _edge_lookup(net: Network) -> dict[int, _EdgeInfo]:
    """``edge_id -> _EdgeInfo``, built once instead of scanning per arc."""

    def _num(row, column: str) -> float:
        # Bushwhack connectors predate the gain columns, so treat a missing or NA
        # value as flat rather than propagating NaN into the exported profile.
        value = getattr(row, column, None)
        return 0.0 if value is None or pd.isna(value) else float(value)

    return {
        int(r.edge_id): _EdgeInfo(
            geometry=r.geometry,
            u=int(r.u),
            length_m=float(r.length_m),
            gain_fwd_m=_num(r, "gain_fwd_m"),
            gain_rev_m=_num(r, "gain_rev_m"),
        )
        for r in net.edges.itertuples(index=False)
    }


def arc_geometry(net: Network, arc, lookup: dict | None = None) -> LineString:
    """Geometry of one arc, oriented in the direction of travel."""
    lookup = lookup if lookup is not None else _edge_lookup(net)
    info = lookup[arc.edge_id]
    geom = info.geometry
    # Edge geometry runs from edge_u to edge_v; flip it when the arc goes the other way.
    if info.u != arc.u:
        geom = LineString(list(geom.coords)[::-1])
    return geom


def arc_relief_m(arc, info: _EdgeInfo) -> tuple[float, float]:
    """``(gain_m, loss_m)`` for one arc, in its direction of travel.

    Descending an edge loses exactly what climbing it the other way gains, so the two
    stored directional gains cover both directions without re-reading the DEM.
    """
    forward = info.u == arc.u
    return (info.gain_fwd_m, info.gain_rev_m) if forward else (info.gain_rev_m, info.gain_fwd_m)


def arc_category(arc, first_visit: bool) -> str:
    """How this traversal counts: ``unique`` (scores), ``repeat``, or ``offtrail``.

    Three kinds of ground exist in this network, not two. Scored trails earn points;
    forest roads and bushwhack connectors both earn nothing and differ only in speed.
    Roads and bushwhacks are therefore folded together as ``offtrail`` — ``Arc.off_trail``
    still distinguishes them for styling and labelling.

    Note this differs slightly from ``Route.evaluate``'s ``offtrail_miles``, which counts
    *every* bushwhack traversal: here a second pass over a connector is a ``repeat``. The
    walk-order definition is the one a racer stepping through the route cares about, and
    ``unique + repeat + offtrail`` still reconciles exactly to the distance walked.
    """
    if not first_visit:
        return "repeat"
    return "unique" if arc.trail_id is not None else "offtrail"


def route_geometry(route: Route) -> LineString:
    """The whole tour as one continuous LineString, in traversal order."""
    lookup = _edge_lookup(route.net)
    coords: list[tuple[float, float]] = []
    for arc in route.arcs:
        piece = list(arc_geometry(route.net, arc, lookup).coords)
        if coords and coords[-1] == piece[0]:
            piece = piece[1:]
        coords.extend(piece)
    return LineString(coords) if len(coords) > 1 else LineString()


def route_gdf(route: Route) -> gpd.GeoDataFrame:
    """One row per arc, with running time and running score — the analytical view."""
    net = route.net
    lookup = _edge_lookup(net)
    rows = []
    elapsed = 0.0
    seen: set[int] = set()
    from .graph import score_edges

    for step, arc in enumerate(route.arcs):
        elapsed += arc.time_s
        info = lookup[arc.edge_id]
        first_visit = arc.edge_id not in seen
        seen.add(arc.edge_id)
        score, miles, trails = score_edges(seen, net, route.race)
        gain_m, loss_m = arc_relief_m(arc, info)
        rows.append(
            {
                "step": step,
                "edge_id": arc.edge_id,
                "trail_id": arc.trail_id,
                "name": arc.name,
                "off_trail": arc.off_trail,
                "first_visit": first_visit,
                "cat": arc_category(arc, first_visit),
                "from_node": arc.u,
                "to_node": arc.v,
                "time_s": round(arc.time_s, 1),
                "elapsed_s": round(elapsed, 1),
                "elapsed_h": round(elapsed / 3600, 3),
                "running_miles": round(miles, 3),
                "running_trails": trails,
                "running_score": round(score, 3),
                "length_m": info.length_m,
                "gain_m": round(gain_m, 2),
                "loss_m": round(loss_m, 2),
                "geometry": arc_geometry(net, arc, lookup),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=net.edges.crs)


# --------------------------------------------------------------------------------------
# Cue sheet
# --------------------------------------------------------------------------------------


def leg_label(row) -> str:
    """How one arc is named on a cue sheet.

    This used to print "BUSHWHACK" for anything off-trail, because generated connectors
    carried machine names like ``bushwhack 12-34`` that meant nothing to a racer. Those
    are gone, and the only generated edge left is the ``Start/Finish`` link from the start
    line to the network — which has a perfectly good name of its own.
    """
    return row.name


def _group_arcs(detail: pd.DataFrame, key_fn: Callable[[object], object]) -> list[dict]:
    """Collapse runs of consecutive arcs sharing a key into legs.

    ``key_fn(row)`` decides what counts as one leg. The cue sheet keys on the printed
    label alone, so a trail walked straight through reads as a single instruction. The
    web export additionally keys on traversal category, because a first pass and an
    immediately following repeat of the same trail must stay separate rows rather than
    merging into one leg that means two different things at once.
    """
    legs: list[dict] = []
    for i, row in enumerate(detail.itertuples(index=False)):
        key = key_fn(row)
        if legs and legs[-1]["key"] == key:
            leg = legs[-1]
            leg["time_s"] += row.time_s
            leg["length_m"] += row.length_m
            leg["gain_m"] += row.gain_m
            leg["loss_m"] += row.loss_m
            leg["n_arcs"] += 1
            # Running totals and the far end are whatever the *last* arc in the leg says.
            leg["elapsed_s"] = row.elapsed_s
            leg["to_node"] = row.to_node
            leg["running_score"] = row.running_score
            leg["running_miles"] = row.running_miles
            leg["running_trails"] = row.running_trails
        else:
            legs.append(
                {
                    "key": key,
                    # Legs are runs of consecutive arcs, so a start index and a count
                    # locate this leg's arcs exactly — which is how the web export
                    # stitches their geometry back together.
                    "arc0": i,
                    "segment": leg_label(row),
                    "name": row.name,
                    "cat": row.cat,
                    "trail_id": row.trail_id,
                    "off_trail": row.off_trail,
                    "from_node": row.from_node,
                    "to_node": row.to_node,
                    "length_m": row.length_m,
                    "gain_m": row.gain_m,
                    "loss_m": row.loss_m,
                    "time_s": row.time_s,
                    "elapsed_s": row.elapsed_s,
                    "running_miles": row.running_miles,
                    "running_trails": row.running_trails,
                    "running_score": row.running_score,
                    "n_arcs": 1,
                }
            )
    return legs


def cue_sheet(route: Route) -> pd.DataFrame:
    """Collapse consecutive arcs on the same trail into one human-readable instruction."""
    detail = route_gdf(route)
    if detail.empty:
        return pd.DataFrame()

    out = pd.DataFrame(_group_arcs(detail, leg_label))
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
    """Write every layer into one OGC GeoPackage.

    Written to a sibling temp file and swapped into place, rather than unlinking the
    target first. On Windows a GeoPackage that any other program has open — marimo, QGIS,
    a stray notebook kernel — cannot be deleted, and the old code's ``path.unlink()``
    raised ``PermissionError`` and took the entire export down with it, discarding a
    finished optimization run. A locked output should cost you that one file, nothing more.
    """
    tmp_path = path.with_suffix(".gpkg.tmp")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            tmp_path = path.with_suffix(f".{os.getpid()}.gpkg.tmp")
    write_path = tmp_path

    edges = net.edges.copy()
    # GeoPackage has no nested types, and pandas NA in an int column upsets the driver.
    edges["trail_id"] = pd.to_numeric(edges["trail_id"], errors="coerce")
    edges.to_file(write_path, layer="edges", driver="GPKG")

    net.nodes.to_file(write_path, layer="nodes", driver="GPKG")

    connectors = edges[edges["off_trail"]]
    if len(connectors):
        connectors.to_file(write_path, layer="connectors", driver="GPKG")

    if trails is None and TRAILS_RAW.exists():
        trails = gpd.read_parquet(TRAILS_RAW)
    if trails is not None:
        # gpx_ele_m is a variable-length list per row — not representable in GPKG.
        trails.drop(columns=[c for c in ("gpx_ele_m",) if c in trails.columns]).to_crs(
            CRS_PROJECTED
        ).to_file(write_path, layer="trails", driver="GPKG")

    depot = net.nodes[net.nodes.get("is_depot", False) == True]  # noqa: E712
    if len(depot):
        depot.to_file(write_path, layer="depot", driver="GPKG")

    if route is not None and route.arcs:
        route_gdf(route).to_file(write_path, layer="route", driver="GPKG")
        gpd.GeoDataFrame(
            [route.evaluate()], geometry=[route_geometry(route)], crs=CRS_PROJECTED
        ).to_file(write_path, layer="route_line", driver="GPKG")

    try:
        os.replace(write_path, path)
    except OSError as exc:
        # The target is open in another program (marimo, QGIS, a notebook kernel). Keep
        # the freshly written file rather than losing it, and say exactly what to do.
        log.error(
            "could not replace %s (%s). The new GeoPackage is complete and saved as %s — "
            "close whatever has %s open and rename it, or delete the old file first.",
            path.name,
            exc.__class__.__name__,
            write_path.name,
            path.name,
        )
        return write_path

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


def write_interactive_map(
    net: Network, route: Route | None = None, path: Path = MAP_HTML_PATH
):
    """Self-contained Leaflet map on USGS topo/imagery basemaps.

    This is the tool for actually checking the model against reality: every bushwhack
    connector can be inspected against satellite imagery to confirm it crosses ground a
    person could plausibly walk, and that it does not cut across the pond.
    """
    import folium

    nodes_wgs = net.nodes.to_crs(CRS_GEOGRAPHIC)
    depot = nodes_wgs[nodes_wgs.get("is_depot", False) == True]  # noqa: E712
    center = (
        [depot.geometry.iloc[0].y, depot.geometry.iloc[0].x]
        if len(depot)
        else [nodes_wgs.geometry.y.mean(), nodes_wgs.geometry.x.mean()]
    )

    fmap = folium.Map(location=center, zoom_start=14, tiles=None)
    folium.TileLayer(
        tiles="https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
        attr="USGS The National Map",
        name="USGS Topo",
    ).add_to(fmap)
    folium.TileLayer(
        tiles="https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}",
        attr="USGS The National Map",
        name="USGS Imagery",
    ).add_to(fmap)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)

    edges_wgs = net.edges.to_crs(CRS_GEOGRAPHIC)
    used_edges = route.edges_used if route is not None else set()

    unused = folium.FeatureGroup(name="trail network (not used)", show=True)
    for row in edges_wgs[~edges_wgs["off_trail"]].itertuples(index=False):
        if row.edge_id in used_edges:
            continue
        folium.PolyLine(
            [(y, x) for x, y in row.geometry.coords],
            color="#8d949e",
            weight=2,
            opacity=0.75,
            tooltip=f"{row.name} (not on route)",
        ).add_to(unused)
    unused.add_to(fmap)

    connectors = folium.FeatureGroup(name="all candidate bushwhacks", show=False)
    for row in edges_wgs[edges_wgs["off_trail"]].itertuples(index=False):
        folium.PolyLine(
            [(y, x) for x, y in row.geometry.coords],
            color="#ff7f0e",
            weight=2,
            dash_array="4,6",
            tooltip=f"{row.name}: {row.length_m:.0f} m",
        ).add_to(connectors)
    connectors.add_to(fmap)

    if route is not None and route.arcs:
        detail = route_gdf(route).to_crs(CRS_GEOGRAPHIC)
        on_route = folium.FeatureGroup(name="route (on trail)", show=True)
        bushwhack = folium.FeatureGroup(name="route (bushwhack)", show=True)
        for row in detail.itertuples(index=False):
            line = folium.PolyLine(
                [(y, x) for x, y in row.geometry.coords],
                color="#d62728" if row.off_trail else "#1f77b4",
                weight=5 if row.off_trail else 4,
                dash_array="6,6" if row.off_trail else None,
                tooltip=(
                    f"step {row.step}: {row.name}<br>"
                    f"{row.elapsed_h:.2f} h elapsed<br>"
                    f"score so far {row.running_score}"
                ),
            )
            line.add_to(bushwhack if row.off_trail else on_route)
        on_route.add_to(fmap)
        bushwhack.add_to(fmap)

    if len(depot):
        folium.Marker(
            center,
            tooltip="START / FINISH",
            icon=folium.Icon(color="green", icon="flag"),
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(str(path))
    log.info("wrote %s", path.name)
    return path


def export_all(net: Network, route: Route | None = None) -> dict[str, Path]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    def _write_cues():
        cues = cue_sheet(route)
        cues.to_csv(CUES_PATH, index=False)
        log.info("cue sheet:\n%s", cues.to_string(index=False))
        return CUES_PATH

    has_route = route is not None and route.arcs
    steps: list[tuple[str, callable]] = [
        ("geopackage", lambda: write_geopackage(net, route)),
        ("map", lambda: plot_route(net, route)),
    ]
    if has_route:
        steps += [("gpx", lambda: write_gpx(route)), ("cues", _write_cues)]
    steps += [
        ("interactive_map", lambda: write_interactive_map(net, route)),
    ]

    # Each artifact is written independently. A locked output file or an unreachable
    # basemap should cost that one artifact, never the whole export — an earlier version
    # let a single PermissionError discard the results of a 20-minute optimization.
    written: dict[str, Path] = {}
    failed: list[str] = []
    for label, fn in steps:
        try:
            result = fn()
            if result is not None:
                written[label] = result
        except Exception as exc:  # noqa: BLE001
            failed.append(label)
            log.error("export step %r failed: %s: %s", label, exc.__class__.__name__, exc)

    if failed:
        log.warning("export finished with %d of %d artifacts: failed = %s",
                    len(written), len(steps), ", ".join(failed))
    return written
