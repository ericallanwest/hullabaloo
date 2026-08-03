"""Phase 8 — the optimized route as something you can step through in a browser.

The static site under ``docs/`` reads two kinds of file from ``docs/data/``:

``network.json``      every edge in the routing graph, drawn as the grey backdrop.
``preset_p<NNN>.json``  one solved itinerary, at pace factor ``NNN / 100``.

The design decision worth stating: **everything the browser needs is computed here.**
Each step carries its own geometry, its traversal category, and the running totals as of
that step, so the page is a renderer and nothing more. The obvious alternative — ship
bare node pairs and rebuild geometry client-side against a separate network file — costs
a few hundred lines of fragile JavaScript whose failure mode is silently drawing the
wrong line. We own the exporter, so we pay that cost once, in Python, where it is tested.

Distances are miles, durations seconds, elevations feet, coordinates ``[lat, lon]`` in
WGS84 — the units the page displays, so the front end never converts anything.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from .config import CONFIG, CRS_GEOGRAPHIC, FT_PER_M, M_PER_MILE, ROOT, RaceParams
from .export import _group_arcs, leg_label, route_gdf
from .graph import Network, Route

log = logging.getLogger(__name__)

WEB_DATA = ROOT / "docs" / "data"

#: Geometry is simplified before export. The stored network is noded at 0.5 m, which is
#: far finer than a screen pixel at any zoom the page offers, and shipping it whole
#: roughly triples the file size for no visible gain.
WEB_SIMPLIFY_M = 2.0

#: ~1.1 m of longitude at this latitude — below the precision of the underlying GPS
#: traces, so rounding here discards noise rather than signal.
COORD_DECIMALS = 5

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------


def _to_latlng(geoms: list[LineString], crs) -> list[list[list[float]]]:
    """Project to WGS84, simplify, and round — as ``[[lat, lon], ...]`` per line.

    Reprojected in one batch rather than per line: pyproj pays a fixed setup cost per
    ``to_crs`` call that dwarfs the transform itself at this scale.
    """
    if not geoms:
        return []
    series = gpd.GeoSeries(geoms, crs=crs).simplify(WEB_SIMPLIFY_M).to_crs(CRS_GEOGRAPHIC)
    return [
        [[round(y, COORD_DECIMALS), round(x, COORD_DECIMALS)] for x, y in geom.coords]
        for geom in series
    ]


def _stitch(geoms) -> LineString:
    """Join consecutive arc geometries into one line, dropping duplicated joints."""
    coords: list[tuple[float, float]] = []
    for geom in geoms:
        part = list(geom.coords)
        if coords and coords[-1] == part[0]:
            part = part[1:]
        coords.extend(part)
    return LineString(coords)


# --------------------------------------------------------------------------------------
# Preset
# --------------------------------------------------------------------------------------


def _trail_names(net: Network) -> dict[int, str]:
    on_trail = net.edges[net.edges["trail_id"].notna()]
    return {
        int(tid): str(grp["name"].iloc[0]) for tid, grp in on_trail.groupby("trail_id")
    }


def _trails_completed(net: Network, route: Route, arc_to_leg: list[int]) -> list[dict]:
    """Every scored trail, with the step at which the route finished it (or ``None``)."""
    names = _trail_names(net)
    miles = {
        int(tid): sum(net.edge_score_mi.get(e, 0.0) for e in edges)
        for tid, edges in net.trail_edges.items()
    }

    completed_at: dict[int, int] = {}
    seen: set[int] = set()
    for i, arc in enumerate(route.arcs):
        seen.add(arc.edge_id)
        for tid, edges in net.trail_edges.items():
            if tid not in completed_at and edges <= seen:
                completed_at[tid] = arc_to_leg[i] + 1  # steps are 1-based on the page

    return sorted(
        (
            {
                "trail_id": int(tid),
                "name": names.get(int(tid), f"trail {tid}"),
                "miles": round(miles.get(int(tid), 0.0), 3),
                "completed_at_step": completed_at.get(int(tid)),
            }
            for tid in net.trail_edges
        ),
        key=lambda t: (t["completed_at_step"] is None, t["completed_at_step"], t["name"]),
    )


def preset_dict(
    net: Network,
    route: Route,
    *,
    pace_factor: float,
    solver: dict | None = None,
    race: RaceParams | None = None,
) -> dict:
    """Build the itinerary document for one solved route.

    ``steps`` are cue-sheet legs rather than raw arcs — consecutive arcs on the same
    trail read as one instruction, which is how the route is actually walked. Legs are
    keyed on ``(label, category)`` so a first pass and an immediately following repeat of
    the same trail stay distinct; keying on the label alone would merge them into one row
    that is half new mileage and half not.
    """
    race = race or route.race or CONFIG.race
    detail = route_gdf(route)
    if detail.empty:
        raise ValueError("cannot export an empty route")

    legs = _group_arcs(detail, lambda row: (leg_label(row), row.cat))

    # Which leg each arc landed in, so trail completions can name a step number.
    arc_to_leg = [0] * len(route.arcs)
    for index, leg in enumerate(legs):
        for offset in range(leg["n_arcs"]):
            arc_to_leg[leg["arc0"] + offset] = index

    geometries = _to_latlng(
        [_stitch(detail.geometry.iloc[leg["arc0"] : leg["arc0"] + leg["n_arcs"]]) for leg in legs],
        detail.crs,
    )

    steps: list[dict] = []
    cum = {"seconds": 0.0, "miles": 0.0, "unique_miles": 0.0, "repeat_miles": 0.0,
           "offtrail_miles": 0.0}
    for index, (leg, coords) in enumerate(zip(legs, geometries), start=1):
        miles = leg["length_m"] / M_PER_MILE
        cum["seconds"] += leg["time_s"]
        cum["miles"] += miles
        cum[f"{leg['cat']}_miles"] += miles
        steps.append(
            {
                "i": index,
                "name": leg["segment"],
                "cat": leg["cat"],
                "bushwhack": bool(leg["off_trail"]),
                "miles": round(miles, 3),
                "seconds": round(float(leg["time_s"]), 1),
                "gain_ft": round(leg["gain_m"] * FT_PER_M),
                "loss_ft": round(leg["loss_m"] * FT_PER_M),
                "from_node": int(leg["from_node"]),
                "to_node": int(leg["to_node"]),
                "cum": {
                    "seconds": round(cum["seconds"], 1),
                    "miles": round(cum["miles"], 3),
                    "unique_miles": round(cum["unique_miles"], 3),
                    "repeat_miles": round(cum["repeat_miles"], 3),
                    "offtrail_miles": round(cum["offtrail_miles"], 3),
                    "trails_completed": int(leg["running_trails"]),
                    "score": round(float(leg["running_score"]), 3),
                },
                "geometry": coords,
            }
        )

    summary = route.evaluate()
    depot = net.nodes.loc[net.nodes["node_id"] == net.depot]
    start = gpd.GeoSeries(depot.geometry.values, crs=net.nodes.crs).to_crs(CRS_GEOGRAPHIC)

    return {
        "schema_version": SCHEMA_VERSION,
        "pace_factor": round(float(pace_factor), 2),
        "race": {
            "time_budget_s": float(race.time_budget_s),
            "start_latlng": [
                round(float(start.iloc[0].y), 6),
                round(float(start.iloc[0].x), 6),
            ],
        },
        "totals": {
            "score": summary["score"],
            "unique_miles": summary["unique_miles"],
            "trails_completed": summary["trails_completed"],
            "walked_miles": float(summary["walked_miles"]),
            # Recomputed from the walk rather than taken from Route.evaluate(): the step
            # categories fold roads in with bushwhacks and count a re-walked connector as
            # a repeat, so these are the numbers the sidebar's rows actually add up to.
            "offtrail_miles": round(cum["offtrail_miles"], 3),
            "repeat_miles": round(cum["repeat_miles"], 3),
            "time_s": round(float(route.time_s), 1),
            "n_steps": len(steps),
            "feasible": bool(summary["feasible"]),
        },
        "network": {
            "total_miles": round(sum(net.edge_score_mi.values()), 2),
            "n_trails": net.n_trails,
        },
        "solver": solver or {},
        "steps": steps,
        "trails": _trails_completed(net, route, arc_to_leg),
    }


def write_preset(
    net: Network,
    route: Route,
    *,
    pace_factor: float,
    solver: dict | None = None,
    directory: Path = WEB_DATA,
) -> Path:
    """Write ``preset_p<NNN>.json`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / preset_filename(pace_factor)
    document = preset_dict(net, route, pace_factor=pace_factor, solver=solver)
    check_preset(document)
    path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    log.info(
        "wrote %s — score %.3f, %d trails, %.2f h, %d KB",
        path.name,
        document["totals"]["score"],
        document["totals"]["trails_completed"],
        document["totals"]["time_s"] / 3600,
        path.stat().st_size // 1024,
    )
    return path


def preset_filename(pace_factor: float) -> str:
    """``1.3 -> 'preset_p130.json'`` — must match ``presetFile()`` in ``docs/js/viz.js``."""
    return f"preset_p{round(pace_factor * 100):03d}.json"


# --------------------------------------------------------------------------------------
# Network backdrop
# --------------------------------------------------------------------------------------


def edge_kind(row) -> str:
    """``trail`` (scores), ``road`` (free to walk, worth nothing), or ``bushwhack``."""
    if row.off_trail:
        return "bushwhack"
    return "trail" if pd.notna(row.trail_id) else "road"


def network_dict(net: Network) -> dict:
    """Every edge in the routing graph, for the grey backdrop layer."""
    edges = net.edges
    geometries = _to_latlng(list(edges.geometry), edges.crs)
    bounds = gpd.GeoSeries(edges.geometry, crs=edges.crs).to_crs(CRS_GEOGRAPHIC).total_bounds

    features = [
        {
            "edge_id": int(row.edge_id),
            "name": str(row.name),
            "kind": edge_kind(row),
            "miles": round(float(row.length_m) / M_PER_MILE, 3),
            "geometry": coords,
        }
        for row, coords in zip(edges.itertuples(index=False), geometries)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "bounds": [
            [round(float(bounds[1]), 6), round(float(bounds[0]), 6)],
            [round(float(bounds[3]), 6), round(float(bounds[2]), 6)],
        ],
        "total_miles": round(sum(net.edge_score_mi.values()), 2),
        "n_trails": net.n_trails,
        "edges": features,
    }


def write_network(net: Network, directory: Path = WEB_DATA) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "network.json"
    path.write_text(json.dumps(network_dict(net), separators=(",", ":")), encoding="utf-8")
    log.info("wrote %s (%d edges, %d KB)", path.name, len(net.edges), path.stat().st_size // 1024)
    return path


# --------------------------------------------------------------------------------------
# Self-checks
# --------------------------------------------------------------------------------------


def check_preset(document: dict, *, tol: float = 0.02) -> None:
    """Fail loudly rather than publish an itinerary whose numbers do not reconcile.

    A preset is a static file a reader will trust without ever running the solver, so
    every arithmetic relationship the page displays is asserted before it ships.
    """
    steps, totals = document["steps"], document["totals"]
    if not steps:
        raise ValueError("preset has no steps")

    last = steps[-1]["cum"]
    parts = last["unique_miles"] + last["repeat_miles"] + last["offtrail_miles"]

    checks = {
        "final cumulative score != totals.score": abs(last["score"] - totals["score"]) > 1e-3,
        "final cumulative time != totals.time_s": abs(last["seconds"] - totals["time_s"]) > 1.0,
        "category miles do not sum to distance walked": abs(parts - totals["walked_miles"]) > tol,
        "cumulative miles != distance walked": abs(last["miles"] - totals["walked_miles"]) > tol,
        "unique miles != scored miles": abs(last["unique_miles"] - totals["unique_miles"]) > tol,
        "final trail count != totals.trails_completed": (
            last["trails_completed"] != totals["trails_completed"]
        ),
        "route exceeds the time budget": last["seconds"] > document["race"]["time_budget_s"] + 1.0,
        "route is marked infeasible": not totals["feasible"],
    }
    failures = [message for message, failed in checks.items() if failed]
    if failures:
        raise ValueError(f"preset failed its self-checks: {'; '.join(failures)}")

    for step in steps:
        if not step["geometry"]:
            raise ValueError(f"step {step['i']} ({step['name']}) has no geometry")
