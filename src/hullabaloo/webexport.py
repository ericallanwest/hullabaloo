"""Phase 8 — the optimized route as something you can step through in a browser.

The static site under ``docs/`` reads two kinds of file from ``docs/data/``:

``network.json``      every edge in the routing graph, drawn as the grey backdrop.
``preset_p<NNN>.json``  one solved itinerary, at pace factor ``NNN / 100``.

The design decision worth stating: **everything the browser needs is computed here.**
Each step carries its own geometry, its traversal category, its turn cue and the running
totals as of that step, so the page is a renderer and nothing more. The obvious
alternative — ship bare node pairs and rebuild geometry client-side against a separate
network file — costs a few hundred lines of fragile JavaScript whose failure mode is
silently drawing the wrong line. We own the exporter, so we pay that cost once, in Python,
where it is tested.

The turn cue is the sharpest case for that rule. It has to be measured on the projected
line *before* export simplifies it, so computing it in the browser would not merely be
untidy — it would be measuring the wrong shape.

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
from .export import (
    TURN_GLYPHS,
    _group_arcs,
    classify_turn,
    leg_geometries,
    leg_label,
    route_gdf,
    turn_cues,
)
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

#: 2 added per-step turn cues (``turn``, ``turn_deg``, ``glyph``).
SCHEMA_VERSION = 2


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

    # Turns come off the *unsimplified* projected lines, before `_to_latlng` touches them.
    # Deriving them from the exported coordinates instead would measure a shape that has
    # been Douglas-Peucker'd at 2 m and rounded to a metre, which is several degrees of
    # slack on a 25 m window — enough to call a fork the wrong way.
    leg_lines = leg_geometries(detail, legs)
    geometries = _to_latlng(leg_lines, detail.crs)
    cues = turn_cues(leg_lines)

    steps: list[dict] = []
    cum = {"seconds": 0.0, "miles": 0.0, "unique_miles": 0.0, "repeat_miles": 0.0,
           "offtrail_miles": 0.0}
    for index, (leg, coords, cue) in enumerate(zip(legs, geometries, cues), start=1):
        turn_deg, turn_label, turn_glyph = cue
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
                # Which way to turn onto this leg, and by how much. The glyph ships
                # alongside the label rather than being derived in JavaScript so that the
                # page, its CSV download and outputs/route_cues.csv all print the same
                # arrow for the same junction.
                "turn": turn_label,
                "turn_deg": turn_deg,
                "glyph": turn_glyph,
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

    totals = {
        "score": summary["score"],
        "unique_miles": summary["unique_miles"],
        "trails_completed": summary["trails_completed"],
        "walked_miles": float(summary["walked_miles"]),
        # Recomputed from the walk rather than taken from Route.evaluate(): the step
        # categories fold roads in with bushwhacks and count a re-walked connector as a
        # repeat, so these are the numbers the sidebar's rows actually add up to.
        "offtrail_miles": round(cum["offtrail_miles"], 3),
        "repeat_miles": round(cum["repeat_miles"], 3),
        "time_s": round(float(route.time_s), 1),
        "n_steps": len(steps),
        "feasible": bool(summary["feasible"]),
    }
    race_block = {
        "time_budget_s": float(race.time_budget_s),
        "start_latlng": [
            round(float(start.iloc[0].y), 6),
            round(float(start.iloc[0].x), 6),
        ],
        # Published so the page and check_preset can both see whether a cap binds.
        "max_mile_points": float(race.max_mile_points),
        "max_trail_points": float(race.max_trail_points),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "pace_factor": round(float(pace_factor), 2),
        "race": race_block,
        "totals": totals,
        "optimality": optimality_block(totals, solver or {}, race_block),
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
    """``trail`` (scores), ``road`` (free to walk, worth nothing), or ``offtrail``.

    ``offtrail`` is now only ever the short link from the start line to the network. The
    generated bushwhack connectors that used to share this label are gone — importing the
    roads that were genuinely missing made every one of them unattractive.
    """
    if row.off_trail:
        return "offtrail"
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

    # A binding cap does not disqualify a route — it is still a real walk inside the
    # budget — but it must never be published still claiming to be proven optimal.
    optimality = document.get("optimality")
    if optimality is not None:
        expected = caps_binding(totals, document.get("race"))
        if optimality["caps_binding"] != expected:
            raise ValueError(
                f"optimality.caps_binding {optimality['caps_binding']} disagrees with the "
                f"totals, which give {expected}"
            )
        if expected and optimality["proven"]:
            raise ValueError(
                "preset claims proven optimality while a scoring cap binds — the MILP's "
                "uncapped objective cannot prove optimality under the real scoring rule"
            )

    for step in steps:
        if not step["geometry"]:
            raise ValueError(f"step {step['i']} ({step['name']}) has no geometry")
        check_turn(step)


def check_turn(step: dict) -> None:
    """A published turn cue must be internally consistent.

    The glyph is what a racer actually reads, and it is the one field on a step that can be
    wrong while every number around it still adds up. Asserting that it matches its own
    label — and that both match the angle — is what stops a hand-edited preset shipping an
    arrow that points the wrong way down a fork.
    """
    label, glyph, degrees = step.get("turn"), step.get("glyph"), step.get("turn_deg")

    if label not in TURN_GLYPHS:
        raise ValueError(f"step {step['i']} ({step['name']}) has unknown turn {label!r}")
    if glyph != TURN_GLYPHS[label]:
        raise ValueError(
            f"step {step['i']} ({step['name']}) is labelled {label!r} but carries the "
            f"glyph {glyph!r}, which means {TURN_GLYPHS[label]!r}"
        )

    # The first leg is walked from a standing start, so it has no incoming bearing and no
    # angle. Every other leg must have one, or the cue was never computed.
    if (label == "start") != (degrees is None):
        raise ValueError(
            f"step {step['i']} ({step['name']}) is {label!r} with turn_deg {degrees!r} — "
            "only the opening leg may have no angle"
        )
    if degrees is None:
        return
    if not -180.0 <= degrees <= 180.0:
        raise ValueError(f"step {step['i']} turn_deg {degrees} is outside [-180, 180]")
    if classify_turn(degrees) != label:
        raise ValueError(
            f"step {step['i']} ({step['name']}) says {label!r} but {degrees}° classifies "
            f"as {classify_turn(degrees)!r}"
        )


def caps_binding(totals: dict, race_block: dict | None = None) -> list[str]:
    """Which 40-point scoring caps this route reaches, if any.

    Scoring is ``min(trails, 40) + min(unique_miles, 40)``, but a minimum is not linear,
    so the MILP maximizes the *uncapped* sum instead. That substitution is free only
    while neither cap binds, and the argument is made on the answer rather than in
    advance:

        The uncapped objective U dominates the true score S everywhere. If the returned
        optimum x* has fewer than 40 trails and under 40 unique miles then S(x*) = U(x*),
        and for any other route x, S(x) <= U(x) <= U(x*) = S(x*). So x* is optimal under
        the real scoring too.

    Reach a cap and that chain breaks. The route is still perfectly valid — it is a real
    walk inside the time budget, and it still scores what it scores — but the solver was
    rewarded for mileage the organizer does not pay for, so "proven optimal" would be a
    claim about a different race. Once a cap binds the interesting objective changes
    shape entirely: the fastest tour that still collects the cap, which is a
    minimum-duration problem this model does not express.

    This starts to matter from a pace factor of about 1.53, where the distance ceiling in
    :func:`optimize_milp.check_caps_nonbinding` first exceeds 40 miles.
    """
    # The caps describe the race, not the solve, so a preset written before they were
    # published is still governed by them.
    race_block = race_block or {}
    mile_cap = race_block.get("max_mile_points", CONFIG.race.max_mile_points)
    trail_cap = race_block.get("max_trail_points", CONFIG.race.max_trail_points)

    binding = []
    if totals["unique_miles"] >= mile_cap:
        binding.append(f"unique miles ({totals['unique_miles']:.2f}) reach the {mile_cap:.0f}-point cap")
    if totals["trails_completed"] >= trail_cap:
        binding.append(f"trails completed ({totals['trails_completed']}) reach the {trail_cap:.0f}-point cap")
    return binding


CAPPED_NOTE = (
    "This route reaches a scoring cap, so points stop accruing before the clock runs "
    "out. The real objective past that point is the *fastest* tour that still collects "
    "the cap — a minimum-duration problem this solver does not express, so the "
    "itinerary below is valid and inside the time budget but is not proven optimal."
)


def optimality_block(totals: dict, solver: dict, race_block: dict | None = None) -> dict:
    """Whether this itinerary's optimality claim survives the scoring caps."""
    binding = caps_binding(totals, race_block)
    return {
        "caps_binding": binding,
        "proven": not binding and solver.get("gap_pct") == 0,
        "note": CAPPED_NOTE if binding else None,
    }
