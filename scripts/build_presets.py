"""Solve the race ahead of time and publish every answer to ``docs/data/``.

    pixi run presets                          # the 18 speed-tier itineraries
    pixi run presets --speeds 6.0             # one tier, all three options
    pixi run presets --speeds 6.0 --options a # one itinerary, while iterating on the page
    pixi run presets --paces 1.0 1.5 2.0      # the legacy pace sweep instead

The site is static: the browser picks a pre-solved itinerary, it never solves anything. So
this script is where every control on the page gets its answers.

Two families, answering two different questions
-----------------------------------------------
**Speed tiers** (``preset_s<NN><option>.json``) are the ones a racer uses. Six top speeds
from 5.0 to 7.5 mph, and at each speed three itineraries that differ in what they are
willing to commit to:

    a   the unconstrained optimum — the honest best answer
    b   the West End must be completed
    c   the West End may not be walked at all

The West End is the only block where the answer genuinely turns on speed. It is a long way
out, so reaching it costs an out-and-back a slow racer cannot afford and a fast one barely
notices — meaning b and c swap places as the expensive constraint somewhere in the middle
of the range. Publishing all three lets a racer see that crossover instead of being handed
one number.

**Pace sweep** (``preset_p<NNN>.json``) is the older family and answers a modelling
question rather than a racing one: how much does the plan depend on being right about your
own speed? That is the one parameter a racer can neither measure in advance nor control on
the day, so it stays published even though the tiers above supersede it for planning.

The expensive inputs — the DEM and the network topology — do not depend on pace, so they
are computed once and reused across every solve.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np

from hullabaloo import adapt, corridors, elevation as elev, webexport
from hullabaloo.config import CONFIG, EDGES, EDGES_TIMED, NODES, OUTPUTS
from hullabaloo.graph import build_network
from hullabaloo.optimize_alns import ALNS, build_trail_chains
from hullabaloo.optimize_milp import solve as milp_solve
from hullabaloo.tobler import pace_for_top_speed_mph
from hullabaloo.webexport import OPTIONS, SPEED_TIERS

log = logging.getLogger("hullabaloo.presets")

#: The legacy pace sweep. 1.0 is textbook Tobler; 2.0 is elite.
PACE_FACTORS = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)

#: The mark the adaptive plan optimises its standing score at — an hour short of the
#: budget. Chosen because that is roughly what being 15% slower than modelled costs you,
#: which is the failure the adaptive plan exists to absorb; optimising at, say, three hours
#: would tune for a scenario in which nothing has gone wrong yet.
FRONT_LOAD_AT_S = 6 * 3600.0


def rule_for(net, option: str):
    """The corridor rule behind each published option.

    ``d`` carries no corridor rule at all — it differs from ``a`` in the order it banks
    points, not in which ground is allowed.
    """
    if option in ("a", "d"):
        return corridors.free_rule()
    if option == "b":
        return corridors.require_rule(net)
    if option == "c":
        return corridors.forbid_rule(net)
    raise ValueError(f"unknown option {option!r}; expected one of {OPTIONS}")


def published_optimum(mph: float, directory: Path) -> float | None:
    """The proven optimum already on disk for this speed, if plan a has been built.

    Lets ``--options d`` rebuild just the adaptive family without re-solving plan a for its
    score alone — a full MILP proof to look up a number that is already published. The
    ceiling has to come from a *proven* solve, so nothing but option a will do.
    """
    path = directory / webexport.preset_speed_filename(mph, "a")
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    return float(document["totals"]["score"])


def solve_adaptive(
    pace: float,
    edges_raw,
    edge_profiles,
    nodes,
    *,
    alns_iterations: int,
    alns_seeds: int,
    optimum: float,
    tag: str,
):
    """Search for the most front-loaded route within the gap budget of ``optimum``.

    No MILP here, deliberately. The MILP has no notion of sequence — it chooses a set of
    arcs and the walk order falls out of a Hierholzer pass afterwards — so it cannot be
    asked for a route that scores *early*. The ALNS can, because its solution representation
    is a visit order. The MILP still sets the ceiling: ``optimum`` comes from plan ``a`` at
    this same speed, so the gap being paid is known exactly rather than estimated.
    """
    tobler = dataclasses.replace(CONFIG.tobler, pace_factor=pace)
    timed = elev.price_edges(edges_raw, edge_profiles, tobler)
    net = build_network(timed, nodes)
    floor = optimum - webexport.ADAPTIVE_GAP_BUDGET

    best = None
    for seed in range(alns_seeds):
        result = ALNS(
            net,
            build_trail_chains(net),
            seed=seed,
            front_load_s=FRONT_LOAD_AT_S,
            score_floor=floor,
        ).solve(iterations=alns_iterations)
        banked = adapt.front_load_score(result.route, FRONT_LOAD_AT_S)
        log.info(
            "  %s | ALNS seed %d: final %.3f, banked by %.1f h %.3f%s",
            tag, seed, result.raw_score, FRONT_LOAD_AT_S / 3600, banked,
            "" if result.raw_score >= floor else "  BELOW FLOOR",
        )
        if result.raw_score < floor:
            continue
        if best is None or banked > best[1]:
            best = (result, banked)

    if best is None:
        raise SystemExit(
            f"{tag}: no seed stayed within {webexport.ADAPTIVE_GAP_BUDGET} points of the "
            f"{optimum:.3f} optimum; raise --alns-iterations or the gap budget"
        )

    result, banked = best
    route = result.route
    problems = route.validate()
    if problems:
        raise SystemExit(f"{tag}: adaptive route is invalid: {problems}")

    meta = {
        "source": "alns-adaptive",
        "status": "heuristic",
        "incumbent": round(result.raw_score, 3),
        "optimum": round(optimum, 3),
        "gap_pct": round(100 * (optimum - result.raw_score) / optimum, 2),
        "banked_by_front_load_h": round(FRONT_LOAD_AT_S / 3600, 2),
        "banked_score": round(banked, 3),
    }
    return net, route, corridors.free_rule(), meta


def _reprice_check(edges_raw, edge_profiles) -> None:
    """Confirm re-pricing at the stored pace reproduces the committed edge table exactly.

    The sweep re-prices every edge from cached DEM profiles rather than re-running the
    elevation stage eleven times. If that re-pricing ever drifts, the sweep is quietly
    solving a different network from the one the repository ships, with nothing to show
    for it — so it is checked against the stored file before any solving starts.
    """
    timed = elev.price_edges(edges_raw, edge_profiles, CONFIG.tobler)
    stored = gpd.read_parquet(EDGES_TIMED)
    for column in ("time_fwd_s", "time_rev_s"):
        drift = np.abs(
            timed[column].to_numpy(dtype=float) - stored[column].to_numpy(dtype=float)
        ).max()
        if drift > 1e-6:
            raise SystemExit(
                f"re-pricing {column} at the stored pace drifted by {drift:.6g}s — the "
                "sweep would be solving a different model than the committed run"
            )
    log.info("re-pricing reproduces the stored edge times exactly")


def solve_at_pace(
    pace: float,
    edges_raw,
    edge_profiles,
    nodes,
    *,
    alns_iterations: int,
    alns_seeds: int,
    milp_seconds: int,
    option: str = "a",
    tag: str | None = None,
):
    """Re-price the network at ``pace`` and return ``(net, route, rule, solver_metadata)``.

    ``option`` selects the corridor rule; ``"a"`` is the unconstrained solve the pace sweep
    has always done, so the legacy family goes through this function unchanged.
    """
    tag = tag or f"pace {pace:.2f}"
    tobler = dataclasses.replace(CONFIG.tobler, pace_factor=pace)
    timed = elev.price_edges(edges_raw, edge_profiles, tobler)
    net = build_network(timed, nodes)
    rule = rule_for(net, option)

    best = None
    for seed in range(alns_seeds):
        result = ALNS(net, build_trail_chains(net), seed=seed, rule=rule).solve(
            iterations=alns_iterations
        )
        log.info(
            "  %s | ALNS seed %d: %.3f%s",
            tag,
            seed,
            result.raw_score,
            "" if result.obeys_rule else f"  REJECTED ({len(result.violations)} violations)",
        )
        # A rule-breaking heuristic route is not a worse answer, it is a different
        # question's answer. Comparing it on score would let it win.
        if not result.obeys_rule:
            continue
        if best is None or result.score > best.score:
            best = result

    route = best.route if best is not None else None
    meta = {"source": "alns", "status": "heuristic"}

    if milp_seconds > 0:
        # Only a rule-respecting incumbent is a valid lower bound for this model. Handing
        # the MILP a score from a route it is not allowed to reproduce would cut off the
        # constrained optimum, or make the model infeasible, and neither failure is loud.
        milp = milp_solve(
            net,
            time_limit_s=milp_seconds,
            incumbent=best.score if best is not None else None,
            msg=False,
            require_trails=rule.require or None,
            forbid_trails=rule.forbid or None,
        )
        summary = milp.summary()
        log.info("  %s | MILP: %s", tag, summary)
        meta = {"source": "alns" if route is not None else "milp", **summary}
        # The MILP is here for the proven bound, but it often also finds a strictly
        # better route. Ship it only once the reconstructed walk validates — it comes out
        # of a Hierholzer pass over an arc multiset, not out of the decoder.
        if milp.route is not None and not milp.route.validate():
            better = (
                route is None
                or milp.route.evaluate()["score"] > route.evaluate()["score"] + 1e-9
            )
            if better:
                route = milp.route
                meta["source"] = "milp"

    if route is None:
        raise SystemExit(
            f"{tag}: neither solver produced a route honouring the rule ({rule.label})"
        )

    problems = route.validate()
    if problems:
        raise SystemExit(f"{tag}: optimizer returned an invalid route: {problems}")

    broken = rule.violations(net, route)
    if broken:
        raise SystemExit(f"{tag}: shipped route breaks its rule: {'; '.join(broken)}")

    return net, route, rule, meta


def build_speed_tiers(speeds, options, ctx, args) -> list[dict]:
    """Solve and publish the speed-tier family. Returns one summary row per itinerary."""
    rows = []
    for mph in speeds:
        pace = pace_for_top_speed_mph(mph)
        # Option a is the free optimum and every other option at this speed is reported
        # relative to it, so it has to be solved first. When a run asks for b or c alone
        # there is nothing to compare against and the delta is simply omitted.
        free_score = None

        for option in sorted(options):
            t0 = time.time()
            tag = f"{mph:.1f} mph / {option}"
            log.info("=" * 72)
            log.info("TOP SPEED %.1f MPH  (pace %.4f)  option %s", mph, pace, option)

            if option == "d":
                if free_score is None:
                    free_score = published_optimum(mph, args.out)
                if free_score is None:
                    raise SystemExit(
                        f"{tag}: the adaptive plan is defined relative to the optimum at "
                        "this speed, so option a must be solved in this run or already "
                        "published"
                    )
                net, route, rule, meta = solve_adaptive(
                    pace,
                    ctx["edges_raw"],
                    ctx["edge_profiles"],
                    ctx["nodes"],
                    alns_iterations=args.alns_iterations,
                    alns_seeds=args.alns_seeds,
                    optimum=free_score,
                    tag=tag,
                )
            else:
                net, route, rule, meta = solve_at_pace(
                    pace,
                    ctx["edges_raw"],
                    ctx["edge_profiles"],
                    ctx["nodes"],
                    alns_iterations=args.alns_iterations,
                    alns_seeds=args.alns_seeds,
                    milp_seconds=args.milp_seconds,
                    option=option,
                    tag=tag,
                )
            meta["wall_seconds"] = round(time.time() - t0, 1)
            summary = route.evaluate()
            if option == "a":
                free_score = summary["score"]

            adaptive = adapt.summary(net, route) if option == "d" else None
            if adaptive is not None:
                log.info(
                    "  %s | %d cuts, %.0f droppable minutes, %.3f banked by %.0f h",
                    tag,
                    len(adaptive["cuts"]),
                    sum(c["minutes_saved"] for c in adaptive["cuts"]),
                    adaptive["front_load"]["score_at_6h"],
                    FRONT_LOAD_AT_S / 3600,
                )

            # write_preset runs the self-checks, including the corridor rule; a preset
            # that fails them never lands on disk.
            webexport.write_preset(
                net,
                route,
                pace_factor=pace,
                solver=meta,
                directory=args.out,
                speed_mph=mph,
                option=option,
                rule=rule,
                free_score=None if option == "a" else free_score,
                adaptive=adaptive,
            )
            rows.append(
                {
                    "speed_mph": mph,
                    "option": option,
                    "rule": rule.label,
                    **summary,
                    "source": meta["source"],
                }
            )

            if not ctx["network_written"]:
                # Geometry does not depend on pace, so one backdrop serves every preset.
                webexport.write_network(net, directory=args.out)
                ctx["network_written"] = True
    return rows


def build_pace_sweep(paces, ctx, args) -> list[dict]:
    """Solve and publish the legacy pace-sweep family."""
    rows = []
    for pace in paces:
        t0 = time.time()
        log.info("=" * 72)
        log.info("PACE FACTOR %.2f", pace)
        net, route, _rule, meta = solve_at_pace(
            pace,
            ctx["edges_raw"],
            ctx["edge_profiles"],
            ctx["nodes"],
            alns_iterations=args.alns_iterations,
            alns_seeds=args.alns_seeds,
            milp_seconds=args.milp_seconds,
        )
        meta["wall_seconds"] = round(time.time() - t0, 1)
        webexport.write_preset(
            net, route, pace_factor=pace, solver=meta, directory=args.out
        )
        rows.append({"pace": pace, **route.evaluate(), "source": meta["source"]})

        if not ctx["network_written"]:
            webexport.write_network(net, directory=args.out)
            ctx["network_written"] = True

    # A faster racer cannot do worse: any dip means a seed got unlucky, not that the model
    # is wrong. Only meaningful within this family — a constrained option legitimately
    # scores below a free one, so the same comparison across the tiers would cry wolf.
    for previous, current in zip(rows, rows[1:]):
        if current["score"] < previous["score"] - 1e-9:
            log.warning(
                "score fell from %.3f at pace %.2f to %.3f at pace %.2f — a faster racer "
                "should never score less; consider more ALNS seeds or MILP time",
                previous["score"], previous["pace"], current["score"], current["pace"],
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--speeds", type=float, nargs="*", default=None,
        help=f"top speeds in mph to publish (default: {' '.join(str(s) for s in SPEED_TIERS)})",
    )
    parser.add_argument(
        "--options", nargs="*", choices=OPTIONS, default=list(OPTIONS),
        help="which alternatives to solve at each speed",
    )
    parser.add_argument(
        "--paces", type=float, nargs="*", default=None,
        help="solve the legacy pace sweep instead; bare flag means all eleven",
    )
    parser.add_argument("--alns-iterations", type=int, default=500)
    parser.add_argument("--alns-seeds", type=int, default=4)
    parser.add_argument("--milp-seconds", type=int, default=300)
    parser.add_argument(
        "--out", type=Path, default=webexport.WEB_DATA, help="where the JSON is written"
    )
    parser.add_argument(
        "--skip-check", action="store_true", help="skip the re-pricing regression check"
    )
    args = parser.parse_args()

    # Bare ``--speeds`` / ``--paces`` mean "all of that family"; naming values means just
    # those. Asking for neither builds the speed tiers, since the pace sweep is the older,
    # slower family and its files are already committed.
    speeds = list(SPEED_TIERS) if args.speeds == [] else (args.speeds or [])
    paces = list(PACE_FACTORS) if args.paces == [] else (args.paces or [])
    if not speeds and not paces:
        speeds = list(SPEED_TIERS)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    started = time.time()

    edges_raw = gpd.read_parquet(EDGES)
    nodes = gpd.read_parquet(NODES)
    array, transform, _, _ = elev.load_dem()

    n_solves = len(speeds) * len(args.options) + len(paces)
    log.info("sampling elevation profiles once for all %d solves", n_solves)
    edge_profiles = elev.edge_profiles(edges_raw, array, transform)

    if not args.skip_check:
        _reprice_check(edges_raw, edge_profiles)

    ctx = {
        "edges_raw": edges_raw,
        "edge_profiles": edge_profiles,
        "nodes": nodes,
        "network_written": False,
    }

    speed_rows = build_speed_tiers(speeds, args.options, ctx, args)
    pace_rows = build_pace_sweep(paces, ctx, args)

    # Built last, by scanning the directory, so it indexes exactly what is on disk —
    # including presets left over from earlier partial runs.
    webexport.write_manifest(args.out)

    log.info("=" * 72)
    if speed_rows:
        log.info(
            f"{'mph':>5} {'opt':>4} {'score':>7} {'trails':>7} {'unique mi':>10} "
            f"{'time h':>7}  rule"
        )
        for row in speed_rows:
            log.info(
                "%5.1f %4s %7.3f %7d %10.3f %7.3f  %s",
                row["speed_mph"], row["option"], row["score"], row["trails_completed"],
                row["unique_miles"], row["time_h"], row["rule"],
            )
    if pace_rows:
        log.info(f"{'pace':>5} {'score':>7} {'trails':>7} {'unique mi':>10} {'time h':>7}  source")
        for row in pace_rows:
            log.info(
                "%5.2f %7.3f %7d %10.3f %7.3f  %s",
                row["pace"], row["score"], row["trails_completed"],
                row["unique_miles"], row["time_h"], row["source"],
            )

    (OUTPUTS / "preset_sweep.json").write_text(
        json.dumps(
            {
                "elapsed_s": round(time.time() - started, 1),
                "speed_tiers": speed_rows,
                "presets": pace_rows,
            },
            indent=2,
            default=str,
        )
    )
    log.info("done in %.1f min", (time.time() - started) / 60)


if __name__ == "__main__":
    main()
