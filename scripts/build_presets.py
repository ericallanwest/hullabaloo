"""Solve the race once per pace factor and publish the results to ``docs/data/``.

    pixi run presets                 # every pace factor
    pixi run presets --paces 1.3     # just one, while iterating on the front end

The site is static: the browser picks a pre-solved itinerary, it never solves anything.
So this script is where the left-hand PACE FACTOR control actually gets its answers.

Why a pace sweep is the interesting knob: ``pace_factor`` scales Tobler's speeds, and
Tobler's constants describe unhurried walking. A fit competitor is meaningfully quicker,
but *how much* quicker is the one parameter a racer can neither measure in advance nor
control on the day. Sweeping it shows how much the plan depends on being right about it.

The expensive inputs — the DEM and the network topology — are pace-independent, so they
are computed once and reused across every pace.
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

from hullabaloo import elevation as elev, webexport
from hullabaloo.config import CONFIG, EDGES, EDGES_TIMED, NODES, OUTPUTS
from hullabaloo.graph import build_network
from hullabaloo.optimize_alns import ALNS, build_trail_chains
from hullabaloo.optimize_milp import solve as milp_solve

log = logging.getLogger("hullabaloo.presets")

#: The radio buttons in ``docs/index.html``. 1.0 is textbook Tobler; 2.0 is elite.
PACE_FACTORS = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)


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
):
    """Re-price the network at ``pace`` and return ``(net, route, solver_metadata)``."""
    tobler = dataclasses.replace(CONFIG.tobler, pace_factor=pace)
    timed = elev.price_edges(edges_raw, edge_profiles, tobler)
    net = build_network(timed, nodes)

    best = None
    for seed in range(alns_seeds):
        result = ALNS(net, build_trail_chains(net), seed=seed).solve(iterations=alns_iterations)
        log.info("  pace %.2f | ALNS seed %d: %.3f", pace, seed, result.score)
        if best is None or result.score > best.score:
            best = result

    route = best.route
    meta = {"source": "alns", "status": "heuristic"}

    if milp_seconds > 0:
        milp = milp_solve(net, time_limit_s=milp_seconds, incumbent=best.score, msg=False)
        summary = milp.summary()
        log.info("  pace %.2f | MILP: %s", pace, summary)
        meta = {"source": "alns", **summary}
        # The MILP is here for the proven bound, but it often also finds a strictly
        # better route. Ship it only once the reconstructed walk validates — it comes out
        # of a Hierholzer pass over an arc multiset, not out of the decoder.
        if milp.route is not None and not milp.route.validate():
            if milp.route.evaluate()["score"] > route.evaluate()["score"] + 1e-9:
                route = milp.route
                meta["source"] = "milp"

    problems = route.validate()
    if problems:
        raise SystemExit(f"pace {pace}: optimizer returned an invalid route: {problems}")
    return net, route, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paces", type=float, nargs="*", default=list(PACE_FACTORS))
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

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    started = time.time()

    edges_raw = gpd.read_parquet(EDGES)
    nodes = gpd.read_parquet(NODES)
    array, transform, _, _ = elev.load_dem()

    log.info("sampling elevation profiles once for all %d paces", len(args.paces))
    edge_profiles = elev.edge_profiles(edges_raw, array, transform)

    if not args.skip_check:
        _reprice_check(edges_raw, edge_profiles)

    rows = []
    for pace in args.paces:
        t0 = time.time()
        log.info("=" * 72)
        log.info("PACE FACTOR %.2f", pace)
        net, route, meta = solve_at_pace(
            pace,
            edges_raw,
            edge_profiles,
            nodes,
            alns_iterations=args.alns_iterations,
            alns_seeds=args.alns_seeds,
            milp_seconds=args.milp_seconds,
        )
        meta["wall_seconds"] = round(time.time() - t0, 1)
        # write_preset runs the self-checks; a preset that fails them never lands on disk.
        webexport.write_preset(net, route, pace_factor=pace, solver=meta, directory=args.out)
        summary = route.evaluate()
        rows.append({"pace": pace, **summary, "source": meta["source"]})

        if pace == args.paces[0]:
            # Geometry does not depend on pace, so one backdrop serves every preset.
            webexport.write_network(net, directory=args.out)

    log.info("=" * 72)
    header = f"{'pace':>5} {'score':>7} {'trails':>7} {'unique mi':>10} {'time h':>7}  source"
    log.info(header)
    for row in rows:
        log.info(
            "%5.2f %7.3f %7d %10.3f %7.3f  %s",
            row["pace"], row["score"], row["trails_completed"],
            row["unique_miles"], row["time_h"], row["source"],
        )

    # A faster racer cannot do worse: any dip means a seed got unlucky, not that the
    # model is wrong. Worth seeing rather than silently shipping.
    for previous, current in zip(rows, rows[1:]):
        if current["score"] < previous["score"] - 1e-9:
            log.warning(
                "score fell from %.3f at pace %.2f to %.3f at pace %.2f — a faster racer "
                "should never score less; consider more ALNS seeds or MILP time",
                previous["score"], previous["pace"], current["score"], current["pace"],
            )

    (OUTPUTS / "preset_sweep.json").write_text(
        json.dumps({"elapsed_s": round(time.time() - started, 1), "presets": rows}, indent=2,
                   default=str)
    )
    log.info("done in %.1f min", (time.time() - started) / 60)


if __name__ == "__main__":
    main()
