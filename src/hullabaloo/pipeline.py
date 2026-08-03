"""End-to-end pipeline: raw site -> optimized route -> deliverables.

    pixi run python -m hullabaloo.pipeline --help

Every stage is idempotent and skips work whose output already exists, so re-running is
cheap. Use ``--force-<stage>`` to redo one deliberately.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict

from .config import CONFIG, EDGES, EDGES_TIMED, NODES, OUTPUTS, TRAILS_RAW

log = logging.getLogger("hullabaloo.pipeline")


def _stage(name: str):
    log.info("=" * 72)
    log.info("STAGE: %s", name)
    log.info("=" * 72)


def run(
    *,
    alns_iterations: int = 500,
    alns_seeds: int = 4,
    milp_seconds: int = 0,
    force_ingest: bool = False,
    force_topology: bool = False,
    force_elevation: bool = False,
) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    started = time.time()
    # asdict recurses into the nested parameter dataclasses already.
    report: dict = {"config": asdict(CONFIG)}

    from . import elevation, export, ingest, topology
    from .graph import build_network, network_traversal_bound
    from .optimize_alns import ALNS, baseline_best_ratio, baseline_greedy, build_trail_chains

    if force_ingest or not TRAILS_RAW.exists():
        _stage("1 - ingest")
        ingest.run(force=force_ingest)

    if force_topology or not EDGES.exists():
        _stage("2 - topology")
        topology.run()

    if force_elevation or not EDGES_TIMED.exists():
        _stage("3 - elevation and Tobler pricing")
        elevation.run()

    _stage("4 - routing graph")
    net = build_network()
    bound = network_traversal_bound(net)
    report["coverage_bound"] = bound
    log.info("coverage reference: %s", bound)

    _stage("5 - optimization")
    chains = build_trail_chains(net)
    greedy = baseline_greedy(net, chains)
    ratio = baseline_best_ratio(net, chains)
    report["baseline_greedy"] = greedy.evaluate()
    report["baseline_best_ratio"] = ratio.evaluate()
    log.info("baseline greedy    : %s", report["baseline_greedy"])
    log.info("baseline best-ratio: %s", report["baseline_best_ratio"])

    best = None
    for seed in range(alns_seeds):
        t0 = time.time()
        result = ALNS(net, chains, seed=seed).solve(iterations=alns_iterations)
        log.info("ALNS seed %d: %.3f (%.0fs)", seed, result.score, time.time() - t0)
        if best is None or result.score > best.score:
            best = result

    route = best.route
    report["alns"] = route.evaluate()
    report["alns_order"] = best.order
    problems = route.validate()
    report["route_validation"] = problems or "OK"
    log.info("ALNS best: %s", report["alns"])
    log.info("route validation: %s", problems or "OK")
    if problems:
        raise RuntimeError(f"optimizer returned an invalid route: {problems}")

    if milp_seconds > 0:
        _stage("5b - MILP upper bound")
        from .optimize_milp import check_caps_nonbinding, solve as milp_solve

        caps = check_caps_nonbinding(net)
        report["caps"] = caps
        log.info("caps: %s", caps)
        milp = milp_solve(net, time_limit_s=milp_seconds, incumbent=best.score, msg=False)
        report["milp"] = milp.summary()
        log.info("MILP: %s", report["milp"])
        if milp.bound is not None:
            gap = (milp.bound - best.score) / best.score
            report["alns_gap_vs_milp_bound_pct"] = round(100 * gap, 2)
            log.info(
                "ALNS score %.3f is within %.2f%% of the proven upper bound %.3f",
                best.score,
                100 * gap,
                milp.bound,
            )

        # The MILP is here for the bound, but it sometimes also finds a strictly better
        # route than the heuristic. Ship the better of the two — but only after checking
        # the reconstructed walk is genuinely valid, since it comes out of a Hierholzer
        # pass over the solver's arc multiset rather than from the decoder.
        if milp.route is not None:
            milp_problems = milp.route.validate()
            report["milp_route_validation"] = milp_problems or "OK"
            milp_eval = milp.route.evaluate()
            report["milp_route"] = milp_eval
            if not milp_problems and milp_eval["score"] > route.evaluate()["score"] + 1e-9:
                log.info(
                    "MILP route (%.3f) beats the ALNS route (%.3f) — shipping the MILP route",
                    milp_eval["score"],
                    route.evaluate()["score"],
                )
                route = milp.route
                report["final_route_source"] = "milp"
            else:
                report["final_route_source"] = "alns"
                if milp_problems:
                    log.warning("MILP route failed validation (%s); keeping ALNS route", milp_problems)

    report["final_route"] = route.evaluate()

    # Persist the results BEFORE exporting. The optimization is the expensive part — a
    # locked output file must never be able to discard a finished run's numbers.
    def _save_report() -> None:
        report["elapsed_s"] = round(time.time() - started, 1)
        (OUTPUTS / "run_report.json").write_text(json.dumps(report, indent=2, default=str))

    _save_report()
    log.info("saved results to %s", OUTPUTS / "run_report.json")

    _stage("6 - exports")
    try:
        written = export.export_all(net, route)
        report["outputs"] = {k: str(v) for k, v in written.items()}
    except Exception as exc:  # noqa: BLE001
        report["export_error"] = f"{exc.__class__.__name__}: {exc}"
        log.error("export stage failed (%s) — results above are still saved", exc)
    _save_report()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alns-iterations", type=int, default=500)
    parser.add_argument("--alns-seeds", type=int, default=4)
    parser.add_argument(
        "--milp-seconds",
        type=int,
        default=0,
        help="Seconds to spend proving an upper bound (0 = skip the MILP entirely).",
    )
    for stage in ("ingest", "topology", "elevation"):
        parser.add_argument(f"--force-{stage}", action="store_true")
    args = parser.parse_args()

    run(
        alns_iterations=args.alns_iterations,
        alns_seeds=args.alns_seeds,
        milp_seconds=args.milp_seconds,
        force_ingest=args.force_ingest,
        force_topology=args.force_topology,
        force_elevation=args.force_elevation,
    )


if __name__ == "__main__":
    main()
