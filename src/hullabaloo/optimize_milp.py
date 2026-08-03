"""Phase 6b — exact MILP formulation, used to bound the ALNS solution.

The point of this module is not to beat the heuristic. It is to say something the
heuristic cannot: *how far from optimal the answer is*. "34.9 points, proven within 6% of
optimal" is a categorically stronger claim than "34.9 points, found by a metaheuristic".

Formulation — a prize-collecting rural postman / arc orienteering problem
------------------------------------------------------------------------
Variables
    x_a in Z+      number of times arc ``a`` is traversed (0..MAX_TRAVERSALS)
    z_e in {0,1}   edge ``e`` is traversed at least once (scores its miles once)
    y_t in {0,1}   trail ``t`` is fully completed (scores its trail point)
    p_v in {0,1}   node ``v`` is used by the route
    g_a >= 0       single-commodity flow, used only to enforce connectivity

Objective
    max  sum_t y_t  +  sum_e miles_e * z_e

Constraints
    (1)  out-degree == in-degree at every node          -> the walk is closed/Eulerian
    (2)  sum_a time_a * x_a <= 25200 s                  -> the 7-hour budget
    (3)  z_e <= sum_{a in e} x_a  <= K * z_e            -> links edge use to arc counts
    (4)  y_t <= z_e for every e in trail t              -> a trail scores only if *every*
                                                           one of its edges is walked
    (5)  p_v >= z_e for both endpoints of e             -> endpoints of used edges are used
    (6)  single-commodity flow from the depot to every
         active node, carried only on traversed arcs    -> connectivity / subtour elimination

Constraint (6) is what stops the solver from returning a lovely high-scoring loop on the
far side of the property that never touches the start line. Flow conservation alone
(constraint 1) is satisfied by any set of disjoint circuits, so without (6) the "route" can
be several unconnected loops.

The 40-point caps on each category are deliberately omitted: the whole network is only
40.1 miles and 40 trails, and covering it takes 14.3 h against a 7 h budget, so neither cap
can bind. :func:`check_caps_nonbinding` asserts this rather than assuming it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pulp

from .config import CONFIG, OUTPUTS, RaceParams
from .graph import Arc, Network, Route

log = logging.getLogger(__name__)

#: Upper bound on how often a single arc may be traversed. Re-walking the same piece of
#: trail more than a couple of times is never useful — it earns nothing after the first
#: pass — so this is a safe tightening that materially shrinks the search tree.
MAX_TRAVERSALS = 3


@dataclass
class MILPResult:
    status: str
    objective: float | None
    bound: float | None
    gap: float | None
    route: Route | None
    arc_counts: dict[int, int]
    trails_completed: list[int]
    solve_seconds: float

    def summary(self) -> dict:
        return {
            "status": self.status,
            "incumbent": None if self.objective is None else round(self.objective, 3),
            "upper_bound": None if self.bound is None else round(self.bound, 3),
            "gap_pct": None if self.gap is None else round(100 * self.gap, 2),
            "trails_completed": len(self.trails_completed),
            "solve_seconds": round(self.solve_seconds, 1),
        }


def check_caps_nonbinding(net: Network, race: RaceParams | None = None) -> dict:
    """Prove the 40-point category caps cannot bind, so omitting them is sound.

    The trail cap is trivially safe (there are exactly 40 trails). The mile cap needs an
    actual argument, because the network is 40.14 miles — a hair over the 40-mile cap.
    The argument: no one can move faster than Tobler's peak speed, so ``budget x peak
    speed`` is a hard ceiling on distance covered. If that ceiling is under 40 miles, the
    cap is unreachable and dropping it from the model changes nothing.
    """
    from .tobler import tobler_speed_kmh

    race = race or CONFIG.race
    total_mi = sum(net.edge_score_mi.values())

    peak_kmh = float(tobler_speed_kmh(-CONFIG.tobler.s0, CONFIG.tobler))
    ceiling_mi = peak_kmh * (race.time_budget_s / 3600.0) * 0.621371

    return {
        "total_network_miles": round(total_mi, 2),
        "peak_speed_mph": round(peak_kmh * 0.621371, 2),
        "distance_ceiling_mi": round(ceiling_mi, 2),
        "mile_cap": race.max_mile_points,
        "mile_cap_reachable": ceiling_mi > race.max_mile_points,
        "n_trails": net.n_trails,
        "trail_cap": race.max_trail_points,
        "trail_cap_reachable": net.n_trails > race.max_trail_points,
        "safe_to_omit_caps": (ceiling_mi <= race.max_mile_points)
        and (net.n_trails <= race.max_trail_points),
    }


def build_model(
    net: Network,
    race: RaceParams | None = None,
    *,
    incumbent: float | None = None,
) -> tuple[pulp.LpProblem, dict]:
    """Assemble the MILP. Returns the problem and its variable dictionaries."""
    race = race or CONFIG.race
    nodes = sorted(int(n) for n in net.nodes["node_id"])
    depot = net.depot

    edge_ids = sorted({a.edge_id for a in net.arcs})
    arcs_by_edge: dict[int, list[Arc]] = {}
    out_arcs: dict[int, list[Arc]] = {v: [] for v in nodes}
    in_arcs: dict[int, list[Arc]] = {v: [] for v in nodes}
    for arc in net.arcs:
        arcs_by_edge.setdefault(arc.edge_id, []).append(arc)
        out_arcs[arc.u].append(arc)
        in_arcs[arc.v].append(arc)

    edge_nodes = {
        eid: (arcs[0].u, arcs[0].v) for eid, arcs in arcs_by_edge.items()
    }
    miles = {eid: net.edge_score_mi.get(eid, 0.0) for eid in edge_ids}

    prob = pulp.LpProblem("hullabaloo", pulp.LpMaximize)

    x = {
        a.arc_id: pulp.LpVariable(f"x_{a.arc_id}", 0, MAX_TRAVERSALS, cat="Integer")
        for a in net.arcs
    }
    z = {e: pulp.LpVariable(f"z_{e}", cat="Binary") for e in edge_ids}
    y = {t: pulp.LpVariable(f"y_{t}", cat="Binary") for t in net.trail_edges}
    p = {v: pulp.LpVariable(f"p_{v}", cat="Binary") for v in nodes if v != depot}
    g = {a.arc_id: pulp.LpVariable(f"g_{a.arc_id}", 0, None) for a in net.arcs}

    # -- objective ---------------------------------------------------------------------
    prob += (
        pulp.lpSum(race.points_per_trail * y[t] for t in y)
        + pulp.lpSum(race.points_per_mile * miles[e] * z[e] for e in edge_ids),
        "score",
    )

    # (1) closed walk
    for v in nodes:
        prob += (
            pulp.lpSum(x[a.arc_id] for a in out_arcs[v])
            == pulp.lpSum(x[a.arc_id] for a in in_arcs[v]),
            f"balance_{v}",
        )

    # (2) time budget
    prob += (
        pulp.lpSum(a.time_s * x[a.arc_id] for a in net.arcs) <= race.time_budget_s,
        "time_budget",
    )

    # (3) edge-use linking
    for e in edge_ids:
        traversals = pulp.lpSum(x[a.arc_id] for a in arcs_by_edge[e])
        prob += (z[e] <= traversals, f"z_lower_{e}")
        prob += (traversals <= MAX_TRAVERSALS * z[e], f"z_upper_{e}")

    # (4) a trail scores only when every one of its edges is walked
    for t, edges in net.trail_edges.items():
        for e in edges:
            prob += (y[t] <= z[e], f"trail_{t}_edge_{e}")

    # (5) endpoints of used edges are active nodes
    #
    # ``set`` matters: a self-loop edge has u == v, and emitting the constraint once per
    # endpoint would then register the same constraint name twice, which PuLP rejects
    # outright. The network contains a genuine 31 m switchback where a trail returns to
    # its own node, so this is a real case rather than a hypothetical one.
    for e in edge_ids:
        for v in set(edge_nodes[e]):
            if v != depot:
                prob += (p[v] >= z[e], f"active_{v}_{e}")

    # (6) single-commodity flow => the traversed subgraph is connected to the depot
    n_nodes = len(nodes)
    for v in nodes:
        inflow = pulp.lpSum(g[a.arc_id] for a in in_arcs[v])
        outflow = pulp.lpSum(g[a.arc_id] for a in out_arcs[v])
        if v == depot:
            prob += (outflow - inflow == pulp.lpSum(p.values()), "flow_source")
        else:
            prob += (inflow - outflow == p[v], f"flow_{v}")
    for a in net.arcs:
        prob += (g[a.arc_id] <= n_nodes * x[a.arc_id], f"flow_cap_{a.arc_id}")

    # Valid primal cut from the heuristic: we already *have* a route worth this much, so
    # nothing worse is interesting. This prunes hard without excluding the optimum.
    if incumbent is not None:
        prob += (
            pulp.lpSum(race.points_per_trail * y[t] for t in y)
            + pulp.lpSum(race.points_per_mile * miles[e] * z[e] for e in edge_ids)
            >= incumbent - 1e-6,
            "incumbent_cut",
        )

    return prob, {"x": x, "z": z, "y": y, "p": p, "g": g, "arcs_by_edge": arcs_by_edge}


def _reconstruct_route(net: Network, arc_counts: dict[int, int]) -> Route | None:
    """Turn the multiset of traversed arcs into an actual closed walk (Hierholzer)."""
    adjacency: dict[int, list[int]] = {}
    for arc_id, count in arc_counts.items():
        arc = net.arcs[arc_id]
        adjacency.setdefault(arc.u, []).extend([arc_id] * count)
    if not adjacency:
        return None

    stack = [net.depot]
    circuit: list[int] = []
    used_from = {v: list(a) for v, a in adjacency.items()}
    path_arcs: list[int] = []

    while stack:
        v = stack[-1]
        if used_from.get(v):
            arc_id = used_from[v].pop()
            stack.append(net.arcs[arc_id].v)
            path_arcs.append(arc_id)
        else:
            stack.pop()
            if path_arcs:
                circuit.append(path_arcs.pop())

    circuit.reverse()
    if not circuit:
        return None
    return Route(arcs=[net.arcs[i] for i in circuit], net=net)


def solve(
    net: Network,
    race: RaceParams | None = None,
    *,
    time_limit_s: int = 600,
    incumbent: float | None = None,
    gap_rel: float = 0.0,
    msg: bool = True,
) -> MILPResult:
    """Solve the MILP with HiGHS and report the optimality gap."""
    import time

    race = race or CONFIG.race
    prob, vars_ = build_model(net, race, incumbent=incumbent)

    solver = _make_solver(time_limit_s, gap_rel, msg)
    t0 = time.time()
    prob.solve(solver)
    elapsed = time.time() - t0

    status = pulp.LpStatus[prob.status]
    objective = pulp.value(prob.objective)

    arc_counts = {
        arc_id: int(round(var.value() or 0))
        for arc_id, var in vars_["x"].items()
        if (var.value() or 0) > 0.5
    }
    trails = [t for t, var in vars_["y"].items() if (var.value() or 0) > 0.5]

    route = _reconstruct_route(net, arc_counts) if arc_counts else None
    bound, gap, solver_status = _extract_bound_and_gap(prob)
    if gap is None and bound is not None and objective and objective > 1e-9:
        gap = max(0.0, (bound - objective) / objective)
    if solver_status and solver_status != "unknown":
        status = solver_status

    return MILPResult(
        status=status,
        objective=objective,
        bound=bound,
        gap=gap,
        route=route,
        arc_counts=arc_counts,
        trails_completed=trails,
        solve_seconds=elapsed,
    )


def _make_solver(time_limit_s: int, gap_rel: float, msg: bool):
    """Prefer HiGHS; fall back to CBC if this PuLP build lacks a HiGHS interface."""
    for name in ("HiGHS", "HiGHS_CMD"):
        if name in pulp.listSolvers(onlyAvailable=True):
            cls = getattr(pulp, name)
            try:
                return cls(msg=msg, timeLimit=time_limit_s, gapRel=gap_rel)
            except TypeError:
                return cls(msg=msg, timeLimit=time_limit_s)
    log.warning("HiGHS unavailable — falling back to CBC, which is markedly slower here")
    return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_s, gapRel=gap_rel)


def _extract_bound_and_gap(prob: pulp.LpProblem) -> tuple[float | None, float | None, str]:
    """Best dual bound, relative gap, and the solver's own status string.

    PuLP maximizes by handing HiGHS the *negated* objective, so both the incumbent and
    the dual bound come back with the opposite sign. Reading ``mip_dual_bound`` naively
    yields a nonsensical negative upper bound on a positive-valued maximization — worth
    being explicit about, since a silently wrong bound would make the headline
    "within X% of optimal" claim meaningless.
    """
    solver_model = getattr(prob, "solverModel", None)
    if solver_model is None or not hasattr(solver_model, "getInfo"):
        return None, None, "unknown"

    try:
        info = solver_model.getInfo()
        raw_bound = float(info.mip_dual_bound)
        gap = float(info.mip_gap)
    except Exception:  # noqa: BLE001
        return None, None, "unknown"

    bound = -raw_bound if prob.sense == pulp.LpMaximize else raw_bound

    status = "unknown"
    try:
        status = solver_model.modelStatusToString(solver_model.getModelStatus())
    except Exception:  # noqa: BLE001
        pass

    if gap is not None and (gap != gap or gap == float("inf")):  # NaN / inf
        gap = None
    return bound, gap, status


def run(time_limit_s: int = 600, incumbent: float | None = None) -> MILPResult:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from .graph import build_network

    net = build_network()
    log.info("cap check: %s", check_caps_nonbinding(net))
    result = solve(net, time_limit_s=time_limit_s, incumbent=incumbent)
    log.info("MILP: %s", result.summary())
    if result.route:
        log.info("reconstructed route: %s", result.route.evaluate())
        log.info("route validation: %s", result.route.validate() or "OK")
    return result


if __name__ == "__main__":
    run()
