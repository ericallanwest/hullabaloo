"""Phase 6a — Adaptive Large Neighbourhood Search for the 7-hour route.

The problem is a **prize-collecting arc routing problem**: points are earned on *arcs*
(unique trail miles) and on *completing whole trails*, not on visiting nodes, and every
arc may be traversed as often as you like while only paying points once. Classic TSP/VRP
machinery does not apply directly.

Solution representation
-----------------------
A solution is a list of **target trails** in visit order. That list is expanded into an
actual closed walk by a deterministic decoder: start at the depot, and for each target
trail in turn, take the shortest path to whichever end of the trail is cheaper to reach,
walk the trail end to end, and continue; finally return to the depot. The decoder always
produces a connected, depot-anchored walk, so every solution the search touches is
structurally valid by construction — only the time budget can be violated, and the
decoder truncates to respect it.

This "order the prizes, let shortest paths do the rest" encoding is what makes ALNS work
well here: the neighbourhood operators only have to reason about *which trails and in
what order*, never about graph connectivity.

Free miles matter
-----------------
Deadhead paths between trails frequently traverse trail edges that have not been used
yet. Those miles score. The evaluator therefore credits every edge the walk actually
touches, which is why the decoder returns the full arc sequence rather than just the
targeted trails.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field

from .adapt import front_load_score
from .config import CONFIG, RaceParams
from .corridors import CorridorRule
from .graph import Arc, Network, Route, score_edges

log = logging.getLogger(__name__)

#: Score charged per corridor-rule violation.
#:
#: Large enough to dominate any real gain — the whole network is only 80 points — so a
#: rule-breaking route can never out-score a compliant one. Finite rather than infinite on
#: purpose: it leaves a gradient, so a search that starts outside the feasible region can
#: still tell "six violations" from "one" and climb back in. With -inf every infeasible
#: neighbour looks equally bad and the search random-walks until it gets lucky.
RULE_PENALTY = 1000.0


# --------------------------------------------------------------------------------------
# Decoder: trail visit order -> concrete closed walk
# --------------------------------------------------------------------------------------


@dataclass
class TrailChain:
    """A trail reduced to what the decoder needs: its two ends and how to walk between."""

    trail_id: int
    name: str
    ends: tuple[int, int]
    #: arc sequence walking from ends[0] to ends[1], and the reverse
    forward: list[Arc]
    backward: list[Arc]
    time_fwd: float
    time_bwd: float
    miles: float


def build_trail_chains(net: Network) -> dict[int, TrailChain]:
    """Order each trail's edges into a walkable chain from one end to the other.

    Each trail was split into consecutive edges in Phase 2, so its edges form a simple
    path (or occasionally a loop). We walk the adjacency to recover the traversal order.
    """
    chains: dict[int, TrailChain] = {}

    arcs_by_edge: dict[int, list[Arc]] = {}
    for arc in net.arcs:
        arcs_by_edge.setdefault(arc.edge_id, []).append(arc)

    for trail_id, edge_ids in net.trail_edges.items():
        adjacency: dict[int, list[tuple[int, int]]] = {}
        for edge_id in edge_ids:
            arc = arcs_by_edge[edge_id][0]
            adjacency.setdefault(arc.u, []).append((arc.v, edge_id))
            adjacency.setdefault(arc.v, []).append((arc.u, edge_id))

        degree_one = [n for n, adj in adjacency.items() if len(adj) == 1]
        start = degree_one[0] if degree_one else next(iter(adjacency))

        # Walk the chain, consuming each edge once.
        order: list[tuple[int, int, int]] = []  # (u, v, edge_id)
        unused = set(edge_ids)
        current = start
        while unused:
            step = next(
                ((nxt, eid) for nxt, eid in adjacency[current] if eid in unused), None
            )
            if step is None:
                break  # trail is not a simple chain; keep what we have
            nxt, eid = step
            unused.discard(eid)
            order.append((current, nxt, eid))
            current = nxt

        if not order:
            continue

        forward: list[Arc] = []
        for u, v, eid in order:
            arc = next(a for a in arcs_by_edge[eid] if a.u == u and a.v == v)
            forward.append(arc)
        backward: list[Arc] = []
        for u, v, eid in reversed(order):
            arc = next(a for a in arcs_by_edge[eid] if a.u == v and a.v == u)
            backward.append(arc)

        chains[trail_id] = TrailChain(
            trail_id=trail_id,
            name=forward[0].name,
            ends=(order[0][0], order[-1][1]),
            forward=forward,
            backward=backward,
            time_fwd=sum(a.time_s for a in forward),
            time_bwd=sum(a.time_s for a in backward),
            miles=sum(net.edge_score_mi.get(eid, 0.0) for _, _, eid in order),
        )

    return chains


def _path_arcs(net: Network, source: int, target: int) -> list[Arc] | None:
    """Arc sequence along the shortest-time path, or ``None`` if unreachable."""
    if source == target:
        return []
    path = net.sp_path.get(source, {}).get(target)
    if path is None:
        return None
    arcs = []
    for u, v in zip(path[:-1], path[1:]):
        arcs.append(net.arcs[net.graph[u][v]["arc_id"]])
    return arcs


def decode(
    order: list[int],
    net: Network,
    chains: dict[int, TrailChain],
    race: RaceParams | None = None,
) -> Route:
    """Expand a trail visit order into a concrete closed walk within the time budget.

    Trails that cannot be fitted (including the mandatory return leg to the depot) are
    skipped rather than aborting, so a too-ambitious order degrades gracefully instead of
    becoming infeasible.
    """
    race = race or CONFIG.race
    budget = race.time_budget_s

    arcs: list[Arc] = []
    position = net.depot
    elapsed = 0.0

    for trail_id in order:
        chain = chains.get(trail_id)
        if chain is None:
            continue

        options = []
        for entry, exit_node, walk, walk_time in (
            (chain.ends[0], chain.ends[1], chain.forward, chain.time_fwd),
            (chain.ends[1], chain.ends[0], chain.backward, chain.time_bwd),
        ):
            approach = _path_arcs(net, position, entry)
            if approach is None:
                continue
            approach_time = sum(a.time_s for a in approach)
            back = net.sp_time.get(exit_node, {}).get(net.depot)
            if back is None:
                continue
            options.append((approach_time + walk_time, approach, walk, exit_node, back))

        if not options:
            continue

        options.sort(key=lambda o: o[0])
        chosen = None
        for cost, approach, walk, exit_node, back in options:
            if elapsed + cost + back <= budget:
                chosen = (cost, approach, walk, exit_node)
                break
        if chosen is None:
            continue

        cost, approach, walk, exit_node = chosen
        arcs.extend(approach)
        arcs.extend(walk)
        elapsed += cost
        position = exit_node

    closing = _path_arcs(net, position, net.depot)
    if closing:
        arcs.extend(closing)

    return Route(arcs=arcs, net=net, race=race)


def evaluate(route: Route) -> float:
    score, _, _ = score_edges(route.edges_used, route.net, route.race)
    return score


def _order_from_route(route: Route, chains: dict[int, TrailChain]) -> list[int]:
    """Recover a trail visit order from a concrete route, for use as an ALNS seed.

    A trail counts as "visited" at the point the route first completes it, which keeps
    the recovered order consistent with what the decoder would reproduce.
    """
    order: list[int] = []
    seen: set[int] = set()
    walked: set[int] = set()
    for arc in route.arcs:
        walked.add(arc.edge_id)
        for trail_id, chain in chains.items():
            if trail_id in seen:
                continue
            if route.net.trail_edges[trail_id] <= walked:
                order.append(trail_id)
                seen.add(trail_id)
    return order


# --------------------------------------------------------------------------------------
# ALNS
# --------------------------------------------------------------------------------------


@dataclass
class ALNSResult:
    route: Route
    order: list[int]
    #: Search score: the race score less any corridor-rule penalty. This is the number the
    #: annealing compared, so it is the number to compare two runs on — but it is *not* the
    #: score the route earns, and must never be published or handed to the MILP as an
    #: incumbent without checking ``violations`` first.
    score: float
    history: list[tuple[int, float]]
    iterations: int
    #: Race score as actually earned, ignoring the rule.
    raw_score: float = 0.0
    #: Empty when the route honours the rule it was searched under.
    violations: list[str] = field(default_factory=list)

    @property
    def obeys_rule(self) -> bool:
        return not self.violations


class ALNS:
    """Adaptive large neighbourhood search with simulated-annealing acceptance."""

    def __init__(
        self,
        net: Network,
        chains: dict[int, TrailChain] | None = None,
        race: RaceParams | None = None,
        seed: int = 0,
        rule: CorridorRule | None = None,
        front_load_s: float | None = None,
        score_floor: float | None = None,
    ) -> None:
        self.net = net
        self.race = race or CONFIG.race
        self.rng = random.Random(seed)
        self.rule = rule or CorridorRule()
        # Adaptive mode: score the route by what it has banked partway through rather than
        # by where it finishes, holding the finish above a floor. The MILP cannot express
        # this at all — it has no notion of sequence, only of which arcs are used — whereas
        # an ALNS solution *is* a visit order, so front-loading is native here.
        self.front_load_s = front_load_s
        self.score_floor = score_floor

        chains = chains if chains is not None else build_trail_chains(net)
        # A forbidden trail is removed from the candidate pool outright, so the search
        # never spends an iteration proposing something it cannot keep. That alone is not
        # sufficient — the decoder's deadhead paths can still wander onto forbidden ground,
        # and the evaluator credits every edge the walk touches — so `_score` also
        # penalizes violations. Filtering is the cheap half; the penalty is the correct half.
        self.chains = {
            tid: chain for tid, chain in chains.items() if tid not in self.rule.forbid
        }
        self.all_trails = list(self.chains)

        self.destroy_ops = [
            self._destroy_random,
            self._destroy_worst,
            self._destroy_segment,
            self._destroy_cluster,
        ]
        self.repair_ops = [
            self._repair_greedy,
            self._repair_regret,
            self._repair_random,
        ]
        self.destroy_weights = [1.0] * len(self.destroy_ops)
        self.repair_weights = [1.0] * len(self.repair_ops)

    # -- scoring -----------------------------------------------------------------------

    def _score(self, order: list[int]) -> float:
        """Decode a visit order and score it, net of any penalty.

        Every operator and the acceptance test go through here, so the rules are applied
        once, in one place, and cannot be forgotten by a code path that scores a candidate
        its own way.
        """
        route = decode(order, self.net, self.chains, self.race)
        score = evaluate(route)

        penalty = 0.0
        if not self.rule.is_free:
            penalty += RULE_PENALTY * self.rule.violation_count(self.net, route)

        if self.front_load_s is None:
            return score - penalty

        # Falling below the floor is charged in proportion to the shortfall rather than
        # flatly: a route two points short and a route twenty points short are not equally
        # wrong, and a flat penalty gives the search no gradient back into the feasible band.
        if self.score_floor is not None and score < self.score_floor:
            penalty += RULE_PENALTY * (self.score_floor - score)
        return front_load_score(route, self.front_load_s, self.race) - penalty

    def decode_order(self, order: list[int]) -> Route:
        """The concrete walk an order expands to. Public so callers can re-check the rule."""
        return decode(order, self.net, self.chains, self.race)

    # -- destroy -----------------------------------------------------------------------

    def _destroy_random(self, order: list[int], k: int) -> list[int]:
        keep = order[:]
        for _ in range(min(k, len(keep))):
            keep.pop(self.rng.randrange(len(keep)))
        return keep

    def _destroy_worst(self, order: list[int], k: int) -> list[int]:
        """Drop the trails giving the least score per second of detour."""
        if len(order) <= 1:
            return order[:]
        base = self._score(order)
        base_time = decode(order, self.net, self.chains, self.race).time_s
        ratios = []
        for trail_id in order:
            trimmed = [t for t in order if t != trail_id]
            route = decode(trimmed, self.net, self.chains, self.race)
            saved = base_time - route.time_s
            lost = base - evaluate(route)
            ratios.append((lost / max(saved, 1.0), trail_id))
        ratios.sort()
        drop = {t for _, t in ratios[:k]}
        return [t for t in order if t not in drop]

    def _destroy_segment(self, order: list[int], k: int) -> list[int]:
        if len(order) <= k or not order:
            return []
        start = self.rng.randrange(0, len(order) - k + 1)
        return order[:start] + order[start + k :]

    def _destroy_cluster(self, order: list[int], k: int) -> list[int]:
        """Remove a geographically coherent group — opens up a whole area for rerouting."""
        if not order:
            return []
        anchor = self.rng.choice(order)
        anchor_node = self.chains[anchor].ends[0]
        distances = []
        for trail_id in order:
            end = self.chains[trail_id].ends[0]
            distances.append((self.net.sp_time.get(anchor_node, {}).get(end, math.inf), trail_id))
        distances.sort()
        drop = {t for _, t in distances[:k]}
        return [t for t in order if t not in drop]

    # -- repair ------------------------------------------------------------------------

    def _insertion_best(self, order: list[int], trail_id: int) -> tuple[float, list[int]]:
        best_score, best_order = -math.inf, None
        positions = range(len(order) + 1)
        for pos in positions:
            candidate = order[:pos] + [trail_id] + order[pos:]
            score = self._score(candidate)
            if score > best_score:
                best_score, best_order = score, candidate
        return best_score, best_order or order

    def _repair_greedy(self, order: list[int]) -> list[int]:
        current = order[:]
        missing = [t for t in self.all_trails if t not in set(current)]
        self.rng.shuffle(missing)
        improved = True
        while improved and missing:
            improved = False
            base = self._score(current)
            best = (base, None, None)
            for trail_id in missing:
                score, candidate = self._insertion_best(current, trail_id)
                if score > best[0] + 1e-9:
                    best = (score, trail_id, candidate)
            if best[1] is not None:
                current = best[2]
                missing.remove(best[1])
                improved = True
        return current

    def _repair_regret(self, order: list[int]) -> list[int]:
        """Regret-2: insert the trail that suffers most from being deferred."""
        current = order[:]
        missing = [t for t in self.all_trails if t not in set(current)]
        while missing:
            base = self._score(current)
            scored = []
            for trail_id in missing:
                candidates = sorted(
                    (
                        self._score(current[:p] + [trail_id] + current[p:])
                        for p in range(len(current) + 1)
                    ),
                    reverse=True,
                )
                best = candidates[0]
                second = candidates[1] if len(candidates) > 1 else best
                scored.append((best - second, best, trail_id))
            scored.sort(key=lambda s: (-s[0], -s[1]))
            _, best_score, trail_id = scored[0]
            if best_score <= base + 1e-9:
                break
            _, current = self._insertion_best(current, trail_id)
            missing.remove(trail_id)
        return current

    def _repair_random(self, order: list[int]) -> list[int]:
        current = order[:]
        missing = [t for t in self.all_trails if t not in set(current)]
        self.rng.shuffle(missing)
        for trail_id in missing[: self.rng.randint(1, 5)]:
            pos = self.rng.randrange(len(current) + 1)
            candidate = current[:pos] + [trail_id] + current[pos:]
            if self._score(candidate) >= self._score(current):
                current = candidate
        return current

    # -- driver ------------------------------------------------------------------------

    def _pick(self, weights: list[float]) -> int:
        total = sum(weights)
        r = self.rng.random() * total
        upto = 0.0
        for i, w in enumerate(weights):
            upto += w
            if r <= upto:
                return i
        return len(weights) - 1

    def _best_start(self) -> list[int]:
        """Pick the strongest of several constructive starts.

        Worth doing rather than always starting from greedy insertion: once the forest
        roads made the whole network reachable, the nearest-trail baseline jumped from
        26.4 to 33.6 and began beating short ALNS runs outright. Starting from the best
        available construction means the search spends its budget improving a good
        solution instead of climbing back to one.
        """
        candidates: list[list[int]] = [self._repair_greedy([])]
        for builder in (baseline_greedy, baseline_best_ratio):
            try:
                route = builder(self.net, self.chains, self.race)
                order = _order_from_route(route, self.chains)
                if order:
                    candidates.append(order)
            except Exception:  # noqa: BLE001 - a failed construction is not fatal
                continue

        # Under a "require" rule, add a variant of each construction that visits the
        # required trails first. None of the constructors know about the rule, so left to
        # themselves they tend to start entirely infeasible — and a required corridor is
        # usually the far one, which is exactly what a budget-truncating decoder drops.
        # Front-loading it is the one placement that reliably survives truncation, and it
        # gives the search a feasible solution to improve rather than one to repair.
        if self.rule.require:
            required = [t for t in self.rule.require if t in self.chains]
            for order in list(candidates):
                rest = [t for t in order if t not in self.rule.require]
                candidates.append(required + rest)

        return max(candidates, key=self._score)

    def solve(
        self,
        iterations: int = 400,
        initial_temp: float = 2.0,
        cooling: float = 0.995,
        decay: float = 0.85,
        seed_order: list[int] | None = None,
    ) -> ALNSResult:
        current = seed_order[:] if seed_order else self._best_start()
        current_score = self._score(current)
        best, best_score = current[:], current_score

        temp = initial_temp
        history = [(0, best_score)]

        for it in range(1, iterations + 1):
            d_idx = self._pick(self.destroy_weights)
            r_idx = self._pick(self.repair_weights)
            k = self.rng.randint(1, max(2, len(current) // 3))

            candidate = self.repair_ops[r_idx](self.destroy_ops[d_idx](current, k))
            cand_score = self._score(candidate)

            reward = 0.0
            if cand_score > best_score + 1e-9:
                best, best_score = candidate[:], cand_score
                reward = 3.0
            elif cand_score > current_score + 1e-9:
                reward = 1.5
            elif self.rng.random() < math.exp((cand_score - current_score) / max(temp, 1e-6)):
                reward = 0.5
            else:
                candidate = None

            if candidate is not None:
                current, current_score = candidate, cand_score

            self.destroy_weights[d_idx] = decay * self.destroy_weights[d_idx] + (1 - decay) * reward
            self.repair_weights[r_idx] = decay * self.repair_weights[r_idx] + (1 - decay) * reward
            temp *= cooling

            if it % 25 == 0:
                history.append((it, best_score))
                log.debug("iter %d best=%.2f temp=%.3f", it, best_score, temp)

        route = decode(best, self.net, self.chains, self.race)
        history.append((iterations, best_score))
        return ALNSResult(
            route=route,
            order=best,
            score=best_score,
            history=history,
            iterations=iterations,
            raw_score=evaluate(route),
            violations=self.rule.violations(self.net, route),
        )


# --------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------


def baseline_greedy(net: Network, chains=None, race: RaceParams | None = None) -> Route:
    """Repeatedly take the nearest unvisited trail that still fits in the budget."""
    chains = chains if chains is not None else build_trail_chains(net)
    race = race or CONFIG.race

    remaining = set(chains)
    order: list[int] = []
    position = net.depot
    elapsed = 0.0

    while remaining:
        best = None
        for trail_id in remaining:
            chain = chains[trail_id]
            for entry, exit_node, walk_time in (
                (chain.ends[0], chain.ends[1], chain.time_fwd),
                (chain.ends[1], chain.ends[0], chain.time_bwd),
            ):
                approach = net.sp_time.get(position, {}).get(entry)
                back = net.sp_time.get(exit_node, {}).get(net.depot)
                if approach is None or back is None:
                    continue
                cost = approach + walk_time
                if elapsed + cost + back <= race.time_budget_s:
                    if best is None or cost < best[0]:
                        best = (cost, trail_id, exit_node)
        if best is None:
            break
        cost, trail_id, exit_node = best
        order.append(trail_id)
        remaining.discard(trail_id)
        elapsed += cost
        position = exit_node

    return decode(order, net, chains, race)


def baseline_best_ratio(net: Network, chains=None, race: RaceParams | None = None) -> Route:
    """Take the trail with the best (points gained / time spent) at each step."""
    chains = chains if chains is not None else build_trail_chains(net)
    race = race or CONFIG.race

    order: list[int] = []
    remaining = set(chains)
    while remaining:
        base_route = decode(order, net, chains, race)
        base_score = evaluate(base_route)
        best = None
        for trail_id in remaining:
            candidate = order + [trail_id]
            route = decode(candidate, net, chains, race)
            gain = evaluate(route) - base_score
            spent = route.time_s - base_route.time_s
            if gain <= 0 or spent <= 0:
                continue
            ratio = gain / spent
            if best is None or ratio > best[0]:
                best = (ratio, trail_id)
        if best is None:
            break
        order.append(best[1])
        remaining.discard(best[1])
    return decode(order, net, chains, race)


def run(iterations: int = 400, seed: int = 0):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from .graph import build_network, network_traversal_bound

    net = build_network()
    chains = build_trail_chains(net)
    log.info("coverage reference: %s", network_traversal_bound(net))

    greedy = baseline_greedy(net, chains)
    ratio = baseline_best_ratio(net, chains)
    log.info("baseline greedy-nearest : %s", greedy.evaluate())
    log.info("baseline best-ratio     : %s", ratio.evaluate())

    seed_order = [t for t in ratio.net.trail_edges if t in chains]
    alns = ALNS(net, chains, seed=seed)
    result = alns.solve(iterations=iterations)
    log.info("ALNS                    : %s", result.route.evaluate())
    problems = result.route.validate()
    log.info("route validation: %s", problems or "OK")
    return result


if __name__ == "__main__":
    run()
