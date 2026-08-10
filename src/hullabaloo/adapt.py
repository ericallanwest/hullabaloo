"""Phase 9 — turning a route into a plan you can change your mind about.

Every itinerary the solver publishes spends the budget to the second. That is what optimal
looks like, and it is also brittle: a racer running 5% slower than modelled is twenty
minutes over with no guidance about what to give up, deciding on tired legs at mile 20 what
the solver spent five minutes deciding at a desk.

This module reads a solved :class:`~hullabaloo.graph.Route` and works out, in advance, every
way it can be shortened — what each cut costs, where it is taken, and the latest time on the
clock at which taking it still gets you home. No solving happens here. The route is already
optimal; the job is to say what it degrades into.

The unit of flexibility: contiguous closed excursions
-----------------------------------------------------
A cut has to be something a racer can act on at a junction, and something that leaves a
route he can still follow. Both point at the same object: a **contiguous span of the walk
that leaves a node and comes back to it**.

Splicing such a span out is always safe. The arcs before it end where the arcs after it
begin, so what remains is still one continuous closed walk from the start line back to it —
no connectivity check, no re-solve, no chance of stranding the route on the far side of the
property. That is worth stating because the obvious alternative — decomposing the walk into
simple cycles — does *not* have this property: those cycles interleave, so "drop this cycle"
means skipping several disjoint pieces of the itinerary, which is neither describable at a
junction nor obviously connected.

Excursions nest. A loop can contain a smaller loop, so they form a laminar family, and
dropping an outer one takes its children with it. That is the natural reading anyway: skip
the big loop and you skip everything on it.

Why costs are not additive
--------------------------
:func:`~hullabaloo.graph.score_edges` scores the **set** of edges walked, and a trail scores
only when every one of its edges is covered. So two excursions can each be individually
cheap while dropping both forfeits a trail that straddled them, and an edge that was a free
repeat becomes load-bearing the moment its twin is cut. Costs therefore have to be measured
on the actual remaining edge set, never summed. Every function here re-scores; nothing adds.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .config import CONFIG, M_PER_MILE, RaceParams
from .graph import Network, Route, score_edges

#: Reduced budgets published alongside each adaptive itinerary, as fractions of the real
#: one. A racer who is behind is usually behind by a quarter or half hour, not by three
#: hours, so the interesting cases cluster near the top.
SALVAGE_FRACTIONS = (0.93, 0.86, 0.79)  # ~6:30, 6:00, 5:30 against a 7-hour budget

#: Cap on how many drop combinations are scored exactly when planning a salvage. Antichains
#: in a laminar family of ~20 excursions stay well inside this; the cap exists so a
#: pathological route degrades to a greedy answer instead of hanging the build.
MAX_COMBINATIONS = 40_000


@dataclass(frozen=True)
class Excursion:
    """A contiguous stretch of the walk that leaves ``hinge`` and returns to it."""

    hinge: int
    #: Half-open arc span ``[start, end)`` into ``route.arcs``.
    start: int
    end: int
    seconds: float
    miles: float
    #: Index of the enclosing excursion in the same list, or ``None`` at top level.
    parent: int | None = None
    depth: int = 0

    @property
    def n_arcs(self) -> int:
        return self.end - self.start

    def contains(self, other: "Excursion") -> bool:
        return self.start <= other.start and other.end <= self.end

    def overlaps(self, other: "Excursion") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class Cut:
    """One publishable way to shorten the route, priced exactly."""

    excursion: Excursion
    seconds_saved: float
    miles_saved: float
    points_lost: float
    trails_lost: int
    #: Latest elapsed seconds at the hinge for which *keeping* this excursion still finishes
    #: inside the budget. ``None`` when keeping it is always affordable.
    keep_before_s: float | None = None

    @property
    def points_per_minute(self) -> float:
        """Cost of the cut per minute it buys. Lower is a better thing to give up."""
        minutes = self.seconds_saved / 60.0
        return self.points_lost / minutes if minutes > 1e-9 else float("inf")


@dataclass
class SalvageSet:
    """A jointly-costed set of cuts that brings the route inside a reduced budget."""

    budget_s: float
    cuts: list[Cut] = field(default_factory=list)
    seconds: float = 0.0
    score: float = 0.0
    trails_completed: int = 0
    unique_miles: float = 0.0
    feasible: bool = True


# --------------------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------------------


def raw_excursions(route: Route) -> list[Excursion]:
    """The excursion spans alone, without working out which contains which.

    Split from :func:`excursions` because linking parents is O(n^2) over the excursions and
    the search objective does not need it — only publishing does. That loop ran on every
    ALNS decode and roughly doubled the cost of scoring a candidate.
    """
    arcs = route.arcs
    if not arcs:
        return []

    found: list[tuple[int, int, int]] = []  # (hinge, start, end)
    stack_nodes: list[int] = [arcs[0].u]
    # ``stack_arcs[k]`` is the index of the arc entering ``stack_nodes[k + 1]``.
    stack_arcs: list[int] = []
    position: dict[int, int] = {arcs[0].u: 0}

    for index, arc in enumerate(arcs):
        node = arc.v
        seen_at = position.get(node)
        if seen_at is None:
            stack_arcs.append(index)
            position[node] = len(stack_nodes)
            stack_nodes.append(node)
            continue

        # Closed a loop: it spans from the arc that first entered this node through here.
        start = stack_arcs[seen_at] if seen_at < len(stack_arcs) else index
        found.append((node, start, index + 1))
        for dropped in stack_nodes[seen_at + 1 :]:
            position.pop(dropped, None)
        del stack_nodes[seen_at + 1 :]
        del stack_arcs[seen_at:]

    result: list[Excursion] = []
    for hinge, start, end in found:
        result.append(
            Excursion(
                hinge=int(hinge),
                start=start,
                end=end,
                seconds=sum(a.time_s for a in arcs[start:end]),
                miles=sum(a.score_mi for a in arcs[start:end]),
            )
        )

    return result


def excursions(route: Route) -> list[Excursion]:
    """Every contiguous closed excursion in the walk, with nesting resolved.

    One pass with a stack of nodes on the current path. Revisiting a node already on the
    stack closes an excursion; everything after that node is unwound, which is exactly what
    makes the returned spans contiguous. Nesting is recovered afterwards by span
    containment, since an excursion is always closed before its parent is.
    """
    result = raw_excursions(route)

    # Parent = the tightest excursion strictly containing this one. Computed for every
    # excursion before any depth is, because a depth is a walk up the parent chain and a
    # chain half-built reports the wrong answer — an inner loop looked top-level simply
    # because its parent had not been linked yet.
    parents: list[int | None] = [None] * len(result)
    for i, child in enumerate(result):
        best, best_span = None, None
        for j, other in enumerate(result):
            if j == i or other.n_arcs <= child.n_arcs or not other.contains(child):
                continue
            if best_span is None or other.n_arcs < best_span:
                best, best_span = j, other.n_arcs
        parents[i] = best

    depths: list[int] = [0] * len(result)
    for i in range(len(result)):
        depth, walk = 0, parents[i]
        while walk is not None:
            depth += 1
            walk = parents[walk]
        depths[i] = depth

    return [
        Excursion(
            hinge=e.hinge,
            start=e.start,
            end=e.end,
            seconds=e.seconds,
            miles=e.miles,
            parent=parents[i],
            depth=depths[i],
        )
        for i, e in enumerate(result)
    ]


def drop(route: Route, spans) -> Route:
    """The route with every arc inside ``spans`` removed.

    ``spans`` is any iterable of ``(start, end)`` half-open arc ranges. Overlapping and
    nested spans are fine — an arc removed twice is simply removed. The result is still a
    closed walk, because each span begins and ends at the same node.
    """
    removed: set[int] = set()
    for start, end in spans:
        removed.update(range(start, end))
    kept = [arc for i, arc in enumerate(route.arcs) if i not in removed]
    return Route(arcs=kept, net=route.net, race=route.race)


# --------------------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------------------


def _evaluate(net: Network, arcs, race: RaceParams) -> tuple[float, float, int]:
    return score_edges({a.edge_id for a in arcs}, net, race)


def cuts(
    net: Network,
    route: Route,
    race: RaceParams | None = None,
    *,
    min_seconds: float = 120.0,
    max_fraction: float = 0.35,
) -> list[Cut]:
    """Price every excursion worth offering, and say when it stops being optional.

    Bounded at both ends, because only the middle of the range is a decision anybody makes
    at a junction. ``min_seconds`` drops the sub-two-minute wiggles that litter any real
    walk. ``max_fraction`` drops the giants: the walk as a whole is trivially an excursion
    from the start line back to it, as is almost all of it, and "skip 5.8 of your 7 hours"
    is not an escape valve — it is abandoning the race, which needs no menu entry.

    Each cost is measured by re-scoring the walk with that excursion actually removed, for
    the reason given in the module docstring: a trail can straddle two excursions, so no
    amount of arithmetic on the excursion's own miles gets this right.
    """
    race = race or route.race or CONFIG.race
    base_score, _, base_trails = _evaluate(net, route.arcs, race)
    ceiling = route.time_s * max_fraction

    priced: list[Cut] = []
    for excursion in excursions(route):
        if not min_seconds <= excursion.seconds <= ceiling:
            continue
        remaining = drop(route, [(excursion.start, excursion.end)])
        score, unique_mi, trails = _evaluate(net, remaining.arcs, race)
        priced.append(
            Cut(
                excursion=excursion,
                seconds_saved=route.time_s - remaining.time_s,
                miles_saved=sum(a.score_mi for a in route.arcs[excursion.start : excursion.end]),
                points_lost=round(base_score - score, 6),
                trails_lost=base_trails - trails,
            )
        )

    priced.sort(key=lambda c: (c.excursion.start, c.excursion.end))
    return _with_thresholds(route, priced, race)


def _with_thresholds(route: Route, priced: list[Cut], race: RaceParams) -> list[Cut]:
    """Attach the 'keep it only if the clock is before X' rule to each cut.

    Backward induction along the walk: everything after an excursion ends is already
    committed, so keeping the excursion is affordable exactly while
    ``elapsed_at_hinge + excursion + committed_remainder <= budget``. Rearranged, that is a
    deadline on the clock at the hinge — which is the form a racer can actually use, because
    it is the only quantity he can read off his watch without doing arithmetic.

    Worth seeing what this reduces to. These routes spend the whole budget, so
    ``budget == total`` and the deadline collapses to the *planned arrival time* at that
    junction. In other words each threshold is a **split time**: hit the junction on
    schedule and the loop is affordable; arrive late and it is not, by exactly the margin
    you are late. That is a familiar object to anyone who has raced, and it is only this
    tidy because the plan has no slack to begin with — which is the same fact that makes
    these itineraries brittle without a menu of cuts to fall back on.
    """
    budget = float(race.time_budget_s)
    elapsed = [0.0]
    for arc in route.arcs:
        elapsed.append(elapsed[-1] + arc.time_s)
    total = elapsed[-1]

    out: list[Cut] = []
    for cut in priced:
        excursion = cut.excursion
        # Time from the end of this excursion to the finish, as planned.
        after = total - elapsed[excursion.end]
        latest = budget - (excursion.seconds + after)
        out.append(
            Cut(
                excursion=excursion,
                seconds_saved=cut.seconds_saved,
                miles_saved=cut.miles_saved,
                points_lost=cut.points_lost,
                trails_lost=cut.trails_lost,
                keep_before_s=None if latest >= total else max(0.0, latest),
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Salvage
# --------------------------------------------------------------------------------------


def _compatible(chosen: tuple[Cut, ...]) -> bool:
    """Reject overlapping picks: a nested cut inside a dropped one saves nothing twice."""
    return all(
        not a.excursion.overlaps(b.excursion)
        for a, b in itertools.combinations(chosen, 2)
    )


def salvage(
    net: Network,
    route: Route,
    budget_s: float,
    available: list[Cut] | None = None,
    race: RaceParams | None = None,
) -> SalvageSet:
    """Best set of cuts that brings the route inside ``budget_s``.

    Exhaustive over non-overlapping combinations, scored exactly on the surviving edge set —
    which is the whole point. Taking the two cheapest-looking cuts is *not* generally the
    cheapest pair, because their costs interact through trail completion. Falls back to a
    greedy pick if the combination count gets out of hand.
    """
    race = race or route.race or CONFIG.race
    available = available if available is not None else cuts(net, route, race)

    if route.time_s <= budget_s:
        score, unique_mi, trails = _evaluate(net, route.arcs, race)
        return SalvageSet(budget_s, [], route.time_s, score, trails, unique_mi, True)

    need = route.time_s - budget_s
    usable = [c for c in available if c.seconds_saved > 0]

    # Enumerated one at a time against a hard cap rather than one size at a time. Building
    # a whole size first and then noticing it was too big means computing C(20,10) before
    # deciding not to use it, which is the pathological case the cap exists to prevent.
    def candidates():
        emitted = 0
        for size in range(1, len(usable) + 1):
            any_at_size = False
            for combo in itertools.combinations(usable, size):
                if not _compatible(combo):
                    continue
                any_at_size = True
                emitted += 1
                if emitted > MAX_COMBINATIONS:
                    return
                yield combo
            if not any_at_size:
                return

    best: SalvageSet | None = None
    for combo in candidates():
        if sum(c.seconds_saved for c in combo) < need:
            continue
        remaining = drop(route, [(c.excursion.start, c.excursion.end) for c in combo])
        if remaining.time_s > budget_s:
            continue
        score, unique_mi, trails = _evaluate(net, remaining.arcs, race)
        if best is None or score > best.score:
            best = SalvageSet(
                budget_s, list(combo), remaining.time_s, score, trails, unique_mi, True
            )

    if best is not None:
        return best

    # Nothing inside the cap reached the target: give up the cheapest minutes we can.
    greedy: list[Cut] = []
    saved = 0.0
    for cut in sorted(usable, key=lambda c: c.points_per_minute):
        if any(cut.excursion.overlaps(g.excursion) for g in greedy):
            continue
        greedy.append(cut)
        saved += cut.seconds_saved
        if saved >= need:
            break
    remaining = drop(route, [(c.excursion.start, c.excursion.end) for c in greedy])
    score, unique_mi, trails = _evaluate(net, remaining.arcs, race)
    return SalvageSet(
        budget_s,
        greedy,
        remaining.time_s,
        score,
        trails,
        unique_mi,
        remaining.time_s <= budget_s,
    )


# --------------------------------------------------------------------------------------
# Bail-out
# --------------------------------------------------------------------------------------


def bailout_curve(net: Network, route: Route, race: RaceParams | None = None) -> list[dict]:
    """For every arc: quit here, walk straight home, and this is what you finish with.

    The single most useful thing a racer can be told mid-race, and it costs nothing to
    compute — ``Network`` already carries all-pairs shortest times from building the graph.

    Score counts the miles picked up **on the way home** as well. The direct line back
    routinely crosses trail that has not been walked yet, and ignoring that would understate
    the bail-out by more than the decision it informs.
    """
    race = race or route.race or CONFIG.race
    budget = float(race.time_budget_s)
    depot = net.depot

    walked_edges: set[int] = set()
    elapsed = 0.0
    curve: list[dict] = []

    for index, arc in enumerate(route.arcs):
        walked_edges.add(arc.edge_id)
        elapsed += arc.time_s
        node = arc.v

        home_s = net.sp_time.get(node, {}).get(depot, float("inf"))
        home_path = net.sp_path.get(node, {}).get(depot, [])
        # Edges crossed on the way home, so their miles are credited if new.
        home_edges = set()
        for u, v in zip(home_path, home_path[1:]):
            data = net.graph.get_edge_data(u, v)
            if data is not None:
                home_edges.add(net.arcs[data["arc_id"]].edge_id)

        score, unique_mi, trails = score_edges(walked_edges | home_edges, net, race)
        curve.append(
            {
                "arc": index,
                "node": int(node),
                "elapsed_s": round(elapsed, 1),
                "home_s": round(home_s, 1),
                "finish_s": round(elapsed + home_s, 1),
                "slack_s": round(budget - (elapsed + home_s), 1),
                "score_if_home_now": round(score, 3),
                "trails_if_home_now": int(trails),
                "unique_miles_if_home_now": round(unique_mi, 3),
            }
        )
    return curve


def droppable_seconds_after(
    route: Route,
    at_s: float,
    *,
    min_seconds: float = 120.0,
    max_fraction: float = 0.35,
) -> float:
    """How much time is still sheddable once the clock passes ``at_s``.

    The measure of a plan's *remaining* flexibility, and the thing worth optimising: a cut
    is only on offer until you reach its junction, so a menu that is all spent by hour four
    is no use to a racer who works out at hour four that he is behind.

    Deliberately does no scoring. :func:`cuts` re-scores the whole walk once per excursion
    to price it exactly, which is right when publishing a menu and far too slow to run on
    every ALNS decode. Here only the time matters, and time is a sum over arcs.

    Nested excursions are counted once. A loop inside a loop is not extra sheddable time —
    dropping the outer one already takes the inner with it — so only excursions with no
    eligible ancestor contribute.
    """
    ceiling = route.time_s * max_fraction
    elapsed = [0.0]
    for arc in route.arcs:
        elapsed.append(elapsed[-1] + arc.time_s)

    eligible = [
        e
        for e in raw_excursions(route)
        if min_seconds <= e.seconds <= ceiling and elapsed[e.start] >= at_s
    ]
    return sum(
        e.seconds
        for e in eligible
        if not any(o is not e and o.contains(e) for o in eligible)
    )


def front_load_score(route: Route, at_s: float, race: RaceParams | None = None) -> float:
    """Score standing after ``at_s`` seconds of the walk — the front-loading measure.

    Used as the search objective when solving an adaptive route. Deliberately cheap: it is
    evaluated on every ALNS decode, so it walks the arc list once and does no set algebra
    beyond what scoring already requires.
    """
    race = race or route.race or CONFIG.race
    net = route.net
    walked: set[int] = set()
    elapsed = 0.0
    for arc in route.arcs:
        if elapsed + arc.time_s > at_s:
            break
        walked.add(arc.edge_id)
        elapsed += arc.time_s
    score, _, _ = score_edges(walked, net, race)
    return score


def summary(
    net: Network,
    route: Route,
    race: RaceParams | None = None,
    *,
    min_seconds: float = 120.0,
    max_fraction: float = 0.35,
    decide_after_s: float | None = None,
) -> dict:
    """Everything the page and the cue sheet need, in one pass.

    ``decide_after_s`` is the hour at which the athlete is expected to judge his pace. The
    salvage table is built only from cuts he can still take at that point: a reduced-budget
    plan whose cheapest route out was a turning back in the first half-hour is arithmetic,
    not advice. The full ``cuts`` menu is unfiltered — an early cut is still worth publishing
    for anyone who knows at the start line that he wants a shorter day.
    """
    race = race or route.race or CONFIG.race
    budget = float(race.time_budget_s)
    priced = cuts(net, route, race, min_seconds=min_seconds, max_fraction=max_fraction)
    priced = sorted(priced, key=lambda c: c.excursion.start)

    # Elapsed clock at the start of each arc, so a cut can say when it is reached.
    elapsed = [0.0]
    for arc in route.arcs:
        elapsed.append(elapsed[-1] + arc.time_s)

    if decide_after_s is None:
        still_open = priced
    else:
        still_open = [c for c in priced if elapsed[c.excursion.start] >= decide_after_s]

    salvages = []
    for fraction in SALVAGE_FRACTIONS:
        target = budget * fraction
        best = salvage(net, route, target, still_open, race)
        # A cut can only be taken if you have not already walked past its junction, so a
        # plan is only usable if you commit to it before the *earliest* of its cuts. Without
        # this the table quietly implies that a racer who realises at hour five that he is
        # behind can still take a turning he passed in the first hour.
        reach = [elapsed[c.excursion.start] for c in best.cuts]
        salvages.append(
            {
                "budget_s": round(target, 1),
                "budget_h": round(target / 3600.0, 2),
                "score": round(best.score, 3),
                "trails_completed": best.trails_completed,
                "unique_miles": round(best.unique_miles, 3),
                "time_s": round(best.seconds, 1),
                "feasible": best.feasible,
                "decide_by_s": round(min(reach), 1) if reach else None,
                "decide_after_s": round(decide_after_s, 1) if decide_after_s else None,
                "cut_hinges": [c.excursion.hinge for c in best.cuts],
                "cut_arcs": [[c.excursion.start, c.excursion.end] for c in best.cuts],
            }
        )

    return {
        "cuts": [
            {
                "hinge": c.excursion.hinge,
                "arc_start": c.excursion.start,
                "arc_end": c.excursion.end,
                "depth": c.excursion.depth,
                # When the plan has you arriving at this junction. Past it the cut is no
                # longer on offer, whatever the clock says — you have already walked it.
                "reach_s": round(elapsed[c.excursion.start], 1),
                "minutes_saved": round(c.seconds_saved / 60.0, 1),
                "miles_saved": round(c.miles_saved, 3),
                "points_lost": round(c.points_lost, 3),
                "trails_lost": c.trails_lost,
                "points_per_minute": round(c.points_per_minute, 4),
                "keep_before_s": None if c.keep_before_s is None else round(c.keep_before_s, 1),
            }
            for c in priced
        ],
        "salvage": salvages,
        "front_load": {
            "score_at_6h": round(front_load_score(route, 6 * 3600.0, race), 3),
            "score_at_5h": round(front_load_score(route, 5 * 3600.0, race), 3),
        },
        # How much room is left at each point a racer might reassess. The 3.5 h figure is
        # what the adaptive search optimises; the rest is here so the cliff is visible
        # rather than asserted — on this network the sheddable time falls off sharply
        # somewhere past four hours, and that is the fact a decision point has to respect.
        "flexibility": [
            {
                "at_h": round(mark / 3600.0, 1),
                "droppable_minutes": round(
                    droppable_seconds_after(
                        route, mark, min_seconds=min_seconds, max_fraction=max_fraction
                    )
                    / 60.0,
                    1,
                ),
            }
            for mark in (2.5 * 3600, 3.0 * 3600, 3.5 * 3600, 4.0 * 3600, 4.5 * 3600)
        ],
    }
