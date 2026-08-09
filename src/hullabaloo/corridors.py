"""Geographic blocks of the trail network, and what a route did in each of them.

The optimizer has no notion of place. It sees arcs, times and scores, and the itinerary it
returns is a list of turns — correct, but not something you can hold in your head or argue
with. Splitting the network into four named blocks gives the answer somewhere to live: a
racer can ask "does this plan go west?" and get an answer, and the published alternatives
in ``scripts/build_presets.py`` can be *defined* by their answer to that question rather
than by an opaque difference in edge sets.

The four blocks, with the start line at the extreme south-east corner of the network:

``Home Block``   the dense cluster on the start line — bike park, Chimney, Gateway.
``North Rim``    everything north of the Poverty Creek valley. The biggest block.
``South Ridge``  the southern spur — Highway, Turkey Trot, Blunderbuss and neighbours.
``West End``     the far western tail, four to five km out. The pivotal commitment.

Membership is written out by name rather than computed at import time. The rule that
generated it was a two-cut test on each trail's length-weighted centroid:

    lon < -80.4900  ->  West End
    lat >= 37.2600  ->  North Rim
    lon < -80.4750  ->  South Ridge
    otherwise       ->  Home Block

but a threshold is a bad thing to leave live in the code. It answers "which side of a line
is this trail's average position on?", which is not the question — Crosscut clears the West
End cut by about five metres and could as fairly be called South Ridge. Freezing the result
as a literal makes that judgement call visible and reviewable, and means a re-noded network
cannot silently reshuffle the published alternatives. ``test_corridors_partition_the_network``
asserts the lists still cover the live network exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph import Network, Route

#: Corridor name -> the trails in it, by the ``name`` column of the edge table.
CORRIDORS: dict[str, frozenset[str]] = {
    "Home Block": frozenset(
        {
            "Bike Park: Black",
            "Bike Park: Blue",
            "Bike Park: Climbing Trail",
            "Bike Park: Green",
            "Gateway",
            "Lower Chimney",
            "Mineral Way",
            "Possum Track",
            "Snyder's Knob",
            "Upper Chimney",
            "Wavelength",
            "Yard Sale",
        }
    ),
    "North Rim": frozenset(
        {
            "Chasers",
            "Grinder",
            "Horse Nettle",
            "Horse Nettle Connector",
            "Jacob's Ladder",
            "Joe Pye",
            "May Apple",
            "Old Road Bed",
            "Poverty Creek (upper)",
            "Prickly Pear",
            "Queen Anne",
            "Royale",
            "Running Cedar",
            "Slytherin",
            "Snakeroot",
        }
    ),
    "South Ridge": frozenset(
        {
            "Blunderbuss",
            "Highway",
            "Ida May",
            "Pine Forest",
            "Turkey Trot",
            "Wilkes Wood",
        }
    ),
    "West End": frozenset(
        {
            "Beauty",
            "Crosscut",
            "Head Hunter",
            "Indian Pipe",
            "Poverty Creek (lower)",
            "Skullcap",
            "Trillium",
        }
    ),
}

#: Display order: outward from the start line. Used for every table and legend.
CORRIDOR_ORDER = ("Home Block", "North Rim", "South Ridge", "West End")

#: The block the published alternatives are built around.
#:
#: It is the only one where the answer genuinely changes with speed. Reaching it costs a
#: long out-and-back that a slow racer cannot afford and a fast one barely notices, so
#: "commit to it" and "skip it" swap places as the binding constraint somewhere in the
#: middle of the published range. Constraining any other block would produce three routes
#: that differ in detail but agree about the interesting decision.
PIVOT_CORRIDOR = "West End"


@dataclass(frozen=True)
class CorridorRule:
    """A constraint on which corridors a route may or must cover.

    The same object is handed to both solvers and to the exporter, so the rule a route was
    searched under, the rule it was proved optimal under, and the rule published alongside
    it on the page are guaranteed to be the same statement. Keeping three copies of that
    in step by hand is exactly the kind of thing that goes wrong quietly.

    An empty rule means the free optimum, which is the case for every legacy pace preset.
    """

    #: Trails that must be completed in full.
    require: frozenset[int] = field(default_factory=frozenset)
    #: Trails whose ground may not be walked at all.
    forbid: frozenset[int] = field(default_factory=frozenset)
    #: Corridor this rule was built from, for display. ``None`` for the free optimum.
    corridor: str | None = None
    #: Which side of the rule this is: ``"require"``, ``"forbid"`` or ``None``.
    kind: str | None = None

    @property
    def is_free(self) -> bool:
        return not self.require and not self.forbid

    @property
    def named(self) -> bool:
        """Whether this rule can describe itself in words.

        Kept separate from :attr:`is_free` because the two can disagree: a rule may name a
        corridor while constraining nothing (nothing left to require after filtering), and
        the display text has to stay coherent either way rather than reading "Skip the None".
        """
        return bool(self.corridor and self.kind)

    @property
    def label(self) -> str:
        """Short name for a control on the page."""
        if not self.named:
            return "Best available"
        verb = "Commit to" if self.kind == "require" else "Skip"
        return f"{verb} the {self.corridor}"

    @property
    def description(self) -> str:
        """One sentence explaining what was asked of the solver, for the sidebar."""
        if not self.named:
            return (
                "No constraint. This is the highest-scoring itinerary the solver could "
                "find anywhere in the network."
            )
        if self.kind == "require":
            return (
                f"Every trail in the {self.corridor} must be finished. The rest of the "
                "route is then optimized around that commitment."
            )
        return (
            f"The {self.corridor} is off the table entirely — no trail in it may be "
            "walked. The route is optimized over what remains."
        )

    def as_dict(self) -> dict:
        """The rule as published alongside the itinerary it produced."""
        return {
            "corridor": self.corridor,
            "kind": self.kind,
            "label": self.label,
            "description": self.description,
            "require_trail_ids": sorted(self.require),
            "forbid_trail_ids": sorted(self.forbid),
        }

    def offenders(self, net: Network, route: Route) -> tuple[list[int], list[int]]:
        """``(required-but-missing, forbidden-but-walked)`` trail ids.

        The hot path. ALNS scores this on every decode — hundreds of thousands of times
        across a build — so it does set arithmetic and nothing else. Naming the trails costs
        a groupby over the whole edge table, which measured ~7 ms against ~0 ms for scoring
        the route itself; doing that here made a constrained search two orders of magnitude
        slower than an unconstrained one. Names are worth paying for when a preset is
        rejected and never otherwise, so they live in :meth:`violations`.
        """
        if self.is_free:
            return [], []
        walked = {arc.edge_id for arc in route.arcs}
        return (
            [t for t in self.require if not net.trail_edges.get(t, frozenset()) <= walked],
            [t for t in self.forbid if net.trail_edges.get(t, frozenset()) & walked],
        )

    def violation_count(self, net: Network, route: Route) -> int:
        missing, walked = self.offenders(net, route)
        return len(missing) + len(walked)

    def violations(self, net: Network, route: Route) -> list[str]:
        """Every way ``route`` breaks this rule, as readable strings.

        Returns a list rather than a bool because the messages are surfaced verbatim when a
        preset is rejected, and "which trail" is the first thing worth knowing.
        """
        missing, walked = self.offenders(net, route)
        if not missing and not walked:
            return []

        names = {
            int(tid): str(group["name"].iloc[0])
            for tid, group in net.edges[net.edges["trail_id"].notna()].groupby("trail_id")
        }
        return [
            f"required trail {names.get(tid, tid)!r} was not completed"
            for tid in sorted(missing)
        ] + [
            f"forbidden trail {names.get(tid, tid)!r} was walked"
            for tid in sorted(walked)
        ]


def free_rule() -> CorridorRule:
    """The unconstrained rule — the honest optimum, with nothing imposed."""
    return CorridorRule()


def require_rule(net: Network, corridor: str = PIVOT_CORRIDOR) -> CorridorRule:
    """Insist every trail in ``corridor`` is completed."""
    return CorridorRule(
        require=trail_ids(net, corridor), corridor=corridor, kind="require"
    )


def forbid_rule(net: Network, corridor: str = PIVOT_CORRIDOR) -> CorridorRule:
    """Bar the route from walking ``corridor`` at all."""
    return CorridorRule(forbid=trail_ids(net, corridor), corridor=corridor, kind="forbid")


def corridor_of(trail_name: str) -> str | None:
    """Which block a trail belongs to, or ``None`` if it is not a scored trail."""
    for corridor, members in CORRIDORS.items():
        if trail_name in members:
            return corridor
    return None


def trail_ids(net: Network, corridor: str) -> frozenset[int]:
    """The ``trail_id``s of one corridor, as the network actually numbers them.

    Resolved against the live network rather than stored, because ``trail_id`` is assigned
    upstream by the scrape and is not a stable thing to hardcode; the names are.
    """
    if corridor not in CORRIDORS:
        raise KeyError(f"unknown corridor {corridor!r}; expected one of {CORRIDOR_ORDER}")

    members = CORRIDORS[corridor]
    on_trail = net.edges[net.edges["trail_id"].notna()]
    return frozenset(
        int(tid)
        for tid, group in on_trail.groupby("trail_id")
        if str(group["name"].iloc[0]) in members
    )


def edge_ids(net: Network, corridor: str) -> frozenset[int]:
    """Every edge belonging to a corridor's trails."""
    return frozenset(
        edge
        for tid in trail_ids(net, corridor)
        for edge in net.trail_edges.get(tid, frozenset())
    )


def breakdown(net: Network, route: Route) -> list[dict]:
    """Per-corridor unique miles and completed trails for one route.

    Unique miles, not miles walked: an edge re-walked to get somewhere else is counted
    once, which is what the race pays for and what the totals elsewhere in the preset
    mean. A corridor the route never enters still gets a row, with zeroes — the absence is
    the interesting part when the whole point of an alternative is that it skipped
    something.
    """
    walked_edges = {arc.edge_id for arc in route.arcs}

    rows = []
    for corridor in CORRIDOR_ORDER:
        tids = trail_ids(net, corridor)
        edges = {e for tid in tids for e in net.trail_edges.get(tid, frozenset())}
        touched = edges & walked_edges

        rows.append(
            {
                "corridor": corridor,
                "unique_miles": round(
                    sum(net.edge_score_mi.get(e, 0.0) for e in touched), 3
                ),
                "total_miles": round(
                    sum(net.edge_score_mi.get(e, 0.0) for e in edges), 3
                ),
                "trails_completed": sum(
                    1
                    for tid in tids
                    if net.trail_edges.get(tid, frozenset()) <= walked_edges
                ),
                "n_trails": len(tids),
            }
        )
    return rows
