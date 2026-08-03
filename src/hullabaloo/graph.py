"""Phase 5 — assemble the directed, time-weighted routing graph and score routes.

Two things live here that the rest of the project leans on:

* :class:`Network` — the directed graph plus the lookup tables the optimizer needs
  (arcs by node, all-pairs shortest times, which edges belong to which trail).
* :func:`score_route` — the *single* definition of what a route is worth. Every
  optimizer, baseline, and report scores through this one function, so the scoring rule
  can be swapped in one place if the organizer clarifies it differently.

Scoring, as confirmed with the organizer's rules:

    score = min(trails_fully_completed, 40) + min(unique_trail_miles, 40)

Repeated traversal earns nothing extra, and forest roads earn nothing at all —
they only cost time. That combination is what makes this a prize-collecting arc routing
problem rather than a shortest-path or a plain TSP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from .config import (
    CONFIG,
    EDGES_TIMED,
    GRAPH_EDGES,
    NODES,
    RaceParams,
)

log = logging.getLogger(__name__)


@dataclass
class Arc:
    """One direction of one edge."""

    arc_id: int
    edge_id: int
    u: int
    v: int
    time_s: float
    score_mi: float
    trail_id: int | None
    off_trail: bool
    name: str


@dataclass
class Network:
    edges: gpd.GeoDataFrame
    nodes: gpd.GeoDataFrame
    arcs: list[Arc]
    depot: int
    #: edge_id -> set of edge_ids that make up that trail
    trail_edges: dict[int, frozenset[int]] = field(default_factory=dict)
    edge_score_mi: dict[int, float] = field(default_factory=dict)
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    #: node -> node -> seconds
    sp_time: dict[int, dict[int, float]] = field(default_factory=dict)
    sp_path: dict[int, dict[int, list[int]]] = field(default_factory=dict)

    @property
    def n_trails(self) -> int:
        return len(self.trail_edges)

    def arcs_from(self, node: int) -> list[Arc]:
        return self._by_tail.get(node, [])

    def __post_init__(self) -> None:
        self._by_tail: dict[int, list[Arc]] = {}
        for arc in self.arcs:
            self._by_tail.setdefault(arc.u, []).append(arc)


def build_network(
    edges: gpd.GeoDataFrame | None = None,
    nodes: gpd.GeoDataFrame | None = None,
) -> Network:
    """Turn the edge and node tables into the directed, time-weighted routing graph.

    Every edge here is legal walkable ground: scored trail, forest road, or the short
    off-trail link from the start line to the network. The graph once also carried
    least-cost bushwhack connectors; importing the roads that were actually missing made
    every one of them unattractive, so the stage that generated them is gone.
    """
    edges = gpd.read_parquet(EDGES_TIMED) if edges is None else edges
    nodes = gpd.read_parquet(NODES) if nodes is None else nodes
    combined = gpd.GeoDataFrame(edges.copy(), geometry="geometry", crs=edges.crs)

    depot_rows = nodes.loc[nodes.get("is_depot", pd.Series(False, index=nodes.index))]
    if depot_rows.empty:
        raise ValueError("No depot node flagged in the node table.")
    depot = int(depot_rows["node_id"].iloc[0])

    arcs: list[Arc] = []
    for row in combined.itertuples(index=False):
        trail_id = None if pd.isna(row.trail_id) else int(row.trail_id)
        for u, v, t in (
            (int(row.u), int(row.v), float(row.time_fwd_s)),
            (int(row.v), int(row.u), float(row.time_rev_s)),
        ):
            arcs.append(
                Arc(
                    arc_id=len(arcs),
                    edge_id=int(row.edge_id),
                    u=u,
                    v=v,
                    time_s=t,
                    score_mi=float(row.score_mi),
                    trail_id=trail_id,
                    off_trail=bool(row.off_trail),
                    name=str(row.name),
                )
            )

    on_trail = combined[combined["trail_id"].notna()]
    trail_edges = {
        int(tid): frozenset(int(e) for e in grp["edge_id"])
        for tid, grp in on_trail.groupby("trail_id")
    }
    edge_score = {
        int(r.edge_id): float(r.score_mi) for r in combined.itertuples(index=False)
    }

    graph = nx.DiGraph()
    graph.add_nodes_from(int(n) for n in nodes["node_id"])
    for arc in arcs:
        if not graph.has_edge(arc.u, arc.v) or graph[arc.u][arc.v]["time_s"] > arc.time_s:
            graph.add_edge(arc.u, arc.v, time_s=arc.time_s, arc_id=arc.arc_id)

    net = Network(
        edges=combined,
        nodes=nodes,
        arcs=arcs,
        depot=depot,
        trail_edges=trail_edges,
        edge_score_mi=edge_score,
        graph=graph,
    )
    compute_shortest_paths(net)
    return net


def compute_shortest_paths(net: Network) -> None:
    """All-pairs shortest travel time, used by the ALNS repair operator."""
    net.sp_time = {}
    net.sp_path = {}
    for source in net.graph.nodes:
        times, paths = nx.single_source_dijkstra(net.graph, source, weight="time_s")
        net.sp_time[source] = times
        net.sp_path[source] = paths


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def score_edges(
    edge_ids, net: Network, race: RaceParams | None = None
) -> tuple[float, float, int]:
    """Return ``(score, unique_miles, trails_completed)`` for a set of traversed edges.

    The single source of truth for what a route is worth.
    """
    race = race or CONFIG.race
    used = frozenset(int(e) for e in edge_ids)
    unique_mi = sum(net.edge_score_mi.get(e, 0.0) for e in used)
    trails_done = sum(1 for edges in net.trail_edges.values() if edges <= used)

    trail_pts = min(trails_done * race.points_per_trail, race.max_trail_points)
    mile_pts = min(unique_mi * race.points_per_mile, race.max_mile_points)
    return trail_pts + mile_pts, unique_mi, trails_done


def route_arcs_to_edges(arc_seq: list[Arc]) -> set[int]:
    return {a.edge_id for a in arc_seq}


@dataclass
class Route:
    """A closed walk from the depot back to the depot."""

    arcs: list[Arc]
    net: Network
    race: RaceParams | None = None

    @property
    def nodes(self) -> list[int]:
        if not self.arcs:
            return [self.net.depot]
        return [self.arcs[0].u] + [a.v for a in self.arcs]

    @property
    def time_s(self) -> float:
        return sum(a.time_s for a in self.arcs)

    @property
    def edges_used(self) -> set[int]:
        return route_arcs_to_edges(self.arcs)

    def evaluate(self) -> dict:
        score, miles, trails = score_edges(self.edges_used, self.net, self.race)
        race = self.race or CONFIG.race
        lengths = self.net.edges.set_index("edge_id")["length_m"]
        walked_mi = sum(lengths.get(a.edge_id, 0.0) for a in self.arcs) / 1609.344
        offtrail_mi = (
            sum(lengths.get(a.edge_id, 0.0) for a in self.arcs if a.off_trail) / 1609.344
        )
        return {
            "score": round(score, 3),
            "unique_miles": round(miles, 3),
            "trails_completed": trails,
            "walked_miles": round(walked_mi, 2),
            "offtrail_miles": round(offtrail_mi, 2),
            "repeat_miles": round(walked_mi - offtrail_mi - miles, 2),
            "time_h": round(self.time_s / 3600, 3),
            "time_budget_h": round(race.time_budget_s / 3600, 2),
            "feasible": self.time_s <= race.time_budget_s + 1e-6,
            "n_arcs": len(self.arcs),
        }

    def validate(self) -> list[str]:
        """Structural checks — a route that fails any of these is not a real route."""
        problems: list[str] = []
        if not self.arcs:
            return ["route is empty"]
        if self.arcs[0].u != self.net.depot:
            problems.append(f"starts at {self.arcs[0].u}, not depot {self.net.depot}")
        if self.arcs[-1].v != self.net.depot:
            problems.append(f"ends at {self.arcs[-1].v}, not depot {self.net.depot}")
        for a, b in zip(self.arcs[:-1], self.arcs[1:]):
            if a.v != b.u:
                problems.append(f"discontinuity: arc ends at {a.v}, next starts at {b.u}")
                break
        race = self.race or CONFIG.race
        if self.time_s > race.time_budget_s + 1e-6:
            problems.append(
                f"over budget: {self.time_s / 3600:.2f} h > {race.time_budget_s / 3600:.2f} h"
            )
        return problems


# --------------------------------------------------------------------------------------
# Reference values
# --------------------------------------------------------------------------------------


def network_traversal_bound(net: Network) -> dict:
    """How long would it take to cover *everything*? Establishes that 7 h is binding.

    This is a lower bound on a full-coverage tour: it sums the cheaper direction of every
    on-trail edge and ignores all the deadheading a real closed walk would require, so a
    true Rural Postman tour can only be slower.
    """
    by_edge: dict[int, float] = {}
    for arc in net.arcs:
        if arc.trail_id is None:  # roads are not required coverage
            continue
        by_edge[arc.edge_id] = min(by_edge.get(arc.edge_id, np.inf), arc.time_s)

    total_s = sum(by_edge.values())
    total_mi = sum(net.edge_score_mi.get(e, 0.0) for e in by_edge)
    race = CONFIG.race
    return {
        "cover_all_lower_bound_h": round(total_s / 3600, 2),
        "budget_h": race.time_budget_s / 3600,
        "fraction_of_network_reachable": round(race.time_budget_s / total_s, 3),
        "total_network_miles": round(total_mi, 2),
        "max_possible_score": 80.0,
    }


def run() -> Network:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    net = build_network()
    net.edges.to_parquet(GRAPH_EDGES)
    log.info("graph: %d nodes, %d arcs, %d trails", len(net.nodes), len(net.arcs), net.n_trails)
    log.info("coverage bound: %s", network_traversal_bound(net))
    return net


if __name__ == "__main__":
    run()
