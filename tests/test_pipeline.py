"""End-to-end checks over the built artifacts.

These run against the materialized pipeline outputs rather than rebuilding from scratch,
so they are fast enough to run on every change. Anything requiring a missing artifact
skips rather than fails, so a partially-built checkout still gives useful signal.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest

from hullabaloo.config import (
    CONFIG,
    CONNECTORS,
    EDGES,
    EDGES_TIMED,
    M_PER_MILE,
    NODES,
    TRAILS_RAW,
)
from hullabaloo.tobler import profile_travel_time, tobler_speed_kmh
from hullabaloo.topology import connected_components

EXPECTED_TRAILS = 40
EXPECTED_MILES = 40.14


def _load(path):
    if not path.exists():
        pytest.skip(f"{path.name} not built yet")
    return gpd.read_parquet(path)


# --------------------------------------------------------------------------------------
# Tobler
# --------------------------------------------------------------------------------------


def test_tobler_peak_is_on_a_gentle_downhill():
    """Tobler's function peaks at -s0, not on the flat. This asymmetry is the reason the
    routing graph must be directed, so it is worth pinning down."""
    params = CONFIG.tobler
    slopes = np.linspace(-0.5, 0.5, 2001)
    peak = slopes[np.argmax(tobler_speed_kmh(slopes, params))]
    assert peak == pytest.approx(-params.s0, abs=1e-3)


def test_tobler_flat_pace_is_plausible():
    mph = float(tobler_speed_kmh(0.0, CONFIG.tobler)) * 0.621371
    assert 2.5 < mph < 4.0, f"flat pace {mph:.2f} mph is not a believable hiking speed"


def test_travel_time_is_direction_dependent_on_a_slope():
    """Climbing must cost more than descending, by exactly the ratio Tobler implies.

    At a 20% grade that ratio is about 1.42, not something larger — the function is far
    gentler on climbs than intuition suggests, which is precisely why the optimizer is
    willing to route uphill when the points justify it.
    """
    grade = 0.20
    distances = np.array([0.0, 100.0, 200.0])
    elevations = np.array([0.0, 20.0, 40.0])  # steady 20% climb

    up = profile_travel_time(distances, elevations, CONFIG.tobler)
    down = profile_travel_time(distances, elevations, CONFIG.tobler, reverse=True)

    expected = float(
        tobler_speed_kmh(-grade, CONFIG.tobler) / tobler_speed_kmh(grade, CONFIG.tobler)
    )
    assert up > down
    assert up / down == pytest.approx(expected, rel=1e-6)
    assert expected == pytest.approx(1.42, abs=0.02)


def test_offtrail_is_slower_by_the_configured_factor():
    distances = np.array([0.0, 100.0])
    elevations = np.array([0.0, 0.0])
    on = profile_travel_time(distances, elevations, CONFIG.tobler)
    off = profile_travel_time(distances, elevations, CONFIG.tobler, off_trail=True)
    assert off == pytest.approx(on / CONFIG.tobler.off_trail_factor, rel=1e-6)


def test_segmentwise_integration_beats_averaging_on_rolling_terrain():
    """A trail that goes up then down must cost more than the flat equivalent, even
    though its net elevation change is zero. Averaging slope first would miss this."""
    distances = np.array([0.0, 100.0, 200.0])
    rolling = profile_travel_time(distances, np.array([0.0, 30.0, 0.0]), CONFIG.tobler)
    flat = profile_travel_time(distances, np.array([0.0, 0.0, 0.0]), CONFIG.tobler)
    assert rolling > flat


# --------------------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------------------


def test_all_trails_ingested():
    trails = _load(TRAILS_RAW)
    assert len(trails) == EXPECTED_TRAILS
    assert trails["trail_id"].is_unique
    assert (trails.geometry.geom_type == "LineString").all()
    assert (trails["n_points"] >= 2).all()


def test_ingested_length_matches_published_total():
    trails = _load(TRAILS_RAW)
    miles = trails.to_crs("EPSG:6346").length.sum() / M_PER_MILE
    assert miles == pytest.approx(EXPECTED_MILES, abs=0.5)


# --------------------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------------------


def test_network_is_structurally_clean():
    edges, nodes = _load(EDGES), _load(NODES)
    assert (edges["length_m"] > 0).all(), "zero-length edges present"
    degenerate = (edges["u"] == edges["v"]) & (edges["length_m"] < CONFIG.topology.snap_tol_m)
    assert not degenerate.any(), "collapsed self-loops present"
    assert edges["edge_id"].is_unique
    assert nodes["node_id"].is_unique
    assert nodes["is_depot"].sum() == 1


def test_splitting_preserved_length():
    trails, edges = _load(TRAILS_RAW), _load(EDGES)
    raw_m = trails.to_crs("EPSG:6346").length.sum()
    split_m = edges.loc[~edges["off_trail"], "length_m"].sum()
    assert split_m == pytest.approx(raw_m, rel=0.005)


def test_every_trail_survived_and_is_contiguous():
    edges = _load(EDGES)
    on_trail = edges[~edges["off_trail"]]
    assert on_trail["trail_id"].nunique() == EXPECTED_TRAILS
    for trail_id, grp in on_trail.groupby("trail_id"):
        used = np.unique(np.concatenate([grp["u"].values, grp["v"].values]))
        comp = connected_components(grp, int(used.max()) + 1)
        assert len({comp[int(n)] for n in used}) == 1, f"trail {trail_id} is fragmented"


def test_trail_network_alone_has_three_components():
    """The three components are separated by genuine 285-700 m gaps. If this ever changes
    the bushwhack phase's reason for existing has changed with it."""
    edges, nodes = _load(EDGES), _load(NODES)
    comp = connected_components(edges, len(nodes))
    assert int(comp.max()) + 1 == 3


def test_connectors_make_the_network_connected():
    import pandas as pd

    edges, nodes = _load(EDGES_TIMED), _load(NODES)
    connectors = _load(CONNECTORS)
    combined = pd.concat([edges[["u", "v"]], connectors[["u", "v"]]], ignore_index=True)
    assert int(connected_components(combined, len(nodes)).max()) + 1 == 1


# --------------------------------------------------------------------------------------
# Elevation / pricing
# --------------------------------------------------------------------------------------


def test_edges_are_priced_in_both_directions():
    edges = _load(EDGES_TIMED)
    assert (edges["time_fwd_s"] > 0).all()
    assert (edges["time_rev_s"] > 0).all()
    # On real terrain the two directions must not be identical everywhere.
    assert not np.allclose(edges["time_fwd_s"], edges["time_rev_s"])


def test_offtrail_edges_score_nothing():
    edges = _load(EDGES_TIMED)
    assert (edges.loc[edges["off_trail"], "score_mi"] == 0).all()
    on = edges[~edges["off_trail"]]
    assert np.allclose(on["score_mi"], on["length_m"] / M_PER_MILE)


def test_connectors_are_slower_than_trail_for_the_same_ground():
    connectors = _load(CONNECTORS)
    edges = _load(EDGES_TIMED)
    off_speed = (connectors["length_m"] / connectors["time_fwd_s"]).mean()
    on = edges[~edges["off_trail"]]
    on_speed = (on["length_m"] / on["time_fwd_s"]).mean()
    assert off_speed < on_speed * 0.75


# --------------------------------------------------------------------------------------
# Graph / routing
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def net():
    if not (EDGES_TIMED.exists() and NODES.exists()):
        pytest.skip("network not built yet")
    from hullabaloo.graph import build_network

    return build_network()


def test_graph_has_two_arcs_per_edge(net):
    assert len(net.arcs) == 2 * len(net.edges)
    assert net.n_trails == EXPECTED_TRAILS


def test_every_trail_forms_a_walkable_chain(net):
    from hullabaloo.optimize_alns import build_trail_chains

    chains = build_trail_chains(net)
    assert len(chains) == EXPECTED_TRAILS
    for trail_id, chain in chains.items():
        assert len(chain.forward) == len(net.trail_edges[trail_id])
        for a, b in zip(chain.forward[:-1], chain.forward[1:]):
            assert a.v == b.u, f"trail {trail_id} chain is discontinuous"


def test_scoring_rewards_completion_not_repetition(net):
    from hullabaloo.graph import score_edges

    trail_id, edges = next(iter(net.trail_edges.items()))
    full, miles, trails = score_edges(edges, net)
    assert trails == 1
    # Listing the same edges twice must not change anything.
    again, miles_again, trails_again = score_edges(list(edges) * 2, net)
    assert (again, round(miles_again, 6)) == (full, round(miles, 6))
    # Dropping one edge forfeits the trail point but keeps the partial mileage.
    if len(edges) > 1:
        partial, partial_mi, partial_trails = score_edges(list(edges)[:-1], net)
        assert partial_trails == 0
        assert partial_mi < miles


def test_baseline_routes_are_valid_and_within_budget(net):
    from hullabaloo.optimize_alns import baseline_best_ratio, baseline_greedy, build_trail_chains

    chains = build_trail_chains(net)
    for route in (baseline_greedy(net, chains), baseline_best_ratio(net, chains)):
        assert route.validate() == []
        assert route.time_s <= CONFIG.race.time_budget_s + 1e-6
        assert route.arcs[0].u == net.depot
        assert route.arcs[-1].v == net.depot


def test_alns_beats_the_greedy_baseline(net):
    from hullabaloo.optimize_alns import ALNS, baseline_greedy, build_trail_chains, evaluate

    chains = build_trail_chains(net)
    baseline = evaluate(baseline_greedy(net, chains))
    result = ALNS(net, chains, seed=0).solve(iterations=15)
    assert result.route.validate() == []
    assert result.score >= baseline


def test_seven_hours_is_a_binding_constraint(net):
    """If the budget were generous enough to cover everything the optimization would be
    pointless, so assert the premise of the whole project."""
    from hullabaloo.graph import network_traversal_bound

    bound = network_traversal_bound(net)
    assert bound["cover_all_lower_bound_h"] > CONFIG.race.time_budget_s / 3600
