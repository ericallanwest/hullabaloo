"""End-to-end checks over the built artifacts.

These run against the materialized pipeline outputs rather than rebuilding from scratch,
so they are fast enough to run on every change. Anything requiring a missing artifact
skips rather than fails, so a partially-built checkout still gives useful signal.
"""

from __future__ import annotations

import dataclasses

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
    """Bound the modelled flat pace at both ends of the pace_factor decision.

    The configured pace allows up to 4.5 mph, not walking speed: ``pace_factor`` is
    deliberately 1.35 for this racer (see ``ToblerParams``), which puts the flat pace at
    4.23 mph — a fit competitor moving with purpose over seven hours, not a stroller.
    Unscaled Tobler must still land in ordinary walking territory, so both are checked;
    asserting only the scaled figure would let a bad ``base_kmh`` hide inside the
    multiplier.
    """
    mph = float(tobler_speed_kmh(0.0, CONFIG.tobler)) * 0.621371
    assert 2.5 < mph < 4.5, f"flat pace {mph:.2f} mph is not a believable racing speed"

    textbook = dataclasses.replace(CONFIG.tobler, pace_factor=1.0)
    base_mph = float(tobler_speed_kmh(0.0, textbook)) * 0.621371
    assert 2.5 < base_mph < 4.0, f"unscaled Tobler {base_mph:.2f} mph is not a walking speed"


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
    """Splitting must neither lose nor duplicate trail geometry. Compare scored trail
    only — forest roads add length that was never in the GPX files."""
    trails, edges = _load(TRAILS_RAW), _load(EDGES)
    raw_m = trails.to_crs("EPSG:6346").length.sum()
    split_m = edges.loc[edges["trail_id"].notna(), "length_m"].sum()
    assert split_m == pytest.approx(raw_m, rel=0.005)


def test_every_trail_survived_and_is_contiguous():
    edges = _load(EDGES)
    on_trail = edges[~edges["off_trail"]]
    assert on_trail["trail_id"].nunique() == EXPECTED_TRAILS
    for trail_id, grp in on_trail.groupby("trail_id"):
        used = np.unique(np.concatenate([grp["u"].values, grp["v"].values]))
        comp = connected_components(grp, int(used.max()) + 1)
        assert len({comp[int(n)] for n in used}) == 1, f"trail {trail_id} is fragmented"


def test_adding_the_depot_never_destroys_edges():
    """Regression: the depot insertion used ``frame.loc[len(frame)] = ...`` to append.

    That is only safe on a clean RangeIndex. After dropping the edge being split,
    ``len(frame)`` still names an existing label, so the "append" silently overwrote a
    real row. It stayed invisible because the actual start point happens to land on an
    existing node, skipping the split branch entirely — so this test forces the split
    branch by anchoring the depot mid-edge.
    """
    from unittest.mock import patch

    from hullabaloo import topology as topo

    trails = _load(TRAILS_RAW)
    plain_edges, plain_nodes, _ = topo.build_network(trails, add_depot=False)

    # Pick a point squarely in the middle of a long edge so the split branch must run.
    longest = plain_edges.loc[plain_edges["length_m"].idxmax()]
    midpoint = longest.geometry.interpolate(longest.geometry.length / 2)
    lon, lat = (
        gpd.GeoSeries([midpoint], crs="EPSG:6346").to_crs("EPSG:4326").iloc[0].coords[0]
    )

    with patch.object(topo, "START_LON", lon), patch.object(topo, "START_LAT", lat):
        edges, nodes, depot = topo.build_network(trails, add_depot=True)

    # One edge becomes two (+1) and the depot access edge is added (+1).
    assert len(edges) == len(plain_edges) + 2
    assert len(nodes) == len(plain_nodes) + 2
    assert edges["edge_id"].is_unique
    assert nodes["node_id"].is_unique
    assert edges.loc[edges["off_trail"], "name"].tolist() == ["depot access"]

    on_trail_m = edges.loc[~edges["off_trail"], "length_m"].sum()
    assert on_trail_m == pytest.approx(plain_edges["length_m"].sum(), rel=1e-6)


def test_trails_alone_split_into_three_components():
    """The 40 trails, considered by themselves, are in three pieces separated by genuine
    285-700 m gaps. This is the fact that motivates bringing in forest roads at all."""
    edges, nodes = _load(EDGES), _load(NODES)
    trails_only = edges[edges["trail_id"].notna()]
    comp = connected_components(trails_only, len(nodes))
    reachable = {comp[int(u)] for u in trails_only["u"]}
    assert len(reachable) == 3


def test_forest_roads_connect_the_whole_network():
    """With the forest roads included every trail is reachable without going off-trail.

    This is the finding that reshaped the project: the gaps between the three trail
    components are spanned by legal, full-speed forest road, so a good route needs
    essentially no bushwhacking.
    """
    edges, nodes = _load(EDGES), _load(NODES)
    assert int(connected_components(edges, len(nodes)).max()) + 1 == 1
    roads = edges[edges["is_road"] == True]  # noqa: E712
    assert len(roads) > 0, "no forest roads in the network"
    assert roads["trail_id"].isna().all(), "roads must not carry a scoring trail_id"


def test_roads_cost_time_but_score_nothing():
    edges = _load(EDGES_TIMED)
    roads = edges[edges["is_road"] == True]  # noqa: E712
    assert (roads["score_mi"] == 0).all(), "forest roads must not earn points"
    assert (roads["time_fwd_s"] > 0).all()
    # Roads are walked at full speed, unlike the 60% bushwhack penalty.
    assert not roads["off_trail"].any()
    road_speed = (roads["length_m"] / roads["time_fwd_s"]).mean()
    trail = edges[edges["trail_id"].notna()]
    trail_speed = (trail["length_m"] / trail["time_fwd_s"]).mean()
    assert road_speed > trail_speed * 0.8


# --------------------------------------------------------------------------------------
# Elevation / pricing
# --------------------------------------------------------------------------------------


def test_edges_are_priced_in_both_directions():
    edges = _load(EDGES_TIMED)
    assert (edges["time_fwd_s"] > 0).all()
    assert (edges["time_rev_s"] > 0).all()
    # On real terrain the two directions must not be identical everywhere.
    assert not np.allclose(edges["time_fwd_s"], edges["time_rev_s"])


def test_only_scored_trails_earn_points():
    """Points come from the 40 scored trails alone. Bushwhack connectors and forest roads
    both score zero; they differ only in speed."""
    edges = _load(EDGES_TIMED)
    assert (edges.loc[edges["off_trail"], "score_mi"] == 0).all()
    assert (edges.loc[edges["trail_id"].isna(), "score_mi"] == 0).all()
    scored = edges[edges["trail_id"].notna()]
    assert np.allclose(scored["score_mi"], scored["length_m"] / M_PER_MILE)
    assert scored["score_mi"].sum() == pytest.approx(EXPECTED_MILES, abs=0.1)


def test_connectors_avoid_water_and_developed_land():
    """Regression: the cost surface is built from a *bare-earth* DEM, which sees only
    gentle slope where houses, driveways and lawns are.

    Before the land-cover mask, the optimal route's longest bushwhack ran 848 m straight
    through a residential neighbourhood. Slope and water checks alone would not have
    caught it — it took looking at aerial imagery.
    """
    from hullabaloo.bushwhack import DEVELOPED_CLASSES, fetch_landcover, fetch_waterbodies

    connectors = _load(CONNECTORS)
    edges = _load(EDGES_TIMED)
    bounds = tuple(edges.to_crs("EPSG:4326").total_bounds)

    try:
        water = fetch_waterbodies(bounds)
        classes, cls_bounds = fetch_landcover(bounds, width=900, height=900)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"external land-cover/hydrography services unavailable: {exc}")

    if len(water):
        assert not len(gpd.sjoin(connectors, water, predicate="intersects", how="inner"))

    # Sample each connector's vertices against the land-cover grid.
    minx, miny, maxx, maxy = cls_bounds
    h, w = classes.shape
    developed_hits = []
    for row in connectors.to_crs("EPSG:4326").itertuples(index=False):
        xs = np.array([c[0] for c in row.geometry.coords])
        ys = np.array([c[1] for c in row.geometry.coords])
        px = np.clip(((xs - minx) / (maxx - minx) * w).astype(int), 0, w - 1)
        py = np.clip(((maxy - ys) / (maxy - miny) * h).astype(int), 0, h - 1)
        sampled = classes[py, px]
        # The depot access legitimately starts in the trailhead parking lot.
        if row.name == "depot access":
            continue
        fraction = float(np.isin(sampled, DEVELOPED_CLASSES).mean())
        if fraction > 0.25:
            developed_hits.append((row.name, round(fraction, 2)))

    assert not developed_hits, f"connectors crossing developed land: {developed_hits}"


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


def test_milp_model_builds_on_a_network_with_self_loops(net):
    """Regression: a self-loop edge (u == v) made ``build_model`` emit one constraint
    name twice, which PuLP rejects, crashing the whole optimization stage.

    The network really does contain a 31 m switchback where a trail returns to its own
    node, and it only surfaced once the forest roads shifted where trails get split — so
    this asserts the model builds at all, and that the self-loop is genuinely present.
    """
    from hullabaloo.optimize_milp import build_model

    loops = [a for a in net.arcs if a.u == a.v]
    prob, vars_ = build_model(net)
    assert len(prob.constraints) > 0
    assert len(vars_["x"]) == len(net.arcs)
    if loops:
        # Each self-loop still gets its endpoint-activation constraint, just once.
        assert any(name.startswith(f"active_{loops[0].u}_") for name in prob.constraints)


def test_seven_hours_is_a_binding_constraint(net):
    """If the budget were generous enough to cover everything the optimization would be
    pointless, so assert the premise of the whole project."""
    from hullabaloo.graph import network_traversal_bound

    bound = network_traversal_bound(net)
    assert bound["cover_all_lower_bound_h"] > CONFIG.race.time_budget_s / 3600


# --------------------------------------------------------------------------------------
# Pace re-pricing
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coarse_dem():
    """The coarsened surface the connectors were routed over, plus its transform."""
    from hullabaloo.bushwhack import elevation_surface
    from hullabaloo.elevation import load_dem

    try:
        array, transform, _, _ = load_dem()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"DEM not available: {exc}")
    return elevation_surface(array, transform)


def test_reprice_reproduces_the_stored_connector_times(coarse_dem):
    """The pace sweep re-times connectors instead of re-routing them, which is only sound
    if re-timing at the stored pace returns the stored numbers exactly.

    This is the sweep's one silent failure mode. ``build_network`` reads connector times
    straight off the connector frame, so a re-pricing bug leaves bushwhacks at one pace
    while every trail moves to another — a mixed-pace model that still solves cleanly and
    produces a plausible-looking route at a pace that exists nowhere.
    """
    from hullabaloo.bushwhack import reprice

    connectors = _load(CONNECTORS)
    elevation, transform = coarse_dem
    again = reprice(connectors, elevation, transform, CONFIG.tobler)

    for column in ("time_fwd_s", "time_rev_s"):
        assert np.allclose(again[column], connectors[column], atol=1e-9), column


def test_pace_factor_scales_travel_time_inversely(coarse_dem):
    """Halving the pace must exactly double the time — the property that lets the sweep
    re-price rather than re-route, since relative costs are then unchanged."""
    import dataclasses

    from hullabaloo.bushwhack import reprice

    connectors = _load(CONNECTORS)
    elevation, transform = coarse_dem
    slow = reprice(
        connectors,
        elevation,
        transform,
        dataclasses.replace(CONFIG.tobler, pace_factor=CONFIG.tobler.pace_factor / 2),
    )
    ratio = slow["time_fwd_s"].to_numpy() / connectors["time_fwd_s"].to_numpy()
    assert np.allclose(ratio, 2.0, rtol=1e-9)


# --------------------------------------------------------------------------------------
# Web export
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_route(net):
    """A real, valid route — cheap enough to build on every test run."""
    from hullabaloo.optimize_alns import baseline_greedy, build_trail_chains

    return baseline_greedy(net, build_trail_chains(net))


def test_traversal_categories_partition_the_walk(net, sample_route):
    """``unique + repeat + offtrail`` must account for every mile walked, exactly.

    These are the four numbers the sidebar shows, so if they do not add up the page is
    lying. Note this is a different split from ``Route.evaluate()``: roads are folded in
    with bushwhacks, and a re-walked connector counts as a repeat rather than off-trail.
    """
    from hullabaloo.export import route_gdf

    detail = route_gdf(sample_route)
    assert set(detail["cat"]) <= {"unique", "repeat", "offtrail"}

    by_cat = detail.groupby("cat")["length_m"].sum() / M_PER_MILE
    assert by_cat.sum() == pytest.approx(sample_route.evaluate()["walked_miles"], abs=0.01)
    # Unique miles are exactly the scoring miles, by construction.
    assert by_cat.get("unique", 0.0) == pytest.approx(
        sample_route.evaluate()["unique_miles"], abs=0.01
    )
    # Only first passes over a scored trail may be marked unique.
    unique = detail[detail["cat"] == "unique"]
    assert unique["trail_id"].notna().all()
    assert unique["first_visit"].all()


def test_group_arcs_splits_a_first_pass_from_an_adjacent_repeat():
    """The web export keys legs on ``(label, category)`` rather than the label alone.

    Walking part of a trail, looping away and returning to finish it produces adjacent
    arcs with the same name but opposite meanings. Keying on the label alone merges them
    into a single row that is half new mileage and half not.
    """
    import pandas as pd

    from hullabaloo.export import _group_arcs, leg_label

    def arc(cat):
        return {
            "name": "Gateway", "off_trail": False, "cat": cat, "trail_id": 1,
            "from_node": 1, "to_node": 2, "length_m": 100.0, "gain_m": 5.0,
            "loss_m": 1.0, "time_s": 60.0, "elapsed_s": 60.0, "running_miles": 0.06,
            "running_trails": 0, "running_score": 0.06,
        }

    detail = pd.DataFrame([arc("unique"), arc("repeat")])
    assert len(_group_arcs(detail, leg_label)) == 1
    assert len(_group_arcs(detail, lambda row: (leg_label(row), row.cat))) == 2


def test_preset_document_reconciles(net, sample_route):
    from hullabaloo import webexport

    document = webexport.preset_dict(net, sample_route, pace_factor=CONFIG.tobler.pace_factor)
    webexport.check_preset(document)  # raises if any published number fails to add up

    assert document["totals"]["n_steps"] == len(document["steps"])
    assert all(step["geometry"] for step in document["steps"])
    # Cumulative fields must be monotonic — a racer cannot un-walk a mile or lose a point.
    scores = [step["cum"]["score"] for step in document["steps"]]
    seconds = [step["cum"]["seconds"] for step in document["steps"]]
    assert scores == sorted(scores)
    assert seconds == sorted(seconds)
    # Every completed trail must be attributed to a real step.
    completed = [t for t in document["trails"] if t["completed_at_step"]]
    assert len(completed) == document["totals"]["trails_completed"]
    assert all(1 <= t["completed_at_step"] <= len(document["steps"]) for t in completed)


def test_preset_filename_matches_the_front_end():
    """``docs/js/viz.js`` builds this name independently; if the two ever disagree the
    page silently 404s on every pace button."""
    from hullabaloo.webexport import preset_filename

    assert preset_filename(1.0) == "preset_p100.json"
    assert preset_filename(1.3) == "preset_p130.json"
    assert preset_filename(1.35) == "preset_p135.json"


def test_cap_ceiling_scales_with_pace(net):
    """Regression: the ceiling was read off ``CONFIG.tobler`` regardless of the pace the
    network was actually priced at, so every pace reported the same answer.

    It matters because ``base_kmh x pace x budget`` crosses the 40-mile cap at a pace
    factor of about 1.53. A sweep running past that would have kept reporting the global
    default's comfortable 35.23 mi ceiling and silently vindicated a cap that had stopped
    being safe.
    """
    from hullabaloo.optimize_milp import check_caps_nonbinding

    slow = check_caps_nonbinding(net, tobler=dataclasses.replace(CONFIG.tobler, pace_factor=1.0))
    fast = check_caps_nonbinding(net, tobler=dataclasses.replace(CONFIG.tobler, pace_factor=2.0))

    assert fast["distance_ceiling_mi"] > slow["distance_ceiling_mi"]
    assert fast["distance_ceiling_mi"] == pytest.approx(2 * slow["distance_ceiling_mi"], rel=1e-6)
    assert slow["safe_to_omit_caps"]
    assert not fast["safe_to_omit_caps"], "a 2.0 pace factor must defeat the a-priori argument"


def test_a_binding_scoring_cap_is_published_but_never_called_optimal():
    """Past pace ~1.53 the a-priori ceiling no longer rules the caps out, so the argument
    moves to the answer.

    A capped route is still a real walk inside the time budget and is still worth
    publishing — the racer wants the itinerary either way. What it must never do is keep
    claiming proven optimality, because past a cap the meaningful objective becomes the
    *fastest* tour that still collects it, which this model does not express.
    """
    import copy
    import json

    from hullabaloo import webexport

    presets = sorted(webexport.WEB_DATA.glob("preset_p*.json"))
    if not presets:
        pytest.skip("no presets published yet")

    document = json.loads(presets[-1].read_text(encoding="utf-8"))
    webexport.check_preset(document)

    for field, total in (("max_mile_points", "unique_miles"),
                         ("max_trail_points", "trails_completed")):
        capped = copy.deepcopy(document)
        capped["race"][field] = capped["totals"][total]  # cap now exactly binds

        binding = webexport.caps_binding(capped["totals"], capped["race"])
        assert binding, f"tightening {field} should bind"

        capped["optimality"] = webexport.optimality_block(
            capped["totals"], capped.get("solver", {}), capped["race"]
        )
        assert not capped["optimality"]["proven"]
        assert capped["optimality"]["note"]
        webexport.check_preset(capped)  # still publishable, just carrying the caveat

        # But a document that binds a cap *and* still claims optimality must be rejected.
        lying = copy.deepcopy(capped)
        lying["optimality"]["proven"] = True
        with pytest.raises(ValueError, match="proven optimality while a scoring cap binds"):
            webexport.check_preset(lying)


def test_published_presets_still_reconcile():
    """Guard the committed artifacts themselves: the site is static, so a stale or
    hand-edited preset would be served to readers with nothing to catch it."""
    import json

    from hullabaloo import webexport

    presets = sorted(webexport.WEB_DATA.glob("preset_p*.json"))
    if not presets:
        pytest.skip("no presets published yet")

    for path in presets:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["schema_version"] == webexport.SCHEMA_VERSION, path.name
        assert path.name == webexport.preset_filename(document["pace_factor"])
        webexport.check_preset(document)
