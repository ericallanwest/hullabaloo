"""End-to-end checks over the built artifacts.

These run against the materialized pipeline outputs rather than rebuilding from scratch,
so they are fast enough to run on every change. Anything requiring a missing artifact
skips rather than fails, so a partially-built checkout still gives useful signal.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools

import geopandas as gpd
import numpy as np
import pytest

from hullabaloo import corridors, webexport
from hullabaloo.config import (
    CONFIG,
    DEFAULT_TOP_SPEED_MPH,
    EDGES,
    EDGES_TIMED,
    M_PER_MILE,
    NODES,
    TRAILS_RAW,
)
from hullabaloo.tobler import (
    pace_for_top_speed_mph,
    profile_travel_time,
    tobler_speed_kmh,
    top_speed_mph,
)
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
    derived from a 5.0 mph top speed for this racer (see ``ToblerParams``), which puts the
    flat pace at 4.20 mph — a fit competitor moving with purpose over seven hours, not a
    stroller. Unscaled Tobler must still land in ordinary walking territory, so both are
    checked; asserting only the scaled figure would let a bad ``base_kmh`` hide inside the
    multiplier.
    """
    mph = float(tobler_speed_kmh(0.0, CONFIG.tobler)) * 0.621371
    assert 2.5 < mph < 4.5, f"flat pace {mph:.2f} mph is not a believable racing speed"

    textbook = dataclasses.replace(CONFIG.tobler, pace_factor=1.0)
    base_mph = float(tobler_speed_kmh(0.0, textbook)) * 0.621371
    assert 2.5 < base_mph < 4.0, f"unscaled Tobler {base_mph:.2f} mph is not a walking speed"


def test_speed_and_pace_convert_exactly_both_ways():
    """The published brand and the modelled parameter must be the same statement.

    Every itinerary on the site is labelled with a top speed but solved with a pace factor.
    If these two drift, a file called ``preset_s60a.json`` is solved for something other
    than 6 mph and nothing downstream would notice — the route would still validate, still
    reconcile, and still be wrong about the one thing its name promises.
    """
    for mph in (5.0, 5.5, 6.0, 6.5, 7.0, 7.5):
        params = dataclasses.replace(
            CONFIG.tobler, pace_factor=pace_for_top_speed_mph(mph)
        )
        assert top_speed_mph(params) == pytest.approx(mph, abs=1e-9)
        # ...and the brand really is the peak of the curve, not a number beside it.
        slopes = np.linspace(-0.5, 0.5, 4001)
        fastest = float(np.max(tobler_speed_kmh(slopes, params))) * 0.621371
        assert fastest == pytest.approx(mph, abs=1e-3)


def test_default_pace_is_the_bottom_published_speed():
    """The shipped default is a rung of the published ladder, not a loose constant.

    Regression: the project previously shipped ``pace_factor = 1.35``, a number with no
    stated relationship to anything a racer could name, and no preset was ever published at
    it — the sweep stepped 1.3, 1.4 straight past the pace that priced every committed
    artefact."""
    assert CONFIG.tobler.pace_factor == pytest.approx(
        pace_for_top_speed_mph(DEFAULT_TOP_SPEED_MPH)
    )
    assert top_speed_mph(CONFIG.tobler) == pytest.approx(DEFAULT_TOP_SPEED_MPH, abs=1e-9)
    assert DEFAULT_TOP_SPEED_MPH in webexport.SPEED_TIERS


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
    on_trail = edges[edges["trail_id"].notna()]
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

    # One edge becomes two (+1) and the start/finish link is added (+1).
    assert len(edges) == len(plain_edges) + 2
    assert len(nodes) == len(plain_nodes) + 2
    assert edges["edge_id"].is_unique
    assert nodes["node_id"].is_unique
    access = edges[edges["name"] == topo.DEPOT_EDGE_NAME]
    assert len(access) == 1
    # Gravel or paved on the ground, so full speed: it scores nothing because it carries
    # no trail_id, which is a separate question from how fast it is walked.
    assert not access["off_trail"].any()
    assert access["trail_id"].isna().all()

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


#: The two ways that close the gap the optimizer used to bushwhack across. Neither is
#: ``highway=track``, so both were invisible to the original road import.
MEADOWBROOK = "Meadowbrook Drive"
STONE_CUTTER = "Stone Cutter's Hollow Access Road"


def test_the_named_osm_ways_reached_the_network():
    """Meadowbrook Drive and Stone Cutter's Hollow Access Road must be in the edge table.

    They enter through a name-matched Overpass clause rather than ``highway=track``:
    Meadowbrook is a paved ``highway=tertiary`` and Stone Cutter a gravel ``highway=path``.
    A silent Overpass failure, a cached ``osm_roads.geojson``, or an OSM rename would each
    drop them without any other symptom — the pipeline would just quietly go back to
    bushwhacking.
    """
    edges = _load(EDGES)
    for name in (MEADOWBROOK, STONE_CUTTER):
        assert (edges["name"] == name).any(), f"{name} is missing from the network"
    assert (edges.loc[edges["name"] == MEADOWBROOK, "trail_id"].isna()).all(), (
        "roads must not carry a scoring trail_id"
    )


def test_the_named_ways_form_the_corridor_they_were_added_for():
    """The point of adding them is a continuous legal route, so assert the joins exist.

    Highway -> Meadowbrook -> Stone Cutter -> Mineral Way / Wavelength is precisely where
    the two longest bushwhack connectors ran. The gaps are 10 m and 9 m, inside the 18 m
    snap tolerance, so topology should fuse them — but "should" is the whole reason to
    check. Nudge ``snap_tol_m`` down and this silently becomes four disconnected stubs.
    """
    edges = _load(EDGES)

    def endpoints(name):
        segment = edges[edges["name"] == name]
        return set(segment["u"]) | set(segment["v"])

    for left, right in (
        (MEADOWBROOK, "Highway"),
        (MEADOWBROOK, STONE_CUTTER),
        (STONE_CUTTER, "Mineral Way"),
        (STONE_CUTTER, "Wavelength"),
    ):
        assert endpoints(left) & endpoints(right), f"{left} does not meet {right}"


def test_meadowbrook_is_continuous_but_stops_short_of_glade_road():
    """Meadowbrook must arrive as one connected road, and only the useful half of it.

    Both halves of this matter, and they pull against each other. The two OSM ways total
    about 4 km; only the ~2 km reaching from the Highway junction to Stone Cutter is
    wanted, so a figure drifting toward 2.5 mi means the clip has stopped trimming and the
    graph is carrying suburban road no route would walk.

    Connectivity is the half that actually bit. Clipping each way to the 250 m buffer
    independently split this road into three pieces, because its middle runs further than
    that from any trail — and the optimizer promptly bushwhacked 0.38 mi across the gap,
    retracing the road's own alignment off-trail. One piece, or the import is pointless.
    """
    import networkx as nx

    edges = _load(EDGES)
    segment = edges[edges["name"] == MEADOWBROOK]
    miles = segment["length_m"].sum() / M_PER_MILE
    assert 0.9 < miles < 1.7, f"Meadowbrook Drive contributes {miles:.2f} mi; expected ~1.27"

    graph = nx.Graph()
    graph.add_edges_from(zip(segment["u"], segment["v"]))
    pieces = nx.number_connected_components(graph)
    assert pieces == 1, f"Meadowbrook Drive is in {pieces} disconnected pieces, expected 1"


def test_the_excluded_ways_stayed_out_and_took_nothing_with_them():
    """The denylist must remove exactly what it names, and nothing adjacent.

    A denylist is the easiest place in this pipeline to do quiet damage: an over-broad
    entry silently deletes ground the optimizer needed, and the only symptom is a score
    that drops for reasons nobody attributes to pruning. So this checks both directions —
    the named ways are gone, *and* the roads that share their neighbourhood survive intact.
    """
    from hullabaloo.roads import EXCLUDED_ROAD_WAYS

    edges = _load(EDGES)
    names = set(edges["name"])
    for gone in ("Woods & Field", "Poverty Creek Connector"):
        assert gone not in names, f"{gone} is on the denylist but reached the network"

    # Meadowbrook is excluded by *way*, not by name: one of its two OSM ways is dropped and
    # the other is load-bearing. Asserting the name is absent would be exactly wrong here.
    assert MEADOWBROOK in names, "the denylist took the wrong half of Meadowbrook Drive"
    assert 59204562 in EXCLUDED_ROAD_WAYS and 490214836 not in EXCLUDED_ROAD_WAYS


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


# --------------------------------------------------------------------------------------
# Corridors
# --------------------------------------------------------------------------------------


def test_corridors_partition_the_network(net):
    """Every trail belongs to exactly one block.

    The corridor lists are frozen literals, so a re-noded or re-scraped network can add or
    rename a trail without the lists noticing. A trail missing from every block would be
    silently unreachable by any corridor rule — it could never be required, and forbidding
    its block would not exclude it — which is the kind of gap that shows up as an
    inexplicably good "skip the West End" route rather than as an error.
    """
    listed = set().union(*corridors.CORRIDORS.values())
    live = {
        str(group["name"].iloc[0])
        for _, group in net.edges[net.edges["trail_id"].notna()].groupby("trail_id")
    }

    assert listed == live, (
        f"unassigned trails: {sorted(live - listed)}; "
        f"listed but not in the network: {sorted(listed - live)}"
    )
    assert set(corridors.CORRIDOR_ORDER) == set(corridors.CORRIDORS)

    seen: set[str] = set()
    for name, members in corridors.CORRIDORS.items():
        overlap = seen & members
        assert not overlap, f"{name} shares {sorted(overlap)} with an earlier corridor"
        seen |= members


def test_corridor_trail_ids_resolve_against_the_live_network(net):
    """Names are the stable key; ``trail_id`` is assigned upstream and is not."""
    total = 0
    for name in corridors.CORRIDOR_ORDER:
        ids = corridors.trail_ids(net, name)
        assert len(ids) == len(corridors.CORRIDORS[name]), name
        assert ids <= set(net.trail_edges), name
        total += len(ids)
    assert total == EXPECTED_TRAILS

    with pytest.raises(KeyError):
        corridors.trail_ids(net, "Nowhere")


def test_corridor_breakdown_accounts_for_every_scored_mile(net, sample_route):
    """The per-corridor miles must add up to the route's own unique-mile total, or the
    sidebar's breakdown quietly disagrees with the headline figure above it."""
    rows = corridors.breakdown(net, sample_route)
    assert [row["corridor"] for row in rows] == list(corridors.CORRIDOR_ORDER)

    summary = sample_route.evaluate()
    assert sum(row["unique_miles"] for row in rows) == pytest.approx(
        summary["unique_miles"], abs=0.02
    )
    assert sum(row["trails_completed"] for row in rows) == summary["trails_completed"]
    assert sum(row["n_trails"] for row in rows) == EXPECTED_TRAILS


# --------------------------------------------------------------------------------------
# Adaptive plans
# --------------------------------------------------------------------------------------


def test_every_excursion_is_a_closed_loop_in_the_walk(net, sample_route):
    """An excursion must leave a node and come back to it, or splicing it out breaks the
    route. This is the property the whole cut mechanism rests on."""
    from hullabaloo import adapt

    nodes = sample_route.nodes
    found = adapt.excursions(sample_route)
    assert found, "a real route always loops somewhere"

    for excursion in found:
        assert nodes[excursion.start] == nodes[excursion.end] == excursion.hinge
        assert 0 <= excursion.start < excursion.end <= len(sample_route.arcs)


def test_excursions_nest_rather_than_straddle(net, sample_route):
    """Excursions form a laminar family: any two are disjoint or one contains the other.
    Partial overlap would make "skip this loop" ambiguous about what else goes with it."""
    from hullabaloo import adapt

    found = adapt.excursions(sample_route)
    for a, b in itertools.combinations(found, 2):
        disjoint = a.end <= b.start or b.end <= a.start
        assert disjoint or a.contains(b) or b.contains(a), (
            f"excursions [{a.start}:{a.end}] and [{b.start}:{b.end}] partially overlap"
        )

    for i, child in enumerate(found):
        if child.parent is not None:
            assert found[child.parent].contains(child)
            assert found[child.parent].depth == child.depth - 1


def test_dropping_a_cut_leaves_a_walkable_route(net, sample_route):
    """The payoff of using contiguous excursions: no connectivity check is needed, because
    the arcs before a cut end exactly where the arcs after it begin."""
    from hullabaloo import adapt

    for cut in adapt.cuts(net, sample_route):
        span = (cut.excursion.start, cut.excursion.end)
        shortened = adapt.drop(sample_route, [span])
        assert shortened.arcs, "a published cut must not empty the route"
        assert not shortened.validate(), f"cut at hinge {cut.excursion.hinge} broke the walk"
        assert shortened.time_s < sample_route.time_s


def test_cut_costs_are_measured_not_assumed(net, sample_route):
    """Each published cost must equal what re-scoring the shortened walk actually gives."""
    from hullabaloo import adapt
    from hullabaloo.graph import score_edges

    race = sample_route.race or CONFIG.race
    base, _, _ = score_edges({a.edge_id for a in sample_route.arcs}, net, race)

    for cut in adapt.cuts(net, sample_route):
        shortened = adapt.drop(
            sample_route, [(cut.excursion.start, cut.excursion.end)]
        )
        actual, _, _ = score_edges({a.edge_id for a in shortened.arcs}, net, race)
        assert cut.points_lost == pytest.approx(base - actual, abs=1e-6)
        assert cut.points_lost >= -1e-9, "skipping ground cannot score more than walking it"


def test_cut_costs_do_not_add_up(net, sample_route):
    """Regression guard for the trap this design exists around.

    Scoring is over the *set* of edges walked and a trail scores only when every one of its
    edges is covered, so a trail can straddle two excursions: drop either and you keep it,
    drop both and it is gone. Summing single-cut costs therefore understates a pair, which
    is why salvage plans are costed jointly. If this ever stops being true the joint search
    is merely redundant — but if it is true and we summed anyway, the menu would lie."""
    from hullabaloo import adapt
    from hullabaloo.graph import score_edges

    race = sample_route.race or CONFIG.race
    base, _, _ = score_edges({a.edge_id for a in sample_route.arcs}, net, race)
    available = adapt.cuts(net, sample_route)

    deviations = []
    for a, b in itertools.combinations(available, 2):
        if a.excursion.overlaps(b.excursion):
            continue
        shortened = adapt.drop(
            sample_route,
            [(a.excursion.start, a.excursion.end), (b.excursion.start, b.excursion.end)],
        )
        actual, _, _ = score_edges({x.edge_id for x in shortened.arcs}, net, race)
        deviations.append((base - actual) - (a.points_lost + b.points_lost))

    if not deviations:
        pytest.skip("route offers no two disjoint cuts to combine")
    # Joint cost is never *less* than the sum: cutting more can only lose more.
    assert min(deviations) > -1e-6


def test_salvage_fits_its_budget_and_degrades_monotonically(net, sample_route):
    """Less time can never buy more points, and a plan claiming to fit must fit."""
    from hullabaloo import adapt

    race = sample_route.race or CONFIG.race
    budget = float(race.time_budget_s)
    available = adapt.cuts(net, sample_route)

    previous = None
    for fraction in adapt.SALVAGE_FRACTIONS:
        target = budget * fraction
        plan = adapt.salvage(net, sample_route, target, available)
        if plan.feasible:
            assert plan.seconds <= target + 1.0
        assert plan.score <= sample_route.evaluate()["score"] + 1e-6
        if previous is not None:
            assert plan.score <= previous + 1e-6
        previous = plan.score


def test_bailout_curve_never_overstates_what_you_keep(net, sample_route):
    """The curve is what a racer trusts when deciding to quit, so it must not flatter."""
    from hullabaloo import adapt

    curve = adapt.bailout_curve(net, sample_route)
    assert len(curve) == len(sample_route.arcs)

    final = sample_route.evaluate()["score"]
    for row in curve:
        assert row["score_if_home_now"] <= final + 1e-6
        assert row["home_s"] >= 0
        assert row["finish_s"] == pytest.approx(row["elapsed_s"] + row["home_s"], abs=0.2)

    assert [r["elapsed_s"] for r in curve] == sorted(r["elapsed_s"] for r in curve)
    # Walking home from the finish is free, so the last row is the full route.
    assert curve[-1]["score_if_home_now"] == pytest.approx(final, abs=1e-6)


def test_front_load_score_is_monotone_and_bounded(net, sample_route):
    from hullabaloo import adapt

    final = sample_route.evaluate()["score"]
    marks = [0.0, 3600.0, 3 * 3600.0, 6 * 3600.0, sample_route.time_s + 1]
    values = [adapt.front_load_score(sample_route, t) for t in marks]

    assert values == sorted(values), "score cannot fall as the clock runs"
    assert values[0] == 0.0
    # ``evaluate`` publishes a rounded score; this one is raw, so compare at that precision.
    assert values[-1] == pytest.approx(final, abs=1e-3)


def test_check_adaptive_rejects_a_dishonest_menu(net, sample_route):
    """The menu is acted on twenty miles from the car with no way to verify it, so the
    claims are asserted before the file ships."""
    from hullabaloo import adapt, webexport

    document = webexport.preset_dict(
        net,
        sample_route,
        pace_factor=CONFIG.tobler.pace_factor,
        option="d",
        adaptive=adapt.summary(net, sample_route),
        free_score=sample_route.evaluate()["score"],
    )
    webexport.check_preset(document)
    assert document["option_label"] == webexport.ADAPTIVE_LABEL

    if document["adaptive"]["cuts"]:
        lying = copy.deepcopy(document)
        lying["adaptive"]["cuts"][0]["points_lost"] = -2.0
        with pytest.raises(ValueError, match="gain"):
            webexport.check_preset(lying)

    if document["adaptive"]["salvage"]:
        impossible = copy.deepcopy(document)
        impossible["adaptive"]["salvage"][0]["score"] = 999.0
        with pytest.raises(ValueError, match="cutting cannot add points"):
            webexport.check_preset(impossible)

    overspent = copy.deepcopy(document)
    overspent["delta_vs_free"] = -(webexport.ADAPTIVE_GAP_BUDGET + 1.0)
    with pytest.raises(ValueError, match="beyond the"):
        webexport.check_preset(overspent)


def test_corridor_rule_detects_both_kinds_of_violation(net, sample_route):
    """The rule must catch a route that breaks it — this is the only thing standing
    between a mislabelled option and the page."""
    walked = {arc.edge_id for arc in sample_route.arcs}
    walked_trail = next(
        tid for tid, edges in net.trail_edges.items() if edges <= walked
    )
    unwalked_trail = next(
        tid for tid, edges in net.trail_edges.items() if not edges & walked
    )

    assert not corridors.CorridorRule().violations(net, sample_route)
    assert not corridors.CorridorRule(
        require=frozenset({walked_trail})
    ).violations(net, sample_route)

    forbidden = corridors.CorridorRule(forbid=frozenset({walked_trail}))
    assert forbidden.violations(net, sample_route), "walking a forbidden trail must fail"

    missing = corridors.CorridorRule(require=frozenset({unwalked_trail}))
    assert missing.violations(net, sample_route), "skipping a required trail must fail"


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


def test_a_closed_loop_gains_exactly_what_it_loses(net, sample_route):
    """Total climb must equal total descent, because the route ends where it started.

    Per arc, gain minus loss telescopes to (end elevation - start elevation), so over a
    closed walk the difference is exactly zero. That makes this the one independent check
    on :func:`export.arc_relief_m`, which picks between the stored forward and reverse
    gains by comparing the arc's tail to the edge's. Get that backwards and every climb
    reads as a descent -- while every individual number still looks entirely plausible.
    """
    from hullabaloo.export import route_gdf

    detail = route_gdf(sample_route)
    assert sample_route.arcs[0].u == net.depot
    assert sample_route.arcs[-1].v == net.depot
    assert detail["gain_m"].sum() == pytest.approx(detail["loss_m"].sum(), abs=0.05)
    assert detail["gain_m"].sum() > 100, "a real route over this terrain must climb"


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


def test_turns_are_named_from_the_angle_between_two_legs():
    """The cue a racer reads at a junction, checked against geometry with a known answer.

    Bearings are planar in EPSG:6346, so a leg drawn straight up the page heads due north
    and every turn off it can be written down by hand.
    """
    from shapely.geometry import LineString

    from hullabaloo.export import TURN_GLYPHS, turn_between

    northbound = LineString([(0, -100), (0, 0)])
    expected = {
        "straight": (0, 100),
        "slight right": (40, 100),
        "right": (100, 0),
        "sharp right": (60, -100),
        "turn around": (0, -100),
        "sharp left": (-60, -100),
        "left": (-100, 0),
        "slight left": (-40, 100),
    }
    for label, end in expected.items():
        degrees, got, glyph = turn_between(northbound, LineString([(0, 0), end]))
        assert got == label, f"{end} off due north should read {label}, not {got}"
        assert glyph == TURN_GLYPHS[label]
        assert -180.0 <= degrees <= 180.0

    # The opening leg is walked from a standing start: no incoming bearing, so no angle.
    assert turn_between(None, northbound) == (None, "start", TURN_GLYPHS["start"])


def test_turn_angles_are_mirror_symmetric():
    """Reflecting the route east-west must swap left for right and nothing else.

    This is the property that catches a sign error, which is the failure mode that matters:
    a turn cue pointing confidently the wrong way down a fork is worse than no cue.
    """
    from shapely.geometry import LineString

    from hullabaloo.export import turn_between

    northbound = LineString([(0, -100), (0, 0)])
    for end in [(40, 100), (100, 0), (60, -100), (0, 100)]:
        right, right_label, _ = turn_between(northbound, LineString([(0, 0), end]))
        left, left_label, _ = turn_between(northbound, LineString([(0, 0), (-end[0], end[1])]))
        assert right == pytest.approx(-left)
        assert right_label.replace("right", "") == left_label.replace("left", "")


def test_a_short_leg_still_gets_a_bearing():
    """Legs shorter than the 25 m bearing window are common — the depot access link is 21 m
    — and must degrade to their own length rather than raising or reading as straight."""
    from shapely.geometry import LineString

    from hullabaloo.export import BEARING_WINDOW_M, turn_between

    stub = LineString([(0, 0), (5, 0)])          # 5 m due east, well under the window
    assert stub.length < BEARING_WINDOW_M
    degrees, label, _ = turn_between(LineString([(0, -50), (0, 0)]), stub)
    assert label == "right"
    assert degrees == pytest.approx(90.0)


def test_every_step_carries_a_consistent_turn(net, sample_route):
    """Turns on a real route: one per leg, self-consistent, and only the first is a start."""
    from hullabaloo import webexport

    steps = webexport.preset_dict(
        net, sample_route, pace_factor=CONFIG.tobler.pace_factor
    )["steps"]

    assert [s["turn"] for s in steps].count("start") == 1
    assert steps[0]["turn"] == "start" and steps[0]["turn_deg"] is None
    for step in steps[1:]:
        assert step["turn_deg"] is not None
        webexport.check_turn(step)  # glyph, label and angle must agree


def test_a_glyph_that_contradicts_its_label_is_rejected(net, sample_route):
    """The glyph is the one field on a step that can be wrong while every number around it
    still adds up, so a hand-edited preset must not be able to publish a bad arrow."""
    import copy

    from hullabaloo import webexport

    document = webexport.preset_dict(net, sample_route, pace_factor=CONFIG.tobler.pace_factor)
    webexport.check_preset(document)

    lying = copy.deepcopy(document)
    lying["steps"][1]["glyph"] = webexport.TURN_GLYPHS["left"]
    lying["steps"][1]["turn"] = "right"
    with pytest.raises(ValueError, match="glyph"):
        webexport.check_preset(lying)

    mislabelled = copy.deepcopy(document)
    mislabelled["steps"][1].update(turn="straight", glyph=webexport.TURN_GLYPHS["straight"],
                                   turn_deg=90.0)
    with pytest.raises(ValueError, match="classifies"):
        webexport.check_preset(mislabelled)


def test_the_cue_sheet_tells_you_which_way_to_turn(sample_route):
    """``route_cues.csv`` is the artifact a racer prints, and its whole job is junctions."""
    from hullabaloo.export import TURN_GLYPHS, cue_sheet

    cues = cue_sheet(sample_route)
    assert list(cues.columns)[:4] == ["leg", "glyph", "turn", "turn_deg"]
    assert cues["turn"].iloc[0] == "start"
    assert set(cues["turn"]) <= set(TURN_GLYPHS)
    assert cues["turn_deg"].iloc[1:].notna().all()


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


def test_preset_speed_filename_matches_the_front_end():
    """Same contract as above, for the family the racer actually uses. ``presetSpeedFile``
    in ``docs/js/viz.js`` reimplements this in JavaScript."""
    from hullabaloo.webexport import preset_speed_filename

    assert preset_speed_filename(5.0, "a") == "preset_s50a.json"
    assert preset_speed_filename(6.5, "b") == "preset_s65b.json"
    assert preset_speed_filename(7.5, "c") == "preset_s75c.json"
    assert preset_speed_filename(6.0, "d") == "preset_s60d.json"

    with pytest.raises(ValueError):
        preset_speed_filename(6.0, "z")


def test_milp_model_carries_the_corridor_constraints(net):
    """Assert the constraints reach the model, without paying for a full solve.

    Cheap but worth pinning: ``require`` and ``forbid`` act on different variables — one on
    the trail indicator, one on every edge indicator — and getting that backwards produces
    a model that solves happily and means the wrong thing. Forbidding via ``y_t`` alone
    would still let the route walk the ground and bank the miles, merely declining the
    trail point.
    """
    from hullabaloo.optimize_milp import build_model

    west = corridors.trail_ids(net, corridors.PIVOT_CORRIDOR)
    west_edges = corridors.edge_ids(net, corridors.PIVOT_CORRIDOR)

    free, _ = build_model(net)
    assert not [n for n in free.constraints if n.startswith(("require_", "forbid_"))]

    required, _ = build_model(net, require_trails=west)
    assert {f"require_trail_{t}" for t in west} <= set(required.constraints)

    forbidden, _ = build_model(net, forbid_trails=west)
    assert {f"forbid_edge_{e}" for e in west_edges} <= set(forbidden.constraints)

    with pytest.raises(KeyError):
        build_model(net, require_trails=frozenset({999_999}))


def test_check_preset_rejects_an_option_that_broke_its_own_rule(net, sample_route):
    """A mislabelled option is the one failure mode where every number still adds up.

    Option c's whole meaning is "this route does not go there". If a route that does go
    there were published under that label, the arithmetic checks would all pass and the
    page would present it, confidently, as the alternative that stays home.
    """
    from hullabaloo import webexport

    walked = {arc.edge_id for arc in sample_route.arcs}
    walked_trail = next(tid for tid, edges in net.trail_edges.items() if edges <= walked)
    corridor = corridors.corridor_of(
        str(net.edges.loc[net.edges["trail_id"] == walked_trail, "name"].iloc[0])
    )

    honest = webexport.preset_dict(
        net, sample_route, pace_factor=CONFIG.tobler.pace_factor, option="a"
    )
    webexport.check_preset(honest)

    lying = webexport.preset_dict(
        net,
        sample_route,
        pace_factor=CONFIG.tobler.pace_factor,
        option="c",
        rule=corridors.CorridorRule(corridor=corridor, kind="forbid"),
    )
    with pytest.raises(ValueError, match="claims to skip"):
        webexport.check_preset(lying)

    # And the same check run against the concrete walk rather than the document.
    with pytest.raises(ValueError, match="breaks its own corridor rule"):
        webexport.check_preset(
            honest,
            net=net,
            route=sample_route,
            rule=corridors.CorridorRule(forbid=frozenset({walked_trail})),
        )


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
    hand-edited preset would be served to readers with nothing to catch it.

    Both families are checked. The speed tiers are the ones a racer reads, and they are the
    ones whose filename encodes a claim — ``preset_s60b.json`` asserts a speed and an
    option, and nothing else in the file would contradict it if the name were wrong."""
    import json

    from hullabaloo import webexport

    presets = sorted(webexport.WEB_DATA.glob("preset_p*.json"))
    if not presets:
        pytest.skip("no presets published yet")

    for path in presets:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert (
            webexport.MIN_SUPPORTED_SCHEMA
            <= document["schema_version"]
            <= webexport.SCHEMA_VERSION
        ), path.name
        assert path.name == webexport.preset_filename(document["pace_factor"])
        webexport.check_preset(document)

    for path in sorted(webexport.WEB_DATA.glob("preset_s*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        # New enough for the fields it carries, not necessarily the latest — the versions
        # are pure additions, and an adaptive plan is checked against its own floor below.
        assert (
            webexport.SCHEMA_WITH_OPTIONS
            <= document["schema_version"]
            <= webexport.SCHEMA_VERSION
        ), path.name
        if document.get("adaptive") is not None:
            assert document["schema_version"] >= webexport.SCHEMA_WITH_ADAPTIVE, path.name
        assert path.name == webexport.preset_speed_filename(
            document["speed_mph"], document["option"]
        )
        # The pace a tier was priced at must be the one its branded speed implies, or the
        # file is labelled with a speed it was not solved for.
        assert document["pace_factor"] == pytest.approx(
            pace_for_top_speed_mph(document["speed_mph"]), abs=1e-3
        ), path.name
        webexport.check_preset(document)


def test_published_manifest_matches_the_files_on_disk():
    """The manifest is what the page builds its controls from, so an entry without a file
    behind it is a control that 404s, and a file without an entry is invisible."""
    import json

    from hullabaloo import webexport

    manifest_path = webexport.WEB_DATA / "presets.json"
    if not manifest_path.exists():
        pytest.skip("no manifest published yet")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = {
        option["file"] for tier in manifest["speeds"] for option in tier["options"]
    } | {entry["file"] for entry in manifest["paces"]}
    on_disk = {
        path.name
        for path in webexport.WEB_DATA.glob("preset_*.json")
    }
    assert listed == on_disk, (
        f"manifest and directory disagree: "
        f"listed only {sorted(listed - on_disk)}, on disk only {sorted(on_disk - listed)}"
    )

    for tier in manifest["speeds"]:
        assert tier["pace_factor"] == pytest.approx(
            pace_for_top_speed_mph(tier["mph"]), abs=1e-3
        )
