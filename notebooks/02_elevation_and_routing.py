"""Elevation, hiking speed, bushwhack connectors, and the optimizer.

Where notebook 01 built the network, this one prices it and solves it.
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium", app_title="Elevation and routing")


@app.cell
def _():
    import marimo as mo
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    from hullabaloo.config import CONFIG, EDGES, EDGES_TIMED, NODES, TRAILS_RAW
    from hullabaloo import elevation as elev
    from hullabaloo.tobler import flat_pace_summary

    return (
        CONFIG,
        EDGES,
        EDGES_TIMED,
        NODES,
        TRAILS_RAW,
        elev,
        flat_pace_summary,
        gpd,
        mo,
        np,
        pd,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Phase 3-6 — elevation, hiking speed, and the route

        ## The DEM

        Source is USGS 3DEP **1 m lidar** (`VA_FEMA-NRCS_SouthCentral_2017`, tile
        `x54y413`), which covers the whole study area in a single 206 MB tile.

        Two lessons from getting this working:

        1. **Fetch the source tile, not the dynamic service.** The `py3dep` async client
           failed intermittently on Windows with spurious DNS errors and *silently fell
           back to a 10 m DEM*. Degrading from 1 m lidar to a 10 m grid without noticing
           corrupts every slope in the model. The pipeline now pulls the tile from S3 and
           treats the service as fallback only.
        2. **Specify smoothing in metres, not pixels.** A `sigma=3` written as pixels
           means 3 m on the lidar but 30 m on the 10 m fallback. That second reading
           flattened genuinely steep trails.
        """
    )
    return


@app.cell
def _(EDGES, TRAILS_RAW, elev, gpd, mo):
    trails = gpd.read_parquet(TRAILS_RAW)
    edges_raw = gpd.read_parquet(EDGES)

    @mo.cache
    def _sensitivity():
        return elev.smoothing_sensitivity(edges_raw, trails)

    sensitivity = _sensitivity()
    mo.vstack(
        [
            mo.md("### How much does the smoothing choice actually matter?"),
            mo.ui.table(sensitivity, selection=None),
            mo.md(
                "Total network traversal time moves only **3.4%** across sigma 0–12 m, so "
                "the model is not hostage to this parameter. sigma = 5 m is chosen "
                "because it drives the bias against the published elevation gains to "
                "roughly zero."
            ),
        ]
    )
    return edges_raw, sensitivity, trails


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Validating the DEM against published gains

        The site publishes forward *and* reverse elevation gain for all 40 trails — 80
        independent ground-truth values, for free.
        """
    )
    return


@app.cell
def _(EDGES_TIMED, elev, gpd, mo, pd, trails):
    timed = gpd.read_parquet(EDGES_TIMED)
    gain_report = elev.validate_against_official(timed, trails)
    stats = elev.gain_correlation(gain_report)
    mo.vstack(
        [
            mo.ui.table(pd.DataFrame([stats]), selection=None),
            mo.ui.table(gain_report, selection=None, page_size=15),
        ]
    )
    return gain_report, stats, timed


@app.cell
def _(gain_report, mo):
    import altair as alt

    _long = gain_report.melt(
        id_vars=["name"],
        value_vars=["official_gain_ft", "dem_gain_ft"],
        var_name="source",
        value_name="ft",
    )
    _scatter = (
        alt.Chart(gain_report)
        .mark_circle(size=90, opacity=0.75)
        .encode(
            x=alt.X("official_gain_ft:Q", title="published gain (ft)"),
            y=alt.Y("dem_gain_ft:Q", title="lidar-derived gain (ft)"),
            tooltip=["name", "official_gain_ft", "dem_gain_ft", "fwd_pct_err"],
        )
    )
    _line = (
        alt.Chart(gain_report)
        .mark_line(color="#999", strokeDash=[4, 4])
        .encode(x="official_gain_ft:Q", y="official_gain_ft:Q")
    )
    mo.ui.altair_chart((_scatter + _line).properties(height=340, title="Lidar vs published elevation gain"))
    return alt,


@app.cell
def _(mo):
    mo.md(
        r"""
        R² ≈ 0.85 across the 80 values, with near-zero bias. The residuals are
        interesting rather than alarming, and they point one way:

        * **Crosscut** is published at **1 ft** of gain over 990 m of singletrack. Its
          GPX elevation profile is *exactly monotone* — physically implausible on real
          trail. The published figures are computed from the GPX elevations, so they
          inherit whatever the recording device did.
        * **Upper Chimney**, a big sustained climb, agrees closely across GPX, published,
          and lidar — because smoothing and noise barely matter when the signal is 1,100 ft.
        * **Prickly Pear** is the largest disagreement (published 688 ft, lidar 185 ft).
          The lidar and GPX agree *exactly* at the low end (696 m) and diverge only near
          the top, which points to GPS elevation drift rather than misplaced geometry.

        Bare-earth lidar is the more trustworthy source, and it is what the model uses.
        """
    )
    return


@app.cell
def _(CONFIG, flat_pace_summary, mo, pd):
    mo.vstack(
        [
            mo.md("## What the Tobler parameters imply"),
            mo.ui.table(
                pd.DataFrame([flat_pace_summary(CONFIG.tobler)]).T.reset_index().rename(
                    columns={"index": "metric", 0: "value"}
                ),
                selection=None,
            ),
            mo.md(
                "3.13 mph on the flat, 19.2 min/mile. These are Tobler's literature "
                "constants, **not** calibrated to a specific person carrying a pack for "
                "seven hours — the `pace_factor` slider exists for exactly that."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The bushwhacking that turned out to be unnecessary

        This project spent most of its life assuming off-trail travel was mandatory. The
        40 scored trails sit in three disconnected pieces, so *something* had to bridge
        them, and the answer looked like least-cost bushwhack connectors: paths routed
        over a Tobler-derived cost surface and walked at 60% speed.

        That machinery is gone now, and the story of why is more interesting than the
        code was.

        ### First: forest roads span the gaps

        The gaps between components are crossed by **forest service roads** — legal,
        full-speed, unambiguous. Once they were imported the network became a single
        component and bushwhacking stopped being mandatory. It remained *attractive*,
        though: the optimizer still took a 0.58 mi shortcut in 8 of 11 published routes.

        ### Then: the shortcuts were an artefact of our own clipping

        Chasing that last shortcut found the real cause. Roads were imported from
        OpenStreetMap `highway=track` only, which misses a paved `highway=tertiary`
        (Meadowbrook Drive), a gravel `highway=path` (Stone Cutter's Hollow Access Road),
        and a `highway=service` forest road (Road 708). Worse, roads were clipped to
        250 m of the trails and whatever survived was kept — which sliced Meadowbrook
        into three disconnected pieces, because its middle runs further than that from
        any trail.

        So the optimizer was bushwhacking across a gap **we had created**, retracing the
        road's own alignment off-trail. The tell was that the on-network alternative
        between those two nodes cost 7330 s, detouring via Beauty and Gateway. A two-hour
        detour to cross 600 m of road is not a routing decision, it is a broken graph.

        With the missing roads imported and the clip fixed to trim ends without severing
        through-routes, candidate connectors fell from **128 to 4**, and re-solving at
        pace 1.0, 1.5 and 2.0 used **none of them**. The stage was removed.

        ### The bug no automated check could catch

        Worth keeping, because it is the best thing this dead end produced.

        Every check passed. Connectors avoided water, respected slope limits, ran at a
        plausible 0.56x on-trail speed. Then drawing them on USGS aerial imagery showed
        the optimal route's longest bushwhack running **848 m through a residential
        neighbourhood** — houses, driveways, lawns, a swimming pool.

        The cause was structural rather than a coding error. The cost surface came from a
        **bare-earth** DEM: terrain with buildings and vegetation stripped out by
        definition. Where the houses are, it saw gentle, inviting slope. No amount of
        slope or hydrography validation finds this, because the input does not contain
        the information. The fix was NLCD land cover with developed classes masked, and
        the illegal shortcut turned out to be worth **0.057 points** — nearly nothing, and
        simply invisible to every check that did not involve looking at a photograph.

        Two smaller lessons from the same stage, both preserved in the road importer:
        Pandapas Pond had to be masked from NHD hydrography (an earlier flatness-based
        heuristic flagged **20% of the map** as water), and connectors joining two points
        on the *same* trail had to be rejected outright as switchback cuts — 167 of them —
        because a model that offers to run the fall line is proposing erosion.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The optimization problem

        Points are earned on **arcs** (unique trail miles) and on **completing whole
        trails**, each arc may be re-walked freely but only scores once, and the tour must
        start and finish at the trailhead within 7 hours. That is a *prize-collecting arc
        routing problem*, not a TSP.

        ### Encoding

        A solution is an **ordered list of target trails**. A deterministic decoder turns
        it into a real walk: from the current position take the shortest path to whichever
        end of the next trail is cheaper, walk that trail end to end, repeat, then return
        to the depot. Every solution is therefore connected and depot-anchored *by
        construction* — the neighbourhood operators never have to reason about graph
        connectivity, only about which trails and in what order.

        Deadhead legs frequently cross trail that has not been walked yet, and those miles
        score, so the evaluator credits every edge the walk actually touches.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### ALNS, and then a proof

        The heuristic finds good routes in seconds. It cannot tell you how good. So the
        same problem is also written as a MILP and handed to HiGHS — and at this size the
        solver does better than bound the problem, it **closes** it: gap 0.00% in 65 s,
        proving no 7-hour route scores above **48.23**. The heuristic's 47.889 turns out
        to be within **0.7%** of that.

        The MILP needs one constraint that is easy to forget: **connectivity**. Flow
        conservation alone is satisfied by any collection of disjoint circuits, so without
        a single-commodity-flow constraint tying the traversed subgraph back to the depot,
        the solver will happily return a lovely high-scoring loop on the far side of the
        property that never touches the start line.

        Two things turned a bound into a proof:

        * feeding the ALNS score in as a primal cut (`objective >= 35.249`), valid because
          the heuristic actually achieved it, which prunes the tree hard;
        * proving the 40-point caps cannot bind rather than modelling them. The network is
          40.14 miles against a 40-mile cap, so it is not *obviously* unreachable — but
          `7 h x 3.73 mph = 26.1 mi` is a hard ceiling at Tobler's peak speed.

        One implementation trap worth recording: PuLP hands HiGHS the *negated* objective
        for maximization, so `mip_dual_bound` comes back with the opposite sign. Reading
        it naively produced an "upper bound" of −38 on a positive-valued maximization and
        a meaningless 0% gap, which would have made the optimality claim pure fiction.
        """
    )
    return


@app.cell
def _(mo):
    solve_button = mo.ui.run_button(label="Run ALNS + MILP bound (slow)")
    solve_button
    return (solve_button,)


@app.cell
def _(mo, solve_button):
    mo.stop(not solve_button.value, mo.md("*Press the button to solve.*"))

    from hullabaloo.graph import build_network, network_traversal_bound
    from hullabaloo.optimize_alns import ALNS, baseline_greedy, build_trail_chains
    from hullabaloo.optimize_milp import check_caps_nonbinding, solve as milp_solve
    import pandas as _pd

    net = build_network()
    chains = build_trail_chains(net)
    greedy = baseline_greedy(net, chains)
    alns = ALNS(net, chains, seed=0).solve(iterations=250)
    milp = milp_solve(net, time_limit_s=240, incumbent=alns.score, msg=False)

    _rows = _pd.DataFrame(
        [
            {"method": "greedy baseline", **greedy.evaluate()},
            {"method": "ALNS", **alns.route.evaluate()},
        ]
    )
    mo.vstack(
        [
            mo.ui.table(_rows, selection=None),
            mo.md(f"**MILP bound:** {milp.summary()}"),
            mo.md(f"**Caps provably non-binding:** {check_caps_nonbinding(net)}"),
        ]
    )
    return (
        ALNS,
        alns,
        baseline_greedy,
        build_network,
        build_trail_chains,
        chains,
        check_caps_nonbinding,
        greedy,
        milp,
        milp_solve,
        net,
        network_traversal_bound,
    )


if __name__ == "__main__":
    app.run()
