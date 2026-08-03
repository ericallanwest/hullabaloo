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

    from hullabaloo.config import CONFIG, CONNECTORS, EDGES, EDGES_TIMED, NODES, TRAILS_RAW
    from hullabaloo import elevation as elev
    from hullabaloo.tobler import flat_pace_summary

    return (
        CONFIG,
        CONNECTORS,
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
        ## Bushwhack connectors

        The network is in three pieces, so off-trail travel is mandatory. Connectors are
        least-cost paths over a Tobler-derived cost surface run at 60% speed.

        Three things to be explicit about:

        * **Water is masked.** Pandapas Pond sits in the middle of the study area. NHD
          hydrography contributes 4.7 ha of impassable cells. An earlier flatness-based
          heuristic flagged **20% of the map** as water, so the code now rejects any
          fallback mask that claims more than 2% of the area rather than quietly warping
          every connector.
        * **Developed land is masked too** — and that one took aerial imagery to find.
        * **`MCP_Geometric` is isotropic.** It prices cells by slope *magnitude*, so path
          *selection* treats up and down alike. We compensate by re-integrating true
          directional Tobler time along the returned polyline, which restores asymmetry
          in the routing graph. `MCP_Flexible` is the fully anisotropic upgrade.

        ### The bug no automated check could catch

        Every check passed. Connectors avoided water, respected slope limits, ran at a
        plausible 0.56x on-trail speed. Then drawing them on USGS aerial imagery showed
        the optimal route's longest bushwhack running **848 m through a residential
        neighbourhood** — houses, driveways, lawns, a swimming pool.

        The cause is structural rather than a coding error. The cost surface comes from a
        **bare-earth** DEM: terrain with buildings and vegetation stripped out by
        definition. Where the houses are, it sees gentle, inviting slope. No amount of
        slope or hydrography validation finds this, because the input does not contain
        the information.

        The fix is NLCD land cover with developed classes (21-24) impassable — 5.4% of the
        study area. The route loses **0.057 points**. The illegal shortcut was worth almost
        nothing; it was simply invisible to every check that did not involve looking at a
        photograph.
        """
    )
    return


@app.cell
def _(CONNECTORS, EDGES_TIMED, gpd, mo, pd, timed):
    connectors = gpd.read_parquet(CONNECTORS)
    _on = timed[~timed["off_trail"]]
    _speeds = pd.DataFrame(
        [
            {
                "surface": "on trail",
                "mean m/s": round(float((_on["length_m"] / _on["time_fwd_s"]).mean()), 3),
            },
            {
                "surface": "off trail",
                "mean m/s": round(
                    float((connectors["length_m"] / connectors["time_fwd_s"]).mean()), 3
                ),
            },
        ]
    )
    mo.vstack(
        [
            mo.ui.table(_speeds, selection=None),
            mo.md(
                "The realised ratio is 0.56 rather than the nominal 0.60 because "
                "connectors cut across slopes that graded trail contours around — a "
                "sanity check that the cost surface is doing real work."
            ),
            mo.ui.table(
                connectors[["name", "kind", "length_m", "straight_m", "sinuosity", "time_fwd_s", "time_rev_s"]]
                .sort_values("time_fwd_s")
                .head(20),
                selection=None,
            ),
        ]
    )
    return (connectors,)


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
        solver does better than bound the problem, it **closes** it: gap 0.00% in 181 s,
        proving no 7-hour route scores above **35.37**. The heuristic's 35.249 turns out
        to be within **0.34%** of that.

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
