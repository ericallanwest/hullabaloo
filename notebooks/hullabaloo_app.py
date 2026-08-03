"""Hullabaloo route optimizer — interactive marimo app.

Run it:
    pixi run app          # read-only app view
    pixi run edit         # editable notebook

Move the sliders and the route re-solves. The expensive artifacts (DEM, network topology,
forest roads) are cached, so only the parts that actually depend on a changed
parameter recompute.
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium", app_title="Hullabaloo Route Optimizer")


@app.cell
def _():
    import marimo as mo

    import dataclasses
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    from hullabaloo.config import CONFIG, EDGES, EDGES_TIMED, NODES, TRAILS_RAW
    from hullabaloo import elevation as elev
    from hullabaloo.graph import build_network, network_traversal_bound, score_edges
    from hullabaloo.optimize_alns import (
        ALNS,
        baseline_best_ratio,
        baseline_greedy,
        build_trail_chains,
        decode,
        evaluate,
    )
    from hullabaloo.tobler import flat_pace_summary, tobler_speed_kmh
    from hullabaloo import export as ex

    return (
        ALNS,
        CONFIG,
        EDGES,
        EDGES_TIMED,
        NODES,
        TRAILS_RAW,
        baseline_best_ratio,
        baseline_greedy,
        build_network,
        build_trail_chains,
        dataclasses,
        decode,
        elev,
        evaluate,
        ex,
        flat_pace_summary,
        gpd,
        mo,
        network_traversal_bound,
        np,
        pd,
        score_edges,
        tobler_speed_kmh,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # The Hullabaloo — optimized 7-hour route

        A rogaine on the Pandapas Pond trail network near Blacksburg, VA. Two ways to
        score, both capped at 40 points:

        * **1 point per trail** you complete end to end (40 trails exist)
        * **1 point per unique mile** of trail you cover (39.7 miles exist)

        The whole network is 40.1 miles and takes **14.3 hours** to cover. You have
        **7**. So this is not a coverage problem — it is a *prize-collecting arc routing*
        problem, and the interesting question is which 45% of the network to spend your
        day on.
        """
    )
    return


@app.cell
def _(mo):
    base_kmh = mo.ui.slider(3.0, 8.0, 0.1, value=6.0, label="Tobler base speed (km/h)", show_value=True)
    k = mo.ui.slider(1.0, 6.0, 0.1, value=3.5, label="Slope sensitivity k", show_value=True)
    s0 = mo.ui.slider(0.0, 0.15, 0.01, value=0.05, label="Peak-speed offset s0", show_value=True)
    off_factor = mo.ui.slider(0.2, 1.0, 0.05, value=0.60, label="Off-trail speed factor", show_value=True)
    pace = mo.ui.slider(0.5, 1.5, 0.05, value=1.0, label="Personal pace multiplier", show_value=True)
    budget_h = mo.ui.slider(2.0, 14.0, 0.5, value=7.0, label="Time budget (hours)", show_value=True)
    iterations = mo.ui.slider(20, 600, 20, value=120, label="ALNS iterations", show_value=True)

    mo.vstack(
        [
            mo.md("## Model parameters"),
            mo.hstack([base_kmh, k, s0], justify="start"),
            mo.hstack([off_factor, pace, budget_h], justify="start"),
            iterations,
        ]
    )
    return base_kmh, budget_h, iterations, k, off_factor, pace, s0


@app.cell
def _(CONFIG, base_kmh, budget_h, dataclasses, k, off_factor, pace, s0):
    tobler_params = dataclasses.replace(
        CONFIG.tobler,
        base_kmh=base_kmh.value,
        k=k.value,
        s0=s0.value,
        off_trail_factor=off_factor.value,
        pace_factor=pace.value,
    )
    race_params = dataclasses.replace(CONFIG.race, time_budget_s=budget_h.value * 3600)
    return race_params, tobler_params


@app.cell
def _(flat_pace_summary, mo, pd, tobler_params):
    pace_table = pd.DataFrame([flat_pace_summary(tobler_params)]).T.rename(columns={0: "value"})
    mo.vstack(
        [
            mo.md("### What these parameters imply"),
            mo.ui.table(pace_table.reset_index().rename(columns={"index": "metric"}), selection=None),
        ]
    )
    return


@app.cell
def _(mo, np, tobler_grid):
    import altair as alt

    chart = (
        alt.Chart(tobler_grid)
        .mark_line(size=3)
        .encode(
            x=alt.X("slope:Q", title="slope (rise / run)", axis=alt.Axis(format="%")),
            y=alt.Y("kmh:Q", title="speed (km/h)"),
            color=alt.Color("surface:N", title=""),
        )
        .properties(height=260, title="Parameterized Tobler hiking function")
    )
    mo.ui.altair_chart(chart)
    return alt, chart


@app.cell
def _(np, pd, tobler_params, tobler_speed_kmh):
    _slopes = np.linspace(-0.45, 0.45, 400)
    tobler_grid = pd.concat(
        [
            pd.DataFrame(
                {"slope": _slopes, "kmh": tobler_speed_kmh(_slopes, tobler_params), "surface": "on trail"}
            ),
            pd.DataFrame(
                {
                    "slope": _slopes,
                    "kmh": tobler_speed_kmh(_slopes, tobler_params) * tobler_params.off_trail_factor,
                    "surface": "off trail",
                }
            ),
        ]
    )
    return (tobler_grid,)


@app.cell
def _(mo):
    mo.md(
        """
        ## The network

        Re-pricing every edge under the current Tobler parameters. The topology and the
        forest roads do not depend on these sliders, so they are loaded from disk
        rather than rebuilt.
        """
    )
    return


@app.cell
def _(EDGES, elev, gpd, mo):
    @mo.cache
    def _dem_and_profiles():
        edges_raw = gpd.read_parquet(EDGES)
        array, transform, _, _ = elev.load_dem()
        return edges_raw, elev.edge_profiles(edges_raw, array, transform)

    edges_raw, profiles = _dem_and_profiles()
    return edges_raw, profiles


@app.cell
def _(build_network, edges_raw, elev, gpd, profiles, tobler_params):
    from hullabaloo.config import NODES as NODES_PATH

    timed_edges = elev.price_edges(edges_raw, profiles, tobler_params)
    nodes = gpd.read_parquet(NODES_PATH)
    net = build_network(timed_edges, nodes)
    return net, nodes, timed_edges


@app.cell
def _(mo, net, network_traversal_bound, pd):
    _bound = network_traversal_bound(net)
    mo.vstack(
        [
            mo.md("### Feasibility reference"),
            mo.ui.table(
                pd.DataFrame([_bound]).T.reset_index().rename(columns={"index": "metric", 0: "value"}),
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    solve_button = mo.ui.run_button(label="Solve route")
    mo.vstack([mo.md("## Optimize"), solve_button])
    return (solve_button,)


@app.cell
def _(
    ALNS,
    baseline_best_ratio,
    baseline_greedy,
    build_trail_chains,
    iterations,
    mo,
    net,
    race_params,
    solve_button,
):
    mo.stop(not solve_button.value, mo.md("*Press **Solve route** to run the optimizer.*"))

    chains = build_trail_chains(net)
    greedy_route = baseline_greedy(net, chains, race_params)
    ratio_route = baseline_best_ratio(net, chains, race_params)
    alns_result = ALNS(net, chains, race_params, seed=0).solve(iterations=int(iterations.value))
    best_route = alns_result.route
    return alns_result, best_route, chains, greedy_route, ratio_route


@app.cell
def _(alns_result, best_route, greedy_route, mo, pd, ratio_route):
    comparison = pd.DataFrame(
        [
            {"method": "greedy nearest trail", **greedy_route.evaluate()},
            {"method": "greedy best ratio", **ratio_route.evaluate()},
            {"method": f"ALNS ({alns_result.iterations} iters)", **best_route.evaluate()},
        ]
    )
    mo.vstack(
        [
            mo.md("### Results"),
            mo.ui.table(comparison, selection=None),
            mo.md(
                f"**Best score: {best_route.evaluate()['score']}** "
                f"({best_route.evaluate()['trails_completed']} trails "
                f"+ {best_route.evaluate()['unique_miles']} unique miles) — "
                f"route validation: {best_route.validate() or 'OK'}"
            ),
        ]
    )
    return (comparison,)


@app.cell
def _(alns_result, mo, pd):
    import altair as alt2

    _hist = pd.DataFrame(alns_result.history, columns=["iteration", "best_score"])
    mo.ui.altair_chart(
        alt2.Chart(_hist)
        .mark_line(size=2, point=False)
        .encode(x="iteration:Q", y=alt2.Y("best_score:Q", scale=alt2.Scale(zero=False), title="best score"))
        .properties(height=220, title="ALNS convergence")
    )
    return (alt2,)


@app.cell
def _(best_route, ex, mo):
    _cues = ex.cue_sheet(best_route)
    mo.vstack([mo.md("### Cue sheet"), mo.ui.table(_cues, selection=None, page_size=25)])
    return


@app.cell
def _(best_route, ex, mo, net):
    _fig_path = ex.plot_route(net, best_route)
    mo.vstack([mo.md("### Route map"), mo.image(str(_fig_path))])
    return


@app.cell
def _(mo):
    export_button = mo.ui.run_button(label="Write GeoPackage / GPX / cue sheet")
    export_button
    return (export_button,)


@app.cell
def _(best_route, ex, export_button, mo, net):
    mo.stop(not export_button.value, mo.md("*Press the button to write outputs to `outputs/`.*"))
    _written = ex.export_all(net, best_route)
    mo.md("Wrote:\n\n" + "\n".join(f"- `{p}`" for p in _written.values()))
    return


if __name__ == "__main__":
    app.run()
