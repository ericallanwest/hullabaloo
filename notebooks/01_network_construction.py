"""Building a routable network out of 40 unrelated GPX tracks.

This notebook documents the hardest and least glamorous part of the project: the input is
not a network, and making it into one is where almost all the risk lives.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="Network construction")


@app.cell
def _():
    import marimo as mo
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    from hullabaloo.config import CONFIG, EDGES, M_PER_MILE, NODES, TRAILS_RAW
    from hullabaloo import topology as topo

    return EDGES, NODES, TRAILS_RAW, gpd, mo, pd, topo


@app.cell
def _(mo):
    mo.md(r"""
    # Phase 1-2 — from 40 polylines to one routable network

    The Trail Not Taken publishes one GPX per trail. Downloading all 40 gives you
    40.14 miles of geometry and **no topology whatsoever**: 40 independent
    `LineString`s that happen to overlap on a map.

    A router needs shared nodes at junctions. Getting there is the make-or-break step,
    and it is worth showing the evidence rather than asserting the result.
    """)
    return


@app.cell
def _(TRAILS_RAW, gpd, mo, pd):
    trails = gpd.read_parquet(TRAILS_RAW)
    _summary = pd.DataFrame(
        {
            "trails": [len(trails)],
            "track points": [int(trails["n_points"].sum())],
            "published miles": [round(trails["official_dist_mi"].sum(), 2)],
            "computed miles": [round(trails.to_crs("EPSG:6346").length.sum() / 1609.344, 2)],
        }
    )
    mo.vstack([mo.md("## The raw input"), mo.ui.table(_summary, selection=None)])
    return (trails,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Why endpoint snapping is not enough

    A first instinct is to snap coincident endpoints together. On this dataset that
    fails badly. Measuring every one of the 80 trail endpoints against every other
    trail:

    | contact type | count (within 10 m) |
    |---|---|
    | endpoint meets another **endpoint** | 20 / 80 |
    | endpoint meets another trail's **geometry** | 55 / 80 |

    So the network is dominated by **T-junctions** — a trail ending partway along
    another trail — plus 10 genuine interior **X-crossings**. Endpoint-only snapping
    would leave the great majority of junctions unconnected, producing a network that
    looks right on a map and routes nothing.

    The fix is to collect split positions along each trail from three sources (true
    intersections, foreign-endpoint projections, and its own endpoints), cut every
    trail at those positions, then cluster the resulting endpoints into shared nodes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Choosing the snap tolerance on evidence

    The tolerance is a real judgement call: too tight and the network stays
    fragmented, too loose and it invents junctions that do not exist. Rather than
    pick a round number, sweep it.
    """)
    return


@app.cell
def _(mo, topo, trails):
    @mo.cache
    def _sweep():
        return topo.tolerance_sweep(trails, (2, 5, 8, 10, 12, 15, 18, 20, 25, 30))

    sweep = _sweep()
    return (sweep,)


@app.cell
def _(mo, sweep):
    import altair as alt

    _chart = (
        alt.Chart(sweep)
        .mark_line(point=True, size=3)
        .encode(
            x=alt.X("tol_m:Q", title="snap tolerance (m)"),
            y=alt.Y("components:Q", title="connected components"),
        )
        .properties(height=260, title="Component count vs snap tolerance")
    )
    mo.vstack([mo.ui.altair_chart(_chart), mo.ui.table(sweep, selection=None)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Component count falls 14 → 6 → 4 → **3** and then flatlines at 3 all the way out
    to 50 m. That plateau is the important part: it is the **structural floor**. The
    three remaining pieces are separated by genuine 285–700 m gaps that no amount of
    snapping will ever close.

    18 m is the smallest tolerance that reaches the floor, so that is the choice.
    Since 18 m is generous by GIS standards, every junction that only exists above
    10 m gets reviewed individually below.
    """)
    return


@app.cell
def _(mo, topo, trails):
    band = topo.junctions_in_band(trails, 10.0, 20.0)
    mo.vstack(
        [
            mo.md("### Junctions created in the 10–18 m band (reviewed individually)"),
            mo.ui.table(band, selection=None),
            mo.md(
                "Each of these is a real trail intersection where two independently "
                "recorded tracks disagree by an amount entirely consistent with "
                "under-canopy consumer GPS error. The paste-able lat/lon column makes "
                "them quick to check against satellite imagery."
            ),
        ]
    )
    return


@app.cell
def _(EDGES, NODES, gpd, mo, topo, trails):
    edges = gpd.read_parquet(EDGES)
    nodes = gpd.read_parquet(NODES)
    _raw_m = float(trails.to_crs("EPSG:6346").length.sum())
    mo.vstack(
        [
            mo.md("## Structural validation of the built network"),
            mo.ui.table(topo.validate(edges, nodes, _raw_m), selection=None),
        ]
    )
    return edges, nodes


@app.cell
def _(mo):
    mo.md(r"""
    Two of those checks earned their keep during development:

    * **degenerate self-loops** — an early version cut trails at split points closer
      together than the snap tolerance. Both ends of the resulting sliver then
      clustered into the *same* node, producing 19 zero-length self-loops. The fix
      was to refuse to create such edges at all: two junctions closer than the
      tolerance are, by our own definition, one junction.
    * **length preserved** — splitting must not lose or duplicate geometry. It comes
      out at 99.9%, the shortfall being the 0.5 m simplification applied to shed GPS
      jitter.
    """)
    return


@app.cell
def _(edges, mo, nodes, topo):
    mo.vstack(
        [
            mo.md("## The three components"),
            mo.ui.table(topo.component_summary(edges, nodes).reset_index(), selection=None),
            mo.md(
                "This is the finding that shapes the rest of the project. The 7 trails of "
                "the northern group and the 5 of the western group cannot be reached from "
                "the main 28 on trail alone. Any route that scores well **must** "
                "bushwhack — which is why the off-trail model is load-bearing rather than "
                "a refinement."
            ),
        ]
    )
    return


@app.cell
def _(edges, mo, nodes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hullabaloo.topology import connected_components

    _comp = connected_components(edges, len(nodes))
    _e = edges.copy()
    _e["component"] = [_comp[int(u)] for u in _e["u"]]

    _fig, _ax = plt.subplots(figsize=(10, 8))
    for _label, _grp in _e.groupby("component"):
        _grp.plot(ax=_ax, linewidth=1.8, label=f"component {_label} ({len(_grp)} edges)")
    nodes[nodes["is_depot"]].plot(ax=_ax, color="black", marker="*", markersize=200, zorder=6)
    _ax.set_title("Trail network coloured by connected component\n(star = start/finish)")
    _ax.legend(frameon=False)
    _ax.set_axis_off()
    mo.mpl.interactive(_fig)
    return


if __name__ == "__main__":
    app.run()
