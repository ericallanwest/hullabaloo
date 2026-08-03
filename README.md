# Hullabaloo — optimizing a 7-hour trail rogaine

Route optimization for the **Blacksburg Adventure Skills Hullabaloo**, a timed rogaine on
the Pandapas Pond / Poverty Creek trail network near Blacksburg, Virginia.

You get 7 hours. You score two ways, each capped at 40 points:

- **1 point per trail** completed end to end — there are 40 trails
- **1 point per unique mile** of trail covered — there are 39.7 miles

Covering the entire network would score a perfect 80. It would also take **14.3 hours** —
and that is a lower bound that ignores all the backtracking a real closed loop forces. With
7 hours the best possible is 35.37, or **44% of a perfect score**, covering 15.4 of the
40.1 miles. The whole game is deciding *which* miles. That makes this a **prize-collecting
arc routing problem**: points sit on edges rather than nodes, re-walking an edge earns
nothing the second time, and the tour must start and finish at the trailhead.

![optimized route](outputs/route_map.png)

## Result

| | score | trails | unique miles | time |
|---|---|---|---|---|
| greedy nearest-trail baseline | 26.42 | 16 | 10.42 | 6.36 h |
| greedy best-ratio baseline | 24.06 | 13 | 11.06 | 6.10 h |
| ALNS (6 seeds × 500 iterations) | 35.25 | 20 | 15.25 | 6.99 h |
| **MILP — proven optimal** | **35.37** | **20** | **15.37** | **7.00 h** |

**The problem is solved to proven global optimality.** HiGHS closed the gap to 0.00% in
181 s: no 7-hour route scores better than **35.37**. That is a **34% improvement** over a
sensible greedy baseline, and the route uses the full budget to the second.

The heuristic is not wasted — it reached 35.249, **within 0.34%** of the optimum, in a few
minutes, and its incumbent is fed to the solver as a valid primal cut that prunes the
search hard. Three of six seeds landed ≥35.0, so the search finds the right basin
reliably rather than getting lucky.

Worth noting how the heuristic gets there: its solution encoding *targets* only 14 trails,
yet the decoded route *completes* 20. The extra six are collected for free on deadhead legs
between targets — which is exactly why the evaluator credits every edge the walk touches
rather than only the ones it set out to collect. Scoring solely the targeted trails would
have thrown away six points.

See [`outputs/run_report.json`](outputs/run_report.json) for the exact figures.

Deliverables land in `outputs/`:

- `hullabaloo.gpkg` — one multi-layer GeoPackage: `edges`, `nodes`, `trails`,
  `route`, `depot`
- `route.gpx` — the tour as a GPX track with a predicted schedule, loadable onto a watch
- `route_cues.csv` — turn-by-turn cue sheet with running time and running score
- `route_map.html` — interactive Leaflet map on USGS topo/imagery layers, for checking the
  model against reality
- `route_map.png`, `run_report.json`

## Stepping through the race

`docs/` is a static site that walks the optimized route one leg at a time: the map fills in
as you step, and the sidebar counts up the miles, the trails completed and the score. It is
the fastest way to see *why* the route is shaped the way it is — where the plan spends a
repeat to reach something worth more, and where it gives up on a trail entirely.

The left-hand **pace factor** control switches between eleven pre-solved itineraries,
from 1.0 (textbook Tobler) to 2.0. Pace is
the one parameter a racer can neither measure in advance nor control on the day: Tobler's
constants describe unhurried walking, and how much quicker a fit competitor actually moves
is a guess. Sweeping it shows how much the plan depends on that guess.

Underneath sits the printed **tanZnavigation Pandapas Pond sheet**, georeferenced on
[MapWarper](https://mapwarper.net/maps/110238) and faded in over USGS topo, imagery or
hillshade. It is the map racers actually carry, so it is the one worth checking a route
against — a line that looks reasonable on a DEM can still cross something the paper map
knows about.

Each itinerary is solved independently and shipped as JSON:

```bash
pixi run presets              # solve every pace factor, write docs/data/
python -m http.server -d docs # then open http://localhost:8000
```

Everything the page needs is computed in [`src/hullabaloo/webexport.py`](src/hullabaloo/webexport.py)
— geometry, traversal categories and running totals all arrive precomputed, so the
JavaScript is a renderer and nothing more. `write_preset` refuses to publish an itinerary
whose numbers do not reconcile, since a static file is trusted by readers who will never
run the solver.

### Where "proven optimal" stops being true

Scoring is `min(trails, 40) + min(unique_miles, 40)`, but a minimum is not linear, so the
MILP maximizes the *uncapped* sum. That substitution is free only while neither cap binds,
and the usual argument for it is a speed limit: nobody beats Tobler's peak, so
`peak_speed x 7 h` bounds the distance covered. At the shipped pace of 1.35 that ceiling is
35.2 miles, comfortably under the 40-mile cap.

The ceiling scales with pace, and it crosses 40 miles at a pace factor of **1.53** — so
the a-priori argument simply expires partway up the sweep. Past that the case has to be
made on the answer instead, which is easy: the uncapped objective `U` dominates the true
score `S` everywhere, so if the returned optimum uses under 40 miles and under 40 trails
then `S = U` there, and `S(x) <= U(x) <= U(x*) = S(x*)` for every other route. The optimum
of the relaxation is the optimum of the real thing.

When a cap does bind, the itinerary is still published — it is a real walk inside the time
budget and a racer wants it either way — but it is labelled **not proven optimal** in the
sidebar rather than quietly keeping a claim it no longer supports. `check_preset` rejects
any preset that binds a cap while still asserting optimality.

The honest reading of that regime is that the question has changed. Once points stop
accruing, the interesting objective is no longer "score the most" but "collect the cap
*fastest*" — a minimum-duration tour, which is a different problem this model does not
express. Solving it properly is future work.

The sweep re-prices every edge from cached DEM profiles rather than re-running the
elevation stage eleven times, and a check asserts that re-pricing at the stored pace
reproduces the committed edge table exactly — otherwise the sweep would quietly be solving
a different network from the one the repository ships.

## Running it

```bash
pixi install
cp .env.example .env          # then set TNT_USER / TNT_PASS

pixi run pipeline             # end to end; skips stages whose output exists
pixi run pipeline-bounded     # also spends 10 min proving an optimality bound
pixi run presets              # solve one itinerary per pace factor for the web viz
pixi run test                 # 34 tests
pixi run app                  # interactive marimo app
```

The first run downloads a 206 MB lidar tile and caches it.

## The three problems worth talking about

### 1. The input is not a network

The site publishes one GPX per trail. Downloading all 40 gives 40.14 miles of geometry and
**no topology at all** — 40 independent `LineString`s that merely overlap on a map.

The instinct is to snap coincident endpoints together. On this data that fails badly:

| contact type | count (within 10 m) |
|---|---|
| endpoint meets another **endpoint** | 20 / 80 |
| endpoint meets another trail's **geometry** | 55 / 80 |

The network is dominated by **T-junctions** — a trail ending partway along another — plus
10 genuine interior X-crossings. Endpoint-only snapping leaves most junctions unconnected
and routes nothing. So each trail is cut at split positions gathered from three sources
(true intersections, foreign-endpoint projections, own endpoints), and the resulting
endpoints are clustered into shared nodes. Every edge keeps its parent `trail_id`, because
"trail completed" is defined over a trail's complete edge set.

The snap tolerance is a real judgement call, so it is picked from a sweep rather than
asserted. Component count falls 14 → 6 → 4 → **3** and then flatlines at 3 all the way out
to 50 m. That plateau is the structural floor; 18 m is the smallest tolerance that reaches
it. Every junction that exists only above 10 m was then reviewed individually against
imagery (`topology.junctions_in_band`) — each is a real intersection where two
independently recorded tracks disagree by plausible under-canopy GPS error.

**Independent check against OSM.** OpenStreetMap is *node-based*: editors connect ways by
reusing a node id, so junctions there are explicit rather than inferred. That makes OSM an
excellent control — a different method on different source data. Comparing:

| tolerance | our junctions confirmed by OSM | OSM shared nodes matched by ours |
|---|---|---|
| 10 m | 71.0% | 28.3% |
| 20 m | **82.6%** | 33.3% |
| 30 m | 89.9% | 40.0% |

83% of the junctions derived by snapping GPX tracks are independently confirmed. The
reverse column is low for a benign reason: OSM also splits ways wherever a tag changes
(surface, access), so consecutive pieces of one logical trail share a node too, and many
OSM junctions involve paths that are not among the 40 scored trails.

**Why not just build the whole network from OSM?** It is a fair question — OSM's topology
is correct by construction, which would delete most of this phase. OSM covers **99.8%** of
the official trail geometry within 25 m, so the geometry is there. The obstacle is
*identity*: scoring is defined over the event's own 40 trails, and only 26 of 40 match an
OSM way by name. Rebuilding on OSM would mean re-attaching scored identity by geometric
matching, and any mislabelling silently corrupts the score. The current network validates
cleanly — every trail contiguous, length preserved to 99.9%, 83% junction agreement — so
OSM earns its keep here as a *validator* and as the source of the forest roads, rather than
as a replacement for the official geometry.

### 2. The network is in three pieces — and forest roads join them

After noding, the 40 trails form **three components** separated by genuine 285–700 m gaps
that no snapping will ever close:

| component | trails | miles |
|---|---|---|
| main | 28 | 30.8 |
| north | 7 | 5.8 |
| west | 5 | 3.5 |

My first instinct was to bridge them by bushwhacking, and I built a whole least-cost
off-trail model to do it. That was solving the wrong problem — twice over, as it turned
out. The gaps are spanned by **forest service roads**: legal, full speed, already on the
ground.

Finding them took two sources, and the difference between them is the interesting part:

- **USFS EDW road layers** are authoritative for *National Forest System* roads, but they
  are a systems inventory and omit non-system roads. They place `BRUSH MOUNTAIN` 2.6 km
  from the nearest trail and leave the entire northern component 887 m from any road.
- **OpenStreetMap** has the roads people actually walk.

Adding the roads collapsed the network from three components to **one** and dropped
cross-component bushwhack candidates to **zero**. Roads earn no points — they are not
among the 40 scored trails — but they cost full-speed time instead of the 60% off-trail
penalty, which is why they dominate bushwhacking so completely.

### The tag is a bad proxy for "walkable"

Bushwhacking was no longer load-bearing, but the optimizer still *wanted* it: a 0.58 mile
shortcut appeared in 8 of 11 published routes. Chasing that one leg is what finally
finished the job, and it had two causes.

The import matched `highway=track` only. Three roads were invisible to it, and each was
the reason the optimizer wanted to leave the trail somewhere:

| road | OSM tag | why it matters |
|---|---|---|
| Meadowbrook Drive | `highway=tertiary`, paved | passes within **10 m** of the Highway trail |
| Stone Cutter's Hollow Access Road | `highway=path`, gravel | reaches Mineral Way and Wavelength, **9 m** from Meadowbrook |
| Forest Service Road 708 | `highway=service`, unpaved | Queen Anne meets it in two places |

Ways are now matched by name as well as tag, which is the more durable fix: a road arrives
whole even when only some of its pieces are tagged `track`, which is exactly the situation
for Forest Service Road.

The second cause was self-inflicted. Roads were clipped to 250 m of the trail network and
whatever survived was kept — which sliced Meadowbrook Drive into **three disconnected
pieces**, because its middle runs further than 250 m from any trail. The optimizer was
bushwhacking across a gap *the clip had created*, retracing the road's own alignment
off-trail.

The tell was the on-network alternative between those two nodes: **7330 seconds**,
detouring via Beauty and Gateway. A two-hour detour to cross 600 m of road is not a
routing decision, it is a broken graph. The clip now trims the ends of a way but keeps the
span between retained sections, so dead-end spurs still go and through-routes survive.

Two smaller rules earned their place at the same time. Connectors joining two points on
the **same trail** are rejected outright as switchback cuts — 167 of them — because a
trail climbs a hillside in traverses precisely so boots do not run the fall line, and a
model that offers the shortcut is proposing erosion. And Pandapas Pond has to be masked
from NHD hydrography; an earlier flatness-based fallback flagged **20% of the map** as
water, so the code rejects any fallback claiming more than 2% rather than quietly letting
routes swim.

Together these took bushwhack candidates from **128 to 4**, and re-solving at pace 1.0,
1.5 and 2.0 used **none of them**. The off-trail model was removed.

Nothing in the network is off-trail any more. The short `depot access` link from the start
line was the last thing modelled that way, and it is gravel or paved on the ground, so it
is walked at full speed too. It was only marked off-trail because it is not one of the 40
scored trails — but that is a question about *points*, and points are already withheld by
`score_mi`. Pace and scoring are independent, and conflating them cost the model a 40%
speed penalty on ground that deserves none.

### The bug that only aerial imagery could catch

This is the best thing the off-trail model produced, and it is worth keeping even though
the code is gone.

Every automated check passed. The connectors avoided water, respected slope limits, ran at
a plausible 0.56× on-trail speed, and the network validated cleanly. Then rendering them
over USGS aerial imagery showed the optimal route's **longest bushwhack running 848 m
straight through a residential neighbourhood** — houses, driveways, mown lawns, a swimming
pool. NLCD confirms 17% of that corridor is Developed.

The cause was structural rather than a coding error: the cost surface came from a
**bare-earth** DEM, which is by definition the terrain with buildings and vegetation
stripped out. Where the houses are, it saw a gentle, inviting slope. No amount of slope or
hydrography checking finds this, because the input simply does not contain the
information.

The fix was NLCD land cover with developed classes marked impassable, and **the route lost
0.057 points**. The illegal shortcut was worth almost nothing. It was just invisible to
every check that did not involve looking at a photograph.

### 3. Scoring on arcs breaks the usual machinery

Points sit on edges, each edge pays only once however often it is walked, and completing a
trail requires *all* of its edges. Standard TSP/VRP formulations do not apply.

**Encoding.** A solution is an ordered list of target trails. A deterministic decoder turns
it into a real walk: from the current position, take the shortest path to whichever end of
the next trail is cheaper, walk it end to end, repeat, return to the depot. Every solution
is connected and depot-anchored *by construction*, so the ALNS operators only reason about
which trails and in what order — never about graph connectivity. Deadhead legs often cross
trail not yet walked, and those miles score, so the evaluator credits every edge the walk
actually touches.

**The bound.** The MILP exists to say what the heuristic cannot: how far from optimal the
answer is. At this size it does better than bound the problem — it closes it. Its one
easy-to-forget constraint is **connectivity**: flow conservation alone is satisfied by any
set of disjoint circuits, so without a single-commodity-flow constraint tying the traversed
subgraph to the depot, the solver returns a lovely high-scoring loop on the far side of the
property that never touches the start line.

Two smaller things that made the difference between a bound and a proof:

- **Feeding the heuristic's score in as a primal cut.** `objective >= 35.249` is valid
  because ALNS actually achieved it, and it prunes an enormous amount of the tree without
  excluding the optimum.
- **Proving the 40-point category caps cannot bind, instead of modelling them.** The
  network is 40.14 miles against a 40-mile cap, so the cap is not *obviously* unreachable
  — but nobody exceeds Tobler's peak speed, and `7 h × 3.73 mph = 26.1 mi` is a hard
  ceiling. `check_caps_nonbinding` asserts this rather than assuming it, which keeps a
  step function out of the objective.

One implementation trap worth recording: PuLP hands HiGHS the *negated* objective for
maximization, so `mip_dual_bound` comes back with the opposite sign. Reading it naively
produced an "upper bound" of −38 on a positive-valued maximization, and a meaningless 0%
gap — which would have made the headline optimality claim pure fiction.

## Elevation

USGS 3DEP **1 m lidar** (`VA_FEMA-NRCS_SouthCentral_2017`, tile `x54y413`), driving a
parameterized Tobler hiking function:

```
W(S) = base * exp(-k * |S + s0|)     km/h,  S = rise/run
defaults: base 6.0, k 3.5, s0 0.05;  off-trail x0.60
```

The `s0` offset puts peak speed on a gentle *downhill*, which is exactly why the routing
graph must be directed — the optimizer can and does exploit which way round to run a loop.
Time is integrated segment by segment along each profile, since Tobler is convex in |slope|
and averaging grade over a whole trail badly underestimates rolling terrain.

Defaults imply **3.13 mph on the flat** (19.2 min/mile). These are literature constants,
**not** calibrated to a specific person carrying a pack for seven hours — `pace_factor`
exists for that.

### Validating against published gains

The site publishes forward *and* reverse elevation gain for all 40 trails: 80 free
ground-truth values. Lidar-derived gain regresses at **R² ≈ 0.85** with near-zero bias.

The residuals are interesting rather than alarming:

- **Crosscut** is published at **1 ft** of gain over 990 m of singletrack, and its GPX
  elevation profile is *exactly monotone* — physically implausible on real trail. The
  published figures are computed from GPX elevations and inherit whatever the recording
  device did.
- **Upper Chimney**, a sustained 1,100 ft climb, agrees closely across GPX, published, and
  lidar — noise barely matters when the signal is that large.
- **Prickly Pear** is the biggest disagreement (published 688 ft, lidar 185 ft). Lidar and
  GPX agree *exactly* at the low end (696 m) and diverge only near the top, which points to
  GPS elevation drift rather than misplaced geometry.

Bare-earth lidar is the more trustworthy source, and it is what the model uses.

Smoothing is specified in **metres, not pixels** — a `sigma=3` read as pixels means 3 m on
the lidar but 30 m on the 10 m fallback, which flattens genuinely steep trails. Total
network traversal time moves only **3.4%** across sigma 0–12 m, so the model is not hostage
to the choice; 5 m drives the bias against published gains to roughly zero.

## Layout

```
src/hullabaloo/
  config.py        every tunable, in one place
  ingest.py        login, scrape trail table, download GPX
  topology.py      planarize -> noded, validated network
  elevation.py     3DEP fetch, smooth, sample profiles
  tobler.py        parameterized speed model
  graph.py         directed time-weighted graph, and the single score() definition
  optimize_alns.py destroy/repair search with SA acceptance
  optimize_milp.py exact formulation, for the bound
  export.py        GeoPackage / GPX / cue sheet / map
  webexport.py     itinerary + network JSON for the web planner
  pipeline.py      end-to-end runner
scripts/
  build_presets.py solve one itinerary per pace factor
notebooks/         marimo (.py, git-diffable)
docs/              static route planner, served by GitHub Pages
  index.html       panes, controls
  css/style.css    light + dark theme
  js/viz.js        Leaflet rendering and step state
  data/            network.json + one preset per pace factor
tests/             34 tests
```

Logic lives in `src/`, notebooks import it. Notebooks stay readable, functions stay
testable.

## Stack

`geopandas` · `shapely 2` · `pyogrio` · `pyproj` · `rasterio` · `rioxarray` ·
`scikit-image` · `scipy` · `networkx` · `PuLP` + `HiGHS` · `marimo` · `pixi`

Working CRS is **EPSG:6346** (NAD83(2011) / UTM 17N, metres), matching the lidar tiling.

## Honest limitations

- **Tobler is uncalibrated.** The constants are from the literature, not from anyone's
  actual pace over seven hours with a pack. Fitting `base`/`k` to real Strava times on
  these trails would be the single highest-value improvement.
- **Every route is now on legal, walkable ground at full speed** — scored trail, forest
  road, or the gravel access link from the start line. That is a result, not an assumption:
  the off-trail model was kept until re-solving showed no route wanted it. It also means
  the plan is only as good as the road data, and OSM tags are an imperfect guide to what
  can actually be walked.
- **The lidar is 2016/17 vintage** and predates any recent trail work.
- **The three components may be joined by forest roads absent from the dataset.** Worth
  checking the gaps against imagery; a real road should be added as a full-speed connector
  rather than a 60% bushwhack.
- **Scoring is interpreted** as fractional miles plus whole trails. It is isolated in one
  `score_edges` function, so an alternate reading is a one-line change.

## Data

Trail data © The Trail Not Taken / tanZ Navigation LLC, accessed with a guest account.
Elevation from the USGS 3D Elevation Program (public domain). Hydrography from USGS NHD.
