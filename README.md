# Hullabaloo — optimizing a 7-hour trail rogaine

Route optimization for the **Blacksburg Adventure Skills Hullabaloo**, a timed rogaine on
the Pandapas Pond / Poverty Creek trail network near Blacksburg, Virginia.

You get 7 hours. You score two ways, each capped at 40 points:

- **1 point per trail** completed end to end — there are 40 trails
- **1 point per unique mile** of trail covered — the organizer's sheet totals 39.7, and
  measuring the tracks puts it at 40.16

Covering the entire network would score a perfect 80. It would also take **10.6 hours** at
the shipped pace — and that is a lower bound that walks every edge in its cheaper direction
and ignores all the backtracking a real closed loop forces. With 7 hours the best possible
is 48.11, or **60% of a perfect score**, covering 22.1 of the 40.16 miles. The whole game is
deciding *which* miles. That makes this a **prize-collecting arc routing problem**: points
sit on edges rather than nodes, re-walking an edge earns nothing the second time, and the
tour must start and finish at the trailhead.

![optimized route](outputs/route_map.png)

## Result

| | score | trails | unique miles | time |
|---|---|---|---|---|
| greedy nearest-trail baseline | 42.07 | 23 | 19.07 | 7.00 h |
| greedy best-ratio baseline | 37.01 | 20 | 17.01 | 6.21 h |
| ALNS (6 seeds × 500 iterations) | 47.60 | 26 | 21.60 | 6.96 h |
| **MILP — proven optimal** | **48.11** | **26** | **22.11** | **7.00 h** |

**The problem is solved to proven global optimality.** HiGHS closed the gap to 0.00% in
45 seconds: no 7-hour route scores better than **48.11** at the shipped 5.0 mph, and the
route uses the full budget to the second.

That is a **14% improvement** over a sensible greedy baseline — a much narrower margin than
this project reported before the forest roads went in, and the narrowing is real rather than
a regression. Roads help the greedy walker too: the baseline climbed from 26.42 to 42.07
once it could move between trail clusters at full speed. An optimizer's advantage is largest
on a network where the obvious move is often wrong, and the roads made a lot of obvious
moves right.

The heuristic is not wasted — it reached 47.603, **within 1.1%** of the optimum, in a few
minutes, and its incumbent is fed to the solver as a valid primal cut that prunes the search
hard. All six seeds landed between 46.23 and 47.60, so the search finds the right basin
reliably rather than getting lucky on one of them.

Worth noting how the heuristic gets there: its solution encoding *targets* only 20 trails,
yet the decoded route *completes* 26. The extra six are collected for free on deadhead legs
between targets — which is exactly why the evaluator credits every edge the walk touches
rather than only the ones it set out to collect. Scoring solely the targeted trails would
have thrown away six points.

See [`outputs/run_report.json`](outputs/run_report.json) for the exact figures.

Deliverables land in `outputs/`:

- `hullabaloo.gpkg` — one multi-layer GeoPackage: `edges`, `nodes`, `trails`,
  `route`, `depot`
- `route.gpx` — the tour as a GPX track with a predicted schedule, loadable onto a watch
- `route_cues.csv` — cue sheet: which way to turn onto each leg, with running time and
  running score
- `route_map.html` — interactive Leaflet map on USGS topo/imagery layers, for checking the
  model against reality
- `route_map.png`, `run_report.json`

## Stepping through the race

`docs/` is a static site that walks the optimized route one leg at a time: the map fills in
as you step, and the sidebar counts up the miles, the trails completed and the score. It is
the fastest way to see *why* the route is shaped the way it is — where the plan spends a
repeat to reach something worth more, and where it gives up on a trail entirely.

Three things are on the map before you take a step. **Forest roads** draw as a dashed pale
red, distinct from the grey trail network, because which roads exist is a fact about the
ground rather than a consequence of the itinerary — they cost full-speed time and earn no
points, and a racer wants to know where they run whether this route uses them or not. Past
zoom 15, **names ride the lines themselves** rather than waiting for a hover, so the map can
be read at a glance; a name is drawn only where its line is long enough to hold it, and the
label rides a reversed copy of any line running east to west, since SVG text follows the
path's own direction and would otherwise read backwards.

The itinerary names a **turn** onto every leg — straight, slight, plain, sharp or turn
around, eight classes in all — computed from the change in bearing at the junction. The
angle is measured over 25 m of trail rather than off the terminal coordinate pair, because
noding snapped every endpoint onto a cluster centroid up to 18 m away and the last segment
of an arc is mostly that displacement. **⬇ Download CSV** hands the whole thing over: the
summary block at the top, then a line per leg carrying the turn, the distance, the clock and
the running score.

The left-hand controls pick from twenty-four pre-solved itineraries: six **top speeds** from
5.0 to 7.5 mph, and at each speed four **plans** — three that differ in what they are
willing to commit to, and a fourth built to be changed while you are running it.

Speed is branded in mph rather than as a multiplier because a multiplier is not something a
racer can feel. The number quoted is the peak of Tobler's curve, reached on a gentle
downhill; flat ground runs about 16% slower, so the 6.0 mph tier is a 11:54 flat mile. The
`pace_factor` underneath is derived from it exactly — at the peak gradient the exponential
term is 1, so peak speed is just `base_kmh x pace_factor` and the two convert without
fitting anything.

The first three exist because the network poses one genuinely open question: whether to go
west. The **West End** — Poverty Creek (lower), Skullcap, Trillium, Beauty and
neighbours, 8.3 miles of trail — sits four to five km from the start line, so reaching it
costs a long out-and-back. Plan **a** is the unconstrained optimum. Plan **b** insists the
West End is completed; plan **c** forbids it outright.

The point of publishing a, b and c is that **the right answer flips inside the published
range**. Each plan is solved to proven optimality under its own rule, and the cost of each
constraint moves monotonically and in opposite directions:

| top speed | a — best | b — commit | c — skip | cost of b | cost of c | cheaper |
|---|---|---|---|---|---|---|
| 5.0 mph | 48.11 | 42.20 | 45.98 | −5.91 | −2.13 | skip |
| 5.5 | 52.48 | 47.64 | 49.52 | −4.84 | −2.96 | skip |
| 6.0 | 56.00 | 52.48 | 52.70 | −3.52 | −3.30 | skip |
| **6.5** | 59.68 | 56.23 | 54.93 | **−3.45** | **−4.76** | **commit** |
| 7.0 | 62.80 | 60.29 | 57.55 | −2.51 | −5.25 | commit |
| 7.5 | 65.66 | 64.14 | 60.13 | −1.52 | −5.54 | commit |

Below about 6.2 mph the long haul west is not worth its out-and-back and skipping is the
cheaper compromise; above it, the West End's 8.3 miles are reachable cheaply enough that
*not* going is the expensive choice. A racer whose speed sits near that crossover is exactly
the racer a single itinerary would serve worst, which is the argument for the grid.

### Plan d — the one you can change your mind about

The three plans above are all knife-edge optimal: each spends the budget to the second and
finishes between 6:59 and 7:00. That is what optimal looks like, and it is also brittle. A
racer running 5% slower than modelled is twenty minutes over with no guidance about what to
give up, deciding at mile 20 what the solver spent five minutes deciding at a desk.

Plan **d** gives up a little of the total — at most three points against the proven
optimum — to bank trail points earlier and to shorten cleanly. It ships with a menu of
**cuts**, each priced in advance: skip this loop, save this many minutes, lose these points.

The unit is a **contiguous closed excursion** — a stretch of the walk that leaves a junction
and comes back to it. Splicing one out is always safe, because the arcs before it end where
the arcs after it begin, so what remains is still one closed walk from the start line back to
it. No re-solve, no chance of stranding the route on the far side of the property. A typical
route offers around nine of these, worth roughly two hours of droppable time in total.

#### What plan d is actually searched for

Having a menu is not the same as having a *useful* menu. A route can carry two hours of
droppable loops and still be useless to a racer if every one of them is behind him by the
time he knows he is behind. So the objective is not droppable time, but **droppable time
still ahead of you at 3.5 hours** — the halfway mark, where a racer first has enough
evidence to judge his own pace and still enough day left to act on it:

```
maximise   droppable minutes reachable after hour 3.5
subject to final score >= proven optimum - 3.0
```

Only *maximal* excursions count, since excursions nest and dropping the outer one already
drops the inner. The MILP cannot express this — it chooses a set of arcs and the walk order
falls out of a Hierholzer pass afterwards — so the ALNS does the searching while the MILP
supplies the ceiling the floor is measured against.

The search needs a running start. Constructive starts at 5.0 mph reach 42.07 and 37.01,
both *below* the 45.11 floor, and the cheapest way to buy room to shorten is to not score —
so a search launched from there wanders further down and never visits a publishable route
at all. Plan d is therefore solved in two phases: a plain score-seeking run supplies an
anchor above the floor, then the flexibility run trades score for room from there.

What it buys, in droppable minutes at 3.5 h, against the front-loading objective it replaced:

| Top speed | before | after |
|---|---:|---:|
| 5.0 mph | 18 | 106 |
| 5.5 mph | 12 | 144 |
| 6.0 mph | 38 | 140 |
| 6.5 mph | 34 | 152 |
| 7.0 mph | 130 | 160 |
| 7.5 mph | 6 | 123 |

The two tiers that mattered most are the two that had nothing: at 5.0 and 5.5 mph a racer
who fell 5% behind previously had no affordable way to recover.

Two things about the menu are worth stating plainly, because both are easy to get wrong:

**The costs do not add up.** Scoring is over the *set* of edges walked and a trail scores
only when every one of its edges is covered, so a trail can straddle two excursions: drop
either and you keep it, drop both and it is gone. On the shipped 5.0 mph route there is a
pair that costs exactly **one point more together than apart**. Every reduced-budget plan is
therefore costed as a whole, never by summing the singles.

**Each threshold is a split time.** Because the plan has no slack, "keep this loop only if
the clock is before X" works out to the planned arrival time at that junction. Hit the
junction on schedule and the loop is affordable; arrive late and it is not, by exactly the
margin you are late.

The same reasoning constrains the reduced-budget table. A cut is only on offer until you
reach its junction, so the "if you only have 6.5 hours" rows are built solely from cuts
still ahead of the 3.5-hour decision point. Left unconstrained the optimiser reliably picks
whatever is cheapest per minute, which tends to sit early: before this was enforced, five of
the six tiers published a 6.5-hour plan that had to be committed to within the first half
hour — correct arithmetic, useless advice.

Folded away below them is the older **pace sweep**, eleven itineraries from 1.0 (textbook
Tobler) to 2.0. It answers a modelling question rather than a racing one: pace is the one
parameter a racer can neither measure in advance nor control on the day, and sweeping it
shows how much the plan depends on that guess.

Underneath sits the printed **tanZnavigation Pandapas Pond sheet**, georeferenced on
MapWarper and faded in over any of nine basemaps. It is the map racers actually carry, so it
is the one worth checking a route against — a line that looks reasonable on a DEM can still
cross something the paper map knows about.

Both printed pages are there, on independent sliders:
[page 1](https://mapwarper.net/maps/110238) at roughly 1:20000 covering the whole area, and
[page 2](https://mapwarper.net/maps/110277) — McDonald / Stonecutter Hollow — at 1:10000.
They need no cross-fading or zoom switching to coexist, because each layer is bounded to its
own sheet extent: page 2 stacks above page 1 and paints only where it has coverage,
revealing page 1 everywhere else. Page 2 also gets one more native zoom level, since the
same size scan covers about a quarter of the ground.

Each itinerary is solved independently and shipped as JSON:

```bash
pixi run presets              # solve all 18 speed-tier itineraries, write docs/data/
python -m http.server -d docs # then open http://localhost:8000
```

Everything the page needs is computed in [`src/hullabaloo/webexport.py`](src/hullabaloo/webexport.py)
— geometry, traversal categories, turn cues and running totals all arrive precomputed, so
the JavaScript is a renderer and nothing more. `write_preset` refuses to publish an
itinerary whose numbers do not reconcile, since a static file is trusted by readers who will
never run the solver.

The turn cues are a case where that division of labour is load-bearing rather than tidy.
Exported geometry is simplified at 2 m and rounded to about a metre, which is several
degrees of slack on a 25 m bearing — enough to call a fork the wrong way. Computing the
angle in Python against the unsimplified projected line and shipping the answer is the only
version that is right, and it means the page, its CSV download and `route_cues.csv` all
print the same arrow for the same junction. `check_preset` asserts the glyph agrees with
both its label and its angle, because a turn arrow is the one field on a step that can be
wrong while every number around it still adds up.

### Where "proven optimal" stops being true

Scoring is `min(trails, 40) + min(unique_miles, 40)`, but a minimum is not linear, so the
MILP maximizes the *uncapped* sum. That substitution is free only while neither cap binds,
and the usual argument for it is a speed limit: nobody beats Tobler's peak, so
`peak_speed x 7 h` bounds the distance covered. At the shipped speed of 5.0 mph that ceiling
is 35.0 miles, comfortably under the 40-mile cap.

The ceiling is just `top speed x 7 h`, so it crosses 40 miles at **5.71 mph** — between the
5.5 and 6.0 mph tiers. The a-priori argument therefore expires two rungs up the published
ladder, and the four fastest tiers cannot use it at all. Past that the case has to be
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
pixi run presets              # solve the 18 published itineraries for the web viz
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

| component | trails | miles | centroid | where |
|---|---|---|---|---|
| main | 28 | 30.9 | 37.264, −80.495 | the spine, running the Poverty Creek valley |
| south | 7 | 5.8 | 37.245, −80.483 | Highway, Turkey Trot, Blunderbuss and neighbours |
| east | 5 | 3.5 | 37.268, −80.461 | Slytherin, Grinder, Chasers, Running Cedar |

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

Together these took bushwhack candidates from **128 to 4**, and re-solving **all eleven
paces** used none of them. That was the condition for deleting the off-trail model, checked
rather than assumed: had a single route still wanted a bushwhack, the stage would have
stayed. `build_presets` now asserts it, so a preset that needs one cannot ship quietly.

Scores rose at every pace — 37.33 at 1.0 and 65.40 at 2.0, against 36.40 and 64.83 before —
which is the direction the change predicts. Ground that used to cost the 60% off-trail
penalty is now walked at full speed, and the freed time buys more trail.

Nothing in the network is off-trail any more. The short `Start/Finish` link from the start
line was the last thing modelled that way, and it is gravel or paved on the ground, so it
is walked at full speed too. It was only marked off-trail because it is not one of the 40
scored trails — but that is a question about *points*, and points are already withheld by
`score_mi`. Pace and scoring are independent, and conflating them cost the model a 40%
speed penalty on ground that deserves none.

### Removing ground the race never touches

With the roads settled, three OSM ways turned out to be used by no route at any pace: `Woods
& Field` north of Glade Road, a `Poverty Creek Connector` that is a user-created trail the
Forest Service discourages, and the arm of Meadowbrook Drive running out to Glade Road.

Dropping them is provably lossless, which is worth spelling out because it is the sort of
claim that usually isn't. The previously-optimal route uses none of the removed edges, so it
stays feasible and the new optimum cannot be lower; and the pruned feasible set is a subset
of the original, so it cannot be higher. Equal, necessarily.

What that argument does *not* cover is its own premise — that pruning removed exactly those
three and nothing else. Node numbering shifts, and the topology stage runs its own orphan
pass, so a segment orphaned *because* of the pruning would be a silent fourth deletion. That
is what re-solving all eleven paces afterwards actually verifies.

The denylist is keyed by OSM way id rather than name, for two independent reasons. `Forest
road` is a fallback label this code invents for unnamed tracks and currently covers twelve
distinct ways, so a name cannot identify one. And Meadowbrook Drive arrives as two ways of
which one is load-bearing — a name-based rule would delete the road the whole import exists
to add.

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

Unscaled, the defaults imply **3.13 mph on the flat** (19.2 min/mile) and a 3.73 mph peak.
These are literature constants, **not** calibrated to a specific person racing for seven
hours — `pace_factor` exists for that, and is set by naming a top speed rather than by
picking a multiplier:

```
pace_factor = top_speed_mph / (base_kmh * 0.621371)      # = mph / 3.728
```

| top speed | pace_factor | flat mph | flat min/mile |
|-----------|-------------|----------|---------------|
| 5.0 mph   | 1.341       | 4.20     | 14:18         |
| 5.5       | 1.475       | 4.62     | 12:59         |
| 6.0       | 1.609       | 5.04     | 11:54         |
| 6.5       | 1.743       | 5.46     | 10:59         |
| 7.0       | 1.878       | 5.88     | 10:12         |
| 7.5       | 2.012       | 6.30     | 9:31          |

The shipped default is the bottom rung, 5.0 mph. Everything committed to the repository —
the timed edge table, `run_report.json`, every figure quoted above — is priced at it.

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
  build_presets.py solve the published itineraries (6 speeds x 3 plans)
notebooks/         marimo (.py, git-diffable)
docs/              static route planner, served by GitHub Pages
  index.html       panes, controls
  css/style.css    light + dark theme
  js/viz.js        Leaflet rendering, step state, CSV download
  icons/           NPS self-guiding-trail mark, used as the favicon
  data/            network.json + presets.json manifest + one file per itinerary
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
- **More walkable ground may still be missing from OSM.** The two ways that removed the
  last bushwhack were found by asking why the optimizer wanted to leave the trail at one
  specific place, which is a diagnostic that only fires when something is *badly* missing.
  A road the routes would merely have preferred leaves no such trace.
- **Four unnamed `Forest road` ways are carried but unused** by every route at every pace.
  They stay in because identifying them needs local knowledge the data does not carry —
  which of them are legitimate connectors and which are user-created trails a land manager
  discourages. Keeping an unused edge costs nothing; deleting a legitimate one costs a route.
- **Scoring is interpreted** as fractional miles plus whole trails. It is isolated in one
  `score_edges` function, so an alternate reading is a one-line change.

## Data

Trail data © The Trail Not Taken / tanZ Navigation LLC, accessed with a guest account.
Elevation from the USGS 3D Elevation Program (public domain). Hydrography from USGS NHD.
