'use strict';

// Hullabaloo route planner — steps through a pre-solved itinerary for a 7-hour rogaine.
//
// This file is deliberately a renderer and nothing more. Every step in a preset already
// carries its own geometry (WGS84 [lat, lon]), its traversal category, and the running
// totals as of that step, all computed in src/hullabaloo/webexport.py. There is no
// geometry resolution, no re-derivation of what counts as a repeat, and no unit
// conversion here — if a number looks wrong, it is wrong in the exporter, not the page.

// ── Constants ──────────────────────────────────────────────────────────────
const NETWORK_URL = 'data/network.json';
const MANIFEST_URL = 'data/presets.json';

// The Pandapas Pond trail map, georeferenced on MapWarper (mapwarper.net/maps/110238).
// Set to null to drop the layer entirely — the Trail Map slider then hides itself and
// the rest of the page is unaffected.
const MAPWARPER_ID = 110238;
// The warped sheet's own extent, from the MapWarper API. Bounding the layer stops
// Leaflet requesting tiles outside it on every pan.
const MAPWARPER_BOUNDS = [[37.2329516, -80.5472426], [37.2920102, -80.4451182]];
// The source scan is 5100x3300 px across that extent, which is roughly z16 of detail.
// Past that MapWarper upscales server-side; letting Leaflet stretch its own tiles
// instead gives the same picture for a fraction of the requests, and keeps the overlay
// from vanishing when a deeper basemap is selected.
const MAPWARPER_NATIVE_ZOOM = 16;

// Page 2 of the printed sheet — "McDonald Hollow / Stonecutter Hollow", drawn at roughly
// 1:10000 against Page 1's 1:20000, so twice the detail over about a quarter of the area.
//
// The two sheets need no cross-fading or zoom switching to coexist: this one is bounded
// to its own extent and stacks above Page 1, so it paints only where it has coverage and
// simply reveals Page 1 everywhere else. Both sliders stay independent.
const MAPWARPER_ID_2 = 110277;
// bbox from /api/v1/maps/110277, reordered to Leaflet's [[south, west], [north, east]].
const MAPWARPER_2_BOUNDS = [[37.2315110, -80.5009348], [37.2609191, -80.4497400]];
// Same 5100x3300 scan over roughly a quarter of Page 1's ground, so one zoom level deeper
// before MapWarper starts upscaling: z17 asks for ~4770 px of tiles across 5100 px of
// source, and z18 would ask for nearly double what the scan actually contains.
const MAPWARPER_2_NATIVE_ZOOM = 17;

const CAT_COLOR = { unique: '#f7882f', offtrail: '#c0392b', repeat: '#c0392b' };
const CAT_LABEL = { unique: '', offtrail: 'road', repeat: 'repeat' };
const GOLD = '#FFD700';

// Two reds, and the difference between them is the whole point of the layer: this pale
// dashed one is road that is *available* — legal, walkable, on the map from the first step
// whether the route uses it or not — while CAT_COLOR.offtrail (#c0392b, solid) is road the
// itinerary actually walks. Matched to the Smokies planner's Connector styling, which
// draws the same distinction on the same kind of network.
const ROAD_AVAILABLE = { color: '#f08080', weight: 4, opacity: 0.65, dashArray: '4,6' };

// Network grays flip with the UI theme: a mid gray that reads as "faint" on a light
// basemap disappears entirely on a dark one. Note the two move in *opposite* directions
// when the goal is legibility — the light-theme gray gets darker, the dark-theme gray gets
// lighter. Both are steps away from the background, which is the thing that matters.
const NET_THEMES = { light: '#6e6e6e', dark: '#8a8a99' };
let netColor = NET_THEMES.light;

// Name labels ride the lines themselves rather than sitting in boxes, so they need a halo
// to stay legible over aerial imagery — the fill is the text, the stroke is painted behind
// it. Same reasoning as the network grays: the pair inverts with the theme.
const LABEL_THEMES = {
  light: { fill: '#2b2b33', halo: '#ffffff' },
  dark:  { fill: '#e6e7ee', halo: '#15171e' },
};
let labelTheme = LABEL_THEMES.light;

// Below this the network is a tangle and every name would overlap its neighbours. At 15 a
// tenth-mile edge is about 40 px across, which is roughly where a short name starts to fit.
const LABEL_MIN_ZOOM = 15;

// ── Utilities ──────────────────────────────────────────────────────────────
// Round to whole minutes *first*, then split. Taking the hour before rounding the
// remainder lets 6 h 59.94 m render as "6h 60m" — which the 6:59:56 optimal route does hit.
function fmtHM(s) {
  const minutes = Math.round(s / 60);
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`;
}

// Per-leg durations. No leg across any published pace comes close to an hour — the
// longest is under 36 minutes — so hours would be a column of zeroes and seconds are
// what actually distinguishes one short connector from another. The hour branch is
// insurance for a slower pace factor being added later.
function fmtMS(s) {
  const total = Math.round(s);          // round before splitting, or 59.6s renders "0m 60s"
  const minutes = Math.floor(total / 60), seconds = total % 60;
  if (minutes < 60) return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m `
       + `${String(seconds).padStart(2, '0')}s`;
}

// The race runs 07:00 to 14:00 local. Presets carry elapsed seconds only, so wall-clock
// time is derived here — change this one constant if the start time ever moves.
const RACE_START_HOUR = 7;

function fmtTimeOfDay(elapsedSeconds) {
  // Floor, not round — a clock reads 7:00 until 7:01 actually arrives, and this shares a
  // line with the elapsed figure, which floors too. Rounding one and flooring the other
  // makes 45 seconds in read "7:01am · 0:00 elapsed".
  const minutes = Math.floor(elapsedSeconds / 60);
  const hour24 = RACE_START_HOUR + Math.floor(minutes / 60);
  const hour12 = ((hour24 + 11) % 12) + 1;           // 12 -> 12pm, 13 -> 1pm
  return `${hour12}:${String(minutes % 60).padStart(2, '0')}${hour24 >= 12 ? 'pm' : 'am'}`;
}

// Scores show one decimal, truncated rather than rounded: a tenth of a mile only pays
// once it has actually been walked, so 21.38 miles is worth 21.3 points, not 21.4. The
// hundredths place is real but the organizer will never score to it.
//
// The epsilon is defensive, not a fix for anything observed — checked against every exact
// tenth from 0 to 80 and every cumulative score in all eleven presets, it never changes
// the answer. It is here because a value landing a hair under a tenth through float error
// would otherwise lose a point that was genuinely earned, and truncation has no rounding
// slack to absorb that.
function fmtScore(value) {
  return (Math.floor(value * 10 + 1e-9) / 10).toFixed(1);
}

function fmtClock(s) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return `${h}:${String(m).padStart(2, '0')}`;
}

function esc(text) {
  return String(text ?? '').replace(/[&<>"]/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

const $ = id => document.getElementById(id);

// Revalidate the data files rather than trusting the browser's copy. GitHub Pages serves
// these with Cache-Control: max-age=600, so after a re-solve a returning visitor would
// otherwise be shown a stale itinerary for up to ten minutes — silently, and with the
// numbers all self-consistent, which is the worst kind of wrong. "no-cache" still uses
// the cached body when the server answers 304, so this costs a conditional request, not a
// re-download.
const FETCH_OPTS = { cache: 'no-cache' };

// ── Module-level state ─────────────────────────────────────────────────────
let map;
let NETWORK = null;     // network.json, loaded once
let PRESET = null;      // the itinerary currently displayed
let currentStep = 1;
let stepPlayTimer = null;

// Module scope rather than inside the DOM-ready handler: the preset controls are built
// from the manifest by top-level functions, and every one of them has to be able to halt
// playback before swapping the itinerary out from under it.
function stopPlaying() {
  if (stepPlayTimer) { clearInterval(stepPlayTimer); stepPlayTimer = null; }
  const btn = $('btnStepPlay');
  if (btn) btn.textContent = '▶ Play';
}
let startMarker = null;
let homeBounds = null;

const roadGroup   = L.layerGroup();   // every road in the network, walked or not
const netGroup    = L.layerGroup();   // full trail network, gray backdrop
const walkedGroup = L.layerGroup();   // steps already taken
const arrowGroup  = L.layerGroup();   // direction-of-travel arrowheads
const currentGroup = L.layerGroup();  // the step under the slider
const labelGroup  = L.layerGroup();   // invisible lines that exist only to carry text

// The labels hang off invisible copies of the network lines rather than off the lines
// themselves, for one reason: SVG text on a path runs in the path's own direction, so a
// trail digitised east-to-west renders its name mirrored and upside down. The carrier can
// be reversed to read left-to-right without disturbing the line the map actually draws,
// which still has to run the way the route walks it for the arrowheads to point right.
let labelLines = [];

// ── Drawing helpers ────────────────────────────────────────────────────────
function stepStyle(step) {
  if (step.cat === 'repeat') return { color: CAT_COLOR.repeat, weight: 5, opacity: 0.9, dashArray: '6,5' };
  // "offtrail" is the scoring category, not a surface: it means forest road or access
  // road, walked at full speed but earning no points. Nothing in the network is actually
  // off-trail, so these draw solid.
  if (step.cat === 'offtrail') return { color: CAT_COLOR.offtrail, weight: 5, opacity: 0.9 };
  return { color: CAT_COLOR.unique, weight: 5, opacity: 1 };
}

function stepVisible(step) {
  if (step.cat === 'repeat')   return $('togRepeat').checked;
  if (step.cat === 'offtrail') return $('togRoads').checked;
  return true;
}

function stepPopup(step) {
  const tag = CAT_LABEL[step.cat];
  const kind = step.cat === 'offtrail' ? 'road' : tag;
  const turn = step.turn ? `${step.glyph} ${esc(step.turn)}<br>` : '';
  return `<b>${esc(step.name)}</b>${kind ? ` <i>(${kind})</i>` : ''}<br>` +
    turn +
    `node ${step.from_node} → ${step.to_node}<br>` +
    `${step.miles.toFixed(2)} mi &nbsp; ${fmtMS(step.seconds)}<br>` +
    `+${step.gain_ft} ft ↑ / −${step.loss_ft} ft ↓<br>` +
    `<small>step ${step.i} · ${fmtClock(step.cum.seconds)} elapsed · score ${fmtScore(step.cum.score)}</small>`;
}

function addStepLine(group, step, options, arrowColor) {
  const line = L.polyline(step.geometry, options)
    .bindPopup(stepPopup(step))
    .bindTooltip(`${step.i}. ${esc(step.name)}`, { sticky: true, opacity: 0.85 });
  line.addTo(group);
  if (!arrowColor) return;
  const arrows = color => L.polylineDecorator(line, {
    patterns: [{ offset: '15%', repeat: '130px', symbol: L.Symbol.arrowHead({
      pixelSize: 11, polygon: false,
      pathOptions: { color, weight: color === '#fff' ? 5 : 2, opacity: 0.9, fillOpacity: 0 },
    }) }],
  });
  arrows('#fff').addTo(arrowGroup);
  arrows(arrowColor).addTo(arrowGroup);
}

// ── Rendering ──────────────────────────────────────────────────────────────
function drawNetwork() {
  netGroup.clearLayers();
  roadGroup.clearLayers();
  labelGroup.clearLayers();
  labelLines = [];
  if (!NETWORK) return;

  // Edges the current route walks are drawn by the step layers on top; the backdrop is
  // deliberately the *whole* network, so the unwalked remainder stays visible as the
  // thing the seven-hour budget could not reach.
  //
  // Roads split off into their own layer because they are a different kind of ground, not
  // a different part of the route: they cost time and earn nothing, so which ones exist is
  // a fact about the network that a racer wants before the first step, not a consequence
  // of the itinerary. `kind` has always been in network.json; nothing read it until now.
  for (const edge of NETWORK.edges) {
    const road = edge.kind !== 'trail';
    L.polyline(edge.geometry,
      road ? ROAD_AVAILABLE : { color: netColor, weight: 3, opacity: 0.75 })
      .bindTooltip(`${esc(edge.name)} — ${edge.miles.toFixed(2)} mi${road ? ' (road)' : ''}`,
        { sticky: true, opacity: 0.85 })
      .addTo(road ? roadGroup : netGroup);

    // Reversed when the edge runs east to west, so the name reads left to right instead of
    // mirrored. Longitude is enough to decide it: the sign of east never changes with zoom.
    const path = edge.geometry;
    const westward = path[path.length - 1][1] < path[0][1];
    // interactive:false so an invisible line can never swallow a click meant for the
    // visible one underneath it, which would break every tooltip and popup on the map.
    labelLines.push({
      name: edge.name,
      line: L.polyline(westward ? [...path].reverse() : path, {
        opacity: 0, weight: 1, interactive: false,
      }).addTo(labelGroup),
    });
  }
  refreshLabels();
}

// Rough width of a string at the label's own font, with room to breathe at either end.
// Only ever used to decide whether a name fits its line, so an estimate is enough — and it
// is the *only* thing standing between this map and a hundred half-drawn names, because
// SVG happily clips text that runs off the end of its path rather than declining to draw
// it. Erring generous costs a few labels; erring tight costs legibility.
function labelWidthPx(text) {
  return text.length * 6.2 + 24;
}

function polylinePixelLength(line) {
  const points = line.getLatLngs().map(ll => map.latLngToLayerPoint(ll));
  let total = 0;
  for (let i = 1; i < points.length; i++) total += points[i].distanceTo(points[i - 1]);
  return total;
}

// Re-applied on every zoom rather than styled once, because whether a name fits is a fact
// about the current scale, not about the edge.
function refreshLabels() {
  if (!labelLines.length) return;
  const on = $('togLabels').checked && map.getZoom() >= LABEL_MIN_ZOOM;

  for (const { name, line } of labelLines) {
    // Cleared first without exception. leaflet-textpath only reclaims its old <text> node
    // on setText(null) — setting fresh text over existing text appends a second node and
    // orphans the first, so labelling the same edge at ten zoom levels leaves ten stacked
    // copies of the name quietly thickening on the map.
    line.setText(null);
    if (!on || labelWidthPx(name) > polylinePixelLength(line)) continue;
    line.setText(` ${name} `, {
      center: true,
      offset: -4,                 // lift the baseline clear of the line it rides
      attributes: {
        'font-size': '11px',
        'font-weight': '600',
        'font-family': 'system-ui, sans-serif',
        fill: labelTheme.fill,
        stroke: labelTheme.halo,
        'stroke-width': 3,
        'stroke-linejoin': 'round',
        'paint-order': 'stroke',   // halo behind the glyphs, not smeared over them
        'pointer-events': 'none',  // a label must never intercept a click on its own line
      },
    });
  }
}

function renderStep(step) {
  walkedGroup.clearLayers();
  currentGroup.clearLayers();
  arrowGroup.clearLayers();
  if (!PRESET) return;

  for (let i = 0; i < step; i++) {
    const s = PRESET.steps[i];
    if (i === step - 1) {
      addStepLine(currentGroup, s, { color: GOLD, weight: 7, opacity: 1 }, GOLD);
    } else if (stepVisible(s)) {
      addStepLine(walkedGroup, s, stepStyle(s), CAT_COLOR[s.cat]);
    }
  }
}

function updateSidebar(step) {
  const s = PRESET.steps[step - 1];
  const cum = s.cum;

  $('sbHeader').textContent = `Step ${step} of ${PRESET.totals.n_steps}`;
  // Wall clock rather than elapsed: the itinerary below shows both, and what a racer
  // standing on the trail wants from the headline is the time on their watch.
  // No escaping needed — textContent never parses markup, and escaping here would
  // render an ampersand in a trail name as a literal "&amp;".
  $('sbClock').textContent = `${fmtTimeOfDay(cum.seconds)} · ${s.name}`;

  const run = cumThrough(step);
  $('sbTotal').textContent  = `${cum.miles.toFixed(1)} mi / ${fmtHM(cum.seconds)}`;
  // Unique mileage truncates, unlike the pure distances around it, because it is also
  // the scoring figure restated below — two rows labelled "unique" disagreeing would be
  // plainly wrong, and this is the one row in the block that earns points. The mileage
  // column does not reliably add up to Total either way: at one decimal place, rounding
  // already breaks the sum on about 38% of steps.
  $('sbUnique').textContent = `${fmtScore(cum.unique_miles)} mi / ${fmtHM(run.unique)}`;
  $('sbRepeat').textContent = `${cum.repeat_miles.toFixed(1)} mi / ${fmtHM(run.repeat)}`;
  $('sbRoads').textContent    = `${cum.offtrail_miles.toFixed(1)} mi / ${fmtHM(run.offtrail)}`;
  $('sbElev').textContent   =
    `${run.gain.toLocaleString()} ft ↑ / ${run.loss.toLocaleString()} ft ↓`;
  $('sbTrails').textContent = `${cum.trails_completed} of ${PRESET.network.n_trails}`;
  // Truncated like the score itself, so the block visibly adds up. It always does: score
  // is trails plus unique miles, and trails is a whole number, so truncating the sum and
  // truncating the mileage give the same tenth.
  $('sbUniqueScore').textContent = fmtScore(cum.unique_miles);
  $('sbScore').textContent  = fmtScore(cum.score);

  $('stepLbl').textContent = `Step ${step} / ${PRESET.totals.n_steps}`;
  $('stepSlider').value = step;
}

// Summed here rather than shipped per step: the exporter already carries cumulative
// miles per category and per-leg relief, and the three categories partition the walk, so
// the matching seconds and the running gain/loss are one pass away over data already on
// the wire. Cheap at this size — under a hundred legs, recomputed per step.
function cumThrough(step) {
  const totals = { unique: 0, repeat: 0, offtrail: 0, gain: 0, loss: 0 };
  for (let i = 0; i < step; i++) {
    const s = PRESET.steps[i];
    totals[s.cat] += s.seconds;
    totals.gain += s.gain_ft;
    totals.loss += s.loss_ft;
  }
  return totals;
}

// Which trails a step finishes off, keyed by step number. Shared by the itinerary and the
// CSV download so the two cannot credit a completion to different steps.
function completionsByStep() {
  const completedAt = new Map();
  for (const trail of PRESET.trails)
    if (trail.completed_at_step) {
      if (!completedAt.has(trail.completed_at_step)) completedAt.set(trail.completed_at_step, []);
      completedAt.get(trail.completed_at_step).push(trail.name);
    }
  return completedAt;
}

function buildItinerary() {
  const completedAt = completionsByStep();

  $('itinerary').innerHTML = PRESET.steps.map(s => {
    const tag = s.cat === 'repeat' ? 'repeat'
              : s.cat === 'offtrail' ? 'road' : '';
    const done = completedAt.get(s.i);
    // The turn is precomputed in webexport.py from the unsimplified geometry — the glyph
    // arrives ready to print, so this stays a renderer.
    const turn = s.glyph
      ? ` <span class="itin-turn" title="${esc(s.turn)}">${s.glyph}</span>` : '';
    return `<div class="itin-step" data-step="${s.i}">
      <span class="itin-n" style="color:${CAT_COLOR[s.cat]}">${s.i}.</span>
      <b>${esc(s.name)}</b>${turn}${tag ? ` <span class="itin-tag">(${tag})</span>` : ''}
      ${done ? ` <span class="itin-done">✓ ${done.length} trail${done.length > 1 ? 's' : ''}</span>` : ''}
      <span class="itin-meta">
        ${s.miles.toFixed(2)} mi &nbsp; ${fmtMS(s.seconds)} &nbsp; ${s.gain_ft} ft ↑ / ${s.loss_ft} ft ↓<br>
        ${fmtTimeOfDay(s.cum.seconds)} &nbsp;·&nbsp; ${fmtClock(s.cum.seconds)} elapsed &nbsp;·&nbsp; score ${fmtScore(s.cum.score)}
      </span>
    </div>`;
  }).join('');

  $('itinerary').querySelectorAll('.itin-step').forEach(row =>
    row.addEventListener('click', () => setStep(+row.dataset.step)));
}

function highlightItinerary(step) {
  document.querySelectorAll('.itin-step').forEach(row => {
    const active = +row.dataset.step === step;
    row.classList.toggle('active', active);
    if (active) row.scrollIntoView({ block: 'nearest' });
  });
}

function setStep(step) {
  if (!PRESET) return;   // a failed load leaves the controls live but with nothing to show
  currentStep = Math.max(1, Math.min(step, PRESET.totals.n_steps));
  renderStep(currentStep);
  updateSidebar(currentStep);
  highlightItinerary(currentStep);
}

function renderPresetInfo() {
  const t = PRESET.totals, solver = PRESET.solver || {};
  // Presets predating the optimality block fall back to the raw solver gap.
  const opt = PRESET.optimality || { caps_binding: [], proven: solver.gap_pct === 0, note: null };

  // The optimality claim has to be scoped to the constraint it was proved under. "No
  // better route exists" is true of option a; for the constrained plans it is only true
  // among routes obeying their rule, and stating it unqualified would claim far too much.
  const constrained = !!(PRESET.corridor_rule || {}).corridor;
  const within = constrained ? ' among routes obeying this constraint' : '';
  // The legacy family is indexed by pace, not speed, and says so.
  const axis = PRESET.speed_mph != null ? 'speed' : 'pace';

  let claim;
  if (opt.caps_binding.length) claim = '';   // the caveat below says it instead
  else if (opt.proven) claim = ` Proven optimal by the MILP — no better route exists at this ${axis}${within}.`;
  else if (solver.gap_pct != null) claim = ` Within ${solver.gap_pct.toFixed(1)}% of a proven upper bound${within}.`;
  else claim = '';

  // What this plan gave up to be what it is. Stated in points against the free optimum at
  // the same speed, which is the only comparison that isolates the cost of the rule —
  // comparing across speeds would fold in how fast the racer is.
  const rule = PRESET.corridor_rule || {};
  const cost = PRESET.delta_vs_free;
  const ruleBlock = !rule.corridor ? '' :
    `<div class="info-rule"><b>${esc(PRESET.option_label || '')}</b>` +
    `<br><span class="info-note">${esc(rule.description || '')}</span>` +
    (cost == null ? '' :
      `<br><span class="info-note">Costs <b>${Math.abs(cost).toFixed(2)}</b> points ` +
      `against the best available plan at this speed.</span>`) +
    `</div>`;

  // Where the miles actually went. A corridor the route never enters keeps its row, at
  // zero — for the two constrained plans the empty row *is* the point.
  const rows = PRESET.corridors || [];
  const corridorBlock = !rows.length ? '' :
    `<div class="info-sub">Where the miles go</div>` +
    rows.map(r =>
      `<div class="info-row"><span>${esc(r.corridor)}</span>` +
      `<span><b>${r.unique_miles.toFixed(1)}</b>` +
      `<span class="info-note"> / ${r.total_miles.toFixed(1)} mi · ` +
      `${r.trails_completed}/${r.n_trails}</span></span></div>`).join('');

  const speed = PRESET.speed_mph != null
    ? `<div class="info-row"><span>Top speed</span><span><b>${PRESET.speed_mph.toFixed(1)} mph</b></span></div>`
    : '';

  $('presetInfo').innerHTML =
    speed +
    `<div class="info-row"><span>Score</span><span><b>${fmtScore(t.score)}</b></span></div>` +
    `<div class="info-row"><span>Trails completed</span><span><b>${t.trails_completed}</b></span></div>` +
    `<div class="info-row"><span>Unique miles</span><span><b>${t.unique_miles.toFixed(2)}</b></span></div>` +
    `<div class="info-row"><span>Distance walked</span><span><b>${t.walked_miles.toFixed(2)} mi</b></span></div>` +
    `<div class="info-row"><span>Finish time</span><span><b>${fmtClock(t.time_s)}</b></span></div>` +
    `<span class="info-note">Score = trails completed + unique miles.${claim}</span>` +
    ruleBlock +
    corridorBlock +
    (opt.caps_binding.length
      ? `<div class="caveat"><b>⚠ Not proven optimal</b><br>` +
        `${opt.caps_binding.map(esc).join('; ')}. ${esc(opt.note || '')}</div>`
      : '');
}

// ── CSV download ───────────────────────────────────────────────────────────
// A racer wants the plan on paper or on a phone, not behind a slider. Everything printed
// here is read straight off the preset and through the same formatters the sidebar uses,
// so the file cannot quietly disagree with the screen it was downloaded from.
function csvCell(value) {
  const text = value == null ? '' : String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

const csvRow = (...cells) => cells.map(csvCell).join(',');

function buildCsv() {
  const t = PRESET.totals, solver = PRESET.solver || {};
  const opt = PRESET.optimality || { caps_binding: [], proven: null };
  const run = cumThrough(t.n_steps);
  const completedAt = completionsByStep();

  const rule = PRESET.corridor_rule || {};
  const meta = [
    ['Hullabaloo Route Planner'],
    ...(PRESET.speed_mph != null ? [['Top speed', `${PRESET.speed_mph.toFixed(1)} mph`]] : []),
    ['Pace factor', PRESET.pace_factor.toFixed(4)],
    ...(rule.corridor ? [
      ['Plan', PRESET.option_label],
      ['Constraint', rule.description],
      ...(PRESET.delta_vs_free != null
        ? [['Cost vs best available', `${PRESET.delta_vs_free.toFixed(2)} points`]] : []),
    ] : []),
    ['Score', fmtScore(t.score)],
    ['Trails completed', `${t.trails_completed} of ${PRESET.network.n_trails}`],
    ['Unique trail miles', t.unique_miles.toFixed(2)],
    ['Distance walked', `${t.walked_miles.toFixed(2)} mi`],
    ['Repeat miles', t.repeat_miles.toFixed(2)],
    ['Road miles', t.offtrail_miles.toFixed(2)],
    ['Elevation', `${run.gain.toLocaleString()} ft up / ${run.loss.toLocaleString()} ft down`],
    ['Finish time', fmtClock(t.time_s)],
    ['Time budget', fmtClock(PRESET.race.time_budget_s)],
    ['Steps', t.n_steps],
    ['Proven optimal', opt.caps_binding.length ? `no — ${opt.caps_binding.join('; ')}`
                                               : (opt.proven ? 'yes' : 'no')],
    ['Solver', `${solver.source || '?'} (${solver.status || '?'}` +
               `${solver.gap_pct != null ? `, gap ${solver.gap_pct.toFixed(1)}%` : ''})`],
    ['Network', `${PRESET.network.total_miles} mi / ${PRESET.network.n_trails} trails`],
    ['Scoring', 'Score = trails completed + unique trail miles'],
    ['Generated', new Date().toISOString()],
    ['Source', 'https://ericallanwest.github.io/hullabaloo/'],
  ];

  const header = csvRow(
    'step', 'turn', 'cue', 'name', 'type', 'miles', 'minutes', 'gain_ft', 'loss_ft',
    'clock', 'elapsed', 'cum_miles', 'cum_unique_miles', 'cum_repeat_miles',
    'cum_road_miles', 'cum_trails', 'cum_score', 'trails_completed_here',
  );

  const rows = PRESET.steps.map(s => csvRow(
    s.i,
    s.glyph || '',
    s.turn || '',
    s.name,
    s.cat === 'offtrail' ? 'road' : s.cat === 'repeat' ? 'repeat' : 'trail',
    s.miles.toFixed(2),
    (s.seconds / 60).toFixed(1),
    s.gain_ft,
    s.loss_ft,
    fmtTimeOfDay(s.cum.seconds),
    fmtClock(s.cum.seconds),
    s.cum.miles.toFixed(3),
    s.cum.unique_miles.toFixed(3),
    s.cum.repeat_miles.toFixed(3),
    s.cum.offtrail_miles.toFixed(3),
    s.cum.trails_completed,
    fmtScore(s.cum.score),
    (completedAt.get(s.i) || []).join('; '),
  ));

  return [...meta.map(cells => csvRow(...cells)), '', header, ...rows].join('\r\n');
}

function downloadCsv() {
  if (!PRESET) return;
  // The BOM is load-bearing: without it Excel on Windows reads the file as the local
  // codepage and turns every turn arrow — and every apostrophe in a trail name — into
  // mojibake, which would defeat the point of putting glyphs in the file at all.
  const blob = new Blob(['﻿' + buildCsv()], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  // Named after whichever family this itinerary came from, so three plans downloaded at
  // the same speed do not overwrite each other in the downloads folder.
  link.download = PRESET.option
    ? `hullabaloo_route_${PRESET.speed_mph.toFixed(1)}mph_${PRESET.option}.csv`
    : `hullabaloo_route_pace${PRESET.pace_factor.toFixed(2)}.csv`;
  link.click();
  // Revoked on the next tick rather than immediately: the click only *starts* the save, and
  // browsers that read the blob asynchronously will hand back an empty file if the URL has
  // already been torn down under them.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function showPreset(preset) {
  PRESET = preset;
  currentStep = 1;
  const slider = $('stepSlider');
  slider.max = preset.totals.n_steps;
  slider.value = 1;

  if (startMarker) startMarker.remove();
  startMarker = L.marker(preset.race.start_latlng, {
    icon: L.divIcon({
      html: '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">' +
            '<polygon points="10,1 19,18 1,18" fill="#27ae60" stroke="#fff" stroke-width="2" stroke-linejoin="round"/></svg>',
      className: 'start-marker', iconSize: [20, 20], iconAnchor: [10, 10],
    }),
    zIndexOffset: 1000,
  }).bindPopup('<b>START / FINISH</b><br>The race is a closed circuit.').addTo(map);

  renderPresetInfo();
  buildItinerary();
  setStep(1);
  $('btnCsv').disabled = false;   // nothing to download until an itinerary is on screen
}

// ── Basemaps ───────────────────────────────────────────────────────────────
// Bing addresses tiles by quadkey rather than x/y, so it needs its own getTileUrl:
// each zoom level contributes one base-4 digit encoding which quadrant the tile is in.
const _BingLayer = L.TileLayer.extend({
  getTileUrl(coords) {
    let quadkey = '';
    for (let i = coords.z; i > 0; i--) {
      let digit = 0;
      const mask = 1 << (i - 1);
      if (coords.x & mask) digit++;
      if (coords.y & mask) digit += 2;
      quadkey += digit;
    }
    return `https://ecn.t3.tiles.virtualearth.net/tiles/a${quadkey}.jpeg?g=1`;
  },
});

// Order and default deliberately mirror the Smokies planner, so the two sites feel like
// the same tool. The USGS layers are Hullabaloo's own and sit at the bottom.
// ArcGIS REST tiles order the path {z}/{y}/{x}, not {z}/{x}/{y}.
const BASEMAPS = {
  'OSM Grayscale': L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    { attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors', maxZoom: 19, zIndex: 1, className: 'grayscale-layer' }),
  'OSM Color': L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    { attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors', maxZoom: 19, zIndex: 1 }),
  'CartoDB Light': L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    { attribution: '&copy; OpenStreetMap contributors &copy; CARTO', maxZoom: 19, zIndex: 1 }),
  'CartoDB Dark': L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    { attribution: '&copy; OpenStreetMap contributors &copy; CARTO', maxZoom: 19, zIndex: 1 }),
  'Google Maps': L.tileLayer('https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
    { attribution: '&copy; Google', maxZoom: 20, zIndex: 1 }),
  'Bing Aerial': new _BingLayer('', { attribution: '&copy; Microsoft Bing', maxZoom: 19, zIndex: 1 }),
  'ESRI World Topo': L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    { attribution: 'Tiles &copy; Esri', maxZoom: 19, zIndex: 1 }),
  'USGS Topo': L.tileLayer('https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}',
    { attribution: 'USGS The National Map', maxZoom: 16, zIndex: 1 }),
  'USGS Imagery': L.tileLayer('https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}',
    { attribution: 'USGS The National Map', maxZoom: 16, zIndex: 1 }),
};

const DEFAULT_BASEMAP = 'OSM Grayscale';

// ── Data loading ───────────────────────────────────────────────────────────
function presetFile(pace) {
  // Mirrors webexport.preset_filename(): 1.3 -> preset_p130.json
  return `data/preset_p${String(Math.round(pace * 100)).padStart(3, '0')}.json`;
}

function presetSpeedFile(mph, option) {
  // Mirrors webexport.preset_speed_filename(): (6.0, 'a') -> preset_s60a.json
  return `data/preset_s${String(Math.round(mph * 10)).padStart(2, '0')}${option}.json`;
}

function selectedSpeed() {
  const el = document.querySelector('input[name="speed"]:checked');
  return el ? +el.value : null;
}

function selectedOption() {
  const el = document.querySelector('input[name="option"]:checked');
  return el ? el.value : 'a';
}

// ── Controls, built from the manifest ──────────────────────────────────────
// The page used to carry a hardcoded list of radio buttons that nothing checked against
// docs/data/. A preset that failed to solve left a button that 404s; one that solved
// without a matching button was invisible. Building the controls from the manifest means a
// control exists exactly when the file behind it does.

let MANIFEST = null;
let ACTIVE_SPEED = null;

function tierFor(mph) {
  return MANIFEST?.speeds.find(t => Math.abs(t.mph - mph) < 1e-9) ?? null;
}

function buildSpeedControls() {
  const box = $('speedOptions');
  box.innerHTML = MANIFEST.speeds.map(tier => {
    const note = tier.mph === MANIFEST.speeds[0].mph ? ' <span class="param-note">steady</span>'
      : tier.mph === MANIFEST.speeds[MANIFEST.speeds.length - 1].mph
        ? ' <span class="param-note">elite</span>' : '';
    return `<label title="pace factor ${tier.pace_factor.toFixed(3)} — about ${(tier.mph * 0.84).toFixed(1)} mph on the flat">` +
      `<input type="radio" name="speed" value="${tier.mph}"> ${tier.mph.toFixed(1)} mph${note}</label>`;
  }).join('');
}

// Rebuilt whenever the speed changes: the labels carry each option's score at *this*
// speed, so the cost of a commitment is visible before you click it.
function buildOptionControls(mph) {
  const tier = tierFor(mph);
  const box = $('optionOptions');
  if (!tier) { box.innerHTML = ''; return; }

  const chosen = selectedOption();
  box.innerHTML = tier.options.map(opt => {
    const delta = opt.delta_vs_free == null || Math.abs(opt.delta_vs_free) < 5e-4
      ? '' : ` <span class="param-note">${opt.delta_vs_free.toFixed(1)} pts</span>`;
    return `<label title="${esc(opt.description || '')}">` +
      `<input type="radio" name="option" value="${opt.option}"` +
      `${opt.option === chosen ? ' checked' : ''}> ${esc(opt.label)}${delta}</label>`;
  }).join('');
  if (!document.querySelector('input[name="option"]:checked')) {
    document.querySelector('input[name="option"]').checked = true;
  }
  box.querySelectorAll('input[name="option"]').forEach(el =>
    el.addEventListener('change', () => { stopPlaying(); loadCurrent(); }));
}

function buildPaceControls() {
  const box = $('paceOptions');
  if (!MANIFEST.paces.length) { $('paceLegacy').style.display = 'none'; return; }
  // The mph equivalent rides alongside the multiplier rather than hiding in a tooltip:
  // anyone reading this panel is comparing it against the speed tiers above, and a bare
  // "1.60" cannot be lined up against "6.0 mph" without doing the conversion by hand.
  box.innerHTML = MANIFEST.paces.map(entry =>
    `<label title="pace ${entry.pace_factor.toFixed(2)} = ${entry.speed_mph.toFixed(2)} mph at Tobler's peak">` +
    `<input type="radio" name="pace" value="${entry.pace_factor}"> ` +
    `${entry.pace_factor.toFixed(2)} <span class="param-note">${entry.speed_mph.toFixed(1)} mph</span></label>`
  ).join('');
  // Picking a pace deselects the speed tiers, for the same reason the reverse holds: two
  // lit controls describing different itineraries would misreport which one is on screen.
  box.querySelectorAll('input[name="pace"]').forEach(el =>
    el.addEventListener('change', () => {
      stopPlaying();
      document.querySelectorAll('input[name="speed"], input[name="option"]')
        .forEach(other => { other.checked = false; });
      loadPreset(presetFile(+el.value));
    }));
}

// Load whatever the speed + option controls currently point at, and drop any legacy pace
// selection — the two families are alternative answers to different questions, so showing
// one selected while the other is on screen would misreport which is being displayed.
//
// The speed is remembered here rather than read back from the DOM every time, because
// choosing a legacy pace clears the speed radios; without it, the option buttons would go
// dead the moment someone looked at the pace sweep and came back.
function loadCurrent() {
  const mph = selectedSpeed() ?? ACTIVE_SPEED;
  if (mph == null) return Promise.resolve();
  ACTIVE_SPEED = mph;

  document.querySelectorAll('input[name="pace"]').forEach(el => { el.checked = false; });
  const speedEl = document.querySelector(`input[name="speed"][value="${mph}"]`);
  if (speedEl) speedEl.checked = true;

  return loadPreset(presetSpeedFile(mph, selectedOption()));
}

// A new speed means new option labels (the score deltas are per-speed), so the option
// controls are rebuilt here and only here — never from inside their own change handler.
function onSpeedChange() {
  stopPlaying();
  ACTIVE_SPEED = selectedSpeed();
  buildOptionControls(ACTIVE_SPEED);
  return loadCurrent();
}

const showLoading = on => $('loading').classList.toggle('visible', on);

function goHome() {
  if (homeBounds) map.fitBounds(homeBounds, { padding: [16, 16] });
}

let loadSeq = 0;   // last-click-wins: a stale fetch must not overwrite a newer one

async function loadPreset(file) {
  const seq = ++loadSeq;
  showLoading(true);
  const errEl = $('presetError');
  errEl.style.display = 'none';
  try {
    if (!NETWORK) {
      NETWORK = await fetch(NETWORK_URL, FETCH_OPTS).then(r => {
        if (!r.ok) throw new Error('trail network failed to load');
        return r.json();
      });
      homeBounds = NETWORK.bounds;
      drawNetwork();
      // Leaflet caches the container size when the map is created, which here is before
      // the flex/absolute layout has settled — without this the first fit lands a whole
      // zoom level short and the network sits in the middle of an empty county.
      map.invalidateSize();
      goHome();
    }
    const preset = await fetch(file, FETCH_OPTS).then(r => {
      if (!r.ok) throw new Error(`${file.split('/').pop()} not found — has it been solved yet?`);
      return r.json();
    });
    if (seq !== loadSeq) return;
    showPreset(preset);
  } catch (err) {
    if (seq !== loadSeq) return;
    errEl.textContent = err.message;
    errEl.style.display = 'block';
  } finally {
    if (seq === loadSeq) showLoading(false);
  }
}

// ── App init ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  map = L.map('map').setView([37.2625, -80.4962], 14);

  let activeBasemap = BASEMAPS[DEFAULT_BASEMAP].addTo(map);
  activeBasemap.setOpacity(+$('basemapOpacity').value / 100);

  let mapwarperLayer = null;
  if (MAPWARPER_ID) {
    mapwarperLayer = L.tileLayer(
      `https://mapwarper.net/maps/tile/${MAPWARPER_ID}/{z}/{x}/{y}.png`,
      { attribution: `Trail map georeferenced via <a href="https://mapwarper.net/maps/${MAPWARPER_ID}">MapWarper</a>`,
        maxZoom: 19, maxNativeZoom: MAPWARPER_NATIVE_ZOOM,
        opacity: 0.4, zIndex: 3, bounds: MAPWARPER_BOUNDS },
    ).addTo(map);
  } else {
    // No georeferenced sheet yet — hide its control rather than leave a dead slider.
    ['mapwarpLabel', 'mapwarpOpacity', 'opacityVal'].forEach(id => $(id).style.display = 'none');
  }

  let mapwarper2Layer = null;
  if (MAPWARPER_ID_2) {
    mapwarper2Layer = L.tileLayer(
      `https://mapwarper.net/maps/tile/${MAPWARPER_ID_2}/{z}/{x}/{y}.png`,
      { attribution: `Trail map p2 via <a href="https://mapwarper.net/maps/${MAPWARPER_ID_2}">MapWarper</a>`,
        maxZoom: 19, maxNativeZoom: MAPWARPER_2_NATIVE_ZOOM,
        opacity: +$('mapwarp2Opacity').value / 100, zIndex: 4, bounds: MAPWARPER_2_BOUNDS },
    ).addTo(map);
  } else {
    document.querySelectorAll('.mapwarp2-ctl').forEach(el => el.style.display = 'none');
  }

  const hillshadeLayer = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}',
    { attribution: 'Hillshade &copy; Esri', maxZoom: 16, opacity: 0.15, zIndex: 2 },
  ).addTo(map);

  // roadGroup first: roads are context for the trails, so the gray network draws over them
  // where the two run close together rather than the other way round.
  [roadGroup, netGroup, walkedGroup, currentGroup, arrowGroup].forEach(g => g.addTo(map));

  // leaflet-textpath appends its <text> to the renderer's <svg> rather than into the <g>
  // that holds the paths, so labels always paint above every line in the pane no matter how
  // often renderStep() tears the walked and current layers down and rebuilds them. That is
  // the stacking we want, and it comes free — a custom pane would not help, because the
  // plugin puts the text in the map's default renderer regardless of the layer's own.
  labelGroup.addTo(map);
  map.on('zoomend', refreshLabels);

  // ── Pane toggles ─────────────────────────────────────────────────────────
  $('left-toggle').addEventListener('click', () => {
    const hidden = document.body.classList.toggle('left-hidden');
    $('left-toggle').textContent = hidden ? '▶' : '◀';
    setTimeout(() => map.invalidateSize(), 260);
  });
  $('sidebar-toggle').addEventListener('click', () => {
    const hidden = document.body.classList.toggle('sidebar-hidden');
    $('sidebar-toggle').textContent = hidden ? '◀' : '▶';
    setTimeout(() => map.invalidateSize(), 260);
  });

  // ── Step controls ────────────────────────────────────────────────────────
  $('stepSlider').addEventListener('input', function () { setStep(+this.value); });
  $('btnStepPrev').addEventListener('click', () => { stopPlaying(); setStep(currentStep - 1); });
  $('btnStepNext').addEventListener('click', () => { stopPlaying(); setStep(currentStep + 1); });

  $('btnStepPlay').addEventListener('click', function () {
    if (stepPlayTimer) return stopPlaying();
    if (!PRESET) return;
    if (currentStep >= PRESET.totals.n_steps) setStep(1);
    this.textContent = '⏸ Pause';
    stepPlayTimer = setInterval(() => {
      if (currentStep >= PRESET.totals.n_steps) stopPlaying();
      else setStep(currentStep + 1);
    }, 700);
  });

  $('btnZoomStep').addEventListener('click', () => {
    const geom = PRESET?.steps[currentStep - 1]?.geometry;
    if (geom?.length) map.fitBounds(L.latLngBounds(geom).pad(0.25));
  });
  $('btnHome').addEventListener('click', goHome);
  $('btnCsv').addEventListener('click', downloadCsv);

  document.addEventListener('keydown', e => {
    if (e.target.matches('input, select, textarea')) return;
    if (e.key === 'ArrowLeft')  { stopPlaying(); setStep(currentStep - 1); }
    if (e.key === 'ArrowRight') { stopPlaying(); setStep(currentStep + 1); }
  });

  // ── Layer toggles ────────────────────────────────────────────────────────
  $('togRepeat').addEventListener('change', () => renderStep(currentStep));
  // One checkbox, both senses of "road": the available network underneath and the road
  // legs of the walk on top. Hiding one while leaving the other would be a puzzle.
  $('togRoads').addEventListener('change', function () {
    if (this.checked) roadGroup.addTo(map); else map.removeLayer(roadGroup);
    renderStep(currentStep);
  });
  $('togLabels').addEventListener('change', refreshLabels);

  // ── Opacity + basemap ────────────────────────────────────────────────────
  $('mapwarpOpacity').addEventListener('input', function () {
    if (mapwarperLayer) mapwarperLayer.setOpacity(+this.value / 100);
    $('opacityVal').textContent = this.value + '%';
  });
  $('mapwarp2Opacity').addEventListener('input', function () {
    if (mapwarper2Layer) mapwarper2Layer.setOpacity(+this.value / 100);
    $('opacity2Val').textContent = this.value + '%';
  });
  $('hillshadeOpacity').addEventListener('input', function () {
    hillshadeLayer.setOpacity(+this.value / 100);
    $('hillshadeOpacityVal').textContent = this.value + '%';
  });
  $('basemapOpacity').addEventListener('input', function () {
    activeBasemap.setOpacity(+this.value / 100);
    $('basemapOpacityVal').textContent = this.value + '%';
  });

  function setBasemap(name) {
    const opacity = +$('basemapOpacity').value / 100;
    map.removeLayer(activeBasemap);
    activeBasemap = BASEMAPS[name];
    activeBasemap.addTo(map);
    activeBasemap.setOpacity(opacity);
    activeBasemap.bringToBack();
    $('basemapSel').value = name;
  }
  $('basemapSel').addEventListener('change', function () { setBasemap(this.value); });

  // ── Dark mode ────────────────────────────────────────────────────────────
  function applyTheme(dark) {
    document.body.classList.toggle('dark', dark);
    netColor = dark ? NET_THEMES.dark : NET_THEMES.light;
    netGroup.eachLayer(l => l.setStyle && l.setStyle({ color: netColor }));
    $('legNet').style.background = netColor;
    labelTheme = dark ? LABEL_THEMES.dark : LABEL_THEMES.light;
    refreshLabels();
    $('btnDark').textContent = dark ? '☀️' : '🌙';
    // Swap the light defaults for the dark one and back, but leave a deliberate pick
    // like aerial or topo alone — someone who chose Bing Aerial meant it.
    const current = $('basemapSel').value;
    if (dark && (current === 'OSM Grayscale' || current === 'CartoDB Light')) setBasemap('CartoDB Dark');
    else if (!dark && current === 'CartoDB Dark') setBasemap(DEFAULT_BASEMAP);
    localStorage.setItem('hullabalooTheme', dark ? 'dark' : 'light');
  }
  $('btnDark').addEventListener('click', () => applyTheme(!document.body.classList.contains('dark')));
  if (localStorage.getItem('hullabalooTheme') === 'dark') applyTheme(true);

  // ── Preset selection ─────────────────────────────────────────────────────
  // Every control is built from the manifest, so nothing is wired up until it arrives.
  // If it cannot be fetched there is no itinerary to show and no set of buttons that
  // would honestly represent one, so the failure is reported rather than papered over.
  fetch(MANIFEST_URL, FETCH_OPTS)
    .then(r => {
      if (!r.ok) throw new Error('presets.json not found — has the build been run?');
      return r.json();
    })
    .then(manifest => {
      MANIFEST = manifest;
      buildSpeedControls();
      buildPaceControls();

      // Default to the middle of the published range rather than an end of it: the
      // extremes are the least likely to describe a given racer.
      const tiers = MANIFEST.speeds;
      const initial = tiers[Math.floor((tiers.length - 1) / 2)];
      const initialEl = document.querySelector(`input[name="speed"][value="${initial.mph}"]`);
      if (initialEl) initialEl.checked = true;

      document.querySelectorAll('input[name="speed"]').forEach(el =>
        el.addEventListener('change', onSpeedChange));

      buildOptionControls(initial.mph);
      return loadCurrent();
    })
    .catch(err => {
      const errEl = $('presetError');
      errEl.textContent = err.message;
      errEl.style.display = 'block';
      showLoading(false);
    });
});
