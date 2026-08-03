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

// Page 2 of the printed sheet, drawn at roughly 1:10000 against Page 1's 1:20000 — twice
// the detail over a smaller area. Set the id once it is georeferenced, and fill in the
// bounds from the MapWarper API (the `bbox` field of /api/v1/maps/<id>).
//
// The two sheets need no cross-fading or zoom switching to coexist: this one is bounded
// to its own extent and stacks above Page 1, so it paints only where it has coverage and
// simply reveals Page 1 everywhere else. Both sliders stay independent.
const MAPWARPER_ID_2 = null;
const MAPWARPER_2_BOUNDS = MAPWARPER_BOUNDS;   // replace with Page 2's own extent
const MAPWARPER_2_NATIVE_ZOOM = 17;            // one more zoom of real detail than Page 1

const CAT_COLOR = { unique: '#f7882f', offtrail: '#c0392b', repeat: '#c0392b' };
const CAT_LABEL = { unique: '', offtrail: 'off-trail', repeat: 'repeat' };
const GOLD = '#FFD700';

// Network grays flip with the UI theme: a mid gray that reads as "faint" on a light
// basemap disappears entirely on a dark one.
const NET_THEMES = { light: '#999999', dark: '#6b6b78' };
let netColor = NET_THEMES.light;

const CAND_STYLE = { color: '#f08080', weight: 3, opacity: 0.55, dashArray: '4,6' };

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

// ── Module-level state ─────────────────────────────────────────────────────
let map;
let NETWORK = null;     // network.json, loaded once
let PRESET = null;      // the itinerary currently displayed
let currentStep = 1;
let stepPlayTimer = null;
let startMarker = null;
let homeBounds = null;

const netGroup    = L.layerGroup();   // full trail network, gray backdrop
const candGroup   = L.layerGroup();   // bushwhack connectors the route did not use
const walkedGroup = L.layerGroup();   // steps already taken
const arrowGroup  = L.layerGroup();   // direction-of-travel arrowheads
const currentGroup = L.layerGroup();  // the step under the slider

// ── Drawing helpers ────────────────────────────────────────────────────────
function stepStyle(step) {
  if (step.cat === 'repeat') return { color: CAT_COLOR.repeat, weight: 5, opacity: 0.9, dashArray: '6,5' };
  if (step.cat === 'offtrail')
    // Bushwhacks are dashed because they are not a path on the ground; roads are solid
    // because they are, even though neither earns a point.
    return { color: CAT_COLOR.offtrail, weight: 5, opacity: 0.9, dashArray: step.bushwhack ? '2,6' : null };
  return { color: CAT_COLOR.unique, weight: 5, opacity: 1 };
}

function stepVisible(step) {
  if (step.cat === 'repeat')   return $('togRepeat').checked;
  if (step.cat === 'offtrail') return $('togOff').checked;
  return true;
}

function stepPopup(step) {
  const tag = CAT_LABEL[step.cat];
  const kind = step.cat === 'offtrail' ? (step.bushwhack ? 'bushwhack' : 'road') : tag;
  return `<b>${esc(step.name)}</b>${kind ? ` <i>(${kind})</i>` : ''}<br>` +
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
  candGroup.clearLayers();
  if (!NETWORK) return;

  // Edges the current route walks are drawn by the step layers on top; the backdrop is
  // deliberately the *whole* network, so the unwalked remainder stays visible as the
  // thing the seven-hour budget could not reach.
  for (const edge of NETWORK.edges) {
    if (edge.kind === 'bushwhack') {
      L.polyline(edge.geometry, { ...CAND_STYLE })
        .bindTooltip(`${esc(edge.name)} — ${edge.miles.toFixed(2)} mi (not used)`,
          { sticky: true, opacity: 0.85 })
        .addTo(candGroup);
    } else {
      L.polyline(edge.geometry, { color: netColor, weight: 3, opacity: 0.75 })
        .bindTooltip(`${esc(edge.name)} — ${edge.miles.toFixed(2)} mi`,
          { sticky: true, opacity: 0.85 })
        .addTo(netGroup);
    }
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
  $('sbOff').textContent    = `${cum.offtrail_miles.toFixed(1)} mi / ${fmtHM(run.offtrail)}`;
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

function buildItinerary() {
  const completedAt = new Map();
  for (const trail of PRESET.trails)
    if (trail.completed_at_step) {
      if (!completedAt.has(trail.completed_at_step)) completedAt.set(trail.completed_at_step, []);
      completedAt.get(trail.completed_at_step).push(trail.name);
    }

  $('itinerary').innerHTML = PRESET.steps.map(s => {
    const tag = s.cat === 'repeat' ? 'repeat'
              : s.cat === 'offtrail' ? (s.bushwhack ? 'bushwhack' : 'road') : '';
    const done = completedAt.get(s.i);
    return `<div class="itin-step" data-step="${s.i}">
      <span class="itin-n" style="color:${CAT_COLOR[s.cat]}">${s.i}.</span>
      <b>${esc(s.name)}</b>${tag ? ` <span class="itin-tag">(${tag})</span>` : ''}
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

  let claim;
  if (opt.caps_binding.length) claim = '';   // the caveat below says it instead
  else if (opt.proven) claim = ' Proven optimal by the MILP — no better route exists at this pace.';
  else if (solver.gap_pct != null) claim = ` Within ${solver.gap_pct.toFixed(1)}% of a proven upper bound.`;
  else claim = '';

  $('presetInfo').innerHTML =
    `<div class="info-row"><span>Score</span><span><b>${fmtScore(t.score)}</b></span></div>` +
    `<div class="info-row"><span>Trails completed</span><span><b>${t.trails_completed}</b></span></div>` +
    `<div class="info-row"><span>Unique miles</span><span><b>${t.unique_miles.toFixed(2)}</b></span></div>` +
    `<div class="info-row"><span>Distance walked</span><span><b>${t.walked_miles.toFixed(2)} mi</b></span></div>` +
    `<div class="info-row"><span>Finish time</span><span><b>${fmtClock(t.time_s)}</b></span></div>` +
    `<span class="info-note">Score = trails completed + unique miles.${claim}</span>` +
    (opt.caps_binding.length
      ? `<div class="caveat"><b>⚠ Not proven optimal</b><br>` +
        `${opt.caps_binding.map(esc).join('; ')}. ${esc(opt.note || '')}</div>`
      : '');
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

function selectedPace() {
  return +(document.querySelector('input[name="pace"]:checked')?.value ?? 1.3);
}

const showLoading = on => $('loading').classList.toggle('visible', on);

function goHome() {
  if (homeBounds) map.fitBounds(homeBounds, { padding: [16, 16] });
}

let loadSeq = 0;   // last-click-wins: a stale fetch must not overwrite a newer one

async function loadPreset(pace) {
  const seq = ++loadSeq;
  showLoading(true);
  const errEl = $('presetError');
  errEl.style.display = 'none';
  try {
    if (!NETWORK) {
      NETWORK = await fetch(NETWORK_URL).then(r => {
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
    const file = presetFile(pace);
    const preset = await fetch(file).then(r => {
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

  [netGroup, candGroup, walkedGroup, currentGroup, arrowGroup].forEach(g => g.addTo(map));

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
  const stopPlaying = () => {
    if (stepPlayTimer) { clearInterval(stepPlayTimer); stepPlayTimer = null; }
    $('btnStepPlay').textContent = '▶ Play';
  };

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

  document.addEventListener('keydown', e => {
    if (e.target.matches('input, select, textarea')) return;
    if (e.key === 'ArrowLeft')  { stopPlaying(); setStep(currentStep - 1); }
    if (e.key === 'ArrowRight') { stopPlaying(); setStep(currentStep + 1); }
  });

  // ── Layer toggles ────────────────────────────────────────────────────────
  $('togRepeat').addEventListener('change', () => renderStep(currentStep));
  $('togOff').addEventListener('change', function () {
    if (this.checked) candGroup.addTo(map); else map.removeLayer(candGroup);
    renderStep(currentStep);
  });

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
  document.querySelectorAll('input[name="pace"]').forEach(el =>
    el.addEventListener('change', () => { stopPlaying(); loadPreset(selectedPace()); }));

  loadPreset(selectedPace());
});
