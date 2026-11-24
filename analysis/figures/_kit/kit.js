// Raising the Roof chart kit: shared tokens, helpers, marks and boot for every animated chart in the post.
// Rules: ../STYLE.md.
//
// Page layout (paths relative to the chart):
//   <head>  <link rel="stylesheet" href="../_kit/kit.css">
//   <body>  the #frame markup (header, #chart > svg#svg, footer#foot), then
//           <script src="../_kit/kit.js"></script>, then the chart's own <script>, which defines
//           CHART, TEXT, BEATS (and DATA), draw(t), and ends with boot().
//
// The kit owns these globals, so a chart must not declare them again:
//   svg, W, H, C, SETUP, MODEL, TASK, TYPE, MARK, MOTION, and every helper below.

// ============================================================================
// 1. Tokens
// ============================================================================

// every --name in kit.css :root (and any a chart adds in its own <style>), as C.name
const C = (() => {
  const cs = getComputedStyle(document.documentElement), out = {};
  for (const p of cs) if (p.startsWith('--')) out[p.slice(2)] = cs.getPropertyValue(p).trim();
  return out;
})();

// One color per setup, in every chart. Names are the exact names the post uses.
const SETUP = {
  single:       { name: 'Single agent' },             // one agent, one session
  restarted:    { name: 'Restarted single agent' },   // one agent, budget split into sessions run one after another
  structured:   { name: 'Protocol teams' },
  selforg:      { name: 'Self-organized teams' },
  orchestrated: { name: 'Orchestrated teams' },
  notalk:       { name: 'No-talk teams' },            // agents working independently, scored on the best one
};
for (const [key, s] of Object.entries(SETUP)) {
  Object.assign(s, { key, fill: C[key], line: C[`${key}-line`], text: C[`${key}-text`], tint: C[`${key}-tint`] });
  console.assert(/^#[0-9a-f]{6}$/i.test(s.fill || ''), `kit: kit.css has no --${key}`);
}

// Models are never told apart by color. Position or labels first; shape only when two models share a row or plot.
const MODEL = {
  sol:    { name: 'GPT-6.1 Sol',       shape: 'circle' },
  terra:  { name: 'GPT-6.1 Terra',     shape: 'circle' },
  opus:   { name: 'Claude Opus 5.5',   shape: 'square' },
  sonnet: { name: 'Claude Sonnet 5.5', shape: 'square' },
  mixed:  { name: 'Mixed Claude and GPT team', shape: 'diamond' },
};
const TASK = { gitzig: 'git-to-zig', ahc014: 'AHC014', ahc038: 'AHC038', takehome: 'Performance take-home' };

// Type scale in px. Title, subtitle and footnote are set in kit.css (31/600, 22.5, 18.5).
const TYPE = {
  title: 31,   // h1, 600
  name: 27,    // row or panel name, 700: ink until its beat, then the setup's text shade
  label: 23,   // row note (400, muted), label inside a bar (600, ink), row label in a dense chart
  annot: 22,   // label beside a mark, callout words, legend key
  small: 21,   // small label on a mark (600)
  tick: 20,    // tick labels and axis titles, muted
  value: 24,   // data value, 700, in the setup's text shade
  hero: 31,    // the one number that carries the claim, 700, ink
  foot: 18.5,  // footnote, muted, line height 1.47
};

// Mark specs.
const MARK = {
  run:      { r: 7, ring: 1.5 },          // one run: setup fill, bg ring
  mean:     { r: 11, rDense: 8, ring: 3 }, // mean of runs: setup line shade, bg ring; rDense when rows are under 32 px apart
  hollow:   { stroke: 2.5 },              // a run that did not spend its budget: bg fill, setup line ring
  interval: { width: 4 },                 // 95% interval: setup line shade, round caps
  series:   { mean: 3.5, run: 1.5, runOpacity: 0.35 },   // line charts
  bar:      { radius: 3, gap: 4 },
  grid: 1.5, baseline: 2,
  ref:      { width: 2, dash: '6 5' },    // reference line: dashed, in the setup's line shade
  callout:  { width: 2.5, dash: [1, 6] }, // callout connector: dotted, ink, round caps
};

// Motion defaults (seconds), taken from the method chart.
const MOTION = {
  duration: 16,       // total runtime
  firstBeat: 0.4,     // the first beat starts here; title, footnote, names and scaffold are on at frame 0
  lastChange: 8.5,    // the build ends by here, so the final hold is at least 45%
  minHold: 0.45,
  grow: 0.9,          // a bar or line sweeping in (ease.inOut)
  pop: 0.25,          // a dot or node appearing (ease.out)
  label: 0.25,        // a label fading in once its mark is about 70% grown (ease.out)
  recolor: 0.5,       // a row name turning from ink to its setup's text shade as its beat starts
  pause: 0.4,         // between beats
};

// ============================================================================
// 2. Helpers (the animated-charts template's section 2)
// ============================================================================
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const lerp = (a, b, u) => a + (b - a) * u;
const ease = {
  linear: (u) => u,
  out: (u) => 1 - Math.pow(1 - u, 3),          // default: fast start, soft landing
  outQuint: (u) => 1 - Math.pow(1 - u, 5),     // snappier landing, for many small marks
  inOut: (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2), // bars sweeping in, marks moving
};
// progress 0..1 of a span that starts at `s` and lasts `d` seconds (d = 0: on at s, no fade)
const prog = (t, s, d, e = ease.out) => e(d > 0 ? clamp01((t - s) / d) : (t >= s ? 1 : 0));
const beat = (t, name, e) => prog(t, BEATS[name][0], BEATS[name][1], e);
// item i of n inside a beat: start times spread across the beat, each item lasts `each` seconds
const stagger = (t, name, i, n, each = 0.7, e = ease.out) => {
  const [s, d] = BEATS[name];
  const gap = n > 1 ? Math.max(0, d - each) / (n - 1) : 0;
  return prog(t, s + i * gap, Math.min(each, d), e);
};
const fade = (t, s, d = 0.3) => prog(t, s, d, ease.out);

function linear(d0, d1, r0, r1) {
  const f = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
  f.domain = [d0, d1]; f.range = [r0, r1];
  return f;
}
function logScale(d0, d1, r0, r1) {
  const a = Math.log(d0), b = Math.log(d1);
  const f = (v) => r0 + ((Math.log(v) - a) / (b - a)) * (r1 - r0);
  f.domain = [d0, d1]; f.range = [r0, r1];
  return f;
}
function band(n, r0, r1, padding = 0.3) {
  const step = (r1 - r0) / n, size = step * (1 - padding);
  return { step, size, at: (i) => r0 + i * step + (step - size) / 2 };
}
function ticks(min, max, count = 5) {
  const span = max - min, raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function logTicks(min, max) {
  const out = [];
  for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++)
    for (const m of [1, 2, 5]) { const v = m * Math.pow(10, e); if (v >= min * 0.999 && v <= max * 1.001) out.push(v); }
  return out;
}
function niceDomain(min, max, count = 5) {
  const raw = (max - min) / count || 1, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag)
    .find((s) => Math.ceil(max / s - 1e-9) - Math.floor(min / s + 1e-9) <= count) || 10 * mag;
  const lo = Math.floor(min / step + 1e-9) * step, hi = Math.ceil(max / step - 1e-9) * step;
  return { min: lo, max: hi, step, ticks: ticks(lo, hi, Math.round((hi - lo) / step)) };
}
const fmt = (v, { dp = 0, pre = '', suf = '' } = {}) =>
  pre + Number(v).toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp }) + suf;
const countUp = (value, u, opts) => fmt(u >= 1 ? value : value * u, opts);
const ordinal = (n) => { const s = ['th', 'st', 'nd', 'rd'], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); };
// print rows as an aligned table to the page log (`chartvid check` shows it)
function logTable(rows, title = '') {
  const keys = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  const cell = (v) => (typeof v === 'number' ? String(+v.toFixed(6)) : String(v ?? ''));
  const w = keys.map((k) => Math.max(k.length, ...rows.map((r) => cell(r[k]).length)));
  const line = (vals) => vals.map((v, i) => v.padEnd(w[i])).join('  ');
  console.log([title, line(keys), ...rows.map((r) => line(keys.map((k) => cell(r[k]))))].filter(Boolean).join('\n'));
}
// blend two hex colors
function mix(c1, c2, u) {
  const p = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
  const a = p(c1), b = p(c2);
  return '#' + a.map((v, i) => Math.round(lerp(v, b[i], clamp01(u))).toString(16).padStart(2, '0')).join('');
}
const NS = 'http://www.w3.org/2000/svg';
// create an SVG element: h(parent, 'rect', { x, y, width, height, fill }, optionalText)
function h(parent, tag, attrs = {}, text) {
  const el = document.createElementNS(NS, tag);
  for (const k in attrs) if (attrs[k] != null) el.setAttribute(k, attrs[k]);
  if (text != null) el.textContent = text;
  parent.appendChild(el);
  return el;
}
const _ctx = document.createElement('canvas').getContext('2d');
function measure(text, size = 22, weight = 400) {
  _ctx.font = `${weight} ${size}px 'Instrument Sans'`;
  return _ctx.measureText(text).width;
}
function drawOn(el, u) {
  const L = el.getTotalLength();
  el.setAttribute('stroke-dasharray', `${L} ${L}`);
  el.setAttribute('stroke-dashoffset', `${L * (1 - clamp01(u))}`);
}
function drawDashed(el, u, dash = [3, 4]) {
  const L = el.getTotalLength(), vis = L * clamp01(u), period = dash[0] + dash[1], parts = [];
  let acc = 0;
  while (acc + period <= vis) { parts.push(dash[0], dash[1]); acc += period; }
  parts.push(Math.min(vis - acc, dash[0]), L);
  el.setAttribute('stroke-dasharray', parts.join(' '));
  el.removeAttribute('stroke-dashoffset');
}
const linePath = (pts) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join('');
function clipToX(pts, xMax) {
  const out = [];
  for (let i = 0; i < pts.length; i++) {
    if (pts[i][0] <= xMax) { out.push(pts[i]); continue; }
    if (i > 0) { const [x0, y0] = pts[i - 1], [x1, y1] = pts[i]; out.push([xMax, y0 + ((xMax - x0) / (x1 - x0)) * (y1 - y0)]); }
    break;
  }
  return out;
}
// a background-colored outline behind a <text>, so lines pass behind it (`check` treats it as protected)
function halo(el, w = 7) {
  el.setAttribute('stroke', C.bg); el.setAttribute('stroke-width', w);
  el.setAttribute('stroke-linejoin', 'round'); el.setAttribute('paint-order', 'stroke');
  return el;
}

// ============================================================================
// 3. Kit marks. Each takes a setup key ('selforg') or a SETUP entry, and u = 0..1 for its entrance.
// ============================================================================
const setupOf = (s) => (typeof s === 'string' ? SETUP[s] : s);

// a model shape centered on (cx, cy), with the area of a circle of radius r
function shape(parent, kind, cx, cy, r, attrs = {}) {
  if (kind === 'square') { const s = r * Math.sqrt(Math.PI); return h(parent, 'rect', { x: cx - s / 2, y: cy - s / 2, width: s, height: s, rx: 1.5, ...attrs }); }
  if (kind === 'diamond') { const d = r * Math.sqrt(Math.PI / 2); return h(parent, 'path', { d: `M${cx},${cy - d}L${cx + d},${cy}L${cx},${cy + d}L${cx - d},${cy}Z`, ...attrs }); }
  return h(parent, 'circle', { cx, cy, r, ...attrs });
}
// one run. hollow: the run did not spend its budget (stopped early)
function run(parent, s, cx, cy, u = 1, { shape: kind = 'circle', hollow = false } = {}) {
  if (u <= 0) return null;
  const S = setupOf(s), r = MARK.run.r * u;
  if (hollow) return shape(parent, kind, cx, cy, Math.max(0.1, r - MARK.hollow.stroke / 2), { fill: C.bg, stroke: S.line, 'stroke-width': MARK.hollow.stroke });
  return shape(parent, kind, cx, cy, r, { fill: S.fill, stroke: C.bg, 'stroke-width': MARK.run.ring });
}
// the mean of a setup's runs (dense: rows under 32 px apart)
function mean(parent, s, cx, cy, u = 1, { shape: kind = 'circle', hollow = false, dense = false } = {}) {
  if (u <= 0) return null;
  const S = setupOf(s), r = (dense ? MARK.mean.rDense : MARK.mean.r) * u;
  if (hollow) {
    shape(parent, kind, cx, cy, r + 1, { fill: C.bg });
    return shape(parent, kind, cx, cy, Math.max(0.1, r - 1.5), { fill: C.bg, stroke: S.line, 'stroke-width': 3 });
  }
  return shape(parent, kind, cx, cy, r, { fill: S.line, stroke: C.bg, 'stroke-width': MARK.mean.ring });
}
// a 95% interval from lo to hi at `at` (y for a horizontal one), growing out from `mid`
function interval(parent, s, lo, hi, at, u = 1, { mid = (lo + hi) / 2, vertical = false } = {}) {
  if (u <= 0) return null;
  const S = setupOf(s), a = lerp(mid, lo, u), b = lerp(mid, hi, u);
  const pts = vertical ? { x1: at, x2: at, y1: a, y2: b } : { x1: a, x2: b, y1: at, y2: at };
  return h(parent, 'line', { ...pts, stroke: S.line, 'stroke-width': MARK.interval.width, 'stroke-linecap': 'round' });
}
// a reference line from (x1, y1) to (x2, y2): dashed in the setup's line shade when it stands for a setup
// (s = 'single' for "single agent"), solid `axis` gray when it does not (s = null: 0, 100%, lambda = 1)
function refLine(parent, s, x1, y1, x2, y2, u = 1) {
  if (u <= 0) return null;
  const el = h(parent, 'path', { d: `M${x1},${y1}L${x2},${y2}`, fill: 'none',
    stroke: s ? setupOf(s).line : C.axis, 'stroke-width': MARK.ref.width });
  if (s) drawDashed(el, u, MARK.ref.dash.split(' ').map(Number)); else drawOn(el, u);
  return el;
}
// text in a setup's text shade (or ink), haloed so lines pass behind it
function tag(parent, x, y, text, { setup, size = TYPE.annot, weight = 600, anchor = 'start', opacity = 1, fill } = {}) {
  if (opacity <= 0) return null;
  return halo(h(parent, 'text', { x, y, 'text-anchor': anchor, 'font-size': size, 'font-weight': weight,
    fill: fill || (setup ? setupOf(setup).text : C.ink), opacity }, text));
}
// The one callout style: a dotted ink connector through `pts` (a bracket, elbow or leader; may be empty),
// then a hero number in ink with up to six muted words after it. u = 0..1: the line draws in the first
// 60%, the label fades in over the rest.
function callout(parent, pts, u, { hero, words, x, y, anchor = 'middle' } = {}) {
  if (u <= 0) return;
  if (pts && pts.length > 1) {
    const el = h(parent, 'path', { d: linePath(pts), fill: 'none', stroke: C.ink, 'stroke-width': MARK.callout.width,
      'stroke-linecap': 'round', 'stroke-linejoin': 'round' });
    drawDashed(el, clamp01(u / 0.6), MARK.callout.dash);
  }
  const o = clamp01((u - 0.55) / 0.45);
  if (o > 0 && hero != null) {
    const el = halo(h(parent, 'text', { x, y, 'text-anchor': anchor, 'font-size': TYPE.hero, 'font-weight': 700, fill: C.ink, opacity: o }), 9);
    h(el, 'tspan', {}, hero);
    if (words) h(el, 'tspan', { 'font-size': TYPE.annot, 'font-weight': 400, fill: C.muted }, '\u00a0' + words);
  }
}
// value gridlines and their tick labels. axis 'x': vertical lines at scale(v) from y `from` to `to`,
// labels under them; axis 'y': horizontal lines from x `from` to `to`, labels left of `from`.
// base: a value that gets the 2 px baseline instead of a gridline (0 on a bar or share axis).
function grid(parent, { axis = 'x', scale, values, from, to, fmt: f = String, base, label = true, opacity = 1 }) {
  for (const v of values) {
    const p = scale(v), isBase = base != null && Math.abs(v - base) < 1e-9;
    const line = axis === 'x' ? { x1: p, x2: p, y1: from, y2: to } : { x1: from, x2: to, y1: p, y2: p };
    h(parent, 'line', { ...line, stroke: isBase ? C.axis : C.grid, 'stroke-width': isBase ? MARK.baseline : MARK.grid, opacity });
    if (!label) continue;
    if (axis === 'x') h(parent, 'text', { class: 'tick', x: p, y: to + 28, 'text-anchor': 'middle', opacity }, f(v));
    else h(parent, 'text', { class: 'tick', x: from - 14, y: p + 7, 'text-anchor': 'end', opacity }, f(v));
  }
}
// a row or panel name: ink at frame 0, turning to the setup's text shade as its beat starts; an optional
// muted note after it. (The method chart's row heading.)
function setupName(parent, { x = 0, y, setup, name, note, t, start, size = TYPE.name, noteSize = TYPE.label, gap = 16 }) {
  const S = setupOf(setup), label = name ?? S.name;
  const el = h(parent, 'text', { x, y, 'font-size': size, 'font-weight': 700, fill: mix(C.ink, S.text, prog(t, start, MOTION.recolor)) }, label);
  if (note) h(parent, 'text', { x: x + measure(label, size, 700) + gap, y, 'font-size': noteSize, fill: C.muted, opacity: fade(t, start + 0.1, 0.4) }, note);
  return el;
}
// legend key HTML: [{ setup, mark: 'dot' | 'line' | 'hollow' | 'mean', shape, label }]. Shapes without a
// setup are drawn in muted gray (a model key).
function keyHTML(items) {
  const sw = ({ setup, mark = 'dot', shape: kind = 'circle' }) => {
    const S = setup ? setupOf(setup) : { fill: C.muted, line: C.muted };
    if (mark === 'line') return `<svg width="26" height="4"><rect width="26" height="4" rx="2" fill="${S.line}"/></svg>`;
    const r = 7.5, size = 20, c = size / 2, holder = document.createElementNS(NS, 'svg');
    if (mark === 'hollow') shape(holder, kind, c, c, r - 1.25, { fill: C.bg, stroke: S.line, 'stroke-width': 2.5 });
    else shape(holder, kind, c, c, r, { fill: mark === 'mean' ? S.line : S.fill });
    return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">${holder.innerHTML}</svg>`;
  };
  return items.map((it) => `<span>${sw(it)}${it.label ?? setupOf(it.setup).name}</span>`).join('');
}

// ============================================================================
// 4. Chart area and boot
// ============================================================================
const svg = document.getElementById('svg');
let W = 0, H = 0;

// Text and timing rules from STYLE.md that code can check. Each failure is a console.warn, which
// `chartvid check` counts as a problem.
function lint() {
  const out = [];
  if (CHART.width !== 1080 || CHART.height !== 1350) out.push(`canvas is ${CHART.width}x${CHART.height}; the kit's canvas is 1080x1350`);
  if (CHART.duration < 15 || CHART.duration > 16.5) out.push(`duration is ${CHART.duration} s; the kit's runtime is 15 to 16 s`);
  const lastEnd = Math.max(0, ...(CHART.beats || []).map((b) => (typeof b === 'number' ? b : b.end ?? 0)));
  if (lastEnd > 0 && (CHART.duration - lastEnd) / CHART.duration < MOTION.minHold)
    out.push(`the last beat ends at ${lastEnd} s, so the final hold is under ${100 * MOTION.minHold}% of ${CHART.duration} s`);
  draw(CHART.duration);
  const text = [...document.querySelectorAll('#title, #sub, #key, #lead, #foot, #svg text')].map((e) => e.textContent).join('\n');
  if (text.includes('\u2014')) out.push('em dash in the text; use a comma, a colon or a new sentence');
  const OLD = [
    [/\bTTC\b/, 'TTC: say Protocol teams'],
    [/\bstructured\b/i, 'Structured teams: say Protocol teams'],
    [/\bUltra\b/, 'Ultra: say Orchestrated teams'],
    [/\bmessage[- ]board teams?\b|^\s*board\s*$/im, 'message-board team: say Self-organized teams (the message board is a channel)'],
    [/\bpeers?\b/i, 'peers: say Self-organized teams'],
    [/\brelay(s|ed)?\b/i, 'relay: say Restarted single agent'],
    [/\d+\s*[×x]\s*\d+M,? one after another/i, '"8 × 4M, one after another": say Restarted single agent'],
  ];
  for (const [re, msg] of OLD) if (re.test(text)) out.push(`old name ${msg}`);
  const bare = text.replace(/GPT-6\.1 (Sol|Terra)|Claude (Opus|Sonnet) 5\.5/g, '').match(/\b(Sol|Terra|Opus|Sonnet)\b/);
  if (bare) out.push(`model name "${bare[0]}" without its full name (GPT-6.1 Sol, GPT-6.1 Terra, Claude Opus 5.5, Claude Sonnet 5.5)`);
  const lines = (id, size, lh) => { const el = document.getElementById(id); return el && el.textContent.trim() ? Math.round(el.getBoundingClientRect().height / (size * lh)) : 0; };
  if (lines('title', TYPE.title, 1.18) > 2) out.push('title runs over two lines');
  if (lines('foot', TYPE.foot, 1.47) > 2) out.push(`footnote runs to ${lines('foot', TYPE.foot, 1.47)} lines; the kit allows 2`);
  for (const m of out) console.warn('kit: ' + m);
}

// The template's boot, plus TEXT.key and the lint. Call boot() as the last line of the chart's script.
function boot() {
  if (!CHART.beats && typeof BEATS !== 'undefined')
    CHART.beats = Object.entries(BEATS).map(([name, [s, d]]) => ({ name, end: +(s + d).toFixed(2) }));
  const root = document.documentElement.style;
  root.setProperty('--w', CHART.width + 'px');
  root.setProperty('--h', CHART.height + 'px');
  const put = (id, text, html) => { const el = document.getElementById(id); if (!el) return; if (html) el.innerHTML = html; else el.textContent = text || ''; };
  put('title', TEXT.title, TEXT.titleHTML);
  put('sub', TEXT.sub, TEXT.subHTML);
  put('lead', '', TEXT.leadHTML);
  put('foot', TEXT.foot, TEXT.footHTML);
  if (TEXT.key && TEXT.key.length) {
    const host = document.getElementById('title').parentElement;
    const k = document.getElementById('key') || host.appendChild(Object.assign(document.createElement('div'), { id: 'key', className: 'key' }));
    k.innerHTML = keyHTML(TEXT.key);
  }
  const q = new URLSearchParams(location.search);
  const capture = q.has('capture');
  window.CHART = CHART;
  window.__seek = (t) => draw(Math.max(0, Math.min(CHART.duration, t)));
  document.fonts.load("600 31px 'Instrument Sans'").then(() => document.fonts.ready).then(() => {
    const box = document.getElementById('chart').getBoundingClientRect();
    W = box.width; H = box.height;
    lint();
    window.__seek(q.has('t') ? Number(q.get('t')) : 0);
    window.__chartReady = true;
    if (capture || q.has('t')) return;
    // Preview: loop in real time. Space pauses, arrows step 0.1s (shift: 1s), Home/End jump.
    const tc = document.createElement('div'); tc.id = 'tc'; document.body.appendChild(tc);
    let t = 0, last = performance.now(), paused = false;
    addEventListener('keydown', (e) => {
      if (e.key === ' ') { paused = !paused; e.preventDefault(); }
      if (e.key === 'ArrowRight') { t += e.shiftKey ? 1 : 0.1; paused = true; }
      if (e.key === 'ArrowLeft') { t -= e.shiftKey ? 1 : 0.1; paused = true; }
      if (e.key === 'Home') t = 0;
      if (e.key === 'End') { t = CHART.duration; paused = true; }
    });
    const loop = (now) => {
      if (!paused) t += (now - last) / 1000;
      last = now;
      if (t > CHART.duration + 0.8) t = 0;
      t = Math.max(0, t);
      window.__seek(t);
      tc.textContent = t.toFixed(2) + 's' + (paused ? ' (paused)' : '');
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  });
}
