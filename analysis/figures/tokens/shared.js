// The shared layout, marks and motion of the matched pair: tokens/chart.html and time/chart.html.
// Each chart.html defines MODE ('tokens' or 'time') and DATA (written by its tally.py), loads the kit, then this file,
// and ends with boot(). Everything a viewer reads is computed here from DATA.
// The kit (../_kit/kit.js) owns svg, W, H, C and every helper.

// ============================================================================
// 1. Config, text, beats
// ============================================================================
const CHART = { width: 1080, height: 1350, fps: 30, duration: 16.5 };
const T_MODE = MODE === 'tokens';

const TASK_ORDER = ['gitzig', 'ahc014', 'ahc038', 'perf'];
const TEAM_SETUPS = ['structured', 'selforg', 'orchestrated'];   // kit order
const SHAPE_OF = (m) => MODEL[m].shape;
const PANEL_NAME = { gitzig: TASK.gitzig, ahc014: TASK.ahc014, ahc038: TASK.ahc038, perf: TASK.takehome };

const BEATS = {
  scaffold: [0.0, 0],    // title, key, footnote, panel names, axes, gridlines, the empty summary
  single:   [0.4, 1.4],  // the single agent's curves draw across all four panels
  teams:    [2.1, 1.7],  // the teams of 8 draw across the same axis
  collapse: [4.1, 0.8],  // each curve shrinks into its end point, leaving a faint trace
  morph:    [5.15, 1.15],// the end points move into the summary; the single agent's four become the reference
  others:   [6.55, 1.45],// every other comparison pops in, teams of 2, then 4, then 8
  callout:  [8.2, 0.85], // time chart: teams of 2 and 4 recede; both: the bracket that names the teams of 8
};

// ============================================================================
// 2. Derived numbers (all from DATA)
// ============================================================================
const med = (a) => { const s = [...a].sort((p, q) => p - q), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
const DOTS = DATA.dots.map((d, i) => ({ ...d, i }));
const COUNTED = DOTS.filter((d) => d.counted);
const BELOW = COUNTED.filter((d) => d.gap < 0);
const K8 = COUNTED.filter((d) => d.k === 8);
const K8_GAP = med(K8.map((d) => d.gap));
const K8_SOONER = T_MODE ? null : med(K8.map((d) => d.sooner));
const SOONER = T_MODE ? null : COUNTED.filter((d) => d.sooner > 1);

console.assert(DOTS.length === DATA.summary.plotted, 'plotted count');
console.assert(COUNTED.length === DATA.summary.counted, 'counted count');
console.assert(BELOW.length === DATA.summary.below, 'below count');
console.assert(Math.abs(K8_GAP - DATA.summary.by_k[8].median_gap) < 1e-9, 'teams of 8 median gap');
if (!T_MODE) {
  console.assert(SOONER.length === DATA.summary.sooner, 'sooner count');
  console.assert(Math.abs(K8_SOONER - DATA.summary.by_k[8].median_sooner) < 1e-9, 'teams of 8 median sooner');
}
logTable(DOTS.map((d) => ({ task: d.task, model: d.model, setup: d.setup, k: d.k, budget: d.budget, gap: d.gap,
  counted: d.counted, ...(T_MODE ? {} : { sooner: d.sooner }), curve: d.curve })), 'dots');
logTable(TASK_ORDER.flatMap((t) => DATA.curves[t].map((c) => ({ task: t, setup: c.setup, n: c.n, final_pct: c.final, end: c.end,
  start: c.pts[0][0], points: c.pts.length }))), 'curves (GPT-6.1 Sol, 8M)');
logTable([{ plotted: DOTS.length, counted: COUNTED.length, below: BELOW.length, k8_n: K8.length, k8_median_gap: K8_GAP,
  ...(T_MODE ? {} : { sooner: SOONER.length, k8_median_sooner: K8_SOONER }) }], 'summary');

const pct1 = (v) => `${v < 0 ? '−' : v > 0 ? '+' : ''}${fmt(Math.abs(v), { dp: 1 })}%`;
const times1 = (v) => `${fmt(v, { dp: 1 })}×`;

const TEXT = {
  title: T_MODE
    ? `At the same budget, teams scored below a single agent in ${BELOW.length} of ${COUNTED.length} comparisons`
    : `At the same budget, teams of 8 typically finished ${Math.round(K8_SOONER)} times sooner, scoring ${Math.round(-K8_GAP)}% lower`,
  key: [{ setup: 'single' }, { setup: 'structured' }, { setup: 'selforg' }, { setup: 'orchestrated' }],
  foot: '',   // set by the chart page
};

// ============================================================================
// 3. Layout (computed once W and H are known)
// ============================================================================
let LAY = null;
function layout() {
  if (LAY && LAY.W === W && LAY.H === H) return LAY;
  const L = { W, H };
  // ---- curve panels: 2 x 2, shared scales
  L.left = 58; L.colGap = 82;
  L.pw = (W - L.left - L.colGap) / 2;
  L.headY = 20;                       // curves heading baseline
  L.ph = 158;                         // plot height
  L.rows = [{ name: 64 }, { name: 64 + 34 + L.ph + 26 }];
  for (const r of L.rows) { r.top = r.name + 14; r.bot = r.top + L.ph; }
  L.ticksY = L.rows[1].bot + 28;
  L.axisTitleY = L.rows[1].bot + 54;
  L.panel = TASK_ORDER.map((t, i) => {
    const col = i % 2, row = L.rows[Math.floor(i / 2)];
    const x0 = L.left + col * (L.pw + L.colGap), x1 = x0 + L.pw;
    const [ylo, yhi, yt] = t === 'gitzig' ? [0, 112, [0, 50, 100]] : [60, 105, [60, 80, 100]];
    return { task: t, i, x0, x1, top: row.top, bot: row.bot, nameY: row.name, ylo, yhi, yt };
  });
  L.px = T_MODE ? (p, v) => linear(0, 8, p.x0, p.x1)(v) : (p, v) => logScale(0.025, 30, p.x0, p.x1)(v);
  L.py = (p, v) => linear(p.ylo, p.yhi, p.bot, p.top)(v);

  // ---- summary
  L.sHeadY = L.axisTitleY + 52;
  L.sTop = L.sHeadY + 26;
  L.sBot = H - (T_MODE ? 62 : 64);
  L.sLeft = L.left;
  L.sy = linear(-45, 32, L.sBot, L.sTop);
  if (T_MODE) {
    // single agent slot, then three groups (teams of 2, 4, 8) with one sub-column per setup
    L.singleX = L.sLeft + 46;
    const g0 = L.sLeft + 112, g1 = W - 178, ggap = 24;
    L.gw = (g1 - g0 - 2 * ggap) / 3;
    const frac = { structured: 0.39, selforg: 0.39, orchestrated: 0.22 }, inner = 6;
    L.groups = [2, 4, 8].map((k, gi) => {
      const x0 = g0 + gi * (L.gw + ggap);
      let acc = x0;
      const subs = {};
      for (const s of TEAM_SETUPS) {
        const w = frac[s] * (L.gw - 2 * inner);
        subs[s] = { x0: acc, x1: acc + w, cx: acc + w / 2 };
        acc += w + inner;
      }
      return { k, x0, x1: x0 + L.gw, cx: x0 + L.gw / 2, subs };
    });
  } else {
    L.sx = logScale(0.45, 21, L.sLeft + 8, W - 8);
    L.singleX = L.sx(1);
  }
  // ---- final positions of every dot in the summary
  const R = MARK.run.r;
  L.dot = {};
  if (T_MODE) {
    for (const g of L.groups) for (const s of TEAM_SETUPS) {
      const sub = g.subs[s];
      const items = DOTS.filter((d) => d.k === g.k && d.kit === s).map((d) => ({ d, y: L.sy(d.gap) }));
      const xs = swarm(items.map((it) => it.y), sub.cx, (sub.x1 - sub.x0) / 2, R + 1.6);
      items.forEach((it, j) => { L.dot[it.d.i] = [xs[j], it.y]; });
    }
  } else {
    for (const d of DOTS) L.dot[d.i] = [L.sx(d.sooner), L.sy(d.gap)];
  }
  LAY = L;
  return L;
}

// a deterministic beeswarm: each value (a y in px) gets the x nearest the center that clears every placed dot
function swarm(ys, cx, half, r) {
  const order = ys.map((y, i) => [y, i]).sort((a, b) => a[0] - b[0]), placed = [], out = new Array(ys.length);
  const D = 2 * r + 1;
  for (const [y, i] of order) {
    let best = cx;
    for (let off = 0; off <= half + 40; off += 1) {
      const cands = off === 0 ? [cx] : [cx - off, cx + off];
      const ok = cands.find((x) => placed.every(([px, py]) => (px - x) ** 2 + (py - y) ** 2 >= D * D));
      if (ok != null) { best = ok; break; }
    }
    placed.push([best, y]);
    out[i] = best;
  }
  return out;
}

// ============================================================================
// 4. Marks
// ============================================================================
// the step path of a mean curve, clipped to x <= xMax (in data units). Best-so-far curves only rise, so the part
// under the panel's floor is a prefix: it is skipped, and the curve enters at the floor where it first crosses it.
function stepPts(p, pts, xMax) {
  const L = LAY, out = [];
  let prev = null;
  for (let i = 0; i < pts.length; i++) {
    const [x, y] = pts[i];
    if (x > xMax) {
      if (prev != null && prev >= p.ylo) out.push([L.px(p, xMax), L.py(p, prev)]);
      break;
    }
    if (prev != null && prev >= p.ylo) out.push([L.px(p, x), L.py(p, prev)]);
    if (y >= p.ylo) {
      if (prev == null || prev < p.ylo) { if (prev != null) out.push([L.px(p, x), L.py(p, p.ylo)]); }
      out.push([L.px(p, x), L.py(p, y)]);
    }
    prev = y;
  }
  return out;
}

// display only: resample a step polyline (screen px) every DX px and join the samples, so steps smaller than a
// pixel or two merge into one clean rising line; the first and last points stay exact
function smoothScreen(pts, DX = 5) {
  if (pts.length < 2) return pts;
  const x0 = pts[0][0], x1 = pts[pts.length - 1][0];
  const yAt = (sx) => { let y = pts[0][1]; for (const [x, yy] of pts) { if (x <= sx + 1e-6) y = yy; else break; } return y; };
  const out = [];
  for (let sx = x0; sx < x1; sx += DX) out.push([sx, yAt(sx)]);
  out.push(pts[pts.length - 1]);
  return out;
}
// where a curve ends (data units)
const curveEnd = (c) => [c.end, c.final];

function dotMark(parent, d, cx, cy, u = 1) {
  if (u <= 0) return;
  const S = SETUP[d.kit], kind = SHAPE_OF(d.model);
  if (!d.counted) return run(parent, d.kit, cx, cy, u, { shape: kind, hollow: true });
  if (d.exception) return shape(parent, kind, cx, cy, MARK.run.r * u, { fill: S.tint, stroke: C.bg, 'stroke-width': MARK.run.ring });
  return run(parent, d.kit, cx, cy, u, { shape: kind });
}

// ============================================================================
// 5. draw(t)
// ============================================================================
function draw(t) {
  svg.replaceChildren();
  svg.setAttribute('fill', C.ink);
  const L = layout();

  // ---------------------------------------------------------------- curve panels: scaffold
  h(svg, 'text', { x: 0, y: L.headY, 'font-size': TYPE.label, 'font-weight': 600, fill: C.ink }, 'GPT-6.1 Sol, single agent and teams of 8');
  h(svg, 'text', { x: W, y: L.headY, 'text-anchor': 'end', class: 'axis-title' }, 'best score so far, % of the single agent’s final');
  const xTicks = T_MODE ? [0, 2, 4, 6, 8] : [0.1, 1, 10];
  const xFmt = T_MODE ? (v) => (v === 0 ? '0' : v + 'M') : (v) => (v < 1 ? v + ' h' : v + ' h');
  for (const p of L.panel) {
    h(svg, 'text', { x: p.x0, y: p.nameY, 'font-size': TYPE.name, 'font-weight': 700, fill: C.ink }, PANEL_NAME[p.task]);
    const isBottom = p.task === 'ahc038' || p.task === 'perf', isLeft = p.task === 'gitzig' || p.task === 'ahc038';
    grid(svg, { axis: 'y', scale: (v) => L.py(p, v), values: p.yt, from: p.x0, to: p.x1, base: p.ylo === 0 ? 0 : undefined, label: false });
    for (const v of p.yt) h(svg, 'text', { class: 'tick', x: p.x0 - 12, y: L.py(p, v) + 7, 'text-anchor': 'end' }, v === 0 ? '0' : v + '%');
    if (!T_MODE) for (const v of xTicks) h(svg, 'line', { x1: L.px(p, v), x2: L.px(p, v), y1: p.top, y2: p.bot, stroke: C.grid, 'stroke-width': MARK.grid });
    if (isBottom) for (const v of xTicks) h(svg, 'text', { class: 'tick', x: L.px(p, v), y: L.ticksY, 'text-anchor': v === 0 && T_MODE ? 'start' : 'middle' }, xFmt(v));
  }
  h(svg, 'text', { class: 'axis-title', x: W, y: L.axisTitleY, 'text-anchor': 'end' }, T_MODE ? 'Weighted tokens' : 'Active hours (log scale)');

  // ---------------------------------------------------------------- curve panels: curves, collapse, traces
  const [s0, sd] = BEATS.single, [t0, td] = BEATS.teams, [c0, cd] = BEATS.collapse, [m0, md] = BEATS.morph;
  const sweep = (start, dur) => {  // data x reached by the sweep at time t (linear in screen x)
    const u = clamp01((t - start) / dur);
    return T_MODE ? 8 * u : Math.exp(Math.log(0.025) + u * (Math.log(30) - Math.log(0.025)));
  };
  const uCollapse = prog(t, c0, cd, ease.inOut);
  const traceO = 0.6 * prog(t, c0, cd * 0.7, ease.out);
  const ends = [];   // panel end points, for the morph
  for (const p of L.panel) {
    const cs = DATA.curves[p.task];
    // faint traces first (they appear as the strong curves shrink), then the strong curves on top
    for (const pass of ['trace', 'strong']) {
      for (const c of cs) {
        const isSingle = c.setup === 'single';
        const [bs, bd] = isSingle ? [s0, sd] : [t0, td];
        if (t < bs) continue;
        const xMax = Math.min(sweep(bs, bd), c.end);
        const pts = smoothScreen(stepPts(p, c.pts, xMax));
        if (pts.length < 2) continue;
        const S = SETUP[c.setup];
        if (pass === 'trace') {
          if (traceO > 0) h(svg, 'path', { d: linePath(pts), fill: 'none', stroke: S.fill, 'stroke-width': 2.5, opacity: traceO, 'stroke-linejoin': 'round' });
          continue;
        }
        if (uCollapse >= 1) continue;
        const el = h(svg, 'path', { d: linePath(pts), fill: 'none', stroke: S.line, 'stroke-width': MARK.series.mean, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' });
        if (uCollapse > 0) {
          const Lp = el.getTotalLength();
          el.setAttribute('stroke-dasharray', `${Lp} ${Lp}`);
          el.setAttribute('stroke-dashoffset', `${-uCollapse * Lp}`);
        }
      }
    }
    for (const c of cs) {
      const isSingle = c.setup === 'single';
      const [bs, bd] = isSingle ? [s0, sd] : [t0, td];
      // the end point pops when the sweep reaches it
      const tEnd = bs + bd * (T_MODE ? 1 : (Math.log(c.end) - Math.log(0.025)) / (Math.log(30) - Math.log(0.025)));
      const [ex, ey] = curveEnd(c);
      ends.push({ task: p.task, setup: c.setup, x: L.px(p, ex), y: L.py(p, ey), pop: prog(t, tEnd - 0.05, MOTION.pop) });
    }
  }

  // ---------------------------------------------------------------- summary: scaffold
  h(svg, 'text', { x: 0, y: L.sHeadY, 'font-size': TYPE.label, 'font-weight': 600, fill: C.ink }, 'Every comparison');
  h(svg, 'text', { x: measure('Every comparison', TYPE.label, 600) + 12, y: L.sHeadY, class: 'axis-title' },
    'team score vs the same model’s single agent');
  // shape key, right-aligned in the heading row
  {
    const items = [['circle', false, 'GPT'], ['square', false, 'Claude'], ['square', true, 'not counted']];
    let x = W;
    for (let j = items.length - 1; j >= 0; j--) {
      const [kind, hollow, label] = items[j];
      const w = measure(label, TYPE.tick, 400);
      h(svg, 'text', { x, y: L.sHeadY, 'text-anchor': 'end', class: 'tick' }, label);
      x -= w + 8 + 7;
      if (hollow) shape(svg, kind, x, L.sHeadY - 7, MARK.run.r - 1.25, { fill: C.bg, stroke: C.muted, 'stroke-width': 2.5 });
      else shape(svg, kind, x, L.sHeadY - 7, MARK.run.r, { fill: C.muted });
      x -= 7 + 20;
    }
  }
  const yTicks = [-40, -20, 0, 20];
  grid(svg, { axis: 'y', scale: L.sy, values: yTicks.filter((v) => v !== 0), from: L.sLeft, to: W, label: false });
  for (const v of yTicks) h(svg, 'text', { class: 'tick', x: L.sLeft - 12, y: L.sy(v) + 7, 'text-anchor': 'end' }, v === 0 ? '0' : pct1(v).replace('.0', ''));
  if (T_MODE) {
    for (const g of L.groups) h(svg, 'text', { x: g.cx, y: L.sBot + 34, 'text-anchor': 'middle', 'font-size': TYPE.label, 'font-weight': 600, fill: C.ink }, `Teams of ${g.k}`);
  } else {
    const ticks = [1, 2, 4, 8, 16];
    for (const v of ticks) {
      h(svg, 'line', { x1: L.sx(v), x2: L.sx(v), y1: L.sTop, y2: L.sBot, stroke: C.grid, 'stroke-width': MARK.grid });
      h(svg, 'text', { class: 'tick', x: L.sx(v), y: L.sBot + 28, 'text-anchor': 'middle' }, v + '×');
    }
    h(svg, 'text', { class: 'axis-title', x: W, y: L.sBot + 54, 'text-anchor': 'end' }, 'Times sooner than the single agent (log scale)');
  }

  // ---------------------------------------------------------------- summary: the single agent reference
  const sy0 = L.sy(0);
  const refU = prog(t, m0 + md * 0.55, 0.6, ease.inOut);
  const refX0 = T_MODE ? L.singleX : L.sLeft;
  refLine(svg, 'single', refX0, sy0, W, sy0, refU);

  // ---------------------------------------------------------------- the callout (drawn under the marks)
  const [k0, kd] = BEATS.callout, uK = prog(t, k0 + (T_MODE ? 0 : 0.3), kd - (T_MODE ? 0 : 0.3), ease.out);
  const recede = T_MODE ? 0 : prog(t, k0, 0.35, ease.out);   // time chart: teams of 2 and 4 step back
  if (T_MODE) {
    // a bracket from the single agent's level (0) to the median of the teams of 8, beside that group
    const g8 = L.groups[2], xb = g8.x1 + 14, y8 = L.sy(K8_GAP);
    callout(svg, [[xb - 10, sy0], [xb, sy0], [xb, y8], [xb - 10, y8]], uK,
      { hero: pct1(K8_GAP), words: 'median', x: xb + 10, y: y8 + 11, anchor: 'start' });
  } else {
    // from the single agent (1x, 0) up and across to the median of the teams of 8 in time, then down to their median score
    const yT = L.sy(25), x1 = L.sx(1), x8 = L.sx(K8_SOONER), y8 = L.sy(K8_GAP);
    callout(svg, [[x1, sy0 - 16], [x1, yT], [x8, yT], [x8, y8], [x8 - 10, y8]], uK,
      { hero: times1(K8_SOONER) + ' sooner', words: `${fmt(Math.abs(K8_GAP), { dp: 1 })}% lower, median team of 8`, x: (x1 + x8) / 2 - 10, y: yT - 12, anchor: 'middle' });
  }

  // ---------------------------------------------------------------- summary: every other comparison
  const [o0, od] = BEATS.others;
  const others = DOTS.filter((d) => !d.curve).sort((a, b) => a.k - b.k || TEAM_SETUPS.indexOf(a.kit) - TEAM_SETUPS.indexOf(b.kit) || b.gap - a.gap);
  const drawOrder = [...others].sort((a, b) => (a.counted === b.counted ? 0 : a.counted ? 1 : -1));
  for (const d of drawOrder) {
    const j = others.indexOf(d), u = prog(t, o0 + (j / others.length) * (od - MOTION.pop), MOTION.pop);
    const [x, y] = L.dot[d.i];
    const el = dotMark(svg, d, x, y, u);
    if (el && d.k !== 8 && recede > 0) el.setAttribute('opacity', 1 - 0.65 * recede);
  }
  // the 32M exception note (lighter dots)
  {
    const ex = DOTS.filter((d) => d.exception), top = ex.reduce((a, b) => (L.dot[a.i][1] < L.dot[b.i][1] ? a : b));
    const [x, y] = L.dot[top.i], o = fade(t, o0 + od - 0.2, 0.3);
    if (T_MODE) {
      if (o > 0) tag(svg, x - 14, y + 7, 'Lighter: 32M budget, apparent exception, see text', { setup: 'selforg', size: TYPE.small, anchor: 'end', opacity: o });
    } else if (o > 0) {
      // beside the two 32M teams of 8 (the teams of 2 and 4 recede in the callout beat), right of the callout's leg
      const nx = L.sx(K8_SOONER) + 16;
      tag(svg, nx, L.sy(21), '32M budget:', { setup: 'selforg', size: TYPE.small, opacity: o });
      tag(svg, nx, L.sy(21) + 25, 'apparent exception, see text', { setup: 'selforg', size: TYPE.small, opacity: o });
    }
  }

  // ---------------------------------------------------------------- the morph: panel end points into the summary
  const uMove = (e) => prog(t, m0 + TASK_ORDER.indexOf(e.task) * 0.08, md - 3 * 0.08, ease.inOut);
  for (const e of ends) {
    if (e.pop <= 0) continue;
    const u = uMove(e);
    const S = SETUP[e.setup];
    if (u > 0) {   // a small faint dot stays where the curve ended
      const gO = 0.85 * clamp01(u * 3);
      h(svg, 'circle', { cx: e.x, cy: e.y, r: 4.5, fill: S.fill, opacity: gO });
    }
    let tx, ty;
    const dd = e.setup === 'single' ? null : DOTS.find((d) => d.curve && d.task === e.task && d.kit === e.setup);
    if (dd) [tx, ty] = L.dot[dd.i]; else [tx, ty] = [L.singleX, sy0];
    const x = lerp(e.x, tx, u), y = lerp(e.y, ty, u);
    if (e.setup === 'single') {
      if (u >= 1 && e.task !== 'gitzig') continue;   // the four land on one spot; draw one
      mean(svg, 'single', x, y, e.pop, { dense: u < 1 });
    } else if (u < 1) {
      mean(svg, e.setup, x, y, e.pop, { dense: true });
    } else {
      dotMark(svg, dd, x, y, 1);
    }
  }
  // tokens: above its dot (teams of 2 sit just right of it below the line); time: below-left (the callout rises above it)
  tag(svg, L.singleX + (T_MODE ? -12 : -10), sy0 + (T_MODE ? -22 : 34), 'Single agent', { setup: 'single', anchor: T_MODE ? 'start' : 'end', opacity: fade(t, m0 + md - 0.1, 0.3) });
}
