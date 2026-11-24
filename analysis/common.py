"""Shared loaders and rules for the analysis scripts, the figures' tallies and the results page.

A configuration (cell) is one task, model, setup, team size, total budget, variant and series. Its value is the mean
final of its counted runs; a no-talk team of k is the exact mean of the best of k over every k-subset of its pool.

Rules (the same everywhere):
  gap       configuration / reference - 1, in %; the take-home as speed (reference cycles / configuration cycles - 1),
            so + is always better for the team. Points: configuration - reference (take-home: reference - config).
  interval  bootstrap percentile 95%, runs resampled within each arm, 10,000 draws, seeded per comparison.
  counted   a comparison with a single agent counts only if that single agent's runs spent at least 85% of their
            budget on average.
"""
import csv
import hashlib
import math
import random
import re
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / 'data'

SPEND_OK = 0.85
NB, SEED = 10_000, 1118
CEILING = 3389                      # AHC014: AtCoder performance of first place
LOWER_BETTER = {'perf'}             # the take-home is scored in clock cycles
POOL = {2: 6, 4: 8, 8: 12}          # no-talk pools: k = 2 from 6 agents at 4M, 4 from 8 at 2M, 8 from 12 at 1M

TASK = {'gitzig': 'git-to-zig', 'ahc014': 'AHC014', 'ahc038': 'AHC038', 'perf': 'Performance take-home'}
MODEL = {'opus': 'Claude Opus 5.5', 'sonnet': 'Claude Sonnet 5.5', 'sol': 'GPT-6.1 Sol', 'terra': 'GPT-6.1 Terra',
         'opus5': 'Claude Opus 5', 'sonnet5': 'Claude Sonnet 5', 'terra56': 'GPT-5.6 Terra', 'sol62': 'GPT-6.2 Sol',
         'mixed': 'Claude Sonnet 5.5 + GPT-6.1 Sol', 'opussol': 'Claude Opus 5.5 + GPT-6.1 Sol'}
SETUP = {'solo': 'Single agent', 'relay': 'Restarted single agent', 'ttc': 'Protocol teams',
         'board': 'Self-organized teams', 'peers': 'Self-organized teams', 'chat': 'Self-organized teams',
         'ultra': 'Orchestrated teams', 'notalk': 'No-talk teams', 'solo_ttc': 'Single agent, protocol prompt',
         'relayctx': 'Single agent, new session whenever the context fills'}
CLAUDE = {'opus', 'sonnet', 'opus5', 'sonnet5'}


def num(x):
    return float(x) if x not in ('', None) else None


def runs(counted_only=False):
    """Every run in data/runs.csv, numbers parsed."""
    out = []
    for r in csv.DictReader(open(DATA / 'runs.csv', encoding='utf-8')):
        for c in ('final', 'raw_per_case_M', 'weighted_tokens_M', 'spend_share', 'active_hours', 'elapsed_hours',
                  'budget_M', 'per_agent_M'):
            r[c] = num(r[c])
        for c in ('team_size', 'repeat', 'counted', 'input_tokens', 'cached_input_tokens', 'cache_write_tokens',
                  'output_tokens', 'model_calls'):
            r[c] = int(r[c]) if r[c] != '' else None
        if counted_only and not r['counted']:
            continue
        out.append(r)
    return out


def key(r):
    """The configuration a run belongs to."""
    return (r['task'], r['model'], r['setup'], int(r['team_size'] or 0), float(r['budget_M'] or 0), r['variant'],
            r['series'])


def cell_id(k):
    task, model, setup, n, budget, variant, series = k
    s = f'{task}/{model}/{setup}/{n}/{budget:g}M'
    if variant:
        s += '/' + re.sub(r'[^a-z0-9]+', '-', variant.lower()).strip('-')
    return s + ('/' + series if series else '')


def seed_of(k):
    return SEED + int(hashlib.sha1(repr(k).encode()).hexdigest()[:8], 16)


def best_of_k(xs, k, lower=False):
    """Exact mean of the best of k over every k-subset of xs."""
    s = sorted(xs, reverse=lower)          # worst first
    if len(s) < k:
        return None
    return sum(s[i] * math.comb(i, k - 1) for i in range(len(s))) / math.comb(len(s), k)


def gap(task, team, base):
    if team is None or base is None:
        return None
    return (base / team - 1) if task in LOWER_BETTER else (team / base - 1)


def points(task, team, base):
    return (base - team) if task in LOWER_BETTER else (team - base)


def boot(rng, arms, fn, nb=NB):
    """Percentile 95% interval of fn(resampled arms)."""
    out = []
    for _ in range(nb):
        v = fn(*[rng.choices(a, k=len(a)) for a in arms])
        if v is not None:
            out.append(v)
    if len(out) < nb // 2:
        return None, None
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]


class Cell:
    """A configuration and its counted runs."""

    def __init__(self, k, rs):
        self.key, self.runs = k, rs
        self.task, self.model, self.setup, self.k, self.budget, self.variant, self.series = k
        self.id = cell_id(k)
        self.vals = [r['final'] for r in rs]
        self.pool = self.setup == 'notalk'
        kb = self.k if self.pool else None
        self.mean = self.stat(self.vals) if self.vals and (not kb or len(self.vals) >= kb) else None
        sh = [r['spend_share'] for r in rs if r['spend_share'] is not None]
        self.spend = st.mean(sh) if sh else None
        hr = [r['active_hours'] for r in rs if r['active_hours'] is not None]
        self.hours_median = st.median(hr) if hr else None

    def stat(self, vals):
        if self.pool:
            return best_of_k(vals, self.k, self.task in LOWER_BETTER)
        return st.mean(vals)

    def ci(self):
        if len(self.vals) < 2 or self.mean is None:
            return None, None
        return boot(random.Random(seed_of(self.key)), [self.vals], self.stat)

    def __repr__(self):
        return f'<{self.id} n={len(self.vals)} mean={self.mean}>'


def cells(rs=None):
    """cell id -> Cell, from the counted runs."""
    rs = runs(counted_only=True) if rs is None else [r for r in rs if r['counted']]
    by = {}
    for r in rs:
        by.setdefault(key(r), []).append(r)
    return {cell_id(k): Cell(k, v) for k, v in by.items()}


def cid(task, model, setup, k, budget, variant='', series=''):
    return cell_id((task, model, setup, int(k), float(budget), variant, series))


def compare(cell, ref, seed_key=None):
    """Gap of a cell against a reference cell, with a bootstrap interval. {} when either side is missing."""
    if not cell or not ref or not cell.vals or not ref.vals or cell.mean is None:
        return {}
    task = cell.task
    rng = random.Random(seed_of(seed_key or (cell.key, ref.key)))
    bm = st.mean(ref.vals)
    g = gap(task, cell.mean, bm)
    lo = hi = None
    if len(cell.vals) > 1 and len(ref.vals) > 1:
        lo, hi = boot(rng, [cell.vals, ref.vals], lambda a, b: gap(task, cell.stat(a), st.mean(b)))
    out = {'gap_pct': 100 * g, 'gap_points': points(task, cell.mean, bm),
           'gap_lo': 100 * lo if lo is not None else None, 'gap_hi': 100 * hi if hi is not None else None,
           'base_n': len(ref.vals), 'base_mean': bm, 'base_spend': ref.spend,
           'counted': ref.spend is not None and ref.spend >= SPEND_OK}
    out['clear'] = None if lo is None else ('above' if lo > 0 else ('below' if hi < 0 else 'within noise'))
    return out


def vs_single(C, cell):
    """The default comparison: the same model's plain single agent at the same total budget."""
    ref = C.get(cid(cell.task, cell.model, 'solo', 1, cell.budget))
    return compare(cell, ref), ref


def boot_points(a, b, task, seed, nb=NB):
    """Bootstrap 95% interval of mean(a) - mean(b) in points (take-home: b - a), runs resampled within each arm."""
    return boot(random.Random(seed), [a, b], lambda x, y: points(task, st.mean(x), st.mean(y)), nb)


# ------------------------------------------------------------------ curves
def snapshots():
    """git-to-zig run -> [(weighted tokens, % of tests passed)] in-run graded snapshots, in file order."""
    out = {}
    for r in csv.DictReader(open(DATA / 'snapshots.csv', encoding='utf-8')):
        out.setdefault(r['run'], []).append((float(r['weighted_tokens']), float(r['score'])))
    return out


def best_curve(pts, grid):
    """Best score so far at each grid point (0 before the first point)."""
    pts = sorted(pts)
    out, best, j = [], 0.0, 0
    for x in grid:
        while j < len(pts) and pts[j][0] <= x:
            best = max(best, pts[j][1])
            j += 1
        out.append(best)
    return out


def x_star(curves, grid, s):
    """First grid point where the mean of the curves reaches s."""
    for i, x in enumerate(grid):
        if sum(c[i] for c in curves) / len(curves) >= s:
            return x
    return None


def lam(x, b, n):
    """Ord's lambda: a team of n agents with b each matches one agent with x; lambda = ln(x/b) / ln n."""
    return math.log(x / b) / math.log(n)


def loglin_x(budgets, means, s):
    """Budget at which a single agent reaches s, with score linear in log budget between the budgets run (extended
    past the ends with the nearest segment)."""
    pts = sorted(zip(budgets, means))
    for (b0, s0), (b1, s1) in zip(pts, pts[1:]):
        if s <= s1 or (b1, s1) == pts[-1]:
            if s1 == s0:
                return None
            return math.exp(math.log(b0) + (s - s0) / (s1 - s0) * (math.log(b1) - math.log(b0)))
    return None


def fmt(x, d=1):
    return '' if x is None else f'{x:.{d}f}'
