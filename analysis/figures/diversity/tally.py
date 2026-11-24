"""Diversity chart tally: does mixing Claude and GPT models in one team help? (appendix "Does model diversity help?")

Reads data/runs.csv (counted runs) through analysis/common.py. Every configuration is at 8M weighted tokens in total,
no variant, original Codex build.
  Mixed teams     protocol teams (setup ttc) of model 'mixed' (git-to-zig: Claude Sonnet 5.5 + GPT-6.1 Sol) and
                  'opussol' (AHC014 and the take-home: Claude Opus 5.5 + GPT-6.1 Sol); self-organized git-to-zig
                  teams of model 'mixed' that shared a SwarmChat chat (setup chat). Half of each team is each model.
  One-model teams protocol teams (ttc) of the Claude model and of GPT-6.1 Sol; on git-to-zig also self-organized
                  teams: Claude Sonnet 5.5 over direct messages (peers), GPT-6.1 Sol over the message board (board).
  Single agents   setup solo, team size 1: Claude Opus 5.5, Claude Sonnet 5.5 and GPT-6.1 Sol.

Units: git-to-zig = % of tests passed; AHC014 = raw private score per case (millions), because AtCoder performance
stops at 3389 (first place); take-home = clock cycles (lower is better).
Gap to the best single agent (the chart's x): team / single - 1 (git-to-zig, AHC014 raw score); take-home as speed,
single cycles / team cycles - 1. In %, + = better than the single agent.
Best single agent = the single agent with the better mean in the task's chart unit (GPT-6.1 Sol on git-to-zig and the
take-home, Claude Opus 5.5 on AHC014). Under budget: a single-agent run that spent under 7.9M of its 8M.
Place of a mixed team: 'between' its two one-model teams of the same size and setup, 'above both' or 'below both'.
Noise of the mixed teams above the best single agent: one-sided exact permutation p (all relabelings of the runs).

Prints the chart's DATA as JSON on stdout and every number behind it, with the numbers the post quotes, on stderr.
--write also pastes DATA into chart.html (between DATA:BEGIN and DATA:END).
Run from the repo root: python3 analysis/figures/diversity/tally.py [--write]
"""
import itertools
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common as cm  # noqa: E402

SIZES = (2, 4, 8)
BUDGET = 8
UNDER = 7.9

# Chart groups: key, task, kit setup, Claude model, setup of the one-model teams per model, (model, setup) of the mixed
GROUPS = [
    dict(key='gitzig-protocol', task='gitzig', kit='structured', claude='sonnet',
         setups={'claude': 'ttc', 'sol': 'ttc'}, mixed=('mixed', 'ttc')),
    dict(key='gitzig-selforg', task='gitzig', kit='selforg', claude='sonnet',
         setups={'claude': 'peers', 'sol': 'board'}, mixed=('mixed', 'chat')),
    dict(key='ahc014', task='ahc014', kit='structured', claude='opus',
         setups={'claude': 'ttc', 'sol': 'ttc'}, mixed=('opussol', 'ttc')),
    dict(key='perf', task='perf', kit='structured', claude='opus',
         setups={'claude': 'ttc', 'sol': 'ttc'}, mixed=('opussol', 'ttc')),
]
C = cm.cells()


def runs(task, model, setup, k):
    c = C.get(cm.cid(task, model, setup, k, BUDGET))
    assert c and c.runs, (task, model, setup, k)
    return [dict(run=r['run'], final=r['final'], raw=r['raw_per_case_M'], tokens=r['weighted_tokens_M'],
                 status=r['status']) for r in c.runs]


def unit(task, r):
    """the chart's value of one run, in the task's own unit (AHC014: raw score per case)"""
    return r['raw'] if task == 'ahc014' else r['final']


def gap(task, team, single):
    """% better than the single agent (+ = better). Take-home: speed = single cycles / team cycles."""
    return 100 * cm.gap(task, team, single)


def perm_p(a, b, lower):
    """one-sided exact permutation p of mean(a) better than mean(b): share of relabelings at least as extreme"""
    pool, n, obs = a + b, len(a), st.mean(a) - st.mean(b)
    if lower:
        obs = -obs
    hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        x = [pool[i] for i in idx]
        y = [pool[i] for i in range(len(pool)) if i not in idx]
        d = st.mean(x) - st.mean(y)
        hits += (-d if lower else d) >= obs - 1e-12
        tot += 1
    return hits / tot


def welch(a, b):
    """Welch t statistic and degrees of freedom"""
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    return (st.mean(a) - st.mean(b)) / math.sqrt(va + vb), (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))


def fmt(task, runs_):
    m = st.mean(r['final'] for r in runs_)
    if task == 'gitzig':
        return f'{m:.1f}'
    if task == 'ahc014':
        return f'{m:.0f} ({st.mean(r["raw"] for r in runs_):.2f})'
    return f'{m:.0f}'


out = dict(groups=[], checks={})
for g in GROUPS:
    task = g['task']
    singles = {m: runs(task, m, 'solo', 1) for m in ('opus', 'sonnet', 'sol')}
    smean = {m: st.mean(unit(task, r) for r in v) for m, v in singles.items()}
    best = (min if task in cm.LOWER_BETTER else max)(smean, key=smean.get)
    S = smean[best]
    srun = singles[best]
    tok = [round(r['tokens'], 3) for r in srun]
    grp = dict(key=g['key'], task=task, kit=g['kit'], claude=g['claude'],
               single=dict(model=best, n=len(srun), mean=S, runs=[unit(task, r) for r in srun],
                           gaps=[gap(task, unit(task, r), S) for r in srun], tokens_M=tok,
                           under_budget=[t < UNDER for t in tok], status=[r['status'] for r in srun],
                           perf_mean=st.mean(r['final'] for r in srun) if task == 'ahc014' else None,
                           names=[r['run'] for r in srun], table=fmt(task, srun)),
               rows=[])
    for k in SIZES:
        cells = {'mixed': runs(task, *g['mixed'], k),
                 'claude': runs(task, g['claude'], g['setups']['claude'], k),
                 'sol': runs(task, 'sol', g['setups']['sol'], k)}
        row = dict(k=k)
        for who, rr in cells.items():
            vals = [unit(task, r) for r in rr]
            m = st.mean(vals)
            row[who] = dict(n=len(rr), mean=m, gap=gap(task, m, S), runs=vals,
                            run_gaps=[gap(task, v, S) for v in vals], names=[r['run'] for r in rr],
                            perf_mean=st.mean(r['final'] for r in rr) if task == 'ahc014' else None,
                            at_ceiling=sum(r['final'] >= cm.CEILING for r in rr) if task == 'ahc014' else None,
                            table=fmt(task, rr))
        # where the mixed team lands against its two one-model teams
        lo, hi = sorted([row['claude']['gap'], row['sol']['gap']])
        mg = row['mixed']['gap']
        row['place'] = 'above both' if mg > hi else 'below both' if mg < lo else 'between'
        better = (min if task in cm.LOWER_BETTER else max)(row['claude']['mean'], row['sol']['mean'])
        row['beats_both_by_pct'] = gap(task, row['mixed']['mean'], better) if mg > hi else None
        row['vs_midpoint_pct'] = gap(task, row['mixed']['mean'], st.mean([row['claude']['mean'], row['sol']['mean']]))
        grp['rows'].append(row)
    out['groups'].append(grp)

# ---- the numbers the text quotes
G = {g['key']: g for g in out['groups']}
ahc = G['ahc014']
at3389 = sum(r[w]['at_ceiling'] for r in ahc['rows'] for w in ('mixed', 'claude', 'sol')) + \
    sum(x['final'] >= cm.CEILING for x in runs('ahc014', 'opus', 'solo', 1))
n_ahc = sum(r[w]['n'] for r in ahc['rows'] for w in ('mixed', 'claude', 'sol')) + ahc['single']['n']
out['checks']['ahc014_at_ceiling'] = [at3389, n_ahc]
above_single = [(g['key'], r['k'], r['mixed']['gap']) for g in out['groups'] for r in g['rows'] if r['mixed']['gap'] > 0]
out['checks']['mixed_above_best_single'] = above_single
p8 = next(r for r in G['perf']['rows'] if r['k'] == 8)
out['checks']['perf8_fewer_cycles_pct'] = 100 * (1 - p8['mixed']['mean'] / G['perf']['single']['mean'])
out['checks']['ahc014_single_tokens_M'] = ahc['single']['tokens_M']
places = [r['place'] for g in out['groups'] for r in g['rows']]
out['checks']['place_counts'] = {p: places.count(p) for p in ('between', 'above both', 'below both')}
gz_single = {m: st.mean(r['final'] for r in runs('gitzig', m, 'solo', 1)) for m in ('sol', 'sonnet')}
out['checks']['gitzig_single_sol_sonnet'] = gz_single
noise = []
for key, k, _ in above_single:
    a = next(r for r in G[key]['rows'] if r['k'] == k)['mixed']['runs']
    b = G[key]['single']['runs']
    t, df = welch(a, b) if key != 'perf' else welch([-x for x in a], [-x for x in b])
    noise.append(dict(cell=f'{key} k={k}', mixed_runs=a, single_runs=b, welch_t=t, welch_df=df,
                      perm_p_one_sided=perm_p(a, b, key == 'perf')))
out['checks']['noise_above_single'] = noise

# ---- the chart's DATA block: only what chart.html draws or checks
r4 = lambda v: round(v, 4)  # noqa: E731
chart = dict(groups=[dict(
    key=g['key'], task=g['task'], kit=g['kit'], claude=g['claude'],
    single=dict(model=g['single']['model'], n=g['single']['n'], gaps=[r4(v) for v in g['single']['gaps']],
                under_budget=g['single']['under_budget'], tokens_M=g['single']['tokens_M']),
    rows=[dict(k=r['k'], place=r['place'], **{w: dict(gap=r4(r[w]['gap']), n=r[w]['n']) for w in ('mixed', 'claude', 'sol')})
          for r in g['rows']]) for g in out['groups']],
    checks=dict(place_counts=out['checks']['place_counts'],
                mixed_above_best_single=[[a, b, r4(c)] for a, b, c in above_single],
                perm_p=[x['perm_p_one_sided'] for x in noise]))
print(json.dumps(chart))

# ---- printout, on stderr
err = sys.stderr
for g in out['groups']:
    s = g['single']
    print(f"\n{g['key']}  best single agent {s['model']} n={s['n']} mean {s['mean']:.4f} ({s['table']})  run gaps "
          f"{[round(x, 1) for x in s['gaps']]}  tokens {s['tokens_M']}", file=err)
    for r in g['rows']:
        print(f"  k={r['k']}  " + '  '.join(f"{w} {r[w]['mean']:.4f} [{r[w]['table']}] ({r[w]['gap']:+.2f}%, n={r[w]['n']})"
                                           for w in ('mixed', 'claude', 'sol')) + f"  -> {r['place']}"
              + (f", +{r['beats_both_by_pct']:.2f}% over the better team" if r['beats_both_by_pct'] else '')
              + f"; vs midpoint {r['vs_midpoint_pct']:+.1f}%", file=err)
for x in noise:
    print(f"  noise {x['cell']}: Welch t {x['welch_t']:.2f} (df {x['welch_df']:.1f}), one-sided permutation p "
          f"{x['perm_p_one_sided']:.3f}", file=err)
ck = out['checks']
n_between = ck['place_counts']['between']
tok = ck['ahc014_single_tokens_M']
print('\nCHECKS (post says -> data gives):', file=err)
print(f'  title "between ... in 8 of 12 cases" -> {n_between} of {len(places)} ({ck["place_counts"]})', file=err)
print(f'  footnote: AHC014 team of 2 "1.9% above the best single agent in raw score" -> '
      + ', '.join(f'{a} k={b} {c:+.1f}%' for a, b, c in above_single), file=err)
print(f'  footnote: take-home team of 8 "needed 1.9% fewer cycles" -> {ck["perf8_fewer_cycles_pct"]:.1f}% fewer '
      f'cycles ({p8["mixed"]["gap"]:+.1f}% speed)', file=err)
print(f'  "16 of the 30 runs reached 3389" -> {ck["ahc014_at_ceiling"][0]} of {ck["ahc014_at_ceiling"][1]}', file=err)
print(f'  Claude Opus 5.5 single agents "spent only 3.6M to 6.9M" -> {min(tok):.1f}M to {max(tok):.1f}M '
      f'(chart footnote rounds {min(tok):.3f} and {max(tok):.3f})', file=err)
print(f'  git-to-zig single agents "Sol 53.3% against Sonnet 41.7%" -> {gz_single["sol"]:.1f}% against '
      f'{gz_single["sonnet"]:.1f}%', file=err)

if '--write' in sys.argv:
    page = HERE / 'chart.html'
    src = page.read_text()
    a, b = src.index('// DATA:BEGIN'), src.index('// DATA:END')
    page.write_text(src[:a] + '// DATA:BEGIN (written by tally.py; do not edit)\nconst DATA = ' + json.dumps(chart)
                    + ';\n' + src[b:])
    print('DATA written into', page.name, file=err)
