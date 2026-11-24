"""Restart chart tally: git-to-zig, best score so far against weighted tokens, for three setups per panel: the
single agent, the self-organized team of 8 (GPT-6.1 message board) and the restarted single agent (8 sessions in a
row, each with a fresh context). Panels: 32M and 8M weighted tokens, GPT-6.1 Sol and GPT-6.1 Terra.

Reads data/runs.csv (counted runs and their finals) and data/snapshots.csv (graded in-run snapshots) through
analysis/common.py. Arms: the configurations gitzig/<model>/solo/1/<budget>M, gitzig/<model>/board/8/<budget>M and
gitzig/<model>/relay/8/<budget>M (no variant, original Codex build).

Curve rule: for each run, the best graded in-run snapshot so far at X weighted tokens, on a grid of 3201 points from 0
to the budget; the line is the mean of those run curves. Before a run's first graded snapshot its curve is 0. A run
whose last graded snapshot comes before the budget holds its best value flat to the end.
Two display-only additions, both reported in the printout:
  1. a snapshot taken just past the budget (overshoot, at most 0.1M) counts at the budget;
  2. each run's final grade counts as one more graded point at the budget, so the lines end at the mean finals.
     Without it, runs whose last graded snapshot came before their final code end their lines below their finals.
     The line up to the last grid point is unchanged by (2).
Runs with no graded snapshot (GPT-6.1 Terra's 32M team r1 to r3) are left out of the line; their finals count in the
end label.
Lambda against the restarted single agent (logged by the page, not drawn): X* = the first grid point where the
restarted agent's mean curve (without the two additions) reaches the team's mean final; lambda = ln(X* / (B/8)) / ln 8.

Prints the chart's DATA as JSON on stdout and the numbers the post quotes on stderr.
--write also pastes DATA into chart.html (between DATA-BEGIN and DATA-END).
Run from the repo root: python3 analysis/figures/restart/tally.py [--write]
"""
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common as cm  # noqa: E402

N_AGENTS = 8
GRID_N = 3201
SETUP = {'single': ('solo', 1), 'selforg': ('board', 8), 'restarted': ('relay', 8)}
C = cm.cells()
SNAP = cm.snapshots()


def arm(model, a, budget):
    setup, k = SETUP[a]
    cell = C[cm.cid('gitzig', model, setup, k, budget)]
    rs = sorted(cell.runs, key=lambda r: r['repeat'])
    return {r['run']: (r['final'], sorted(SNAP.get(r['run'], []))) for r in rs}


def mean_curve(curves):
    return [sum(c[i] for c in curves) / len(curves) for i in range(len(curves[0]))]


def steps(grid, y):
    keep = [0] + [i for i in range(1, len(y)) if y[i] != y[i - 1]] + [len(y) - 1]
    return [[round(grid[i] / 1e6, 4), round(y[i], 3)] for i in keep]


out = {'panels': [], 'notes': {}}
for budget in (32.0, 8.0):
    B = budget * 1e6
    grid = [B * i / (GRID_N - 1) for i in range(GRID_N)]
    for model in ('sol', 'terra'):
        arms = {a: arm(model, a, budget) for a in SETUP}
        panel = {'budgetM': int(budget), 'model': model, 'arms': {}}
        means, lines = {}, {}
        for a, runs in arms.items():
            withc = {n: v for n, v in runs.items() if v[1]}
            over = max(w for _, c in withc.values() for w, _ in c) - B
            assert over < 0.1e6, (model, budget, a, over)
            raw = mean_curve([cm.best_curve(c, grid) for _, c in withc.values()])
            capped = mean_curve([cm.best_curve([(min(w, B), r) for w, r in c], grid) for _, c in withc.values()])
            shown = mean_curve([cm.best_curve([(min(w, B), r) for w, r in c] + [(B, f)], grid)
                                for f, c in withc.values()])
            assert raw[:-1] == capped[:-1] == shown[:-1]
            finals = [f for f, _ in runs.values()]
            means[a] = (raw, st.mean(finals))
            lines[a] = shown
            last_graded = {n: max(w for w, _ in c) for n, (_, c) in withc.items()}
            panel['arms'][a] = {
                'runs': list(runs), 'finals': [round(f, 4) for f in finals],
                'n_line': len(withc), 'no_line': [n for n in runs if n not in withc],
                'line_end': round(shown[-1], 3),
                'line_end_without_finals': round(capped[-1], 3),
                'line_end_raw': round(raw[-1], 3),
                'line_at_last_but_one': round(shown[-2], 3),
                'last_graded_min_M': round(min(last_graded.values()) / 1e6, 3),
                'curve': steps(grid, shown),
            }
        # lambda against the restarted single agent: X* on the restarted mean curve at the team's mean final
        X = cm.x_star([cm.best_curve(c, grid) for _, c in arms['restarted'].values() if c], grid, means['selforg'][1])
        assert all(abs(u - v) < 1e-9 for u, v in zip(
            mean_curve([cm.best_curve(c, grid) for _, c in arms['restarted'].values() if c]), means['restarted'][0]))
        panel['lambda'] = {'X_star_M': round(X / 1e6, 3), 'lam': round(cm.lam(X, B / N_AGENTS, N_AGENTS), 3)}

        # where the lines cross: the first point after which line a stays above line b (lines as drawn)
        def ahead_from(a, b):
            d = [u - v for u, v in zip(lines[a][1:], lines[b][1:])]
            neg = [i for i, v in enumerate(d) if v <= 0]
            if not neg:
                return 0.0
            return None if neg[-1] == len(d) - 1 else round(grid[neg[-1] + 2] / 1e6, 3)
        panel['cross'] = {'team_above_single_from_M': ahead_from('selforg', 'single'),
                          'restarted_above_single_from_M': ahead_from('restarted', 'single'),
                          'restarted_above_team_from_M': ahead_from('restarted', 'selforg')}
        # values at each session boundary (B/8 apart)
        step_i = (GRID_N - 1) // 8
        panel['at_sessions'] = {f'{k * budget / 8:g}M': {a: round(lines[a][k * step_i], 2) for a in arms}
                                for k in range(1, 9)}
        out['panels'].append(panel)

out['notes'] = {
    'rule': 'mean over runs of the best graded in-run snapshot so far at X weighted tokens, grid of 3201 points from 0 '
            'to the budget; final grade added as a graded point at the budget; overshoot capped',
    'sources': {'finals': 'data/runs.csv (counted runs)', 'curves': 'data/snapshots.csv'},
}

print(json.dumps(out, separators=(',', ':')))

# ---- the numbers the post quotes, on stderr
name = {'single': 'Single agent', 'selforg': 'Team of 8', 'restarted': 'Restarted'}
POST = {('sol', 32): (83.7, 93.2, 97.0, 3.8, 0.94), ('terra', 32): (26.6, 32.0, 41.9, 9.9, 0.75),
        ('sol', 8): (53.3, 55.0, 62.1, 7.1, 0.89), ('terra', 8): (16.5, 17.8, 28.0, 10.2, 0.60)}
err = sys.stderr
for p in out['panels']:
    A = p['arms']
    m = {a: st.mean(A[a]['finals']) for a in A}
    want = POST[(p['model'], p['budgetM'])]
    got = (m['single'], m['selforg'], m['restarted'], m['restarted'] - m['selforg'], p['lambda']['lam'])
    print(f"\n=== {p['model']} {p['budgetM']}M   X*={p['lambda']['X_star_M']}M lambda={p['lambda']['lam']}", file=err)
    for a, s in A.items():
        print(f"  {name[a]:12} n={len(s['finals']):2} line n={s['n_line']:2} final mean={m[a]:7.3f}  line end="
              f"{s['line_end']:7.3f} (no finals {s['line_end_without_finals']:7.3f}, raw grid "
              f"{s['line_end_raw']:7.3f}, one step before end {s['line_at_last_but_one']:7.3f})  earliest "
              f"last-graded {s['last_graded_min_M']}M  steps={len(s['curve'])}  no line: {s['no_line']}", file=err)
    print('  crossings', p['cross'], file=err)
    for k, v in p['at_sessions'].items():
        print(f"    at {k:>5}: " + '  '.join(f"{name[a]} {v[a]:6.2f}" for a in v), file=err)
    shown = (f'{got[0]:.1f}', f'{got[1]:.1f}', f'{got[2]:.1f}', f'+{got[3]:.1f}', f'{got[4]:.2f}')
    post = (f'{want[0]:.1f}', f'{want[1]:.1f}', f'{want[2]:.1f}', f'+{want[3]:.1f}', f'{want[4]:.2f}')
    print(f"  CHECK single/team/restarted/gap vs team/lambda: post {'/'.join(post)}  data {'/'.join(shown)}  "
          f"{'OK' if shown == post else 'DIFFERENT'}", file=err)
ns = {(p['model'], p['budgetM'], a): (len(s['finals']), s['n_line']) for p in out['panels'] for a, s in p['arms'].items()}
print('\nCHECK footnote "mean of 10 runs (restarted 5; GPT-6.1 Terra at 8M 3, its 32M team 7)": runs (line runs) '
      + '; '.join(f'{m} {b}M {a} {n}({nl})' for (m, b, a), (n, nl) in ns.items()), file=err)
gaps = [st.mean(p['arms']['restarted']['finals']) - st.mean(p['arms']['selforg']['finals']) for p in out['panels']]
print(f'CHECK title "by about 4 to 10 points": {round(min(gaps))} to {round(max(gaps))}', file=err)

if '--write' in sys.argv:
    html = HERE / 'chart.html'
    src = html.read_text()
    new, k = re.subn(r'/\* DATA-BEGIN \*/.*?/\* DATA-END \*/',
                     lambda _: '/* DATA-BEGIN */' + json.dumps(out, separators=(',', ':')) + '/* DATA-END */', src,
                     flags=re.S)
    assert k == 1
    html.write_text(new)
    print('DATA written into', html.name, file=err)
