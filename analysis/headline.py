"""The post's headline numbers: how many runs and comparisons, how teams scored against a single agent, how much
sooner they finished, and how much of their budget the single agents spent.

    python3 analysis/headline.py

The comparisons are the 82 team configurations of the tokens and time charts: teams of 2, 4 and 8 in every setup at
8M on the four tasks (Claude Opus 5.5 and Sonnet 5.5, GPT-6.1 Sol and Terra), plus GPT-6.1's self-organized
git-to-zig teams at 32M. Each is compared with the same model's single agent at the same budget (common.compare).
Counted for tokens: that single agent spent at least 85% of its budget on average. Counted for time: that single
agent did not spend most of its time waiting on its own jobs (flag 'waited': every Claude single agent on the AHC
tasks and the take-home). Times sooner = the single agent's median active hours / the team's.
"""
import statistics as st
from collections import Counter

from common import CLAUDE, MODEL, SETUP, TASK, cells, runs, vs_single

C = cells()
R = runs()
counted = [r for r in R if r['counted']]


def comparisons():
    out = []
    for c in C.values():
        if c.k < 2 or c.setup not in ('ttc', 'board', 'peers', 'ultra') or c.variant or c.series:
            continue
        if c.model not in ('opus', 'sonnet', 'sol', 'terra'):
            continue
        if not (c.budget == 8 or (c.budget == 32 and c.task == 'gitzig' and c.model in ('sol', 'terra') and c.k <= 8)):
            continue
        g, ref = vs_single(C, c)
        waited = any('waited' in r['flags'].split() for r in ref.runs)
        out.append({'cell': c, 'ref': ref, 'gap': g['gap_pct'], 'counted': g['counted'], 'clear': g['clear'],
                    'sooner': ref.hours_median / c.hours_median, 'time_counted': not waited})
    return out


if __name__ == '__main__':
    print('== Runs')
    print(f'Counted runs: {len(counted)}   (all rows in data/runs.csv: {len(R)}; left out with a reason: '
          f'{len(R) - len(counted)})')
    for part, n in sorted(Counter(r['part'] for r in counted).items(), key=lambda x: -x[1]):
        print(f'  {part:24} {n}')
    print(f'Configurations with counted runs: {len(C)}')

    cmp_ = comparisons()
    cnt = [x for x in cmp_ if x['counted']]
    print('\n== Teams vs the single agent at the same budget (tokens chart)')
    print(f'Team configurations compared: {len(cmp_)} ({sum(x["cell"].budget == 8 for x in cmp_)} at 8M, '
          f'{sum(x["cell"].budget == 32 for x in cmp_)} at 32M)')
    print(f'Counted (single agent spent >= 85%): {len(cnt)}; teams below the single agent: '
          f'{sum(x["gap"] < 0 for x in cnt)} of {len(cnt)}')
    for task in TASK:
        t = [x for x in cnt if x['cell'].task == task]
        print(f'  {TASK[task]:22} {sum(x["gap"] < 0 for x in t)} of {len(t)} below')
    for k in (2, 4, 8):
        g = [x['gap'] for x in cnt if x['cell'].k == k]
        print(f'  teams of {k}: median gap {st.median(g):+.1f}% over {len(g)} counted configurations')
    eight = [x for x in cnt if x['cell'].k == 8]
    med8 = st.median(x['gap'] for x in eight)
    print(f'Median gap, counted teams of 8: {med8:+.1f}%  -> teams of 8 get {100 + med8:.0f}% of the single agent\'s score')
    print('Hollow (single agent under 85%):')
    for x in cmp_:
        if not x['counted'] and x['cell'].k == 2 and x['cell'].setup == 'ttc':
            print(f"  {MODEL[x['cell'].model]}, {TASK[x['cell'].task]}: single agent spent {100 * x['ref'].spend:.0f}%")

    print('\n== Time (time chart)')
    tc = [x for x in cmp_ if x['time_counted']]
    t8 = [x for x in tc if x['cell'].k == 8]
    sp = [x['sooner'] for x in t8]
    print(f'Counted for time: {len(tc)} of {len(cmp_)}; all finished sooner: {all(x["sooner"] > 1 for x in tc)} '
          f'(least {min(x["sooner"] for x in tc):.2f}x)')
    print(f'Teams of 8: {len(t8)} counted, median {st.median(sp):.2f}x sooner (range {min(sp):.2f}x to {max(sp):.2f}x), '
          f'median gap {st.median(x["gap"] for x in t8):+.1f}%')
    for k in (2, 4):
        print(f'  teams of {k}: median {st.median(x["sooner"] for x in tc if x["cell"].k == k):.2f}x sooner')
    a8 = [x for x in cmp_ if x['cell'].k == 8]
    print('Footnote 4, every team of 8 (counted or not):')
    print(f'  range {min(x["sooner"] for x in a8):.2f}x to {max(x["sooner"] for x in a8):.2f}x; '
          f'below 3x: ' + ', '.join(f"{MODEL[x['cell'].model]} {SETUP[x['cell'].setup]} {TASK[x['cell'].task]} "
                                    f"{x['sooner']:.2f}x" for x in a8 if x['sooner'] < 3))
    print('  Claude Opus 5.5 single agents that hit the 240-hour cap: ' + ', '.join(
        f"{r['run']} ({r['active_hours']:.0f} h, {r['weighted_tokens_M']:.1f}M)" for r in counted
        if r['model'] == 'opus' and r['setup'] == 'solo' and 'time-cap' in r['flags'].split()))

    print('\n== Single agents that stopped early (8M, counted runs, mean share of budget spent)')
    for m in ('sonnet', 'opus'):
        for task in ('ahc014', 'ahc038', 'perf'):
            c = C[f'{task}/{m}/solo/1/8M']
            print(f'  {MODEL[m]:18} {TASK[task]:22} {100 * c.spend:.0f}%')
    o = [C[f'{t}/opus/solo/1/8M'].spend for t in ('ahc014', 'ahc038')]
    print(f'  Claude Opus 5.5 on the two AHC tasks together: {100 * st.mean(o):.0f}%')

    print('\n== Claude Opus 5.5 on AHC014 at 8M (first place = AtCoder performance 3389)')
    for setup in ('solo', 'ttc'):
        rs = [r for r in counted if (r['task'], r['model'], r['setup'], r['budget_M']) == ('ahc014', 'opus', setup, 8.0)]
        print(f'  {SETUP[setup]:16} {sum(r["final"] >= 3389 for r in rs)} of {len(rs)} runs at first place')
