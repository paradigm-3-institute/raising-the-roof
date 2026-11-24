"""The communication appendix: how much a team gains from talking, against a no-talk team of the same size.

    python3 analysis/communication.py

At 8M in total. A no-talk team of k is the best of k agents that each had 8M/k and never talked: the exact mean of
the best of k over every k-subset of a pool (6 agents at 4M, 8 at 2M, 12 at 1M). Protocol teams are matched with
no-talk agents given the protocol prompt, self-organized teams with no-talk agents given the plain task prompt.
Gaps are % of the same model's single agent with 8M (take-home as speed); "talking added" = team gap - no-talk gap,
in points of that single agent's score. AHC014 is compared in raw score per case (AtCoder performance stops at 3389).
"""
import statistics as st

from common import MODEL, TASK, best_of_k, cells, cid, gap

C = cells()
PAIRS = [('ttc', 'protocol prompt', ['opus', 'sonnet', 'sol']), ('board', 'plain prompt', ['sol', 'terra'])]
NAME = {'ttc': 'Protocol', 'board': 'Self-organized'}


def value(c, task):
    if task != 'ahc014':
        return c.mean
    raws = [r['raw_per_case_M'] for r in c.runs]
    return best_of_k(raws, c.k) if c.setup == 'notalk' else st.mean(raws)


def rows():
    out = []
    for task in TASK:
        for setup, prompt, models in PAIRS:
            for m in models:
                solo = C.get(cid(task, m, 'solo', 1, 8))
                for k in (2, 4, 8):
                    team, nt = C.get(cid(task, m, setup, k, 8)), C.get(cid(task, m, 'notalk', k, 8, prompt))
                    if not (solo and team and nt) or nt.mean is None:
                        continue
                    s = value(solo, task)
                    gt, gn = 100 * gap(task, value(team, task), s), 100 * gap(task, value(nt, task), s)
                    out.append({'task': task, 'setup': setup, 'model': m, 'k': k, 'team': gt, 'notalk': gn,
                                'talk': gt - gn, 'solo_spend': solo.spend, 'pool': len(nt.vals), 'n': len(team.vals)})
    return out


if __name__ == '__main__':
    R = rows()
    for task in TASK:
        print(f'== {TASK[task]}')
        for x in [x for x in R if x['task'] == task]:
            print(f"  {NAME[x['setup']]:15} {MODEL[x['model']]:18} k={x['k']}  team {x['team']:+6.1f}%  no-talk "
                  f"{x['notalk']:+6.1f}% (pool {x['pool']})  talking added {x['talk']:+5.1f} points"
                  + ('' if x['solo_spend'] >= 0.85 else f"   (single agent spent {100 * x['solo_spend']:.0f}%)"))
    print('\n== The appendix and main text')
    g = [x for x in R if x['task'] == 'gitzig']
    print(f"git-to-zig, teams of 4 and 8: talking added {min(x['talk'] for x in g if x['k'] > 2):+.0f} to "
          f"{max(x['talk'] for x in g if x['k'] > 2):+.0f} points; teams of 2: "
          + ', '.join(f"{x['talk']:+.0f}" for x in g if x['k'] == 2))
    s8 = {x['setup']: x['talk'] for x in g if x['model'] == 'sol' and x['k'] == 8}
    print(f"GPT-6.1 Sol, teams of 8: self-organized {s8['board']:+.0f}, protocol {s8['ttc']:+.0f}")
    ahc = [x for x in R if x['task'] in ('ahc014', 'ahc038')]
    gpt = [x['talk'] for x in ahc if x['model'] in ('sol', 'terra')]
    print(f"AHC, GPT teams: talking added {min(gpt):+.1f} to {max(gpt):+.1f} points ({sum(v < 0 for v in gpt)} of "
          f"{len(gpt)} negative); Claude Opus 5.5: " + ', '.join(
              f"{x['talk']:+.1f}" for x in ahc if x['model'] == 'opus'))
    p = [x['talk'] for x in R if x['task'] == 'perf']
    print(f"Take-home: talking added {min(p):+.1f} to {max(p):+.1f} points")
    n8 = [x for x in R if x['k'] == 8]
    gz = [x['notalk'] for x in n8 if x['task'] == 'gitzig']
    print(f"No-talk teams of 8 vs the single agent: git-to-zig {min(gz):+.0f}% to {max(gz):+.0f}%; other tasks "
          f"{min(x['notalk'] for x in n8 if x['task'] != 'gitzig'):+.0f}% to "
          f"{max(x['notalk'] for x in n8 if x['task'] != 'gitzig'):+.0f}%")
    p8 = [x for x in n8 if x['task'] == 'gitzig' and x['setup'] == 'ttc']
    print('git-to-zig protocol teams of 8: ' + '; '.join(
        f"{MODEL[x['model']]} no-talk {x['notalk']:+.0f}%, talking added {x['talk']:+.0f}, team {x['team']:+.0f}%"
        for x in p8))
    print('git-to-zig GPT self-organized teams of 8: ' + '; '.join(
        f"{MODEL[x['model']]} team {x['team']:+.1f}%" for x in n8 if x['task'] == 'gitzig' and x['setup'] == 'board'))
    print(f"Every comparison: talking added {min(x['talk'] for x in R):+.0f} to {max(x['talk'] for x in R):+.0f} points "
          f"({len(R)} comparisons)")
