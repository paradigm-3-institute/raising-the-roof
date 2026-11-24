"""The model-diversity appendix: teams that mix Claude and GPT-6.1 Sol agents, at 8M weighted tokens.

    python3 analysis/diversity.py

Each team is compared with the best single agent on its task: git-to-zig GPT-6.1 Sol (tests passed), AHC014 Claude
Opus 5.5 (raw private score per case, since AtCoder performance stops at 3389), take-home GPT-6.1 Sol (speed:
reference cycles / team cycles - 1). Claude-only and Sol-only teams use the same setup as the mixed team; for
self-organized teams that means direct messages (Claude), the message board (Sol) and SwarmChat (mixed).
"""
import statistics as st

from common import MODEL, cells, cid, gap, runs

C = cells()
R = runs(counted_only=True)
GROUPS = [('gitzig', 'Protocol teams', 'sonnet', {'claude': 'ttc', 'sol': 'ttc', 'mixed': ('mixed', 'ttc')}),
          ('gitzig', 'Self-organized teams', 'sonnet', {'claude': 'peers', 'sol': 'board', 'mixed': ('mixed', 'chat')}),
          ('ahc014', 'Protocol teams', 'opus', {'claude': 'ttc', 'sol': 'ttc', 'mixed': ('opussol', 'ttc')}),
          ('perf', 'Protocol teams', 'opus', {'claude': 'ttc', 'sol': 'ttc', 'mixed': ('opussol', 'ttc')})]


def value(c, task):
    """The cell's mean in the chart's unit: raw score per case on AHC014, else the final."""
    if task == 'ahc014':
        return st.mean(r['raw_per_case_M'] for r in c.runs)
    return c.mean


if __name__ == '__main__':
    print(f"git-to-zig single agents: GPT-6.1 Sol {C['gitzig/sol/solo/1/8M'].mean:.1f}%, Claude Sonnet 5.5 "
          f"{C['gitzig/sonnet/solo/1/8M'].mean:.1f}%, Claude Opus 5.5 {C['gitzig/opus/solo/1/8M'].mean:.1f}%")
    print('git-to-zig single agents given the protocol prompt: ' + ', '.join(
        f"{MODEL[m]} {C[f'gitzig/{m}/solo_ttc/1/8M/protocol-prompt'].mean:.1f}%" for m in ('opus', 'sol')))
    print('git-to-zig, self-organized vs protocol teams at 8M:')
    for m in ('opus', 'sonnet', 'sol'):
        so = 'peers' if m != 'sol' else 'board'
        print(f'  {MODEL[m]:18}' + '  '.join(
            f"{k}: {C[cid('gitzig', m, so, k, 8)].mean:.1f} vs {C[cid('gitzig', m, 'ttc', k, 8)].mean:.1f}"
            for k in (2, 4, 8)))
    places = []
    for task, label, claude, s in GROUPS:
        best = {'gitzig': 'sol', 'ahc014': 'opus', 'perf': 'sol'}[task]
        ref = C[cid(task, best, 'solo', 1, 8)]
        rv = value(ref, task)
        print(f'\n{task} {label}: reference = best single agent, {MODEL[best]} ({rv:.4g})')
        for k in (2, 4, 8):
            cs = {'claude': C[cid(task, claude, s['claude'], k, 8)], 'sol': C[cid(task, 'sol', s['sol'], k, 8)],
                  'mixed': C[cid(task, s['mixed'][0], s['mixed'][1], k, 8)]}
            g = {w: 100 * gap(task, value(c, task), rv) for w, c in cs.items()}
            lo, hi = sorted((g['claude'], g['sol']))
            place = 'between' if lo <= g['mixed'] <= hi else ('above both' if g['mixed'] > hi else 'below both')
            places.append(place)
            print(f"  {k} agents: Claude-only {g['claude']:+.1f}%, Sol-only {g['sol']:+.1f}%, mixed {g['mixed']:+.1f}% "
                  f"(n = {len(cs['mixed'].vals)}) -> {place}")
    print(f"\nMixed team between its Claude-only and Sol-only teams: {places.count('between')} of {len(places)}; "
          f"above both {places.count('above both')}, below both {places.count('below both')}")
    a14 = [r for r in R if r['task'] == 'ahc014' and r['budget_M'] == 8 and (
        (r['setup'] == 'solo' and r['model'] == 'opus') or (r['setup'] == 'ttc' and r['model'] in ('opus', 'sol', 'opussol')))]
    print(f'AHC014 runs in this comparison at 3389 (first place): {sum(r["final"] >= 3389 for r in a14)} of {len(a14)}')
    o = C['ahc014/opus/solo/1/8M']
    print('Claude Opus 5.5 single agents on AHC014: ' + ', '.join(
        f"{r['weighted_tokens_M']:.1f}M of 8M ({r['end_reason']})" for r in o.runs))
