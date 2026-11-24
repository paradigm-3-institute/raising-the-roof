"""What was run: models, setups, tasks and budgets, repeats per configuration, and Codex builds.

    python3 analysis/methodology.py
"""
from collections import Counter, defaultdict

from common import MODEL, SETUP, TASK, cells, runs

C = cells()
R = runs(counted_only=True)
SHORT = {'gitzig': 'git-to-zig', 'ahc014': 'AHC014', 'ahc038': 'AHC038', 'perf': 'take-home'}


def coverage(model, setups):
    by = defaultdict(set)
    for c in C.values():
        if c.model == model and c.setup in setups and (not c.variant or setups & {'solo_ttc', 'notalk'}) and not c.series:
            by[c.budget].add(c.task)
    return '; '.join(f'{b:g}M: ' + ', '.join(SHORT[t] for t in TASK if t in ts) for b, ts in sorted(by.items()))


if __name__ == '__main__':
    print('== Which model ran in which setup (counted runs; budgets and tasks)')
    for m in ('opus', 'sonnet', 'sol', 'terra', 'opus5', 'sonnet5', 'terra56', 'sol62', 'mixed', 'opussol'):
        print(f'  {MODEL[m]}')
        for label, setups in (('Single agent', {'solo'}), ('Restarted single agent', {'relay'}),
                              ('Single agent, protocol prompt', {'solo_ttc'}), ('Protocol teams', {'ttc'}),
                              ('Self-organized teams', {'board', 'peers', 'chat'}), ('Orchestrated teams', {'ultra'}),
                              ('No-talk pools', {'notalk'})):
            s = coverage(m, setups)
            if s:
                print(f'    {label:30} {s}')
    print('\n== Team sizes per setup')
    for s in ('ttc', 'board', 'peers', 'chat', 'ultra'):
        print(f'  {SETUP[s]} ({s}): ' + ', '.join(str(k) for k in sorted({c.k for c in C.values() if c.setup == s})))
    print('\n== Runs per configuration (pools of no-talk agents not included)')
    n = Counter(len(c.vals) for c in C.values() if c.setup != 'notalk')
    print('  ' + ', '.join(f'{k} runs: {v} configurations' for k, v in sorted(n.items())))
    for k in (2, 5, 10):
        print(f'  {k} runs: ' + ', '.join(sorted(c.id for c in C.values() if len(c.vals) == k and c.setup != 'notalk')))
    print('\n== Codex builds (counted GPT runs)')
    cb = Counter((r['codex'], SETUP[r['setup']]) for r in R if r['codex'])
    for (v, s), k in sorted(cb.items()):
        print(f'  {v:16} {s:52} {k}')
    print('\n== Caps')
    tc = [r for r in R if 'time-cap' in r['flags'].split()]
    print(f'  Runs that hit the cap on working time: {len(tc)}: ' + ', '.join(
        f"{r['run']} ({r['active_hours']:.0f} h)" for r in tc))
    print('\n== Weighted tokens: output + 0.23 x uncached input + 0.02 x all input')
    ok, bad = 0, []
    for r in runs():
        if r['input_tokens'] is None or not r['weighted_tokens_M']:
            continue
        w = (r['output_tokens'] + 0.23 * (r['input_tokens'] - r['cached_input_tokens']) + 0.02 * r['input_tokens']) / 1e6
        if abs(w - r['weighted_tokens_M']) <= 0.01 * r['weighted_tokens_M'] + 0.02:
            ok += 1
        else:
            bad.append(f"{r['run']} ({r['weighted_tokens_M']:.2f}M tracked, {w:.2f}M from the logs)")
    print(f'  Recomputed from the token columns: {ok} runs within 1% of the budget tracker; differ: {", ".join(bad)}')
