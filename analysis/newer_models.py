"""Older and newer models: each model's teams against its own single agent at 8M, and GPT-6.2 Sol on git-to-zig.

    python3 analysis/newer_models.py

Pairs: Claude Opus 5 -> Opus 5.5, Claude Sonnet 5 -> Sonnet 5.5, GPT-5.6 Terra -> GPT-6.1 Terra, GPT-6.1 Sol ->
GPT-6.2 Sol. Gap = team / own single agent - 1 (common.compare); "not counted" when that single agent spent under 85%.
"""
from common import MODEL, SETUP, TASK, cells, cid, vs_single

C = cells()
PAIRS = [('opus5', 'opus'), ('sonnet5', 'sonnet'), ('terra56', 'terra'), ('sol', 'sol62')]


def line(c):
    g, ref = vs_single(C, c)
    flag = '' if g['counted'] else '  (not counted: single agent spent under 85%)'
    return (f'{c.mean:8.1f} (n = {len(c.vals)})  gap {g["gap_pct"]:+6.1f}% [{g["gap_lo"]:+.1f}, {g["gap_hi"]:+.1f}]{flag}'
            if g['gap_lo'] is not None else f'{c.mean:8.1f} (n = {len(c.vals)})  gap {g["gap_pct"]:+6.1f}%{flag}')


if __name__ == '__main__':
    for old, new in PAIRS:
        print(f'== {MODEL[old]} -> {MODEL[new]}')
        for task in TASK:
            so, sn = C.get(cid(task, old, 'solo', 1, 8)), C.get(cid(task, new, 'solo', 1, 8))
            om = old if old != 'sol' else new
            if not any(c.model == om and c.task == task for c in C.values()):
                continue
            print(f'  {TASK[task]}')
            for m, s in ((old, so), (new, sn)):
                if s:
                    print(f'    {MODEL[m]:17} single agent {s.mean:8.1f} (n = {len(s.vals)}), spent {100 * s.spend:.0f}% '
                          f'of its budget')
            for setup in ('ttc', 'board', 'peers'):
                for k in (2, 4, 8):
                    co = C.get(cid(task, old, setup, k, 8))
                    cn = C.get(cid(task, new, setup, k, 8))
                    if not co or not cn or not (co.model == om or cn.model == om):
                        continue
                    print(f'    {SETUP[setup]}, {k} agents')
                    print(f'      {MODEL[old]:17}{line(co)}')
                    print(f'      {MODEL[new]:17}{line(cn)}')
        print()
    t = C['gitzig/terra56/solo/1/8M']
    print('GPT-5.6 Terra single agents on git-to-zig: ' + ', '.join(
        f"{r['weighted_tokens_M']:.1f}M of 8M ({r['end_reason']})" for r in t.runs))
