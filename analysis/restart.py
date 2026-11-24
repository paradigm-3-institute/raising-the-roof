"""The restarted single agent on git-to-zig: its finals against the single agent and the self-organized teams, λ
against both references, model calls, and what λ means for speed and tokens under Ord's model.

    python3 analysis/restart.py

The restarted single agent splits its budget into 8 sessions run one after another, each starting fresh with the
files the last one left. λ: analysis/lam.py. Model calls: data/runs.csv (from the agents' logs).
"""
import statistics as st

import lam as L
from common import MODEL, cells, cid, snapshots

C = cells()
SN = snapshots()


def mean_calls(c):
    v = [r['model_calls'] for r in c.runs if r['model_calls']]
    return st.mean(v) if v else None, len(v)


if __name__ == '__main__':
    for budget in (32, 8):
        print(f'== git-to-zig, {budget}M')
        for model in ('sol', 'terra'):
            solo = C[cid('gitzig', model, 'solo', 1, budget)]
            relay = C[cid('gitzig', model, 'relay', 8, budget)]
            team = C[cid('gitzig', model, 'board', 8, budget)]
            print(f'  {MODEL[model]}: restarted single agent {relay.mean:.1f}% (n = {len(relay.vals)}), '
                  f'self-organized team of 8 {team.mean:.1f}% (n = {len(team.vals)}), single agent {solo.mean:.1f}% '
                  f'(n = {len(solo.vals)})')
            print(f'    restarted vs team: {relay.mean - team.mean:+.1f} points; vs single agent: '
                  f'{relay.mean - solo.mean:+.1f} points')
            for k in ((16, 32) if budget == 32 and model == 'terra' else ()):
                t = C[cid('gitzig', model, 'board', k, budget)]
                print(f'    restarted vs team of {k} ({t.mean:.1f}%): {relay.mean - t.mean:+.1f} points')
            print(f'    Codex: single agent {sorted({r["codex"] for r in solo.runs})}, team '
                  f'{sorted({r["codex"] for r in team.runs})}, restarted {sorted({r["codex"] for r in relay.runs})}')
            ls, lr = L.vs_single(C, team), L.vs_restarted(C, team, SN)
            print(f'    λ, team of 8: vs single agent {ls["lam"]:.2f} [{ls["lo"]:.2f}, {ls["hi"]:.2f}]'
                  f'{" (extrapolated past 32M)" if ls["extrapolated"] else ""}; vs restarted single agent '
                  f'{lr["lam"]:.2f} [{lr["lo"]:.2f}, {lr["hi"]:.2f}] (the restarted agent reaches the team\'s mean at '
                  f'{lr["x_M"]:.2f}M)')
            sooner, tokens = L.ord_tradeoff(round(lr['lam'], 2), 8)
            print(f'    Ord\'s model at λ = {lr["lam"]:.2f}: a team of 8 finishes {sooner:.1f}x sooner for '
                  f'{tokens:.2f}x the tokens')
            cs, ns = mean_calls(solo)
            cr, nr = mean_calls(relay)
            print(f'    model calls per run: single agent {cs:,.0f} (n = {ns}), restarted {cr:,.0f} (n = {nr}): '
                  f'{100 * (cr / cs - 1):+.0f}%')
            if budget == 8:
                for k in (2, 4):
                    t = C[cid('gitzig', model, 'board', k, budget)]
                    x = L.vs_restarted(C, t, SN)
                    print(f'    λ, team of {k} ({t.mean:.1f}%) vs restarted: {x["lam"]:.2f} [{x["lo"]:.2f}, {x["hi"]:.2f}]')
        print()
