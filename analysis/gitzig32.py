"""git-to-zig at 32M weighted tokens: GPT-6.1's self-organized teams against the single agent, teams of 16 and 32,
the reruns on the patched Codex build, and the best of 8 single agents each scored after 4M.

    python3 analysis/gitzig32.py

Intervals for the gap in points: bootstrap percentile 95%, runs resampled within each arm, 10,000 draws.
Best of 8 at 4M: each 32M single agent is scored by its last graded in-run snapshot at or before 4M weighted tokens;
the value is the mean of the best of 8 over every 8-of-10 subset of the single agents.
"""
import statistics as st

from common import MODEL, best_of_k, boot_points, cells, cid, seed_of, snapshots

C = cells()


def cell(model, setup, k, series=''):
    return C.get(cid('gitzig', model, setup, k, 32, '', series))


def score_at(pts, x):
    """Last graded snapshot at or before x tokens (by tokens), and the best one up to x."""
    before = sorted(p for p in pts if p[0] <= x)
    return (before[-1][1] if before else 0.0), max((s for _, s in before), default=0.0)


if __name__ == '__main__':
    snaps = snapshots()
    for model in ('sol', 'terra'):
        solo = cell(model, 'solo', 1)
        print(f'== {MODEL[model]}, git-to-zig, 32M')
        print(f'  Single agent: {solo.mean:.1f}% (n = {len(solo.vals)})')
        for k in (2, 4, 8, 16, 32):
            for series in ('', 'rerun'):
                c = cell(model, 'board', k, series)
                if not c:
                    continue
                lo, hi = boot_points(c.vals, solo.vals, 'gitzig', seed_of(('points', c.key, solo.key)))
                codex = sorted({r['codex'] for r in c.runs})
                label = f'Self-organized team of {k}' + (' (rerun)' if series else '')
                print(f'  {label:34} {c.mean:5.1f}% (n = {len(c.vals):2})  gap {c.mean - solo.mean:+5.1f} points '
                      f'[{lo:+.1f}, {hi:+.1f}]   Codex {", ".join(codex)}')
        at4 = [score_at(snaps[r['run']], 4e6) for r in solo.runs]
        last, best = [a for a, _ in at4], [b for _, b in at4]
        print(f'  Single agents after 4M: mean {st.mean(last):.1f}% (last snapshot), {st.mean(best):.1f}% (best so far)')
        print(f'  Best of 8 of them: {best_of_k(last, 8):.1f}% (last snapshot), {best_of_k(best, 8):.1f}% '
              f'(best so far)   from {len(last)} runs')
        print()
