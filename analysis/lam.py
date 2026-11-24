"""Ord's lambda: how much of a team's combined budget turns into progress.

A team of N agents, each with budget b, is matched with one agent that would need budget X to reach the team's mean
score; lambda = ln(X / b) / ln N. lambda = 1 means the team does as well as one agent with N times b; below 1, the
team gets less from its combined budget than a single agent would.

Two references:
  single agent             X from the single agent's mean finals at the budgets we ran (2M, 8M, 32M), with score
                           linear in log budget between neighbouring budgets, extended past the ends with the nearest
                           segment (flagged as extrapolated).
  restarted single agent   X = the first point where the restarted agent's mean curve (best graded snapshot so far,
                           averaged over its runs) reaches the team's mean final. Grid: 5,000 tokens at 8M, 10,000 at
                           32M.
Intervals: bootstrap percentile 95%, resampling runs in every arm.
"""
import math
import random
import statistics as st

import common as C

NB_LAM = 4000


def _inv(points, s):
    """X (budget) at which the single agent reaches s; points = [(budget, mean score)], score up = better."""
    pts = [(math.log(b), m) for b, m in sorted(points)]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if min(y0, y1) <= s <= max(y0, y1) and y1 != y0:
            return math.exp(x0 + (s - y0) * (x1 - x0) / (y1 - y0)), False
    (x0, y0), (x1, y1) = (pts[0], pts[1]) if s < pts[0][1] else (pts[-2], pts[-1])
    if y1 <= y0:
        return None, True
    return math.exp(x0 + (s - y0) * (x1 - x0) / (y1 - y0)), True


def vs_single(cells, team, nb=NB_LAM):
    """lambda of a team cell against the same model's single agents (all budgets run), with a bootstrap interval."""
    solos = {c.budget: c for c in cells.values() if (c.task, c.model, c.setup, c.variant, c.series) ==
             (team.task, team.model, 'solo', '', '') and c.vals}
    if len(solos) < 2 or team.mean is None or team.k < 2:
        return None
    sign = -1 if team.task in C.LOWER_BETTER else 1
    b = team.budget / team.k
    X, ext = _inv([(B, sign * c.mean) for B, c in solos.items()], sign * team.mean)
    if X is None:
        return None
    rng = random.Random(C.seed_of(('lambda', team.key)))
    draws = []
    for _ in range(nb):
        pts = [(B, sign * st.mean(rng.choices(c.vals, k=len(c.vals)))) for B, c in solos.items()]
        if not all(y0 < y1 for (_, y0), (_, y1) in zip(sorted(pts), sorted(pts)[1:])):
            continue
        X2, _ = _inv(pts, sign * st.mean(rng.choices(team.vals, k=len(team.vals))))
        if X2:
            draws.append(C.lam(X2, b, team.k))
    draws.sort()
    lo, hi = (draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]) if len(draws) > 100 else (None, None)
    return {'vs': 'Single agent', 'lam': C.lam(X, b, team.k), 'lo': lo, 'hi': hi, 'x_M': X, 'extrapolated': ext,
            'single_budgets': sorted(solos)}


def grid_for(budget_M):
    step = 5_000 if budget_M <= 8 else 10_000
    return list(range(0, int(budget_M * 1e6) + 1, step))


def relay_curves(relay, snaps, grid):
    return [C.best_curve(snaps[r['run']], grid) for r in relay.runs if r['run'] in snaps]


def vs_restarted(cells, team, snaps=None, nb=NB_LAM):
    """lambda of a git-to-zig team cell against the restarted single agent at the same budget."""
    relay = cells.get(C.cid(team.task, team.model, 'relay', 8, team.budget))
    if not relay or team.mean is None or team.k < 2:
        return None
    snaps = snaps if snaps is not None else C.snapshots()
    grid = grid_for(team.budget)
    curves = relay_curves(relay, snaps, grid)
    b = team.budget * 1e6 / team.k
    X = C.x_star(curves, grid, team.mean)
    if X is None or X <= 0:
        return {'vs': 'Restarted single agent', 'lam': None, 'note': 'team above the restarted agent at every point'}
    rng = random.Random(C.seed_of(('lambda-restarted', team.key)))
    draws, above = [], 0
    for _ in range(nb):
        cs = [rng.choice(curves) for _ in curves]
        s = st.mean(rng.choices(team.vals, k=len(team.vals)))
        X2 = C.x_star(cs, grid, s)
        if X2 is None:
            above += 1
        elif X2 > 0:
            draws.append(C.lam(X2, b, team.k))
    draws.sort()
    lo, hi = (draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]) if draws else (None, None)
    return {'vs': 'Restarted single agent', 'lam': C.lam(X, b, team.k), 'lo': lo, 'hi': hi, 'x_M': X / 1e6,
            'relay_n': len(curves), 'share_above': above / nb}


def ord_tradeoff(lam, n):
    """Under Ord's model a team of n finishes n**lam times sooner and spends n**(1 - lam) times the tokens."""
    return n ** lam, n ** (1 - lam)
