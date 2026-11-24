"""The numbers behind chart.html (next to this file) and ../time/chart.html, a matched pair drawn by shared.js.

    python3 analysis/figures/tokens/tally.py            prints the tokens chart's DATA as JSON
    python3 analysis/figures/tokens/tally.py --write    also copies it into chart.html, between the DATA markers
    python3 analysis/figures/time/tally.py [--write]    the same for the time chart (it runs this file)

Reads data/runs.csv and data/score_events.csv through analysis/common.py.

Each chart has two parts:
  curves  GPT-6.1 Sol at 8M weighted tokens: the single agent and teams of 8 in each team setup (protocol 'ttc',
          self-organized 'board', orchestrated 'ultra'), one panel per task. Best score so far against weighted tokens
          (tokens chart) or active hours (time chart).
  dots    every comparison, 82 in all: teams of 2, 4 and 8 of Claude Opus 5.5, Claude Sonnet 5.5, GPT-6.1 Sol and
          GPT-6.1 Terra in the three setups at 8M on the four tasks (self-organized: 'board' for GPT, 'peers' for
          Claude), plus git-to-zig's GPT self-organized teams at 32M (drawn lighter). Each is placed by its gap to the
          same model's single agent at the same budget.

Curves
  A run's score events carry the active hour (waits removed) and the weighted tokens spent (team total) when each
  score counted. Per run: best score so far; a finished run holds its last value; on git-to-zig the final grade
  (runs.csv final) replaces the in-run best at the run's end. Per cell: mean over its counted runs, from the first x at
  which every run has a score. Tokens are capped at the 8M budget (the last scores land at most 0.06M past it), so
  every curve ends at 8M; hours are drawn to the hour the cell's last run ended.
  y: % of the single agent's mean final, in the curve's own units, up is better: 100 v / S on git-to-zig (tests
  passed) and AHC014 (raw score), 100 S / v on AHC038 (raw score) and the take-home (cycles). AHC curves are in-run
  public test-case scores; the dots use the final AtCoder performance, so an AHC curve's end and its dot differ.

Dots
  gap     common.vs_single: team / single agent - 1 (take-home: single cycles / team cycles - 1), in %.
  tokens  hollow = not counted (common.compare): the single agent's runs spent under 85% of the budget on average.
  time    sooner = the single agent's median active hours / the team's (median: with 3 runs a mean follows the slowest
          run). Hollow when most of the single agent's runs carry the flag 'waited' (half or more of its time inside
          tool calls of 10 minutes or more, mostly sleeping while its own background jobs ran): every Claude single
          agent on AHC014, AHC038 and the take-home.
"""
import argparse
import csv
import json
import statistics as st
import sys
from bisect import bisect_right
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common  # noqa: E402

TASKS = ('gitzig', 'ahc014', 'ahc038', 'perf')
HIGHER = {'gitzig': True, 'ahc014': True, 'ahc038': False, 'perf': False}   # in the curves' units
MODELS = ('opus', 'sonnet', 'sol', 'terra')
TEAM_SETUPS = ('ttc', 'board', 'peers', 'ultra')
KIT = {'solo': 'single', 'ttc': 'structured', 'board': 'selforg', 'peers': 'selforg', 'ultra': 'orchestrated'}
CURVE_CELLS = (('solo', 1), ('ttc', 8), ('board', 8), ('ultra', 8))
CURVE_MODEL, BUDGET = 'sol', 8
CAP = 8.0      # the tokens axis ends at the budget (M)
SMALL = 10.0   # a "small loss": less than this many % below the single agent


# ------------------------------------------------------------------------------------------------ curves
def score_events():
    """run -> [(active hours, weighted tokens, score)]"""
    out = {}
    for r in csv.DictReader(open(common.DATA / 'score_events.csv', encoding='utf-8')):
        out.setdefault(r['run'], []).append((float(r['active_hours']), float(r['weighted_tokens']), float(r['score'])))
    return out


def run_steps(r, events, axis):
    """best-so-far breakpoints [[x, value]]; x = active hours or weighted tokens (M) when the score counted"""
    task = r['task']
    # sort by x, then by hour, so scores at equal token counts keep their time order
    pts = sorted((h if axis == 'hours' else tok * 1e-6, h, v) for h, tok, v in events)
    out, best = [], None
    for x, _, v in pts:
        if axis == 'tokens':
            x = min(x, CAP)
        if best is None or (v > best if HIGHER[task] else v < best):
            best = v
            if out and abs(out[-1][0] - x) < 1e-12:
                out[-1] = [x, v]
            else:
                out.append([x, v])
    if task == 'gitzig' and r['final'] is not None:
        end = r['active_hours'] if axis == 'hours' else min(r['weighted_tokens_M'], CAP)
        out = [p for p in out if p[0] < end] + [[end, r['final']]]
    return out


def val_at(steps, x):
    i = bisect_right([p[0] for p in steps], x) - 1
    return steps[i][1] if i >= 0 else None


def mean_curve(cell, events, axis):
    steps = [run_steps(r, events[r['run']], axis) for r in cell.runs]
    out = []
    for x in sorted({p[0] for s in steps for p in s}):
        v = [val_at(s, x) for s in steps]
        if any(y is None for y in v):
            continue
        m = st.mean(v)
        if not out or abs(m - out[-1][1]) > 1e-12:
            out.append([x, m])
    end = CAP if axis == 'tokens' else max(r['active_hours'] for r in cell.runs)
    if out[-1][0] < end:
        out.append([end, out[-1][1]])
    final = st.mean(s[-1][1] for s in steps)
    assert abs(out[-1][1] - final) < 1e-9, (cell.id, axis)
    return out, final


def pct(task, v, S):
    return 100 * v / S if HIGHER[task] else 100 * S / v


def thin(curve, eps=0.05):
    """drop breakpoints that move the curve by under eps points (keeps the first and last)"""
    out = [curve[0]]
    for x, y in curve[1:-1]:
        if abs(y - out[-1][1]) >= eps:
            out.append([x, y])
    out.append(curve[-1])
    return out


def curves(C, axis):
    events = score_events()
    out = {}
    for task in TASKS:
        cs = [C[common.cid(task, CURVE_MODEL, s, k, BUDGET)] for s, k in CURVE_CELLS]
        S = mean_curve(cs[0], events, axis)[1]   # the single agent's mean final, in the curve's units
        out[task] = []
        for (s, _), cell in zip(CURVE_CELLS, cs):
            c, fin = mean_curve(cell, events, axis)
            out[task].append(dict(
                setup=KIT[s], n=len(cell.runs), final=round(pct(task, fin, S), 3),
                end=CAP if axis == 'tokens' else round(max(r['active_hours'] for r in cell.runs), 4),
                pts=[[round(x, 5), round(y, 3)] for x, y in thin([[x, pct(task, v, S)] for x, v in c])]))
    return out


# ------------------------------------------------------------------------------------------------ dots
def in_scope(c):
    return (c.task in TASKS and c.model in MODELS and c.setup in TEAM_SETUPS and c.k in (2, 4, 8)
            and not c.variant and not c.series
            and (c.budget == BUDGET or (c.budget == 32 and c.task == 'gitzig' and c.setup == 'board')))


def dots(C):
    out = []
    for cell in sorted((c for c in C.values() if in_scope(c)), key=lambda c: (c.task, c.model, c.setup, c.k, c.budget)):
        cmp, ref = common.vs_single(C, cell)
        assert cmp, cell.id
        waited = sum('waited' in r['flags'].split() for r in ref.runs)
        out.append(dict(
            task=cell.task, model=cell.model, setup=cell.setup, kit=KIT[cell.setup], k=cell.k, budget=int(cell.budget),
            gap=round(cmp['gap_pct'], 2), exception=cell.budget == 32,
            curve=cell.model == CURVE_MODEL and cell.k == 8 and cell.budget == BUDGET,
            tok_counted=cmp['counted'], time_counted=2 * waited <= len(ref.runs),
            sooner=round(ref.hours_median / cell.hours_median, 4)))
    return out


# ------------------------------------------------------------------------------------------------ summaries
def tokens_summary(D):
    TC = [d for d in D if d['tok_counted']]
    return dict(
        plotted=len(D), counted=len(TC), below=sum(d['gap'] < 0 for d in TC), hollow=len(D) - len(TC),
        by_k={k: dict(n=sum(d['k'] == k for d in TC), below=sum(d['k'] == k and d['gap'] < 0 for d in TC),
                      median_gap=st.median([d['gap'] for d in TC if d['k'] == k])) for k in (2, 4, 8)},
        median_gap=st.median([d['gap'] for d in TC]))


def time_summary(D):
    HC = [d for d in D if d['time_counted']]
    not_sonnet_perf = [d['sooner'] for d in D if d['k'] == 8 and not (d['task'] == 'perf' and d['model'] == 'sonnet')]
    return dict(
        plotted=len(D), counted=len(HC), hollow=len(D) - len(HC),
        sooner=sum(d['sooner'] > 1 for d in HC), below=sum(d['gap'] < 0 for d in HC),
        small_loss=sum(-SMALL < d['gap'] < 0 for d in HC), loss_10_or_more=sum(d['gap'] <= -SMALL for d in HC),
        at_or_above=sum(d['gap'] >= 0 for d in HC), within_10=sum(d['gap'] > -SMALL for d in HC),
        median_sooner=st.median([d['sooner'] for d in HC]), median_gap=st.median([d['gap'] for d in HC]),
        by_k={k: dict(n=sum(d['k'] == k for d in HC),
                      median_sooner=st.median([d['sooner'] for d in HC if d['k'] == k]),
                      min_sooner=min(d['sooner'] for d in HC if d['k'] == k),
                      max_sooner=max(d['sooner'] for d in HC if d['k'] == k),
                      median_gap=st.median([d['gap'] for d in HC if d['k'] == k]),
                      below=sum(d['k'] == k and d['gap'] < 0 for d in HC),
                      within_10=sum(d['k'] == k and d['gap'] > -SMALL for d in HC)) for k in (2, 4, 8)},
        k8_all_range_excl_sonnet_perf=[min(not_sonnet_perf), max(not_sonnet_perf)],
        sonnet_perf_8=next(d['sooner'] for d in D if d['k'] == 8 and d['task'] == 'perf' and d['model'] == 'sonnet'))


# ------------------------------------------------------------------------------------------------ DATA
DOT_KEYS = ('task', 'model', 'setup', 'kit', 'k', 'budget', 'gap', 'exception', 'curve')


def data(chart):
    C = common.cells()
    D = dots(C)
    if chart == 'tokens':
        return dict(summary=tokens_summary(D), curves=curves(C, 'tokens'),
                    dots=[{**{k: d[k] for k in DOT_KEYS}, 'counted': d['tok_counted']} for d in D])
    return dict(summary=time_summary(D), curves=curves(C, 'hours'),
                dots=[{**{k: d[k] for k in DOT_KEYS}, 'counted': d['time_counted'], 'sooner': d['sooner']} for d in D])


def main(chart, html):
    ap = argparse.ArgumentParser(description=f"Print the {chart} chart's DATA as JSON.")
    ap.add_argument('--write', action='store_true', help='also copy DATA into chart.html, between the DATA markers')
    a = ap.parse_args()
    block = data(chart)
    print(json.dumps(block, indent=1))
    if a.write:
        src = html.read_text(encoding='utf-8')
        b, e = '// DATA-BEGIN (written by tally.py)\n', '// DATA-END\n'
        i, j = src.index(b) + len(b), src.index(e)
        html.write_text(src[:i] + 'const DATA = ' + json.dumps(block, separators=(',', ':')) + ';\n' + src[j:],
                        encoding='utf-8')
        print(f'DATA copied into {html.name}', file=sys.stderr)


if __name__ == '__main__':
    main('tokens', Path(__file__).resolve().parent / 'chart.html')
