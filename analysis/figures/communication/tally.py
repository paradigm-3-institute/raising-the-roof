"""Communication chart tally: how much does talking add, beyond splitting the budget?
(appendix "How much does communication add?")

Reads data/runs.csv (counted runs) through analysis/common.py. For each task, model, setup and team size k (2, 4, 8),
at 8M weighted tokens in total:
  single   the same model's plain single agent with the full 8M (setup solo, k 1, no variant)
  no-talk  k independent agents with 8M/k each, scored on the best one: the exact mean of the best of k over every
           k-subset of the pool (6 agents at 4M for k = 2, 8 at 2M for k = 4, 12 at 1M for k = 8)
  team     the talking team: protocol teams (setup ttc) against no-talk agents given the protocol prompt,
           self-organized teams (setup board, GPT-6.1 message board) against no-talk agents given the plain prompt
Gaps are % of the single agent's mean: config / single - 1 (take-home as speed: single cycles / config cycles - 1).
"Talking adds" = team gap - no-talk gap, in points of the single agent's score (both sides share the denominator).
Intervals: bootstrap percentile 95%, 10,000 draws, runs resampled within each arm, seeded per comparison with
common.seed_of (seed 1118 + sha1 of the pair): the no-talk and team gaps are common.compare(); in raw score the same
draws with the seed + 7; the talking-adds interval resamples all three arms (team, pool, single) with its own seed.
Not counted: a gap against a single agent whose runs spent under 85% of their budget on average. That rule only
touches comparisons with the single agent; team vs no-talk is counted when both of those spent their budget.
AHC014 is also tallied in raw private score per case (millions), because AtCoder performance stops at 3389; the chart
uses raw score there.
The chart's *: an AHC comparison whose no-talk pool has only in-run grades (no later re-grade).

Prints the chart's DATA as JSON on stdout and every comparison, with the numbers the post quotes, on stderr.
--write also pastes DATA into chart.html (between DATA:BEGIN and DATA:END).
Run from the repo root: python3 analysis/figures/communication/tally.py [--write]
"""
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common as cm  # noqa: E402

TASKS = ['gitzig', 'ahc014', 'ahc038', 'perf']
PAIRS = [('ttc', 'protocol prompt', ['opus', 'sonnet', 'sol']), ('board', 'plain prompt', ['sol', 'terra'])]
KS = [2, 4, 8]
BUDGET = 8
LOWER = cm.LOWER_BETTER
C = cm.cells()


def cell(task, model, setup, k, variant=''):
    c = C.get(cm.cid(task, model, setup, k, BUDGET, variant))
    if not c or not c.runs:
        return None
    rs = c.runs
    return {'cell': c, 'key': c.key, 'id': c.id, 'task': task, 'k': k, 'setup': setup,
            'runs': [r['run'] for r in rs], 'vals': c.vals,
            'raw': [r['raw_per_case_M'] for r in rs if r['raw_per_case_M'] is not None],
            'spend': c.spend, 'src': sorted({r['final_source'] for r in rs}),
            'at_ceiling': sum(r['final'] >= cm.CEILING for r in rs) if task == 'ahc014' else None}


def stat(vals, task, kb=None):
    return cm.best_of_k(vals, kb, task in LOWER) if kb else st.mean(vals)


def compare(c, ref, field='vals'):
    """gap of c to ref in %, with its interval: common.compare() in the chart's unit, the same draws (seed + 7) in raw"""
    if field == 'vals':
        g = cm.compare(c['cell'], ref['cell'])
        return {'gap': g['gap_pct'], 'lo': g['gap_lo'], 'hi': g['gap_hi']}
    task, kb = c['task'], (c['k'] if c['setup'] == 'notalk' else None)
    a, b = c[field], ref[field]
    rng = random.Random(cm.seed_of((c['key'], ref['key'])) + 7)
    g = cm.gap(task, stat(a, task, kb), st.mean(b))
    lo, hi = cm.boot(rng, [a, b], lambda x, y: cm.gap(task, stat(x, task, kb), st.mean(y)))
    return {'gap': 100 * g, 'lo': 100 * lo, 'hi': 100 * hi}


def talk_adds(team, nt, ref, field='vals'):
    task, k = team['task'], nt['k']
    rng = random.Random(cm.seed_of((team['key'], nt['key'], ref['key'], field, 'talk')))
    f = lambda a, p, b: 100 * (cm.gap(task, st.mean(a), st.mean(b))  # noqa: E731
                               - cm.gap(task, cm.best_of_k(p, k, task in LOWER), st.mean(b)))
    v = f(team[field], nt[field], ref[field])
    lo, hi = cm.boot(rng, [team[field], nt[field], ref[field]], f)
    return {'pts': v, 'lo': lo, 'hi': hi}


rows = []
for task in TASKS:
    for setup, prompt, models in PAIRS:
        for m in models:
            ref = cell(task, m, 'solo', 1)
            for k in KS:
                team, nt = cell(task, m, setup, k), cell(task, m, 'notalk', k, prompt)
                if not team or not nt:
                    if team or nt:
                        rows.append({'task': task, 'setup': setup, 'model': m, 'k': k, 'missing': 'no no-talk pool'
                                     if team else 'no team', 'team_n': len(team['vals']) if team else 0})
                    continue
                assert len(nt['vals']) == cm.POOL[k], (task, m, k, len(nt['vals']))
                row = {'task': task, 'setup': setup, 'model': m, 'k': k, 'prompt': prompt,
                       'single': {'id': ref['id'], 'n': len(ref['vals']), 'mean': st.mean(ref['vals']),
                                  'spend': ref['spend'], 'counted': ref['spend'] >= cm.SPEND_OK, 'src': ref['src'],
                                  'vals': ref['vals']},
                       'notalk': {'id': nt['id'], 'n': len(nt['vals']), 'best_of_k': stat(nt['vals'], task, k),
                                  'spend': nt['spend'], 'src': nt['src'], **compare(nt, ref)},
                       'team': {'id': team['id'], 'n': len(team['vals']), 'mean': st.mean(team['vals']),
                                'spend': team['spend'], 'src': team['src'], **compare(team, ref)},
                       'talk': talk_adds(team, nt, ref)}
                row['pending_regrade'] = task.startswith('ahc') and any('in-run' in s for s in nt['src'])
                row['talk']['counted'] = nt['spend'] >= cm.SPEND_OK and team['spend'] >= cm.SPEND_OK
                if task == 'ahc014':
                    row['ceiling'] = {'single': ref['at_ceiling'], 'team': team['at_ceiling'],
                                      'pool': nt['at_ceiling'], 'pool_n': len(nt['vals'])}
                    if len(nt['raw']) == len(nt['vals']) and len(team['raw']) == len(team['vals']) and \
                            len(ref['raw']) == len(ref['vals']):
                        row['raw'] = {'single': st.mean(ref['raw']), 'notalk_best_of_k': cm.best_of_k(nt['raw'], k, False),
                                      'team': st.mean(team['raw']),
                                      'notalk': compare(nt, ref, 'raw'), 'team_gap': compare(team, ref, 'raw'),
                                      'talk': talk_adds(team, nt, ref, 'raw')}
                rows.append(row)

full = [r for r in rows if 'missing' not in r]
# ---- the numbers the finding rests on (AtCoder performance on AHC014, as the catalog)
gpt_ahc = [r for r in full if r['task'] in ('ahc014', 'ahc038') and r['model'] in ('sol', 'terra')]
opus_ahc = [r for r in full if r['task'] in ('ahc014', 'ahc038') and r['model'] == 'opus']
gz = [r for r in full if r['task'] == 'gitzig']
perf = [r for r in full if r['task'] == 'perf']
summary = {
    'gpt_ahc_n': len(gpt_ahc), 'gpt_ahc_below': sum(r['talk']['pts'] < 0 for r in gpt_ahc),
    'gpt_ahc_selforg': sum(r['setup'] == 'board' for r in gpt_ahc), 'gpt_ahc_protocol': sum(r['setup'] == 'ttc' for r in gpt_ahc),
    'gpt_ahc_talk_min': min(r['talk']['pts'] for r in gpt_ahc), 'gpt_ahc_talk_max': max(r['talk']['pts'] for r in gpt_ahc),
    'gpt_ahc_interval_below_0': sum(r['talk']['hi'] < 0 for r in gpt_ahc),
    'ahc_notalk_gap_min': min(r['notalk']['gap'] for r in full if r['task'].startswith('ahc') and r['single']['counted']),
    'ahc_notalk_gap_max': max(r['notalk']['gap'] for r in full if r['task'].startswith('ahc') and r['single']['counted']),
    'opus_ahc_talk': [(r['task'], r['k'], round(r['talk']['pts'], 2)) for r in opus_ahc],
    'opus_ahc014_raw_talk': [(r['k'], round(r['raw']['talk']['pts'], 2)) for r in opus_ahc if 'raw' in r],
    'gitzig_notalk_gap_min': min(r['notalk']['gap'] for r in gz),
    'gitzig_talk_share': [(r['setup'], r['model'], r['k'], round(r['talk']['pts'] / -r['notalk']['gap'], 3)
                           if r['notalk']['gap'] < 0 else None) for r in gz],
    'perf_talk_min': min(r['talk']['pts'] for r in perf), 'perf_talk_max': max(r['talk']['pts'] for r in perf),
    'perf_notalk_gap_range': [min(r['notalk']['gap'] for r in perf), max(r['notalk']['gap'] for r in perf)],
}


def rnd(o):
    if isinstance(o, float):
        return round(o, 4)
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rnd(v) for v in o]
    return o


DATA = rnd({'rows': rows, 'summary': summary})
print(json.dumps(DATA, separators=(',', ':')))

# ---- printout, on stderr
err = sys.stderr
NAME = cm.MODEL
for r in rows:
    if 'missing' in r:
        print(f"{r['task']:7} {r['setup']:5} {NAME[r['model']]:18} k={r['k']}  MISSING: {r['missing']}", file=err)
        continue
    s, n, t, a = r['single'], r['notalk'], r['team'], r['talk']
    print(f"{r['task']:7} {r['setup']:5} {NAME[r['model']]:18} k={r['k']}  single {s['mean']:8.2f} (sp {s['spend']:.2f}"
          f"{'' if s['counted'] else ' NOT COUNTED'})  no-talk {n['gap']:+6.1f} [{n['lo']:+.1f},{n['hi']:+.1f}]  team "
          f"{t['gap']:+6.1f} [{t['lo']:+.1f},{t['hi']:+.1f}]  talk {a['pts']:+6.2f} [{a['lo']:+.1f},{a['hi']:+.1f}]"
          f"{'  PENDING' if r['pending_regrade'] else ''}"
          + (f"  | raw: no-talk {r['raw']['notalk']['gap']:+.1f} team {r['raw']['team_gap']['gap']:+.1f} talk "
             f"{r['raw']['talk']['pts']:+.2f} [{r['raw']['talk']['lo']:+.1f},{r['raw']['talk']['hi']:+.1f}]"
             f" ceil {r['ceiling']}" if 'raw' in r else ''), file=err)
print(json.dumps(DATA['summary']), file=err)


# the chart's values (AHC014 in raw score per case), as the page reads them
def chart_val(r, what):
    if r['task'] == 'ahc014':
        return {'nt': r['raw']['notalk']['gap'], 'team': r['raw']['team_gap']['gap'], 'talk': r['raw']['talk']['pts']}[what]
    return {'nt': r['notalk']['gap'], 'team': r['team']['gap'], 'talk': r['talk']['pts']}[what]


whole = lambda v: f"{'+' if round(v) > 0 else '-' if round(v) < 0 else ''}{abs(round(v))}"  # noqa: E731
print('\nCHECK right-hand table "Talking added, pts" (teams of 2 / 4 / 8; * = no-talk pool graded in-run only):', file=err)
for task in TASKS:
    for setup, _, models in PAIRS:
        for m in models:
            rs = {r['k']: r for r in full if (r['task'], r['setup'], r['model']) == (task, setup, m)}
            if not rs:
                continue
            star = ' *' if any(r['pending_regrade'] for r in rs.values()) else ''
            cells = ' / '.join(whole(chart_val(rs[k], 'talk')) if k in rs else '-' for k in KS)
            exact = ', '.join(f"{chart_val(rs[k], 'talk'):.2f}" if k in rs else '-' for k in KS)
            print(f"  {task:7} {setup:5} {NAME[m] + star:20} {cells:16} ({exact})", file=err)
best = max(gz, key=lambda r: r['talk']['pts'])
print(f"CHECK callout \"+53 pts with talking\": {best['talk']['pts']:+.2f} ({best['setup']} {best['model']} k={best['k']})",
      file=err)
for task in ('ahc014', 'ahc038'):
    cs = [r for r in gpt_ahc if r['task'] == task]
    print(f'CHECK callout "9 of 9 lower with talking" on {task}: {sum(chart_val(r, "talk") < 0 for r in cs)} of {len(cs)}',
          file=err)
print(f"CHECK title \"in all 18 cases and Claude Opus 5.5 in none\": GPT lower {sum(chart_val(r, 'talk') < 0 for r in gpt_ahc)}"
      f" of {len(gpt_ahc)}; Claude Opus 5.5 lower {sum(chart_val(r, 'talk') < 0 for r in opus_ahc)} of {len(opus_ahc)}",
      file=err)
print('CHECK "* No-talk pool graded in-run only": rows graded in-run only: '
      + ', '.join(sorted({f"{r['task']} {r['setup']} {r['model']}" for r in full if r['pending_regrade']})), file=err)
odd = [r for r in gz if r['notalk']['gap'] > 0]
print('CHECK note "No-talk team of 2 was +16%": '
      + ', '.join(f"{r['model']} k={r['k']} {r['notalk']['gap']:+.1f}%" for r in odd), file=err)
nc = [r for r in full if not r['single']['counted']]
runs_n = [x for r in full for x in (r['team']['n'], r['single']['n'])]
print(f"CHECK footnote \"3 to 10 runs per team\": {min(runs_n)} to {max(runs_n)}; \"spent 66% of its budget\": "
      + ', '.join(sorted({f"{r['task']} {r['model']} {100 * r['single']['spend']:.0f}%" for r in nc})), file=err)
g48 = [r['talk']['pts'] for r in gz if r['k'] in (4, 8)]
g2 = [r['talk']['pts'] for r in gz if r['k'] == 2]
print(f"CHECK appendix: git-to-zig teams of 4 and 8 gained \"16 to 53 points\": {min(g48):.1f} to {max(g48):.1f}; "
      f"teams of 2: {', '.join(f'{v:.1f}' for v in g2)} (\"one team lost points\": {sum(v < 0 for v in g2 + g48)})", file=err)
s8 = {r['setup']: r['talk']['pts'] for r in gz if r['model'] == 'sol' and r['k'] == 8}
print(f"CHECK appendix: GPT-6.1 Sol k=8 self-organized \"53\" vs protocol \"20\": {s8['board']:.1f} vs {s8['ttc']:.1f}",
      file=err)
ga = [chart_val(r, 'talk') for r in gpt_ahc]
opus_talk = ', '.join(f"{chart_val(r, 'talk'):+.1f}" for r in opus_ahc)
print(f"CHECK appendix: AHC talking cost GPT teams \"2 to 12 points\": {-max(ga):.1f} to {-min(ga):.1f} (chart units; "
      f"AtCoder performance on AHC014: {-summary['gpt_ahc_talk_max']:.1f} to {-summary['gpt_ahc_talk_min']:.1f}); "
      f"Claude Opus 5.5: {opus_talk}", file=err)
print(f"CHECK appendix: take-home \"within ~4 points\": {summary['perf_talk_min']:+.1f} to {summary['perf_talk_max']:+.1f}",
      file=err)
n8 = [r for r in full if r['k'] == 8]
gz8 = [r['notalk']['gap'] for r in n8 if r['task'] == 'gitzig']
ot8 = [chart_val(r, 'nt') for r in n8 if r['task'] != 'gitzig']
print(f"CHECK appendix: no-talk@8 on git-to-zig \"31% to 60% below\": {-max(gz8):.1f}% to {-min(gz8):.1f}%; other tasks "
      f"\"within about 10%\": {min(ot8):+.1f}% to {max(ot8):+.1f}%", file=err)
p8 = [r for r in gz if r['setup'] == 'ttc' and r['k'] == 8]
p8_nt = [r['notalk']['gap'] for r in p8]
print(f"CHECK appendix: protocol teams@8 on git-to-zig lost \"more than half\" to splitting: "
      f"{', '.join(f'{v:.1f}%' for v in p8_nt)}; "
      f"talking won back \"18 to 26 points\": {min(r['talk']['pts'] for r in p8):.1f} to {max(r['talk']['pts'] for r in p8):.1f}; "
      f"\"still finished 27% to 41% below\": {-max(r['team']['gap'] for r in p8):.2f}% to {-min(r['team']['gap'] for r in p8):.2f}%",
      file=err)
b8 = [r for r in gz if r['setup'] == 'board' and r['k'] == 8]
print(f"CHECK appendix: GPT self-organized@8 \"ended level with the single agent or slightly above\": "
      + ', '.join(f"{r['model']} {r['team']['gap']:+.1f}%" for r in b8), file=err)
t8 = [chart_val(r, 'talk') for r in n8]
ta = [chart_val(r, 'talk') for r in full]
print(f"CHECK main text \"-9% to +53%\": teams of 8 {min(t8):+.1f} to {max(t8):+.1f}; all sizes {min(ta):+.1f} to "
      f"{max(ta):+.1f}", file=err)

if '--write' in sys.argv:
    html = HERE / 'chart.html'
    src = html.read_text()
    new, k = re.subn(r'// DATA:BEGIN.*?// DATA:END', lambda _: '// DATA:BEGIN (written by tally.py; do not edit)\nconst DATA = '
                     + json.dumps(DATA, separators=(',', ':')) + ';\n// DATA:END', src, flags=re.S)
    assert k == 1
    html.write_text(new)
    print('DATA written into', html.name, file=err)
