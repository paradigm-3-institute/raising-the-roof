"""The results page: every configuration and run behind "Raising the Roof", laid out by the questions the post asks.

    python3 site/build.py [-o DIR]

Writes DIR/index.html (default: site/out/), one self-contained file. It inlines data/runs.csv verbatim and the
configuration table from analysis/configs.py, and offers both as downloads. Every gap, interval, flag count and λ comes
from analysis/common.py, analysis/configs.py and analysis/lam.py, so the page and the analysis scripts agree. The only
request the page makes is for the Instrument Sans font from Google Fonts. Layout and code: site/template.html.

A view answers one question. It lays out configurations as grids, one per task and budget; each grid cell holds a
configuration, its reference (normally the same model's single agent at the same total budget) and the comparison.
"""
import argparse
import json
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'analysis'))
import common as C  # noqa: E402
import configs as CF  # noqa: E402
import lam as L  # noqa: E402

DATA_AS_OF = '2026-11-24'

ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
ap.add_argument('-o', '--out', type=Path, default=HERE / 'out', help='output folder (default: site/out/)')
args = ap.parse_args()
t0 = time.time()

TASK_UNIT = {'gitzig': '% of tests passed', 'ahc014': 'AtCoder performance', 'ahc038': 'AtCoder performance',
             'perf': 'clock cycles, lower is better'}
TASK_SLUG = {'gitzig': 'git-to-zig', 'ahc014': 'ahc014', 'ahc038': 'ahc038', 'perf': 'take-home'}
MODEL_SLUG = {'opus': 'claude-opus-5.5', 'sonnet': 'claude-sonnet-5.5', 'sol': 'gpt-6.1-sol', 'terra': 'gpt-6.1-terra',
              'sol62': 'gpt-6.2-sol', 'opus5': 'claude-opus-5', 'sonnet5': 'claude-sonnet-5', 'terra56': 'gpt-5.6-terra',
              'mixed': 'mixed', 'opussol': 'mixed'}
TASKS = ['gitzig', 'ahc014', 'ahc038', 'perf']

# Flags a run can carry: (label, meaning). Cells count them.
FLAGS = {
    'time-cap': ('hit time cap', 'Stopped at the cap on working time (240 h for Claude runs on AHC and the take-home) '
                 'before spending its budget.'),
    'gave-up': ('gave up', 'An agent ended on its own (it stopped taking actions) before spending its budget.'),
    'under-budget': ('under budget', 'Finished with less than 85% of its token budget spent.'),
    'failed': ('crashed', 'Ended with an error. If it still has a final score, the score is used.'),
    'revived': ('revived', 'The host process died and the run was resumed from where it stopped. Down time is not '
                'counted in its hours.'),
    'paused': ('paused', 'Waited on a usage limit or was paused by hand. The pause is not counted in its hours.'),
    'content-filter': ('content filter', "An agent's output was blocked by the provider's content filter."),
    'in-run-grade': ('in-run grade only', 'AHC final from the private grade taken during the run, not from the later '
                     'regrade of the final code that most AHC finals come from.'),
    'regraded': ('regraded', "The run's own final grading failed or was lost, so its final comes from a later regrade "
                 'of the final code (named in the notes).'),
    'at-ceiling': ('at 3389', 'AHC014 performance at 3389, the first-place ceiling; see the raw score.'),
    'slow-judge': ('slow judge', 'AHC grading feedback was slow during the run (median wait 10 minutes or more).'),
    'waited': ('mostly waiting', 'Spent most of its working time inside long tool calls, mostly waiting on its own '
               'background jobs. Measured only for some 8M runs on the AHC tasks and the take-home: the single agents '
               "and Claude's protocol teams."),
}
STATUS = {'done': 'Finished and graded.', 'stopped': 'Stopped before it finished.',
          'failed': 'Ended with an error and has no final score.'}

# ------------------------------------------------------------------ the catalog
RUNS_CSV = (C.DATA / 'runs.csv').read_text(encoding='utf-8')
CELLS = C.cells()
ROWS = CF.table(CELLS)
CAT = {r['id']: r for r in ROWS}


def rounded(d):
    """Comparison fields rounded as in the configuration table."""
    return {k: round(v, CF.DECIMALS[k]) if k in CF.DECIMALS and v is not None else v for k, v in d.items()}


CMP = {}   # (cell id, reference id) -> comparison; the default comparisons are the configuration table's own
for r in ROWS:
    if r['gap_pct'] is not None:
        CMP[(r['id'], CF.default_ref(CELLS, CELLS[r['id']]).id)] = rounded({k: r[k] for k in CF.GAP_COLUMNS})


def compare(cell, ref):
    if (cell, ref) not in CMP:
        CMP[(cell, ref)] = rounded(C.compare(CELLS[cell], CELLS[ref]))
    return CMP[(cell, ref)]


def cid(task, model, setup, k, budget, variant='', series=''):
    i = C.cid(task, model, setup, k, budget, variant, series)
    return i if i in CELLS else None


def selforg(model):
    return 'peers' if model in C.CLAUDE else 'board'


def slug(s):
    return re.sub(r'[^a-z0-9.]+', '-', str(s).lower()).strip('-')


# ------------------------------------------------------------------ λ (analysis/lam.py)
def lam_entry(x, how):
    r3 = lambda v: round(v, 3) if v is not None else None
    return {'vs': x['vs'], 'lam': r3(x['lam']), 'lo': r3(x['lo']), 'hi': r3(x['hi']), 'how': how}


def lambdas():
    """git-to-zig teams against the single agent (where the model has single agents at two or more budgets), and
    self-organized teams against the restarted single agent (8M: teams of 2, 4, 8; 32M: team of 8, both builds)."""
    out = {}
    for c in CELLS.values():
        if c.task != 'gitzig' or c.k < 2 or c.setup in ('notalk', 'relay') or c.variant or c.series:
            continue
        x = L.vs_single(CELLS, c)
        if x:
            bs = [f'{b:g}M' for b in x['single_budgets']]
            budgets = ', '.join(bs[:-1]) + ' and ' + bs[-1]
            how = (f"X = {x['x_M']:.1f}M: the budget at which one agent would reach this team's mean score, read off "
                   f"the single agent's mean scores at {budgets}, with score linear in log budget"
                   + (' (extrapolated past the budgets run)' if x['extrapolated'] else '')
                   + '. λ = ln(X / b) / ln N for N agents with budget b each. Bootstrap 95% interval. Code: '
                   'analysis/lam.py.')
            out.setdefault(c.id, []).append(lam_entry(x, how))
    snaps = C.snapshots()
    for model in ('sol', 'terra'):
        for budget, sizes, series in ((8, (2, 4, 8), ('',)), (32, (8,), ('', 'rerun'))):
            for k in sizes:
                for s in series:
                    c = CELLS.get(C.cid('gitzig', model, 'board', k, budget, '', s))
                    x = c and L.vs_restarted(CELLS, c, snaps)
                    if x and x.get('lam') is not None:
                        how = (f"X = {x['x_M']:.1f}M: the first point where the restarted single agent's mean curve "
                               "(its best graded score so far, averaged over its runs) reaches this team's mean score. "
                               'λ = ln(X / b) / ln N for N agents with budget b each. Bootstrap 95% interval. Code: '
                               'analysis/lam.py.')
                        out.setdefault(c.id, []).append(lam_entry(x, how))
    return out


LAM = lambdas()


# ------------------------------------------------------------------ grid entries
def build_diff(a, b):
    ca, cb = CAT[a]['codex'], CAT[b]['codex']
    return bool(ca and cb and ca != cb)


def entry(cell, ref=None, show='gap'):
    """One grid cell: the configuration, its reference and the comparison. show: gap, abs (score) or abs+gap."""
    if not cell:
        return None
    e = {'cell': cell, 'ref': ref, 'show': show}
    if ref and show in ('gap', 'abs+gap'):
        e.update(compare(cell, ref))
        if build_diff(cell, ref):
            e['build'] = f"Codex {CAT[cell]['codex']} here, {CAT[ref]['codex']} for the reference"
    if cell in LAM:
        e['lam'] = LAM[cell]
    return e


def change(new, new_ref, old, old_ref):
    """Newer model's gap minus the older model's gap, in percentage points, with a bootstrap interval."""
    if not all((new, new_ref, old, old_ref)):
        return {}
    cs = [CELLS[x] for x in (new, new_ref, old, old_ref)]
    task = cs[0].task
    f = lambda a, b, c, d: 100 * (C.gap(task, st.mean(a), st.mean(b)) - C.gap(task, st.mean(c), st.mean(d)))
    arms = [c.vals for c in cs]
    lo, hi = C.boot(random.Random(C.seed_of(tuple(c.key for c in cs))), arms, f) \
        if all(len(a) > 1 for a in arms) else (None, None)
    r2 = lambda v: round(v, 2) if v is not None else None
    return {'show': 'change', 'change': r2(f(*arms)), 'change_lo': r2(lo), 'change_hi': r2(hi)}


def score_change(new, old):
    """The newer single agent's score against the older one's, in % (the take-home as speed)."""
    if not (new and old):
        return None
    g = compare(new, old)
    return {'show': 'change', 'change': g['gap_pct'], 'change_lo': g['gap_lo'], 'change_hi': g['gap_hi'],
            'change_kind': 'score'}


def grid(anchor, cols, rows, ref=None, note='', layout='generic'):
    rows = [r for r in rows if any(v for v in r['cells'].values())]
    used = [c for c in cols if any(r['cells'].get(c['key']) for r in rows)]
    return {'anchor': anchor, 'cols': used, 'rows': rows, 'ref': ref, 'note': note, 'layout': layout} if rows else None


def row(label, model, setup, cells, sub='', slug_=None):
    return {'label': label, 'sub': sub, 'model': model, 'setup': setup,
            'cells': {k: v for k, v in cells.items() if v},
            'slug': slug_ or f'{MODEL_SLUG.get(model, model)}/{slug(label)}'}


SIZES = [1, 2, 4, 8, 16, 32]
SIZE_COLS = [{'key': str(k), 'label': '1 agent' if k == 1 else f'{k} agents'} for k in SIZES]


# ------------------------------------------------------------------ views
def teams_grid(anchor, task, budget, models):
    rows = []
    for m in models:
        solo = cid(task, m, 'solo', 1, budget)
        ways = [('Single agent', 'solo', {'1': entry(solo, None, 'abs')}),
                ('Restarted single agent', 'relay', {'1x8': entry(cid(task, m, 'relay', 8, budget), solo)})]
        for label, setup in (('Protocol teams', 'ttc'), ('Self-organized teams', selforg(m)),
                             ('Orchestrated teams', 'ultra')):
            ways.append((label, setup, {str(k): entry(cid(task, m, setup, k, budget), solo) for k in SIZES[1:]}))
        rows += [row(label, m, setup, cells) for label, setup, cells in ways]
    # the restarted single agent is one agent: its own column after the single agent
    cols = SIZE_COLS[:1] + [{'key': '1x8', 'label': '1 agent, 8 sessions'}] + SIZE_COLS[1:]
    return grid(anchor, cols, rows, layout='model')


def view_teams():
    tasks = []
    for task in TASKS:
        budgets = []
        for budget in (8, 32):
            g = teams_grid(f'teams-vs-single/{TASK_SLUG[task]}/{budget}M', task, budget,
                           ['opus', 'sonnet', 'sol', 'terra'])
            if g:
                budgets.append({'budget': budget, 'grid': g})
        tasks.append({'task': task, 'budgets': budgets})
    return {'id': 'teams-vs-single', 'title': 'Teams vs single agent', 'backs': 'In the post: Main results',
            'blocks': [{'id': '', 'title': '', 'tasks': tasks}]}


def view_diversity():
    tasks = []
    for task, claude, mixed in (('gitzig', 'sonnet', 'mixed'), ('ahc014', 'opus', 'opussol'),
                                ('perf', 'opus', 'opussol')):
        sc, ss = cid(task, claude, 'solo', 1, 8), cid(task, 'sol', 'solo', 1, 8)
        pick = min if task in C.LOWER_BETTER else max
        best = pick((sc, ss), key=lambda x: CELLS[x].mean)
        cols = [{'key': 'claude', 'label': f'{C.MODEL[claude]} only'}, {'key': 'sol', 'label': 'GPT-6.1 Sol only'},
                {'key': 'mixed', 'label': 'Mixed'}]
        rows = []
        for label, setups in (('Protocol teams', {'claude': 'ttc', 'sol': 'ttc', 'mixed': 'ttc'}),
                              ('Self-organized teams', {'claude': 'peers', 'sol': 'board', 'mixed': 'chat'})):
            for k in (2, 4, 8):
                cells = {'claude': entry(cid(task, claude, setups['claude'], k, 8), best),
                         'sol': entry(cid(task, 'sol', setups['sol'], k, 8), best),
                         'mixed': entry(cid(task, mixed, setups['mixed'], k, 8), best)}
                if not cells['mixed']:
                    continue
                sub = f'{k} agents' + (' · direct messages / message board / SwarmChat'
                                       if label.startswith('Self') else '')
                rows.append(row(label, mixed, setups['mixed'], cells, sub, slug_=f'{slug(label)}/{k}'))
        spend = CELLS[best].spend
        ref = {'cell': best, 'label': f'best single agent, {C.MODEL[CELLS[best].model]}',
               'under': spend is not None and spend < C.SPEND_OK}
        g = grid(f'model-diversity/{TASK_SLUG[task]}/8M', cols, rows, ref)
        tasks.append({'task': task, 'budgets': [{'budget': 8, 'grid': g}] if g else []})
    return {'id': 'model-diversity', 'title': 'Model diversity',
            'backs': 'In the post: Cyberhayek: Does model diversity help? (appendix)',
            'blocks': [{'id': '', 'title': '', 'tasks': tasks}]}


def view_communication():
    tasks = []
    for task in TASKS:
        rows = []
        for setup, prompt, models in (('ttc', 'protocol prompt', ['opus', 'sonnet', 'sol']),
                                      ('board', 'plain prompt', ['sol', 'terra'])):
            for m in models:
                solo = cid(task, m, 'solo', 1, 8)
                cells = {}
                for k in (2, 4, 8):
                    team = entry(cid(task, m, setup, k, 8), solo)
                    nt = entry(cid(task, m, 'notalk', k, 8, prompt), solo)
                    if team or nt:
                        cells[str(k)] = {'show': 'pair', 'team': team, 'notalk': nt}
                label = 'Protocol teams' if setup == 'ttc' else 'Self-organized teams'
                rows.append(row(label, m, setup, cells, f'no-talk twin: {prompt}'))
        g = grid(f'communication/{TASK_SLUG[task]}/8M', SIZE_COLS[1:4], rows, layout='model')
        tasks.append({'task': task, 'budgets': [{'budget': 8, 'grid': g}] if g else []})
    return {'id': 'communication', 'title': 'Communication',
            'backs': 'In the post: How much does communication add? (appendix)',
            'note': ('A no-talk team of k is the best of k agents that work alone, each with 1/k of the budget, '
                     'averaged over every group of k drawn from a pool (6 agents at 4M, 8 at 2M, 12 at 1M).'),
            'blocks': [{'id': '', 'title': '', 'tasks': tasks}]}


def view_older():
    blocks = []
    for old, new in (('opus5', 'opus'), ('sonnet5', 'sonnet'), ('terra56', 'terra'), ('sol', 'sol62')):
        tasks = []
        one = next(iter({old, new} & {'opus5', 'sonnet5', 'terra56', 'sol62'}))   # the model run only for this
        for task in TASKS:
            if not any(c.model == one and c.task == task for c in CELLS.values()):
                continue
            so, sn = cid(task, old, 'solo', 1, 8), cid(task, new, 'solo', 1, 8)
            cols = [{'key': 'old', 'label': C.MODEL[old]}, {'key': 'new', 'label': C.MODEL[new]},
                    {'key': 'change', 'label': 'Change'}]
            rows = [row('Single agent', new, 'solo', {'old': entry(so, None, 'abs'), 'new': entry(sn, None, 'abs'),
                                                      'change': score_change(sn, so)}, 'score',
                        slug_='single-agent/1')]
            for label, setup in (('Protocol teams', 'ttc'), ('Self-organized teams', selforg(new))):
                for k in (2, 4, 8):
                    if not cid(task, one, setup, k, 8):
                        continue
                    co, cn = cid(task, old, setup, k, 8), cid(task, new, setup, k, 8)
                    rows.append(row(label, new, setup, {'old': entry(co, so), 'new': entry(cn, sn),
                                                        'change': change(cn, sn, co, so) or None},
                                    f'{k} agents', slug_=f'{slug(label)}/{k}'))
            g = grid(f'older-vs-newer/{MODEL_SLUG[old]}/{TASK_SLUG[task]}/8M', cols, rows)
            if g:
                tasks.append({'task': task, 'budgets': [{'budget': 8, 'grid': g}]})
        if tasks:
            blocks.append({'id': MODEL_SLUG[old], 'title': f'{C.MODEL[old]} → {C.MODEL[new]}', 'tasks': tasks})
    return {'id': 'older-vs-newer', 'title': 'Older vs newer models',
            'backs': 'In the post: Are models improving at collaborating? (appendix)',
            'note': 'Each model is compared with its own single agent. Change: newer gap minus older gap, in points.',
            'blocks': blocks}


def view_checks():
    blocks = []
    tasks = []
    for task in TASKS:
        rows = []
        for m in ('opus', 'sonnet', 'sol'):
            c = cid(task, m, 'solo_ttc', 1, 8, 'protocol prompt')
            if c:
                rows.append(row('Single agent, protocol prompt', m, 'solo_ttc',
                                {'v': entry(c, cid(task, m, 'solo', 1, 8), 'abs+gap')}))
        g = grid(f'checks/protocol-prompt/{TASK_SLUG[task]}/8M', [{'key': 'v', 'label': 'vs the plain single agent'}],
                 rows)
        if g:
            tasks.append({'task': task, 'budgets': [{'budget': 8, 'grid': g}]})
    blocks.append({'id': 'protocol-prompt', 'title': 'Does the protocol prompt alone change a single agent?',
                   'tasks': tasks})
    rows = []
    for m in ('sol', 'terra'):
        solo = cid('gitzig', m, 'solo', 1, 32)
        rows.append(row('Self-organized teams', m, 'board',
                        {'original': entry(cid('gitzig', m, 'board', 8, 32), solo),
                         'patched': entry(cid('gitzig', m, 'board', 8, 32, '', 'rerun'), solo)}, '8 agents'))
    g = grid('checks/patched-codex/git-to-zig/32M', [{'key': 'original', 'label': 'Codex 0.162.1'},
                                                    {'key': 'patched', 'label': 'Patched build (rerun)'}], rows)
    blocks.append({'id': 'patched-codex', 'title': 'Does the Codex compaction bug explain the 32M result?',
                   'tasks': [{'task': 'gitzig', 'budgets': [{'budget': 32, 'grid': g}]}]})
    tasks = []
    for task in ('ahc014', 'ahc038'):
        g = teams_grid(f'checks/ahc-32m/{TASK_SLUG[task]}/32M', task, 32, ['terra'])
        if g:
            tasks.append({'task': task, 'budgets': [{'budget': 32, 'grid': g}]})
    blocks.append({'id': 'ahc-32m', 'title': 'Do teams catch up on the AHC tasks with a 32M budget?', 'tasks': tasks})
    tasks = []
    for task in ('gitzig', 'ahc014', 'ahc038'):
        g = teams_grid(f'checks/2m/{TASK_SLUG[task]}/2M', task, 2, ['opus', 'sonnet', 'sol', 'terra'])
        if g:
            tasks.append({'task': task, 'budgets': [{'budget': 2, 'grid': g}]})
    blocks.append({'id': '2m', 'title': 'The 2M single agents behind λ, and GPT-6.1 Terra teams at 2M',
                   'tasks': tasks})
    return {'id': 'checks', 'title': 'Checks', 'backs': 'In the post: Run setup (methodology appendix)',
            'blocks': blocks}


def view_context():
    ref = cid('gitzig', 'terra', 'solo', 1, 8)
    rows = []
    for v in ('compacts at 61k', 'compacts at 122k', 'compacts at 244k'):
        rows.append(row(f'Single agent, {v}', 'terra', 'solo',
                        {'v': entry(cid('gitzig', 'terra', 'solo', 1, 8, v), ref, 'abs+gap')},
                        slug_=f'single-agent/{v.split()[-1]}'))
    rows.append(row('Single agent, default compaction', 'terra', 'solo', {'v': entry(ref, None, 'abs')},
                    slug_='single-agent/default'))
    rows.append(row('Restarted single agent', 'terra', 'relay',
                    {'v': entry(cid('gitzig', 'terra', 'relay', 8, 8), ref, 'abs+gap')}, '8 sessions',
                    slug_='restarted/8'))
    g = grid('context-window/git-to-zig/8M', [{'key': 'v', 'label': 'GPT-6.1 Terra, 8M'}], rows,
             note='By default the agent compacts its context at about 450k tokens.')
    return {'id': 'context-window', 'title': 'Context window', 'backs': 'An extra result, not discussed in the post',
            'blocks': [{'id': '', 'title': '', 'tasks': [{'task': 'gitzig', 'budgets': [{'budget': 8, 'grid': g}]}]}]}


def cells_used(v):
    ids = set()
    for b in v['blocks']:
        for t in b['tasks']:
            for bu in t['budgets']:
                g = bu['grid']
                if g.get('ref'):
                    ids.add(g['ref']['cell'])
                for r in g['rows']:
                    for e in r['cells'].values():
                        for x in ([e.get('team'), e.get('notalk')] if e.get('show') == 'pair' else [e]):
                            if x and x.get('cell'):
                                ids.add(x['cell'])
                            if x and x.get('ref'):
                                ids.add(x['ref'])
    return sorted(ids)


VIEWS = [view_teams(), view_diversity(), view_communication(), view_older(), view_checks(), view_context()]
for v in VIEWS:
    v['cells_used'] = cells_used(v)

# ------------------------------------------------------------------ the page
payload = {
    'asof': DATA_AS_OF,
    'flags': {k: {'label': a, 'meaning': b} for k, (a, b) in FLAGS.items()},
    'status': STATUS,
    'names': {'task': C.TASK, 'unit': TASK_UNIT, 'model': C.MODEL, 'setup': C.SETUP},
    'pool': C.POOL,
    'runsCSV': RUNS_CSV,
    'configsCSV': CF.csv_text(ROWS),
    'runCell': [C.cell_id(C.key(r)) for r in C.runs()],
    'views': VIEWS,
}
data = json.dumps(payload, separators=(',', ':'), ensure_ascii=False).replace('</', '<\\/')
template = (HERE / 'template.html').read_text(encoding='utf-8')
assert template.count('/*PAYLOAD*/null') == 1
page = template.replace('/*PAYLOAD*/null', data)
args.out.mkdir(parents=True, exist_ok=True)
(args.out / 'index.html').write_text(page, encoding='utf-8')
n_lam = sum(len(x) for x in LAM.values())
print(f"wrote {args.out / 'index.html'} ({len(page.encode()) / 1024:.0f} kB): {len(ROWS)} configurations, "
      f"{len(payload['runCell'])} runs, {len(VIEWS)} views, {n_lam} λ values, {time.time() - t0:.0f} s")
