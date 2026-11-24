"""The configuration table: one row per configuration with counted runs, and its gap to the same model's single agent.

    python3 analysis/configs.py              # a short summary
    python3 analysis/configs.py --csv PATH   # also write the table as CSV

A configuration is one task, model, setup, team size, total budget, variant and series (common.cells()). Its row has
the counted runs' n, mean (no-talk teams: the expected best of k from the pool, with n the pool size), median, sd,
bootstrap 95% interval of the mean, the AHC014 raw score per case, how many runs sit at the AHC014 ceiling, the mean
share of budget spent, median working hours, flag counts and Codex builds. The default comparison is with the same
model's plain single agent at the same total budget (common.vs_single), by the rules in common.py: gap in % with a
bootstrap interval, the gap in points, the single agent's n, mean and spend, whether the comparison counts and whether
the interval clears 0. Plain single agents and mixed-model teams have no default comparison.

table() keeps numbers unrounded; csv_rows() and csv_text() round them (means 4 decimals, gaps 2). The results page
(site/build.py) and the analysis scripts both read this table.
"""
import argparse
import csv
import io
import statistics as st
from collections import Counter

import common as C

CHANNEL = {'board': 'message board', 'peers': 'direct messages', 'chat': 'SwarmChat'}
TASK_ORDER = ['gitzig', 'ahc014', 'ahc038', 'perf']
MODEL_ORDER = ['opus', 'sonnet', 'sol', 'terra', 'sol62', 'opus5', 'sonnet5', 'terra56', 'mixed', 'opussol']
SETUP_ORDER = ['solo', 'relay', 'relayctx', 'solo_ttc', 'ttc', 'board', 'peers', 'chat', 'ultra', 'notalk']
MIXED = {'mixed', 'opussol'}

COLUMNS = ['id', 'task', 'model', 'setup', 'channel', 'team_size', 'budget_M', 'variant', 'series', 'n', 'mean',
           'median', 'sd', 'ci_lo', 'ci_hi', 'raw_mean_M', 'at_ceiling', 'spend_share', 'hours_median', 'flag_counts',
           'codex', 'gap_pct', 'gap_lo', 'gap_hi', 'gap_points', 'base_n', 'base_mean', 'base_spend', 'counted',
           'clear', 'caveat']
GAP_COLUMNS = ['gap_pct', 'gap_lo', 'gap_hi', 'gap_points', 'base_n', 'base_mean', 'base_spend', 'counted', 'clear']
DECIMALS = {'mean': 4, 'median': 4, 'sd': 4, 'ci_lo': 4, 'ci_hi': 4, 'raw_mean_M': 5, 'spend_share': 3,
            'hours_median': 3, 'gap_pct': 2, 'gap_lo': 2, 'gap_hi': 2, 'gap_points': 3, 'base_mean': 4,
            'base_spend': 3}


def codex_of(cell):
    return ', '.join(sorted({r['codex'] for r in cell.runs if r['codex']}))


def default_ref(cells, cell):
    """The single agent a configuration is compared with by default, or None (plain single agents, mixed teams)."""
    if (cell.setup == 'solo' and not cell.variant) or cell.model in MIXED:
        return None
    return cells.get(C.cid(cell.task, cell.model, 'solo', 1, cell.budget))


def row_of(cells, c):
    lo, hi = c.ci()
    raws = [r['raw_per_case_M'] for r in c.runs if r['raw_per_case_M'] is not None]
    raw = None
    if raws:
        raw = C.best_of_k(raws, c.k) if c.pool else st.mean(raws)
    flags = Counter(f for r in c.runs for f in r['flags'].split())
    row = {'id': c.id, 'task': c.task, 'model': c.model, 'setup': c.setup, 'channel': CHANNEL.get(c.setup, ''),
           'team_size': c.k, 'budget_M': c.budget, 'variant': c.variant, 'series': c.series, 'n': len(c.vals),
           'mean': c.mean, 'median': None if c.pool else st.median(c.vals),
           'sd': st.stdev(c.vals) if len(c.vals) > 1 and not c.pool else None, 'ci_lo': lo, 'ci_hi': hi,
           'raw_mean_M': raw, 'at_ceiling': sum(v >= C.CEILING for v in c.vals) if c.task == 'ahc014' else None,
           'spend_share': c.spend, 'hours_median': c.hours_median,
           'flag_counts': ' '.join(f'{k}:{v}' for k, v in sorted(flags.items())), 'codex': codex_of(c),
           'caveat': ''}
    row.update(dict.fromkeys(GAP_COLUMNS))
    ref = default_ref(cells, c)
    g = C.compare(c, ref)
    if g:
        row.update({k: g[k] for k in GAP_COLUMNS})
        if row['codex'] != codex_of(ref):
            row['caveat'] = f"Codex {row['codex'] or 'n/a'} here, {codex_of(ref) or 'n/a'} for the single agent"
    return row


def order(row):
    idx = lambda seq, x: seq.index(x) if x in seq else len(seq)
    return (idx(TASK_ORDER, row['task']), row['budget_M'], idx(MODEL_ORDER, row['model']),
            idx(SETUP_ORDER, row['setup']), row['variant'], row['series'], row['team_size'])


def table(cells=None):
    """One row per configuration with counted runs, numbers unrounded (None when missing)."""
    cells = C.cells() if cells is None else cells
    return sorted((row_of(cells, c) for c in cells.values()), key=order)


def fmt(col, v):
    """A table value as written to CSV."""
    if v is None:
        return ''
    if isinstance(v, bool):
        return '1' if v else '0'
    if col in DECIMALS:
        return f'{round(v, DECIMALS[col])}'
    return str(v)


def csv_rows(rows):
    return [{c: fmt(c, r[c]) for c in COLUMNS} for r in rows]


def csv_text(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, COLUMNS, lineterminator='\n')
    w.writeheader()
    w.writerows(csv_rows(rows))
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--csv', metavar='PATH', help='write the table to PATH')
    a = ap.parse_args()
    rows = table()
    n_runs = sum(r['n'] for r in rows)
    print(f'{len(rows)} configurations from {n_runs} counted runs')
    print('by task:', ', '.join(f"{C.TASK[t]} {sum(r['task'] == t for r in rows)}" for t in TASK_ORDER))
    cmp_ = [r for r in rows if r['gap_pct'] is not None]
    counted = [r for r in cmp_ if r['counted']]
    clear = Counter(r['clear'] or 'one run' for r in counted)
    print(f'{len(cmp_)} compared with their single agent; {len(counted)} counted: '
          f"{clear['above']} clearly above, {clear['below']} clearly below, {clear['within noise']} within noise")
    if a.csv:
        with open(a.csv, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_text(rows))
        print('wrote', a.csv)


if __name__ == '__main__':
    main()
