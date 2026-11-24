"""What the runs would cost at API prices.

    python3 analysis/cost.py

Tokens: data/tokens.csv, per run, model and price tier, summed from each agent's logs (one row per model call there).
Prices: data/prices.csv, USD per million tokens, as listed on models.dev on 2026-11-23. OpenAI charges its long-context
tier for every call whose prompt is over 272k tokens; 'long' rows hold those calls. Claude cache writes are priced at
the listed cache-write price (5-minute cache).
Rows marked 'no log' are runs whose per-call logs were not kept, only the run's token totals (Claude Code: uncached
input counted as cache writes). They are priced at base prices, and OpenAI ones are raised by the share of
long-context spending measured on runs of the same kind that kept their logs (same run name without the repeat
number, else the same model).
"""
import csv
import re
from collections import defaultdict

from common import DATA, runs

P = {(r['model_id'], r['tier']): {k: float(v) for k, v in r.items() if k not in ('model_id', 'tier')}
     for r in csv.DictReader(open(DATA / 'prices.csv'))}


def price(model, tier, row):
    p = P.get((model, tier)) or P[(model, 'base')]
    i, c, w, o = (int(row[k]) for k in ('input_tokens', 'cached_input_tokens', 'cache_write_tokens', 'output_tokens'))
    return ((i - c - w) * p['input'] + c * p['cached_input'] + w * p['cache_write'] + o * p['output']) / 1e6


def family(run):
    return re.sub(r'-(r|p)\d+.*$', '', run)


def costs():
    """run -> USD."""
    rows = list(csv.DictReader(open(DATA / 'tokens.csv')))
    fam, mod = defaultdict(lambda: [0.0, 0.0]), defaultdict(lambda: [0.0, 0.0])
    logged = defaultdict(list)
    for r in rows:
        if r['tier'] != 'no log':
            logged[r['run']].append(r)
    for run, rs in logged.items():
        models = {r['model_id'] for r in rs}
        if len(models) != 1 or not next(iter(models)).startswith('gpt'):
            continue
        cost = sum(price(r['model_id'], r['tier'], r) for r in rs)
        flat = sum(price(r['model_id'], 'base', r) for r in rs)
        for d, k in ((fam, family(run)), (mod, rs[0]['model_id'])):
            d[k][0] += cost
            d[k][1] += flat
    out = defaultdict(float)
    for r in rows:
        c = price(r['model_id'], r['tier'], r)
        if r['tier'] == 'no log' and r['model_id'].startswith('gpt'):
            src = fam.get(family(r['run'])) or mod.get(r['model_id'])
            if src and src[1]:
                c *= src[0] / src[1]
        out[r['run']] += c
    return out


if __name__ == '__main__':
    cost = costs()
    R = runs()
    every = sum(cost.get(r['run'], 0) for r in R)
    counted = sum(cost.get(r['run'], 0) for r in R if r['counted'])
    print(f'Every run in data/runs.csv ({len(R)}): ${every:,.0f}')
    print(f'Counted runs only ({sum(r["counted"] for r in R)}): ${counted:,.0f}')
    tok = list(csv.DictReader(open(DATA / 'tokens.csv')))
    per_model = defaultdict(float)
    for run, c in cost.items():           # a mixed team's cost split by each model's share
        rs = [r for r in tok if r['run'] == run]
        tot = sum(price(r['model_id'], r['tier'], r) for r in rs) or 1
        for r in rs:
            per_model[r['model_id']] += c * price(r['model_id'], r['tier'], r) / tot
    print('By model:')
    for m, c in sorted(per_model.items(), key=lambda x: -x[1]):
        print(f'  {m:20} ${c:,.0f}')
    nolog = {r['run'] for r in tok if r['tier'] == 'no log'}
    print(f'Runs priced from totals (no per-call log): {len(nolog)}, ${sum(cost[r] for r in nolog):,.0f}')
