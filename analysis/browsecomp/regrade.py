"""Grade every final answer of the BrowseComp appendix with simple-evals' judge, GPT-4.1, and write
data/browsecomp_trials.csv.

This is the code that made the verdicts behind the post's scores. It cannot run on this release alone: it needs the
run folders (jobs/pilot126-* and jobs/pilot253-*; their transcripts and answers are not public), the dataset
(`python browsecomp_task.py fetch`) and an OpenAI API key in OPENAI_API_KEY. From the repo root:

    python analysis/browsecomp/regrade.py prompts jobs/pilot126-* jobs/pilot253-* --work WORK
    python analysis/browsecomp/regrade.py judge WORK
    python analysis/browsecomp/regrade.py combine WORK --out data/browsecomp_trials.csv

WORK holds the questions, answers and the judge's text: keep it private. Only `combine`'s output is public.

The rule:
  1. A setting's trial on a question is its latest graded trial, preferring one that ended without a harness error
     (some questions ran twice).
  2. The trial's answer is the team's last submission. A trial without one is wrong.
  3. GPT-4.1 grades every answer once, as simple-evals does: its verdict is the trial's score (1 or 0). combine stops
     if an answer has no verdict (a call that failed is never counted as "no").
The in-run verdict (the harness judge's, browsecomp_task.JUDGE_MODEL) is kept for comparison only.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import browsecomp_task as bc  # noqa: E402

SETUP = {'solo': 'single agent', 'ultra': 'orchestrated', 'paper': 'protocol', 'board': 'self-organized'}
SETTINGS = {('solo', 'low'), ('solo', 'high'), ('ultra', 'low'), ('ultra', 'high'), ('paper', 'high'),
            ('board', 'high')}
JOB = re.compile(r'pilot\d+-(\w+)-(\w+)-r1')
ORDER = random.Random(20261018).sample(range(1, bc.OFFICIAL_TASK_COUNT + 1), bc.OFFICIAL_TASK_COUNT)
RANK = {q: i + 1 for i, q in enumerate(ORDER)}

# simple-evals' grader, as pinned in browsecomp_task.py: gpt-4.1-2025-04-14 through Chat Completions, system message
# "You are a helpful assistant.", the filled GRADER_TEMPLATE as the user message, temperature 0.5, 2,048 tokens,
# verdict from `correct: (yes|no)` (group(1), failing closed to "no").
GRADER = dict(model=bc.OFFICIAL_GRADER_MODEL, temperature=bc.OFFICIAL_GRADER_TEMPERATURE,
              max_tokens=bc.OFFICIAL_GRADER_MAX_TOKENS)
SYSTEM = bc.OFFICIAL_SYSTEM_MESSAGE
USD_IN, USD_CACHED, USD_OUT = 2.00 / 1e6, 0.50 / 1e6, 8.00 / 1e6   # gpt-4.1 per token, for the cost cap

# ---------------------------------------------------------------------------------------------------------------
# prompts: pick each setting's trial per question and write GPT-4.1's prompt for its final answer


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def ts(s):
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
    except (AttributeError, ValueError):
        return None


def trial_row(job, arm, effort, trial):
    state = trial / bc.STATE
    final = bc.trial_final(trial) or {}
    if final.get('score') is None:
        return None
    res = read_json(trial / 'result.json')
    a, b = ts(res.get('started_at')), ts(res.get('finished_at'))
    ex = res.get('exception_info')
    weighted = sum(read_json(p).get('weighted_tokens', 0) or 0 for p in trial.glob('agent/*/token-usage.json'))
    searches = sum(1 for e in bc.read_jsonl(trial / 'search-filter' / 'search_filter.jsonl')
                   if e.get('kind') == 'standalone_search')
    subs = bc.read_jsonl(state / 'submissions.jsonl')
    return dict(arm=arm, effort=effort, question=int(bc.trial_task(trial)['index']), job=job, trial=str(trial),
                start=(res.get('started_at') or '')[:16], seconds=round(b - a) if a and b else '',
                error=(ex.get('exception_type', 'yes') if isinstance(ex, dict) else 'yes') if ex else '',
                final_sha=final.get('sha256'), weighted_tokens=round(weighted), searches=searches,
                submissions=len(subs), budget_hit=int(any(trial.glob('agent/*/token-budget-exhausted.json'))),
                audit=read_json(state / 'audit.json').get('status', ''))


def cmd_prompts(args):
    questions = {q.index: q for q in bc.load_questions()}
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    graded = defaultdict(list)
    for job_dir in map(Path, args.jobs):
        m = JOB.match(job_dir.name)
        if not m or 'cap5m' in job_dir.name or (m.group(1), m.group(2)) not in SETTINGS:
            print(f'skipped {job_dir.name}')
            continue
        for trial in bc.trials(job_dir):
            row = trial_row(job_dir.name, m.group(1), m.group(2), trial)
            if row:
                graded[(row['arm'], row['effort'], row['question'])].append(row)
    finals, prompts = [], {}
    for key, rows in sorted(graded.items()):
        row = sorted(rows, key=lambda r: (bool(r['error']), not r['start'], [-ord(c) for c in r['start']]))[0]
        row['attempts'] = len(rows)
        trial = Path(row['trial'])
        subs = {s.get('sha256'): s for s in reversed(bc.read_jsonl(trial / bc.STATE / 'submissions.jsonl'))}
        grades = {g['sha256']: g for g in bc.read_jsonl(trial / bc.STATE / 'grades.jsonl') if g.get('score') is not None}
        sha = row.pop('final_sha')
        row['inrun_verdict'] = grades[sha]['verdict'] if sha in grades else ''
        row['item'] = ''
        if sha in subs:
            q = questions[row['question']]
            row['item'] = f"{row['arm']}-{row['effort']}-{row['question']:04d}"
            prompts[row['item']] = bc.grader_prompt(q.problem, q.answer, subs[sha]['response'])
        finals.append(row)
    cols = ['arm', 'effort', 'question', 'job', 'trial', 'start', 'attempts', 'item', 'inrun_verdict',
            'weighted_tokens', 'searches', 'submissions', 'seconds', 'budget_hit', 'error', 'audit']
    with open(work / 'finals.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, cols, lineterminator='\n')
        w.writeheader()
        w.writerows(finals)
    fd = os.open(work / 'prompts.jsonl', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as fh:
        for item, prompt in prompts.items():
            fh.write(json.dumps({'item': item, 'prompt': prompt}) + '\n')
    print(f'{len(finals)} trials; {len(prompts)} answers for GPT-4.1')


# ---------------------------------------------------------------------------------------------------------------
# judge: one GPT-4.1 call per answer; verdicts to results.jsonl, the judge's text to responses.jsonl. Run it again to
# retry the calls that failed.


def cmd_judge(args):
    work = Path(args.work)
    prompts = {p['item']: p['prompt'] for p in map(json.loads, open(work / 'prompts.jsonl'))}
    if args.limit:
        prompts = dict(list(prompts.items())[:args.limit])
    res_path, txt_path = work / 'results.jsonl', work / 'responses.jsonl'
    done = {r['item'] for r in bc.read_jsonl(res_path) if r.get('status') == 200}
    todo = [item for item in prompts if item not in done]
    est = sum(len(prompts[i]) / 4 + 15 for i in todo) * USD_IN + 150 * len(todo) * USD_OUT
    print(f'{len(prompts)} answers: {len(todo)} calls to make ({len(done)} done); '
          f'about ${est:.2f} (cap ${args.max_usd:.2f})')
    if args.dry_run or not todo:
        return
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key:
        sys.exit('set OPENAI_API_KEY')
    lock = threading.Lock()
    state = {'usd': 0.0, 'ok': 0, 'fail': 0, 'stop': None}

    def call(prompt):
        body = json.dumps({**GRADER, 'messages': [{'role': 'system', 'content': SYSTEM},
                                                  {'role': 'user', 'content': prompt}]}).encode()
        req = urllib.request.Request(args.base_url.rstrip('/') + '/chat/completions', data=body, method='POST',
                                     headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
        code, detail = 0, ''
        for attempt in range(8):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return 200, json.loads(r.read())
            except urllib.error.HTTPError as e:
                code, detail = e.code, e.read()[:300].decode(errors='replace')
                if code in (400, 401, 403, 404):
                    return code, {'error': detail}
            except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
                code, detail = 0, str(e)[:200]
            time.sleep(min(60, 2 ** attempt) * random.uniform(1, 1.5))
        return code, {'error': detail}

    def work_one(item):
        if state['stop']:
            return
        status, data = call(prompts[item])
        rec, text = {'item': item, 'status': status}, None
        if status == 200:
            u = data.get('usage') or {}
            cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
            pin, pout = u.get('prompt_tokens', 0), u.get('completion_tokens', 0)
            text = data['choices'][0]['message'].get('content') or ''
            rec.update(model=data.get('model'), verdict=bc.parse_grader_response(text),
                       parsed=bool(re.search(bc.GRADER_PATTERN, text)), finish=data['choices'][0].get('finish_reason'),
                       prompt_tokens=pin, cached_tokens=cached, completion_tokens=pout,
                       usd=round((pin - cached) * USD_IN + cached * USD_CACHED + pout * USD_OUT, 6))
        else:
            rec['error'] = str(data.get('error'))[:300]
        with lock:
            with open(res_path, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
            if text is not None:
                with open(txt_path, 'a') as fh:
                    fh.write(json.dumps({'item': item, 'text': text}) + '\n')
                state['ok'] += 1
                state['usd'] += rec['usd']
                if state['usd'] >= args.max_usd:
                    state['stop'] = f'cost cap ${args.max_usd:.2f} reached'
            else:
                state['fail'] += 1
                if status in (400, 401, 403, 404):
                    state['stop'] = f'HTTP {status}'
            n = state['ok'] + state['fail']
            if n % 100 == 0:
                print(f"{n}/{len(todo)} calls, ${state['usd']:.2f}, {state['fail']} failed", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work_one, todo))
    print(f"done: {state['ok']} ok, {state['fail']} failed, ${state['usd']:.2f}"
          + (f"; stopped: {state['stop']}" if state['stop'] else ''))


# ---------------------------------------------------------------------------------------------------------------
# combine: the rule, to the public per-trial table (ids, verdicts, counts and timings only)

PUBLIC = ['setup', 'effort', 'question', 'sample_rank', 'job', 'attempts', 'inrun_verdict', 'gpt41_verdict',
          'correct', 'weighted_tokens', 'searches', 'submissions', 'seconds', 'budget_hit', 'error', 'audit']


def cmd_combine(args):
    work = Path(args.work)
    verdict = {r['item']: r['verdict'] for r in bc.read_jsonl(work / 'results.jsonl') if r.get('status') == 200}
    rows, missing = [], []
    for r in csv.DictReader(open(work / 'finals.csv')):
        v = verdict.get(r['item'], '') if r['item'] else ''
        if r['item'] and not v:
            missing.append(r['item'])
        rows.append({**{k: r[k] for k in PUBLIC if k in r}, 'setup': SETUP[r['arm']],
                     'sample_rank': RANK[int(r['question'])], 'gpt41_verdict': v, 'correct': int(v == 'yes')})
    if missing:
        sys.exit(f'{len(missing)} answers have no GPT-4.1 verdict yet: run judge again')
    order = list(SETUP.values())
    rows.sort(key=lambda x: (order.index(x['setup']), x['effort'] != 'low', x['sample_rank']))
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, PUBLIC, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} trials to {args.out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prompts')
    p.add_argument('jobs', nargs='+')
    p.add_argument('--work', required=True)
    p = sub.add_parser('judge')
    p.add_argument('work')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--max-usd', type=float, default=20.0)
    p.add_argument('--limit', type=int, default=0, help='grade only the first N answers (a check before the rest)')
    p.add_argument('--base-url', default='https://api.openai.com/v1')
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('combine')
    p.add_argument('work')
    p.add_argument('--out', required=True)
    args = ap.parse_args()
    {'prompts': cmd_prompts, 'judge': cmd_judge, 'combine': cmd_combine}[args.cmd](args)


if __name__ == '__main__':
    main()
