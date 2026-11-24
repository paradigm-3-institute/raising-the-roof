"""The BrowseComp appendix: our six scores, their 95% intervals and the comparisons with OpenAI's, from
data/browsecomp_trials.csv.

    python3 analysis/browsecomp.py

One row per trial: GPT-5.6 Sol on the first 253 questions of a seeded order of BrowseComp's 1,266, in six settings.
`correct` is GPT-4.1's verdict on the trial's final answer (one call, simple-evals' grader; analysis/browsecomp/regrade.py
made it), 0 for a trial without an answer. A setting's score is its mean over the 253 questions; the interval is a
bootstrap percentile 95% interval over questions (10,000 draws, seeded). The second table counts the trials the audit
flags as `suspected-contamination` as wrong.
"""
import csv
import random
import statistics as st

from common import DATA, boot, seed_of

# OpenAI's published BrowseComp scores for GPT-5.6 Sol, alone and as a team of agents in Codex, on all 1,266
# questions: the GPT-5.6 announcement, https://openai.com/index/gpt-5-6/
OPENAI = {('single agent', 'low'): 69.0, ('single agent', 'high'): 88.2,
          ('orchestrated', 'low'): 81.8, ('orchestrated', 'high'): 90.8}
SETTINGS = [('single agent', 'low'), ('single agent', 'high'), ('orchestrated', 'low'), ('orchestrated', 'high'),
            ('protocol', 'high'), ('self-organized', 'high')]
N = 253
FLAG = 'suspected-contamination'

rows = list(csv.DictReader(open(DATA / 'browsecomp_trials.csv')))
order = random.Random(20261018).sample(range(1, 1267), 1266)
by = {s: [r for r in rows if (r['setup'], r['effort']) == s] for s in SETTINGS}
assert len(rows) == sum(map(len, by.values()))
for s, rs in by.items():
    assert sorted(int(r['question']) for r in rs) == sorted(order[:N]), s
for r in rows:
    assert r['correct'] == str(int(r['gpt41_verdict'] == 'yes')), r
    assert r['gpt41_verdict'] or r['inrun_verdict'] == '', r   # only a trial without an answer has no verdict

name = lambda s: f'{s[0]}, {s[1]} effort'


def table(title, value):
    """Each setting's score and interval with value(row) as a trial's score."""
    score, ci = {}, {}
    print(f"{title}\n{'Setting':28} {'Score':>6}  {'95% interval':14} {'OpenAI':>7}")
    for s, rs in by.items():
        vals = [value(r) for r in rs]
        score[s] = 100 * st.mean(vals)
        lo, hi = boot(random.Random(seed_of(('browsecomp',) + s)), [vals], st.mean)
        ci[s] = (100 * lo, 100 * hi)
        o = f'{OPENAI[s]:.1f}%' if s in OPENAI else '-'
        print(f'{name(s):28} {score[s]:5.1f}%  {ci[s][0]:4.1f} to {ci[s][1]:4.1f}  {o:>7}')
    return score, ci


score, ci = table('GPT-4.1 on every final answer', lambda r: int(r['correct']))
print()
flagged = sum(r['audit'] == FLAG for r in rows)
table(f'The same, with the {flagged} trials the audit flags as {FLAG} counted wrong',
      lambda r: int(r['correct']) * (r['audit'] != FLAG))

answered = [r for r in rows if r['gpt41_verdict']]
agree = sum(r['gpt41_verdict'] == r['inrun_verdict'] for r in answered)
print(f'\n{len(answered)} of {len(rows)} trials have an answer, each graded by one GPT-4.1 call; the in-run judge agrees '
      f'on {agree} ({100 * agree / len(answered):.1f}%): it said yes where GPT-4.1 said no on '
      f"{sum(r['inrun_verdict'] == 'yes' and r['gpt41_verdict'] == 'no' for r in answered)}, the reverse on "
      f"{sum(r['inrun_verdict'] == 'no' and r['gpt41_verdict'] == 'yes' for r in answered)}")

half = [(hi - lo) / 2 for lo, hi in ci.values()]
print(f'Intervals: ±{min(half):.1f} to ±{max(half):.1f} points')

# The post compares the one-decimal scores it shows.
r1 = {s: round(v, 1) for s, v in score.items()}
solo = ('single agent', 'high')
print(f"\nGain from a team at high effort, OpenAI: {OPENAI[('orchestrated', 'high')] - OPENAI[solo]:.1f} points")
gains = {s[0]: r1[s] - r1[solo] for s in SETTINGS if s[1] == 'high' and s != solo}
exact = {s[0]: score[s] - score[solo] for s in SETTINGS if s[1] == 'high' and s != solo}
print(f'Ours: {min(gains.values()):.1f} to {max(gains.values()):.1f} points from the scores as shown '
      f'({", ".join(f"{k} {v:.1f}" for k, v in gains.items())}); '
      f'{min(exact.values()):.1f} to {max(exact.values()):.1f} unrounded')
below = {s: OPENAI[s] - r1[s] for s in OPENAI}
print('Below OpenAI: ' + ', '.join(f'{name(s)} {v:.1f}' for s, v in below.items()))
