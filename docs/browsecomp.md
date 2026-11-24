# BrowseComp

`run.py browsecomp` (`browsecomp_task.py`, `benchmarks/browsecomp/`) runs questions of
[BrowseComp](https://openai.com/index/browsecomp/) (Wei et al., OpenAI, 2025): 1,266 hard-to-find facts, each with
a short answer that an LLM judge grades. It started from Harbor PR #2355 ("Adapter: browsecomp", Brian Guo);
credits and licences are in `benchmarks/browsecomp/NOTICE.md`. Search is Codex's own web search, so it runs with
`--cli codex` only.

**The dataset is never part of this repository.** `python browsecomp_task.py fetch` downloads OpenAI's encrypted
test set to `.cache/browsecomp/` and checks its SHA-256 and row count; questions are decrypted in memory only.
Do not publish decrypted questions, answers, generated task directories or trial logs that quote them.

## The post's runs

GPT-5.6 Sol on 253 questions: the first 253 of a seeded random order of all question ids
(`random.Random(20261018).sample(range(1, 1267), 1266)`; ids are 1-based rows of the CSV), at low and high
effort, with the search filter on. Commands: [experiments.md](experiments.md), section BrowseComp.

| Arm (job names) | Setup | Budget, time cap |
|---|---|---|
| `solo` | single agent, the question alone (`--protocol task_only --agents 1`) | 1M weighted tokens, 2 h |
| `ultra` | orchestrated team, at most 4 agents (`--protocol ultra --agents 4`) | 1M for the whole tree, 2 h |
| `paper` | protocol team of 4 (`--protocol antiherding_matched --agents 4`) | 1.25M per agent, 4 h |
| `board` | self-organized team of 4 (`--protocol board --agents 4`) | 1.25M per agent, 4 h |

Some questions ran more than once in a setting (61 of the 1,518 setting-question pairs). The latest graded trial
counts, preferring one that ended without a harness error.

## The task

One trial is one question: one container, the agents (one or a team) on that question, one answer. The task text
is simple-evals' `QUERY_TEMPLATE` with the question, then how the run works. Agents submit with `submit-answer`;
`answers` lists every submission. The answer log (`benchmarks/browsecomp/service/server.py`) is its own container
next to the agents': it records each submission in `<trial>/browsecomp/submissions.jsonl` on the host and holds no
answer key. The team's answer is the last submission. An agent's work ends when its turn ends after it has
submitted (the harness's episode watcher); an agent that ends a turn without submitting gets a continuation asking
it to. Each agent gets 2 CPUs and 4 GB. Nothing the agents see names the benchmark, and the container holds no
dataset file, answer or canary (checked when each task is written).

## The search filter

`BROWSECOMP_SEARCH_FILTER=on` runs each trial with a relay (`benchmarks/browsecomp/relay/proxy.py`), its own
container next to the agents' and the only host their network reaches while they run. The agents' Codex reaches
its model API through it (its base URLs and token refresh point at the relay, over TLS with a CA made for the
relay). In every search the relay:

- blocks Hugging Face and GitHub (`benchmarks/browsecomp/relay/blocklist.txt`);
- withholds every result and opened page that quotes one of the 1,266 questions: at least 15 words of a question
  in runs of 10 or more consecutive words (outside the question's own quotation marks), or half of its 5-word
  phrases;
- removes Codex's hosted `web_search` tool, whose results it cannot see, so searches go through Codex's standalone
  search, which it can;
- forwards only Codex's model endpoints and token refresh, and logs every search with the URLs it returned
  (`<trial>/search-filter/`).

The question texts are mounted only into the relay. `BROWSECOMP_SEARCH_FILTER=dry-run` withholds nothing and logs
what it would have withheld. `benchmarks/browsecomp/relay/relay-selftest.sh` tests the relay on its own.

## Grading

After the run, on the host, `run.py` grades each distinct response with simple-evals' `GRADER_TEMPLATE` (parsed
with the `group(1)` fix of openai/simple-evals#67, failing closed to "no"), then audits the trial. The judge in
this code is a Codex model on the harness's ChatGPT login (`browsecomp_task.JUDGE_MODEL`, `gpt-6-luna` at high
effort, every tool off, in a throwaway container), because simple-evals' judge, `gpt-4.1-2025-04-14`, needs an
OpenAI API key. Grades go to `<trial>/browsecomp/grades.jsonl` and `final.json`. Harbor's own reward is 1 when the
team submitted an answer, not the grade.

**The post's scores.** After the runs, GPT-4.1 (`gpt-4.1-2025-04-14`) graded the final answer of every trial once,
with simple-evals' prompt and settings (temperature 0.5, 2,048 tokens, "You are a helpful assistant."), as in
OpenAI's evaluation; its verdict is the trial's score, and the 28 trials that ended without an answer count as wrong.
The code is [`analysis/browsecomp/regrade.py`](../analysis/browsecomp/regrade.py) (it needs the run folders and an
OpenAI API key), the per-trial result is [`data/browsecomp_trials.csv`](../data/browsecomp_trials.csv), and
`python3 analysis/browsecomp.py` computes the appendix's numbers from it.

```bash
python browsecomp_task.py grade jobs/<job> [--force]   # grade again
python browsecomp_task.py audit jobs/<job>             # answer-source indicators
python browsecomp_task.py summary jobs/<job>           # raw and audited accuracy, tokens, time
```

**The audit.** The audit scans what the agents saw and wrote (their session logs with search results, outputs and
submissions) for signs that an agent looked for the benchmark or its answers: any mention of BrowseComp, the
dataset's canary, simple-evals or decryption code (strong indicators), and dataset mirrors, OpenAI's GitHub and the
word "canary" (weak ones, for review). `audit.json` names files, indicators and line numbers, never the text.
`summary` reports raw accuracy and audited accuracy, in which a trial with a strong indicator counts as wrong.
The post's scores are raw: they count every trial, whatever its audit (each trial's status is in
`data/browsecomp_trials.csv`). `analysis/browsecomp.py` also prints them with the trials flagged
`suspected-contamination` counted as wrong.
