# Data

The results behind the post, one row per run. The scripts in [`../analysis/`](../analysis/) turn these files into
every number and chart in the post. Data as of 2026-11-24.

Not included: runs that were not part of the post (pilots, smoke tests, other tasks), runs still going, and Claude's
self-organized teams on AHC014, AHC038 and the take-home, which did not finish in time for the post. Transcripts and
run folders are not in this repo.

## runs.csv

One row per run. Runs left out of the post's numbers stay as rows, with `counted` = 0 and the reason.

| Column | Meaning |
|---|---|
| `run` | Run name: setup, team size x budget per agent, task, model, repeat (e.g. `b162-8x4m-board-gitzig-sol-r6`) |
| `part` | Which comparison the run belongs to: `main`, `restart`, `teams-16-32`, `patched-rerun`, `no-talk`, `protocol-prompt-single`, `mixed-models`, `model-generations`, `context-window` (an extra check, not discussed in the post) |
| `task` | `gitzig` (git-to-zig), `ahc014`, `ahc038`, `perf` (performance take-home) |
| `model` | `opus`, `sonnet` (Claude 5.5), `opus5`, `sonnet5`, `sol`, `terra` (GPT-6.1), `sol62` (GPT-6.2 Sol), `terra56` (GPT-5.6 Terra), `mixed` (Claude Sonnet 5.5 + GPT-6.1 Sol), `opussol` (Claude Opus 5.5 + GPT-6.1 Sol) |
| `model_id` | The model's API name; `+` joins the two models of a mixed team |
| `setup` | `solo` single agent; `relay` restarted single agent; `ttc` protocol team; `board`, `peers`, `chat` self-organized team; `ultra` orchestrated team; `notalk` one agent of a no-talk pool; `solo_ttc` single agent given the protocol prompt; `relayctx` single agent that starts a new session whenever its context fills |
| `setup_name` | The setup's name in the post |
| `channel` | How a self-organized team talks: `message board` (Codex), `direct messages` (Claude Code), `SwarmChat` (mixed teams) |
| `team_size` | Agents in the team (1 for single agents, 8 sessions for the restarted single agent, k of the pool for no-talk agents) |
| `budget_M` | Total budget, millions of weighted tokens (a no-talk agent's own budget) |
| `per_agent_M` | Budget of each agent; empty for orchestrated teams, which share one budget |
| `variant` | `protocol prompt` / `plain prompt` (no-talk agents and single agents given the protocol prompt), `compacts at 61k` etc. (context-window check) |
| `series` | `rerun`: the 32M self-organized teams of 8 rerun on the patched Codex build |
| `repeat` | Repeat number of the configuration (pool slot for no-talk agents) |
| `codex` | Codex version for GPT agents; `patched` = with the fix for the compaction bug described in the methodology; empty for Claude Code runs |
| `status` | `done`, `failed` (ended with an error and has no final), `stopped` (never finished) |
| `counted` | 1 if the run counts in the post's numbers |
| `why_not` | Why a run does not count |
| `final` | Final score: git-to-zig % of tests passed; AHC AtCoder performance; take-home clock cycles (lower is better) |
| `raw_per_case_M` | AHC014 only: raw private score per test case, millions (AtCoder performance stops at 3389, first place) |
| `final_source` | Where the final comes from: `result.json` (the run's own grading), `AHC regrade` (every AHC final regraded on one machine after the runs; mean of the passes that scored), `in-run grade` (AHC run not regraded), `final-regrade.json` or `regrade` (the run's own final grading failed, so its final code was graded again) |
| `weighted_tokens_M` | Weighted tokens spent, millions, from the budget tracker: output + 0.23 x uncached input + 0.02 x all input (so an uncached input token weighs a quarter of an output token and a cached one 2%) |
| `spend_share` | `weighted_tokens_M` / budget |
| `input_tokens`, `cached_input_tokens`, `cache_write_tokens`, `output_tokens` | Raw token counts summed over the run's agents (cache writes: Claude only). `tokens.csv` splits them by model and price tier |
| `model_calls` | Model calls, from the agents' logs; empty where the logs were not kept |
| `active_hours` | Working hours: time from start to end minus waits for usage limits, pauses and host downtime |
| `elapsed_hours` | Wall-clock hours from start to end |
| `end_reason` | How the agents stopped: `budget`, `time cap`, `gave up` (or `gave up (2 of 8 agents)`), `error`, `other`, `not recorded` |
| `flags` | `time-cap`, `gave-up`, `under-budget` (spent under 85%), `failed`, `revived` (resumed after the host process died), `paused`, `content-filter`, `in-run-grade`, `regraded`, `at-ceiling` (AHC014 at 3389), `slow-judge` (AHC grading feedback slow, median wait 10 minutes or more), `waited` (spent half its working time or more inside tool calls of 10 minutes or more, mostly waiting on its own background jobs; measured for runs of the main comparison) |
| `notes` | Details behind the flags |
| `started` | Start date (UTC) |

## Other files

| File | What |
|---|---|
| `tokens.csv` | Tokens per run, model and price tier: `base`, `long` (OpenAI calls with a prompt over 272k tokens, priced higher), `no log` (only the run's totals were kept). Columns: `run`, `model_id`, `tier`, `model_calls`, `input_tokens`, `cached_input_tokens`, `cache_write_tokens`, `output_tokens` |
| `prices.csv` | API prices in USD per million tokens, as listed on models.dev on 2026-11-23 |
| `snapshots.csv` | git-to-zig: every graded in-run snapshot (`run`, `weighted_tokens`, `score` in % of tests passed), for the runs that recorded tokens with it |
| `score_events.csv` | GPT-6.1 Sol at 8M, single agent and teams of 8 on all four tasks: each new score during the run (`run`, `active_hours`, `weighted_tokens`, `score`). AHC scores here are public-test-case scores, not AtCoder performance |
| `browsecomp_trials.csv` | The BrowseComp appendix: one row per trial, below |

## browsecomp_trials.csv

GPT-5.6 Sol on 253 BrowseComp questions in six settings, one row per setting and question (1,518 rows). During
the run, the harness's judge (gpt-6-luna) graded each answer with simple-evals' grader prompt. After the run,
simple-evals' own judge, GPT-4.1, graded every final answer once with the same prompt and settings, as in OpenAI's
evaluation; the post's scores use its verdicts, and a trial without an answer counts as wrong.
[`analysis/browsecomp/regrade.py`](../analysis/browsecomp/regrade.py) made this file. No question, answer or judge
text is here.

| Column | Meaning |
|---|---|
| `setup` | `single agent`, `orchestrated` (Codex Ultra, at most 4 agents), `protocol` (team of 4), `self-organized` (team of 4) |
| `effort` | Reasoning effort: `low` or `high` |
| `question` | Row of the question in OpenAI's BrowseComp CSV, from 1 |
| `sample_rank` | Its place in the seeded order of all 1,266 questions (`random.Random(20261018).sample(range(1, 1267), 1266)`); the sample is ranks 1 to 253 |
| `job` | The job the trial ran in ([experiments.md](../docs/experiments.md)) |
| `attempts` | Graded trials of this setting on this question. Where there were several, the latest counts, preferring one that ended without a harness error |
| `inrun_verdict` | The in-run judge's verdict on the final answer: `yes`, `no`, empty for no answer |
| `gpt41_verdict` | GPT-4.1's verdict on the final answer (one call): `yes`, `no`, empty for no answer |
| `correct` | The trial's score: 1 if `gpt41_verdict` is `yes`, else 0 |
| `weighted_tokens` | Weighted tokens spent by all the trial's agents (the weighting of `runs.csv`) |
| `searches` | Web searches through the search filter |
| `submissions` | Answers submitted (the last one is the team's) |
| `seconds` | Wall-clock time of the trial; empty where not recorded |
| `budget_hit` | 1 if an agent ran out of its token budget |
| `error` | The harness error the trial ended with, if any (`AgentTimeoutError`: the time cap) |
| `audit` | The transcript scan ([browsecomp.md](../docs/browsecomp.md)): `clean`, `review` (weak indicators only), `inconclusive` (files skipped), `suspected-contamination` (a strong indicator); empty for one trial with no audit. The post's scores count every trial, whatever its audit |
