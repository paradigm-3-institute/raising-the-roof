# Running the harness

The harness runs single agents and teams of Claude Code and Codex agents on long tasks under a fixed budget of
weighted tokens. It is a fork of the harness of Park et al., "Scaling Discovery through Test-Time Communication"
([github.com/jerryjonghopark/test-time-communication](https://github.com/jerryjonghopark/test-time-communication)),
cut down to what the post's runs used.

`run.py` builds a [Harbor](https://github.com/laude-institute/harbor) job: a task directory, a Docker
environment and an agent. `agent.py` runs the agents inside the task's container: each agent is a headless CLI
session (`claude -p` or `codex exec`) driven by a bash runner that sends the task prompt, then continuation turns,
until the agent's budget, its working time or its continuations run out. A team's agents share one container.

| File | What it does |
|---|---|
| `run.py` | command line; builds the task and the Harbor command |
| `agent.py` | the runners, for Claude Code, Codex and mixed teams |
| `multiagent_protocols/` | the prompts of each setup ([setups.md](setups.md)) |
| `frontier_swe.py`, `ale_bench_task.py`, `perf_takehome_task.py` + `scored_artifact.py`, `browsecomp_task.py` | the task adapters |
| `token_budget.py` | token counting from the CLIs' own logs, and the budget watcher |
| `resources_tool.py` | the agents' `resources` command (budget and time left) |
| `limit_resume.py`, `team_hold.py`, `lockstep.py` | usage-limit waits, team pauses, budget pacing |
| `harness_guard.py` | keeps an agent's own `pkill`/`kill -9 -1` from stopping the harness |
| `message_board.py`, `claude_peers_journal.py`, `swarmchat_glue.py`, `ultra.py` | the team channels |
| `pids_limit.py`, `network_allowlist.py` | process cap and network allowlist of every container |
| `revive.py` | continues a trial whose host process died |
| `benchmarks/` | what goes into each task's containers (tools, judges, scorers, Dockerfiles) |
| `vendor/swarmchat/` | upstream SwarmChat, unmodified |
| `codex-patch/` | the Codex patch some runs used ([codex-patch.md](codex-patch.md)) |

## Requirements

- Linux on x86-64, Docker with Compose v2 (`docker info` and `docker compose version` must work). The post's
  runs used hosts with 32 or more cores; a single agent on git-to-zig takes 4 CPUs and 16 GB, a team 8 CPUs and
  32 GB, and an ALE-Bench or take-home agent 2 CPUs and 6 GB.
- [uv](https://docs.astral.sh/uv/). `uv sync` installs Python 3.12 and the pinned packages (Harbor 0.23.0) into
  `.venv`.
- A Claude subscription (Pro or Max) for Claude Code agents, a ChatGPT subscription for Codex agents. API keys work
  too (below).

```bash
uv sync
uv run python run.py --help
```

## The CLIs

Each run mounts a pinned standalone Linux release of the CLI into the task's containers. The post's runs used
Claude Code 2.1.289 and Codex 0.160.0, 0.162.1 and 0.170.0, and later runs a patched build of Codex 0.162.1 and
0.170.0 ([codex-patch.md](codex-patch.md); [experiments.md](experiments.md) lists which run used which).
These helpers download and check a release into `.cache/`, where the commands in [experiments.md](experiments.md)
expect it:

```bash
install_claude() {
  CLAUDE_VERSION="$1"
  CLAUDE_URL="https://downloads.claude.ai/claude-code-releases/${CLAUDE_VERSION}"
  CLAUDE_DIR="$PWD/.cache/claude-${CLAUDE_VERSION}"
  mkdir -p "$CLAUDE_DIR"
  curl -fL "$CLAUDE_URL/linux-x64/claude" -o "$CLAUDE_DIR/claude"
  curl -fsSL "$CLAUDE_URL/manifest.json" | python3 -c "import json, sys; \
print(json.load(sys.stdin)['platforms']['linux-x64']['checksum'] + '  $CLAUDE_DIR/claude')" \
    | sha256sum -c -
  chmod 755 "$CLAUDE_DIR/claude"
}
install_codex() {
  CODEX_VERSION="$1"
  CODEX_URL="https://github.com/openai/codex/releases/download/rust-v${CODEX_VERSION}"
  CODEX_DIR="$PWD/.cache/codex-${CODEX_VERSION}"
  mkdir -p "$CODEX_DIR"
  curl -fL "$CODEX_URL/codex-package-x86_64-unknown-linux-musl.tar.gz" -o "$CODEX_DIR/package.tar.gz"
  curl -fsSL "$CODEX_URL/codex-package_SHA256SUMS" \
    | awk -v a="codex-package-x86_64-unknown-linux-musl.tar.gz" -v f="$CODEX_DIR/package.tar.gz" '$2 == a {print $1 "  " f}' \
    | sha256sum -c -
  tar -xzf "$CODEX_DIR/package.tar.gz" -C "$CODEX_DIR"
  rm "$CODEX_DIR/package.tar.gz"
}
install_claude 2.1.289
install_codex 0.160.0
install_codex 0.162.1
install_codex 0.170.0
```

Pass `--claude-binary` and `--codex-binary` to choose a release. Use the Codex release package, not a bare
`codex` binary: newer models run tools through `codex-code-mode-host`, which ships in the same package.

## Signing in

Runs use your own subscription.

- **Claude Code**: `claude setup-token` prints a long-lived token for your plan. Export it as
  `CLAUDE_CODE_OAUTH_TOKEN` or save it in `~/.claude/oauth-token` (mode 600). Without a token the harness falls
  back to `ANTHROPIC_API_KEY`.
- **Codex**: `codex login` writes `~/.codex/auth.json`; `--codex-auth FILE` picks another login. Without it the
  harness falls back to `CODEX_API_KEY`. The file is mounted read-write and shared by every agent of every
  concurrent trial: Codex replaces its single-use refresh token when it refreshes, and one shared file keeps all
  agents on the same token. To keep experiments apart from your everyday login, sign in again with
  `CODEX_HOME=~/.codex-runs codex login` and pass `--codex-auth ~/.codex-runs/auth.json`.

Agents reach their model API and nothing else while they run (Harbor's egress allowlist, `network_allowlist.py`).

## Running a task

Add `--dry-run` to any command to print the Harbor command without starting anything.

```bash
# git-to-zig (FrontierSWE v2), a single GPT-6.1 Sol agent with 8M weighted tokens and a 48-hour working-time cap
uv run python run.py frontier-swe --task git-to-zig --cli codex --codex-binary .cache/codex-0.162.1/bin/codex \
  --model gpt-6.1-sol --effort high --protocol bare_solo --agents 1 --token-budget 8000000 --resources-tool \
  --timeout 172800 --name my-first-run

# AHC014 (ALE-Bench), a protocol team of 4 Claude Opus 5.5 agents, 2M each
uv run python run.py ale-bench --problem ahc014 --ale-no-private-lane --cli claude \
  --claude-binary .cache/claude-2.1.289/claude --model claude-opus-5-5 --effort high \
  --protocol antiherding_matched --agents 4 --token-budget 2000000 --resources-tool --timeout 864000

# the performance take-home, an orchestrated team (Codex's Ultra mode) of at most 4 agents, 8M in total
uv run python run.py perf-takehome --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-6.1-sol \
  --effort high --protocol ultra --agents 4 --token-budget 8000000 --resources-tool --timeout 864000
```

On first use each task fetches what it needs at a pinned commit and builds its images: FrontierSWE v2 from
Proximal Labs' repository (a sparse clone in `.cache/frontier-swe-v2`, the task's pinned images), ALE-Bench
from Sakana AI's repository and dataset (the judge images are built locally), the take-home from Anthropic's
repository. [tasks.md](tasks.md) has what each task gives the agents and how it is graded. BrowseComp:
[browsecomp.md](browsecomp.md).

### Options

| Option | What it does |
|---|---|
| `--cli claude\|codex --model M` | the agents' CLI and model (`--team` for a mixed team, [setups.md](setups.md)) |
| `--protocol P --agents N` | the setup ([setups.md](setups.md)) |
| `--effort` | reasoning effort (the post: `high`) |
| `--token-budget B` | weighted tokens per agent (Ultra mode: for the whole tree) |
| `--timeout S` | working time per agent in seconds, not counting time spent waiting for usage limits |
| `--resources-tool` | gives agents a `resources` command (tokens and time left) and states the budget left in every continuation |
| `--continuations N` | the most continuation turns (default 1000) |
| `--relay LEGS` | the restarted single agent ([setups.md](setups.md)) |
| `--cpus`, `--memory-mb` | the team's total resources (default: the task's per-agent share times N) |
| `--trials`, `--concurrent` | repeats in one job, and how many run at once |
| `--name` | the job's name, `jobs/<name>/` |
| `--fastgrade` | git-to-zig: grade snapshots with the task's own verifier in a plain container (same scores, far fewer disk writes) |
| `--ale-judge-cpus`, `--ale-no-private-lane` | ALE-Bench: the host CPUs the judge pins cases to; grade only at the end |
| `--codex-compact-limit`, `--claude-compact-window` | where each CLI compacts its context |
| `--limit-wait-max H`, `--no-limit-resume` | usage-limit waits (below) |
| `--no-harness-guard`, `--pids-limit` | harness protection, the containers' process cap |

## Budget, continuations and stopping

**Weighted tokens.** The budget counts CAISI's weighted tokens (appendix A2 of CAISI's evaluation of DeepSeek
models): output tokens at 100%, unique (uncached) input tokens at 23%, all input tokens at 2%. Tokens are read
from the CLIs' own session logs as they are written (`token_budget.py`), so a turn stopped part way counts too.
Once an agent reaches its budget, a watcher stops its CLI and everything the CLI started.

**Continuations.** When an agent ends its turn, its runner sends the task's continuation text
(`benchmarks/<task>/continuation.txt`), with the budget left when `--resources-tool` is on, in the same session.
An agent that ends 10 continuation turns in a row, each under 60 seconds and with at most 2 tool calls, is
stopped as having given up (`agent_gave_up` in its log).

**Working time.** `--timeout` caps each agent's working time. Time spent waiting for a usage limit to reset does
not count (below).

**Usage limits** (on by default, `limit_resume.py`). An agent that hits its subscription's usage limit waits for
the reset and then resumes its interrupted turn without its prompt being sent again. Waiting counts against
`--limit-wait-max` (default 48 hours in total per agent), not against `--timeout`. With `--team-limit-hold`
(default for `--team`) the whole team pauses while one agent waits.

**Harness protection** (on by default, `harness_guard.py`). Agents may run as root in the task image, and a
`pkill -f python` or `kill -9 -1` of theirs would otherwise stop the harness's watchers and runners. A keeper in
the container runs each agent's runner and watchers under names agents' patterns do not match and starts again
whatever dies; a CLI ended by a signal resumes its turn with a note naming the command that was running. A trial
whose protection cannot be kept ends cleanly and is graded.

**Content filter.** A reply the API's content filter blocks ends that turn; the agent continues with the usual
continuation. After 5 blocks in a row (`--content-filter-max-blocks`) with no output in between, the agent is
stopped.

## Reviving a trial

`revive.py` continues a trial whose host process died (a host restart, say) while its containers and logs
survived: each agent resumes its own CLI session with one note, within the run's budgets counting what it had
already used. For git-to-zig it can also restore a trial whose containers are gone, from its snapshot store.

```bash
uv run python revive.py plan jobs/<job>/<trial> -- <the run's run.py arguments>   # what it would do
uv run python revive.py start jobs/<job>/<trial> -- <the run's run.py arguments>
```

## Outputs

Each job is `jobs/<name>/`, with Harbor's `config.json` and `result.json` and one directory per trial:

```
jobs/<name>/<task>__<id>/
  agent/                          what the agents' container wrote under /logs/agent
    sessions-<i>-<agent>/         one per agent (single agent of a protocol without a solo prompt: codex-single-session/)
      codex-home/ or claude-home/ the CLI's home: its session logs (Codex rollouts, Claude Code transcripts)
      <cli>-attempt-<turn>.jsonl  the CLI's output for each turn
      token-usage.json            the agent's tokens as the budget watcher last counted them
      limit-wait.json             waiting and working time
    agent-<i>-<agent>-output.log  the runner's log, with every harness event (continuations, waits, stops)
    multiagent-collab/            the protocol (protocol.json), the shared prompt and the runners
    message-board.jsonl           board teams: the board's journal (and .sqlite after the run)
    claude-peers.jsonl            Claude teams: the team's messages
    swarmchat/                    mixed self-organized teams: the chat's data
    ultra.json                    Ultra mode: the run's settings
  frontier-swe/                   git-to-zig: promotions and snapshot grades over time (snapshots.jsonl, grades/)
  ale-bench/                      AHC: every submission and its grade, the final grade
  scorer/                         take-home: every submission and its cycle count, the final grade
  browsecomp/                     BrowseComp: the answer log, grades and audit
  verifier/                       Harbor's verifier output (the reward)
```

The task-specific folders are written by the judge or scorer on the host, out of the agents' reach.

## Tests

```bash
uv run pytest
```

Many tests run the pinned Claude Code and Codex releases against a local mock of the model API (no login, no
model); they are skipped when the releases are not in `.cache/`. A few need Docker and a task image and are
skipped without them.
