# Setups

How each setup in the post maps to the harness. The post's names come first; the code name (protocol, flag)
is in brackets. Every setup runs the same task, the same continuation turns and the same weighted token budget;
only the prompt, the channel between agents and the agent count change. Exact commands for every run are in
[experiments.md](experiments.md).

| Post | Code | Flags |
|---|---|---|
| Single agent | `bare_solo` (BrowseComp: `task_only`) | `--protocol bare_solo --agents 1` |
| Protocol team of N | `antiherding_matched`, N agents | `--protocol antiherding_matched --agents N` |
| Matched single agent | `antiherding_matched`, 1 agent | `--protocol antiherding_matched --agents 1` |
| No-talk team (protocol prompt) | `antiherding_matched`, lone agents | N separate runs of `--agents 1` at budget B/N |
| No-talk team (plain prompt) | `bare_solo`, lone agents | N separate runs of `--agents 1` at budget B/N |
| Self-organized team, GPT | `board` (Codex's message board) | `--cli codex --protocol board --agents N` |
| Self-organized team, Claude | `claude_peers` (Claude Code's cross-session messaging) | `--cli claude --protocol claude_peers --agents N` |
| Self-organized team, mixed | `swarmchat` (SwarmChat) | `--team ... --protocol swarmchat --lockstep-gap 250000` |
| Protocol team, mixed | `antiherding_matched` with `--team` | `--team ... --protocol antiherding_matched` |
| Orchestrated team of N | `ultra` (Codex's Ultra mode) | `--cli codex --protocol ultra --agents N` |
| Restarted single agent | relay | `--protocol bare_solo --relay 8` |

`--token-budget` is always per agent, except in Ultra mode, where it is the budget of the whole tree (the
orchestrator and its helpers). A team of N at a total of 8M weighted tokens is `--agents N --token-budget
8000000/N`.

## Single agent (`bare_solo`)

One agent, one session, with the task's instruction alone. It runs through the team runner with one agent, so
its session layout, budget, usage-limit waits and `resources` command are the same as a team member's
(`multiagent_protocols/bare_solo.py`).

## Protocol teams (`antiherding_matched`)

N agents share one container and a folder of notes. Each gets the anti-herding prompt of Park et al.,
"Scaling Discovery through Test-Time Communication" (`multiagent_protocols/antiherding.py`, unchanged from their
repository), before the task's instruction: claim a numbered slot with a distinct approach, log findings with
evidence, spend part of the time disconfirming the leading approach, and adopt a teammate's approach only past an
evidence bar. With N ≥ 2 `antiherding_matched` sends exactly that prompt (`multiagent_protocols/antiherding_matched.py`).

## Matched single agent (`antiherding_matched --agents 1`)

The same prompt without the parts about teammates: the agent still claims its one slot and keeps the findings,
disconfirmation and board files, as private notes. Each edit to the team text is listed in
`multiagent_protocols/antiherding_matched.py` (`DROPPED_SECTIONS`, `EDITS`).

## No-talk teams

A no-talk team of N is N lone agents, each with budget B/N and no channel to the others; the team's score is
the best agent's. The post did not run each no-talk team on its own: for each N it ran a pool of lone agents
(6 at 4M, 8 at 2M, 12 at 1M on a total of 8M) and averaged the best of every group of N drawn from the pool.
Pool runs are ordinary single-agent runs, with the protocol prompt (`antiherding_matched --agents 1`, run names
`nt...`) or the plain prompt (`bare_solo`, run names `ntbare...`). On git-to-zig each pool agent got its share of
a team's resources (`--cpus 8/N --memory-mb 32768/N`).

## Self-organized teams on Codex's message board (`board`)

Codex contains a message board shared by a team of agents (channels, threads, posts, search, subscriptions, and
notifications in the running turn), disabled by default. The harness turns it on in each agent's
`config.toml` (`features.multi_agent_v2.message_board_remote`, Codex 0.162.1 or later) and runs a board server in
the container (`message_board.py`) that reproduces the built-in board's behaviour for agents that run as separate
`codex exec` processes. The agents see Codex's own board tools, tool descriptions and notifications. Each agent's
prompt sets a shared goal and introduces the board, nothing more (`multiagent_protocols/board.py`):

```
You are one of N agents working on this task at the same time, in this container, all
with this same prompt. You are one team: the team is scored on its best result, so
whatever any one of you achieves counts for all of you.

The team shares a message board, like a team Slack: channels, threads and posts that
everyone can read, search and subscribe to.
Your teammates are /root/a1, /root/a2, .... The board's tools (post, read_thread, subscribe, ...) are collaboration tools.

How you split the work, what you share, and when you talk is up to you. Work it out
among yourselves.
```

The board's journal is kept in the trial's agent directory (`message-board.jsonl`, and `message-board.sqlite`
after the run).

## Self-organized Claude teams (`claude_peers`)

Each agent is a `claude -p` session named by its id; agents message one another with Claude Code's
cross-session messaging (its `SendMessage` and `ListAgents` tools). The prompt is the board prompt with the board
paragraph replaced by "You can message one another, like direct messages in a team Slack: each message goes to
one teammate." (`multiagent_protocols/claude_peers.py`). Needs exactly Claude Code 2.1.289, the release it was
built and tested on. After the run the team's messages are written to `claude-peers.jsonl`
(`claude_peers_journal.py`).

## Self-organized mixed teams (`swarmchat`)

Claude Code and Codex agents share a chat run by upstream SwarmChat (`vendor/swarmchat`, unmodified;
`swarmchat_glue.py` is the only glue). The chat is a shell command, `swarmchat`, and new messages are pushed into
each agent's conversation after its tool calls by a PostToolUse hook in both CLIs. The prompt is the board
prompt with the board paragraph replaced by upstream's own `prompts/chat.md`
(`multiagent_protocols/swarmchat.py`). A mixed team spends its budgets in lockstep (`--lockstep-gap 250000`,
`lockstep.py`): an agent more than 250k weighted tokens ahead of the slowest agent still working is paused
between model responses until that agent catches up; the pause counts as neither working time nor waiting.
Its Codex agents ran the patched build ([codex-patch.md](codex-patch.md)).

## Mixed protocol teams (`--team`)

`--team CLI:MODEL[:EFFORT],...` gives each agent its own CLI and model, in agent order (`agent.py`
`MixedTeamAgent`). Each agent keeps its CLI's login, budget, usage-limit waits and revival. With `--team`,
`--team-limit-hold` is on: while any agent waits for a usage limit to reset, the others pause too, so the team
never works short-handed (`team_hold.py`).

## Orchestrated teams (`ultra`)

One Codex agent that delegates to its own helpers (threads of the same `codex exec`), at most N agents at once,
the orchestrator included. The harness turns on Codex's multi-agent mode with the instruction Codex itself sends
in Ultra mode (`ultra.PROACTIVE_MODE_TEXT`, verbatim) and caps the threads at N (`ultra.py`,
`multiagent_protocols/ultra.py`). The prompt starts with "At most N agents, you included, can work on this task
at the same time." and "Your helpers stop when you end your turn, so wait for them to finish before you end it."
Unlike Codex's own Ultra setting, every agent runs at the run's `--effort`. One token budget covers the whole
tree, and the run gets the resources of a team of N. Each helper works in its own named copy (`workcopy NAME`).

## Restarted single agent (`--relay`)

One agent's run split into LEGS sessions, one after another in the same container, each a fresh CLI session
with `--token-budget` of its own (`agent.py`, `RELAY_ENV`). A leg ends on its budget, by giving up, or after its
continuations; the next leg gets the same prompt preceded by "You are building on top of a previous agent's
work: another agent worked on this task before you, in this same environment, and what it left is still here.
Continue the task." Files on disk carry over; the conversation does not. `--timeout` and `--limit-wait-max`
cover the whole run.
