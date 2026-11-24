"""Lockstep token pacing (run.py --lockstep-gap, lockstep.py).

The rule, the watcher and the runner's wait are exercised by hand. The end-to-end tests run teams of Claude
Code and Codex agents as Harbor does, on this machine (test_harness_guard.py's Local), with the keeper, every
CLI on a mock API that reports the tokens its plan gives each response (tests/limit_mock_api.py): no model is
called and no quota spent. They are skipped without the pinned releases."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from cli_render import INSTRUCTION, ROOT
from test_harness_guard import GUARD, LIMIT, Local, kill, written
from test_limit_resume import CLAUDE_BINARY, CODEX_BINARY, Mock, needs_claude, needs_codex, user_texts
from test_mixed_team import HOLD, SONNET_SOL, render_team, run_py, runners, team_env

sys.path.insert(0, str(ROOT))
import harness_guard  # noqa: E402
import limit_resume  # noqa: E402
import lockstep  # noqa: E402
import team_hold  # noqa: E402

M = 1_000_000
GAP = 250_000
ON = {"RELEASE_LOCKSTEP_GAP": str(GAP), "RELEASE_LOCKSTEP_REF": "slowest", "RELEASE_LOCKSTEP_STALL": "3600",
      "RELEASE_TOKEN_BUDGET": "4000000"}


def agents(*tokens, budget=4 * M, ended=()) -> list[dict]:
    budgets = budget if isinstance(budget, (list, tuple)) else [budget] * len(tokens)
    return [{"index": i, "agent": f"a{i + 1}", "tokens": float(t), "budget": float(budgets[i]), "ended": i in ended}
            for i, t in enumerate(tokens)]


def paused(team: list[dict], kind: str = "slowest", gap: float = GAP, now: set = frozenset()) -> set:
    return {a["index"] for a in team if lockstep.assess(team, a["index"], kind, gap, a["index"] in now)["pause"]}


# ---- the rule ---------------------------------------------------------------------------------------------

def test_a_pair_pauses_the_one_ahead_until_the_other_catches_up():
    assert paused(agents(M + GAP, M)) == set()              # GAP ahead is not more than GAP
    found = lockstep.assess(agents(M + GAP + 1, M), 0, "slowest", GAP, False)
    assert found["pause"] and found["lead"] == pytest.approx(GAP + 1) and found["reference"] == pytest.approx(M)
    assert found["gap"] == GAP and found["working"] == 2
    assert paused(agents(M + GAP + 1, M)) == {0}            # the reference agent is never paused
    # A paused agent stays paused while it is ahead at all, and goes on once it is not.
    assert paused(agents(M + 1, M), now={0}) == {0}
    assert paused(agents(M, M), now={0}) == set()
    assert paused(agents(M, M + 5), now={0}) == set()


def test_four_and_four_with_each_reference():
    # Four Sonnet agents behind, four Sol agents ahead.
    team = agents(1.0 * M, 1.1 * M, 1.2 * M, 1.3 * M, 1.6 * M, 1.8 * M, 2.0 * M, 2.2 * M)
    # slowest: 1.0M, so everyone more than 1.25M; median: 1.45M (between the groups), so more than 1.7M;
    # mean: 1.525M, so more than 1.775M.
    assert paused(team, "slowest") == {3, 4, 5, 6, 7}
    assert paused(team, "median") == {5, 6, 7}
    assert paused(team, "mean") == {5, 6, 7}
    assert lockstep.reference([1, 2, 3, 4], "median") == 2.5 and lockstep.reference([3, 1, 2], "median") == 2
    assert lockstep.reference([1, 2, 3, 6], "mean") == 3 and lockstep.reference([4, 2, 9], "slowest") == 2


def simulate(kind: str, rates: list[float], budget: float = 4 * M, gap: float = GAP, step: float = 1000.0) -> dict:
    """Agents spending at RATES (tokens per tick) under the rule until every budget is spent: the pauses, the
    largest lead of any agent over the slowest still working, and the tokens spent after the first agent of
    the fastest group ended (the slow agents' solo tail)."""
    tokens, paused_now, pauses, lead_max, tail_from = [0.0] * len(rates), set(), 0, 0.0, None
    fast = max(rates)
    while any(t < budget for t in tokens):
        team = agents(*tokens, budget=budget, ended=[i for i, t in enumerate(tokens) if t >= budget])
        for agent in team:
            found = lockstep.assess(team, agent["index"], kind, gap, agent["index"] in paused_now)
            if found["pause"] and agent["index"] not in paused_now:
                pauses += 1
                paused_now.add(agent["index"])
            elif not found["pause"]:
                paused_now.discard(agent["index"])
        working = [t for t in tokens if t < budget]
        lead_max = max(lead_max, max(working) - min(working)) if len(working) > 1 else lead_max
        for i, rate in enumerate(rates):
            if i not in paused_now and tokens[i] < budget:
                tokens[i] = min(tokens[i] + rate * step, budget)
        if tail_from is None and any(t >= budget for t, r in zip(tokens, rates) if r == fast):
            tail_from = sum(t for t, r in zip(tokens, rates) if r != fast)
    slow = [r for r in rates if r != fast]
    return {"pauses": pauses, "lead_max": lead_max,
            "tail": (budget * len(slow) - tail_from) / len(slow) if slow else 0.0}


def test_why_the_slowest_agent_is_the_default_reference():
    """4 Sonnet + 4 Sol (Sol 1.5 times as fast, the Sonnet agents slightly apart), over budgets that end the run
    at different points of the pause cycle. With the median or the mean the reference sits between the
    groups, and a paused Sol agent goes on only once it is back at the reference, which (counting it) means once
    the Sonnet group has caught up: the Sol group runs up to 2 x GAP ahead, as with the slowest reference and
    twice the gap. So it pauses half as often, but each Sonnet agent works on alone at the end for several times
    as many tokens."""
    rates = [1.0, 1.01, 1.02, 1.03, 1.5, 1.5, 1.5, 1.5]
    budgets = (3.8 * M, 4.0 * M, 4.2 * M)
    results = {kind: [simulate(kind, rates, budget) for budget in budgets] for kind in lockstep.REFERENCES}
    lead = {kind: max(r["lead_max"] for r in found) for kind, found in results.items()}
    tail = {kind: sum(r["tail"] for r in found) / len(found) for kind, found in results.items()}
    pauses = {kind: sum(r["pauses"] for r in found) / len(found) for kind, found in results.items()}
    assert lead["slowest"] <= GAP + 2000 and lead["median"] > 2 * GAP and lead["mean"] > 2 * GAP
    assert tail["slowest"] < GAP / 2 and tail["median"] > 4 * tail["slowest"] and tail["mean"] > 4 * tail["slowest"]
    assert pauses["slowest"] > 2 * pauses["median"] - 1
    # A pair: without lockstep the slow agent works on alone for 4M x (1 - 1/1.5) ~ 1.33M; with it, < GAP.
    for budget in budgets:
        pair = simulate("slowest", [1.0, 1.5], budget)
        assert pair["tail"] < GAP and pair["lead_max"] <= GAP + 2000 and 4 <= pair["pauses"] <= 6


def test_ended_agents_drop_out_and_one_agent_alone_never_pauses():
    # The slowest agent has ended: the next slowest is the reference.
    assert paused(agents(2.0 * M, 1.0 * M, 1.6 * M, ended={1})) == {0}
    assert lockstep.assess(agents(2.0 * M, 1.0 * M, 1.6 * M, ended={1}), 0, "slowest", GAP, False)["reference"] \
        == pytest.approx(1.6 * M)
    # One agent left: nothing pauses, and a paused one goes on.
    assert paused(agents(2.0 * M, 1.0 * M, ended={1})) == set()
    assert paused(agents(2.0 * M, 1.0 * M, ended={1}), now={0}) == set()
    # An agent that has ended is never paused.
    assert paused(agents(2.0 * M, 1.0 * M, 1.0 * M, ended={0})) == set()


def test_uneven_budgets_compare_shares():
    # Budgets 4M and 2M (mean 3M): GAP is for a 3M budget, so 333k of the 4M agent's tokens, 167k of the 2M one's.
    team = agents(1.3 * M, 0.5 * M, budget=[4 * M, 2 * M])
    found = lockstep.assess(team, 0, "slowest", GAP, False)
    assert found["lead"] == pytest.approx(0.3 * M) and found["gap"] == pytest.approx(GAP * 4 / 3)
    assert not found["pause"]                                   # 32.5% vs 25% of their budgets: 7.5% < 8.33%
    assert paused(agents(1.4 * M, 0.5 * M, budget=[4 * M, 2 * M])) == {0}
    # The 2M agent ahead in share, though behind in tokens.
    assert paused(agents(1.0 * M, 0.8 * M, budget=[4 * M, 2 * M])) == {1}
    # Equal budgets: plain tokens and GAP.
    assert lockstep.assess(agents(1.3 * M, M, budget=[2 * M, 2 * M]), 0, "slowest", GAP, False)["gap"] == GAP


# ---- the watcher, stepped by hand ---------------------------------------------------------------------------

class Team:
    """N agents' session directories, token-usage.json as the token watcher writes it, and their lockstep
    watchers; the CLI is "running" while RUNNING has its index, and stopping it is recorded."""

    def __init__(self, tmp: Path, clis=("claude", "codex"), hold: bool = False, stall: float = 3600, budget=4 * M,
                 kind: str = "slowest"):
        self.tmp, self.stopped, self.running = tmp, [], set()
        self.record = tmp / "team-holds.jsonl"
        budgets = budget if isinstance(budget, (list, tuple)) else [budget] * len(clis)
        self.sessions = []
        for index, cli in enumerate(clis):
            session = tmp / f"sessions-{index}-a{index + 1}"
            (session / f"{cli}-home").mkdir(parents=True)
            self.sessions.append(session)
        team = [{"index": i, "agent": f"a{i + 1}", "session_dir": str(s), "budget": budgets[i]}
                for i, s in enumerate(self.sessions)]
        self.watches = [lockstep.Lockstep({
            "index": i, "agent": f"a{i + 1}", "cli": cli, "session_dir": str(self.sessions[i]),
            "home_env": "HOME_ENV", "home": str(self.sessions[i] / f"{cli}-home"), "binary": f"/bin/{cli}",
            "record": str(self.record), "poll": 2, "hold": hold, "holds_dir": str(tmp / "holds"),
            "team_dir": str(tmp / "lockstep"), "keeper_state": str(tmp / "keeper.json"), "metric": "weighted",
            "gap": GAP, "ref": kind, "stall_seconds": stall, "team": team}) for i, cli in enumerate(clis)]

    def spend(self, index: int, weighted: float) -> None:
        """token-usage.json with WEIGHTED tokens, all of them output."""
        (self.sessions[index] / "token-usage.json").write_text(json.dumps({"usage": {
            "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": int(weighted),
            "reasoning_output_tokens": 0}, "weighted_tokens": int(weighted)}))

    def step(self, now: float, since: float | None = None) -> None:
        """Every watcher steps at NOW; from SINCE, every 30 s up to it first (a watcher steps every 2 s)."""
        times = [] if since is None else [since + t for t in range(0, int(now - since), 30)]
        for when in [*times, now]:
            for watch in self.watches:
                watch.step(when)

    def records(self, event: str | None = None) -> list[dict]:
        found = [json.loads(line) for line in self.record.read_text().splitlines()] if self.record.exists() else []
        return [r for r in found if event is None or r["event"] == event]

    def paused(self) -> list[int]:
        return [i for i, s in enumerate(self.sessions) if (s / lockstep.PAUSE).exists()]


@pytest.fixture
def team(tmp_path, monkeypatch):
    found = Team(tmp_path)
    install(found, monkeypatch)
    return found


def install(found: Team, monkeypatch) -> None:
    def processes(env, home, binary):
        index = next(i for i, s in enumerate(found.sessions) if home.startswith(str(s)))
        return [100 + index] if index in found.running else []

    def stop(env, home):
        index = next(i for i, s in enumerate(found.sessions) if home.startswith(str(s)))
        found.stopped.append(index)
        found.running.discard(index)
        return True

    monkeypatch.setattr(lockstep, "cli_processes", processes)
    monkeypatch.setattr(team_hold, "stop_cli", stop)


def claude_log(session: Path, *entries) -> Path:
    path = session / "claude-home" / "projects" / "-app" / "s1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        for entry in entries:
            stream.write(json.dumps(entry) + "\n")
    return path


PROMPT = {"type": "user", "message": {"role": "user", "content": "Solve the task."}}
TOOL_USE = {"type": "assistant", "message": {"id": "m1", "stop_reason": "tool_use", "content": [
    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "make"}}]}}
TOOL_RESULT = {"type": "user", "message": {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}}
END_TURN = {"type": "assistant", "message": {"id": "m2", "stop_reason": "end_turn", "content": [
    {"type": "text", "text": "done"}]}}


def test_the_watcher_pauses_stops_between_responses_and_resumes(team):
    now = time.time()
    fast, slow = team.watches[0], team.watches[1]
    team.running = {0, 1}
    claude_log(team.sessions[0], PROMPT)            # agent 0's request is under way: its response streams
    team.spend(0, 300_000)
    team.spend(1, 40_000)
    team.step(now)
    assert team.paused() == [0] and team.records("pause_start")[0]["agent"] == "a1"
    start = team.records("pause_start")[0]
    assert start["reason"] == "lockstep" and start["tokens"] == {"a1": 300_000, "a2": 40_000}
    assert start["reference"] == 40_000 and start["lead"] == 260_000 and start["gap"] == GAP
    fast.tick(now + 0.1)
    assert team.stopped == []                       # not while the response streams
    claude_log(team.sessions[0], TOOL_USE)          # the response has ended and its tool runs
    fast.tick(now + 0.2)
    assert team.stopped == [0]
    pause = json.loads((team.sessions[0] / lockstep.PAUSE).read_text())
    assert pause["stop"] == "between responses" and pause["stopped_cli"] and pause["stopped"] == now + 0.2
    # The other agent catches up; the pause lasts at least MIN_PAUSE_SECONDS after the stop.
    team.spend(1, 300_000)
    team.step(now + 5)
    assert team.paused() == [0]
    team.step(now + 0.2 + lockstep.MIN_PAUSE_SECONDS)
    assert team.paused() == []
    end = team.records("pause_end")[0]
    assert end["released"] == "caught_up" and end["stopped_cli"] and end["lead"] == 0 and end["reference"] == 300_000
    assert end["stopped_at"] == pytest.approx(now + 0.2, abs=0.01) and end["stop"] == "between responses"
    [interval] = lockstep.paused_intervals(str(team.sessions[0]), now + 100)
    assert interval == pytest.approx((now + 0.2, now + 0.2 + lockstep.MIN_PAUSE_SECONDS))
    assert not slow.pause and team.stopped == [0]


def test_a_pause_of_a_cli_that_is_not_running_stops_nothing(team):
    now = time.time()
    team.spend(0, 300_000)
    team.step(now)
    assert team.paused() == [0] and team.stopped == []
    pause = json.loads((team.sessions[0] / lockstep.PAUSE).read_text())
    assert pause["stop"] == "not running" and pause["stopped"] == now and not pause["stopped_cli"]
    # The runner starts its CLI anyway (it had just passed its check): it is stopped between responses.
    team.running = {0}
    claude_log(team.sessions[0], PROMPT)
    team.step(now + 2)
    team.watches[0].tick(now + 2.1)
    assert team.stopped == []
    claude_log(team.sessions[0], TOOL_USE)
    team.watches[0].tick(now + 2.2)
    assert team.stopped == [0]
    # Caught up before MIN_PAUSE_SECONDS after a stop of the CLI: the pause lasts until then.
    team.spend(1, 300_000)
    team.step(now + 3)
    assert team.paused() == [0]


def test_a_turn_that_ends_by_itself_is_not_stopped(team):
    now = time.time()
    team.running = {0}
    claude_log(team.sessions[0], PROMPT, END_TURN)
    team.spend(0, 300_000)
    team.step(now)
    team.watches[0].tick(now + 0.1)
    assert team.stopped == []
    # Past SAFE_WAIT_SECONDS without a moment between responses, it is stopped anyway.
    team.watches[0].tick(now + lockstep.SAFE_WAIT_SECONDS + 1)
    assert team.stopped == [0]
    assert json.loads((team.sessions[0] / lockstep.PAUSE).read_text())["stop"] == "timed out"


def test_a_stop_is_recorded_before_the_cli_is_signalled(team, monkeypatch):
    """The agent can end while its CLI is being stopped (its budget spent by the response that just ended),
    and the keeper then ends the watcher in the middle of the stop: the pause has the stop already."""
    now = time.time()
    team.running = {0}
    claude_log(team.sessions[0], PROMPT, TOOL_USE)
    team.spend(0, 300_000)
    team.step(now)

    def interrupted(env, home):
        raise lockstep.Stopped

    monkeypatch.setattr(team_hold, "stop_cli", interrupted)
    with pytest.raises(lockstep.Stopped):
        team.watches[0].tick(now + 0.1)
    pause = json.loads((team.sessions[0] / lockstep.PAUSE).read_text())
    assert pause["stop"] == "between responses" and pause["stopped"] == now + 0.1 and pause["stopped_cli"]
    team.watches[0].end(now + 3, team.watches[0].read_team(), "agent_ended")
    [end] = team.records("pause_end")
    assert end["stop"] == "between responses" and end["stopped_at"] == pytest.approx(now + 0.1, abs=0.01)
    assert lockstep.paused_intervals(str(team.sessions[0]), now + 9) == [pytest.approx((now + 0.1, now + 3))]


def test_stream_reads_claude_code_and_codex_logs(tmp_path):
    claude = lockstep.Stream("claude", str(tmp_path / "c"))
    assert not claude.idle(0)
    path = tmp_path / "c" / "projects" / "-app" / "s.jsonl"
    path.parent.mkdir(parents=True)
    states = []
    two = {"type": "assistant", "message": {"id": "m3", "stop_reason": "tool_use", "content": [
        {"type": "tool_use", "id": "t2"}, {"type": "tool_use", "id": "t3"}]}}
    one_result = {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t2"}]}}
    other_result = {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t3"}]}}
    for entry in (PROMPT, TOOL_USE, {"type": "attachment"}, TOOL_RESULT, two, one_result, other_result, END_TURN):
        with path.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        states.append(claude.idle(10))
    assert states == [False, True, True, False, True, True, False, False]

    codex = lockstep.Stream("codex", str(tmp_path / "x"))
    rollout = tmp_path / "x" / "sessions" / "2026" / "10" / "15" / "rollout-1.jsonl"
    rollout.parent.mkdir(parents=True)

    def item(kind, **payload):
        return {"type": "response_item", "payload": {"type": kind, **payload}}

    events = [item("message", role="user"), item("reasoning"), item("custom_tool_call", call_id="c1"),
              {"type": "token_usage_record", "payload": {}}, {"type": "event_msg", "payload": {"type": "item_completed"}},
              item("custom_tool_call_output", call_id="c1"), {"type": "event_msg", "payload": {"type": "token_count"}},
              item("function_call", call_id="c2"), item("function_call", call_id="c3"),
              {"type": "token_usage_record", "payload": {}}, item("function_call_output", call_id="c2"),
              item("function_call_output", call_id="c3"), item("message", role="assistant"),
              {"type": "token_usage_record", "payload": {}}]
    states = []
    for event in events:
        with rollout.open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        states.append(codex.idle(10))
    # Idle from a finished response (token_usage_record) until its tools' outputs are in; a final answer
    # (no tool call) ends the turn, and is not a moment to stop.
    assert states == [False, False, False, True, True, False, False, False, False, True, True, False, False, False]


def test_stall_guard_releases_every_pause_and_rearms_when_tokens_move(tmp_path, monkeypatch):
    found = Team(tmp_path, clis=("claude", "codex", "codex"), stall=600)
    install(found, monkeypatch)
    now = time.time()
    found.spend(0, 100_000)                 # the slowest: stuck (a long build)
    found.spend(1, 400_000)
    found.spend(2, 420_000)
    found.step(now)
    assert found.paused() == [1, 2]
    found.step(now + 599, since=now)
    assert found.paused() == [1, 2]
    found.step(now + 601)                   # no unpaused agent's tokens grew for 600 s
    assert found.paused() == []
    assert [r["released"] for r in found.records("pause_end")] == ["stall", "stall"]
    [stall] = found.records("lockstep_stall")
    assert stall["stalled"] == ["a1"] and (tmp_path / "lockstep" / lockstep.STALL).exists()
    # Free to work while it lasts: no new pause, however far ahead.
    found.spend(1, 900_000)
    found.step(now + 700, since=now + 601)
    assert found.paused() == []
    # The stuck agent spends again: pausing starts again.
    found.spend(0, 101_000)
    found.step(now + 710)
    assert found.paused() == [1, 2] and not (tmp_path / "lockstep" / lockstep.STALL).exists()
    assert len(found.records("lockstep_rearm")) == 1 and len(found.records("pause_start")) == 4


def test_no_stall_while_the_team_limit_hold_holds(tmp_path, monkeypatch):
    found = Team(tmp_path, stall=600, hold=True)
    install(found, monkeypatch)
    now = time.time()
    found.spend(0, 300_000)
    found.step(now)
    assert found.paused() == [0]
    # Agent 1 waits for a usage limit and holds the team (team_hold.py's hold file, fresh).
    holds = tmp_path / "holds"
    holds.mkdir()
    for t in range(0, 1300, 20):
        (holds / "1.json").write_text(json.dumps({"index": 1, "agent": "a2", "session_dir": str(found.sessions[1]),
                                                  "heartbeat": now + t, "until": now + 2000}))
        found.step(now + t)
    assert found.paused() == [0] and not found.records("lockstep_stall")
    (holds / "1.json").unlink()            # the hold's last step was at now + 1280
    found.step(now + 1280 + 599, since=now + 1300)
    assert found.paused() == [0]
    found.step(now + 1280 + 601)
    assert found.paused() == [] and found.records("lockstep_stall")


def test_ended_agents_and_a_dead_runner_end_the_pause(tmp_path, monkeypatch):
    found = Team(tmp_path, clis=("claude", "codex", "codex"))
    install(found, monkeypatch)
    now = time.time()
    found.spend(0, 100_000)
    found.spend(1, 400_000)
    found.spend(2, 300_000)
    found.step(now)
    assert found.paused() == [1]
    # The slowest has spent its budget: the reference is agent 2, still 100k behind, then caught up.
    (found.sessions[0] / "token-budget-exhausted.json").write_text("{}")
    found.step(now + 1)
    assert found.paused() == [1] and found.records("pause_end") == []
    found.spend(2, 400_000)
    found.step(now + 2)
    assert found.paused() == [] and found.records("pause_end")[0]["released"] == "caught_up"
    assert found.records("pause_end")[0]["reference"] == 400_000
    found.spend(1, 900_000)
    found.step(now + 3)
    assert found.paused() == [1]
    # Agent 2's runner was killed too often and the keeper gave up on it: one agent left, nothing pauses.
    (tmp_path / "keeper.json").write_text(json.dumps({"units": {"2": {"done": 137}}}))
    found.step(now + 4)
    assert found.paused() == [] and found.records("pause_end")[-1]["released"] == "alone"
    (tmp_path / "keeper.json").write_text(json.dumps({"units": {"2": {"done": None}}}))
    found.step(now + 5)
    assert found.paused() == [1]
    (found.sessions[1] / "harness-stop.json").write_text("{}")
    found.step(now + 6)
    assert found.paused() == [] and found.records("pause_end")[-1]["released"] == "agent_ended"


def test_a_watcher_started_again_goes_on_and_a_stale_pause_is_closed(team):
    now = time.time()
    team.spend(0, 300_000)
    team.step(now)
    again = lockstep.Lockstep(team.watches[0].config)
    assert again.pause["since"] == now
    again.step(now + 30)
    assert team.paused() == [0] and len(team.records("pause_start")) == 1
    # Left by a watcher long gone (a revived trial): closed at its heartbeat, and decided anew.
    later = lockstep.Lockstep(team.watches[0].config)
    later.step(now + 30 + 3600)
    assert [r["released"] for r in team.records("pause_end")] == ["stale"]
    assert team.records("pause_end")[0]["seconds"] == 30.0 and len(team.records("pause_start")) == 2


# ---- the working-time watcher and the `resources` command ---------------------------------------------------

def test_working_time_leaves_lockstep_pauses_out_of_both(tmp_path, monkeypatch):
    session = tmp_path / "s"
    session.mkdir()
    with (session / limit_resume.WAITS).open("w") as stream:
        stream.write(json.dumps({"start": 100, "end": 200}) + "\n")          # a usage-limit wait
    with (session / lockstep.PAUSES).open("w") as stream:
        stream.write(json.dumps({"start": 150, "end": 300}) + "\n")          # a pause, half in the wait
    (session / lockstep.PAUSE).write_text(json.dumps({"since": 590, "stopped": 600}))  # one that lasts
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    waited, active, waiting = lockstep.limit_record(limit_resume.CliWaits("codex", str(session)), str(session),
                                                    0.0, 3600.0, 172800.0)
    # 1000 s: 100 waited, 100 + 400 paused (not counting the 50 inside the wait), 400 worked.
    assert (waited, active, waiting) == (100.0, 400.0, True)
    state = json.loads((session / limit_resume.STATE).read_text())
    assert state["lockstep_paused_seconds"] == 500.0 and state["waited_seconds"] == 100.0
    # The time cap is not reached while paused, nor are the most waits allowed.
    stopped = []
    monkeypatch.setattr(limit_resume, "stop_cli", lambda *args: stopped.append(args))
    monkeypatch.setattr(limit_resume, "record", lockstep.limit_record)
    limit_resume.check(limit_resume.CliWaits("codex", str(session)), str(session), "E", "/h", 0.0, 450.0, 150.0)
    assert not stopped and not (session / limit_resume.TIME_CAP).exists()
    assert not (session / limit_resume.EXHAUSTED).exists()


def test_resources_counts_lockstep_pauses_as_time_not_used(tmp_path):
    sys.modules.pop("agent", None)
    with mock.patch.dict(os.environ, team_env()):
        module = importlib.import_module("agent")
    home = tmp_path / "sessions-0-a1" / "codex-home"
    home.mkdir(parents=True)
    (home.parent / "limit-wait.json").write_text(json.dumps({"waited_seconds": 100.0,
                                                             "lockstep_paused_seconds": 500.0}))
    config = {"cli": "codex", "homes": [str(home)], "budget": None, "metric": "weighted", "timeout": 3600.0,
              "started": time.time() - 1000}
    # 1000 s since the start: 900 s used without lockstep (100 s waited), 400 s with it (500 s paused too).
    for lockstep_pauses, used in ((False, 900), (True, 400)):
        script = tmp_path / f"resources-{lockstep_pauses}"
        script.write_text(module.resources_script(config, lockstep_pauses))
        out = subprocess.run([sys.executable, str(script), "--continuation"],
                             env={"CODEX_HOME": str(home), "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        left = out.stdout.split("Remaining resources: ")[1].split(" of your 1h")[0]
        seconds = sum(int(part[:-1]) * {"h": 3600, "m": 60, "s": 1}[part[-1]] for part in left.split())
        assert abs(seconds - (3600 - used)) <= 3, out.stdout
    assert module.resources_script(config) == module.resources_script(config, False)


# ---- the runner's wait --------------------------------------------------------------------------------------

def runner_helpers(hold: bool) -> str:
    sys.modules.pop("agent", None)
    extra = {**LIMIT, **GUARD, **ON, **(HOLD if hold else {})}
    with mock.patch.dict(os.environ, team_env(extra=extra)):
        module = importlib.import_module("agent")
        agent = module.ClaudeCodeCollaborationAgent(logs_dir=ROOT / ".cache/render-logs", model_name="m")
        helpers = agent._action_feedback_helpers()
    start = helpers.index("lockstep_paused() {")
    return helpers[start:helpers.index("\n}\n", helpers.index("team_pause_stopped() {")) + 3]


def wait_script(tmp: Path, hold: bool, body: str) -> str:
    return (f'limit_dir={shlex.quote(str(tmp))}\nlog_file="$limit_dir/log"\nwaits=0\n'
            'limit_wait() { waits=$((waits + 1)); [ "$waits" -lt 3 ] || rm -f "$limit_dir/team-hold.json"; '
            'command sleep 0.1; }\nlimit_stop_reached() { echo stopped >> "$limit_dir/stops"; }\n'
            f"{runner_helpers(hold).replace('sleep 2', 'sleep 0.2')}\n{body}")


def test_the_runner_waits_for_lockstep_and_the_hold_together(tmp_path):
    """Paused by lockstep and held by the team limit hold at once: the agent waits until both are over; only
    the hold's steps count as waiting (limit_wait)."""
    (tmp_path / lockstep.PAUSE).write_text('{"lead": 260000}')
    (tmp_path / team_hold.HELD).write_text('{"holders": ["a1"]}')
    # The lockstep pause ends 1.5 s in; the hold after its third step.
    body = ("( sleep 1.5; rm -f \"$limit_dir/lockstep-pause.json\" ) &\n"
            "team_pause /x/codex-attempt-3.jsonl; echo \"code=$? waits=$waits\"")
    done = subprocess.run(["bash", "-c", wait_script(tmp_path, True, body)], capture_output=True, text=True, timeout=30)
    assert done.stdout.strip() == "code=0 waits=3", done.stderr
    events = [json.loads(line)["data"] for line in (tmp_path / "log").read_text().splitlines()]
    assert [e["event"] for e in events] == ["lockstep_pause", "team_limit_hold", "lockstep_pause_ended",
                                            "team_limit_hold_ended"]
    assert events[0]["pause"] == {"lead": 260000} and events[0]["attemptLog"] == "codex-attempt-3.jsonl"
    assert events[2]["pausedSeconds"] >= 1


def test_the_runner_waits_for_lockstep_alone_without_counting_it(tmp_path):
    (tmp_path / lockstep.PAUSE).write_text("{}")
    body = ("( sleep 1; rm -f \"$limit_dir/lockstep-pause.json\" ) &\n"
            "team_pause; echo \"code=$? waits=$waits\"")
    done = subprocess.run(["bash", "-c", wait_script(tmp_path, False, body)], capture_output=True, text=True, timeout=30)
    assert done.stdout.strip() == "code=0 waits=0", done.stderr
    # The agent ends while it waits (here its budget): the wait stops at once.
    (tmp_path / lockstep.PAUSE).write_text("{}")
    (tmp_path / "token-budget-exhausted.json").write_text("{}")
    done = subprocess.run(["bash", "-c", wait_script(tmp_path, False, "team_pause; echo \"code=$?\"")],
                          capture_output=True, text=True, timeout=30)
    assert done.stdout.strip() == "code=1" and (tmp_path / "stops").exists()


# ---- run.py and agent.py --------------------------------------------------------------------------------------

def test_run_py_lockstep():
    import run
    _, env = run_py("--team", "claude:claude-sonnet-5-5,codex:gpt-6.1-sol", "--token-budget", "4000000")
    assert not any(name.startswith("RELEASE_LOCKSTEP") for name in env)
    _, env = run_py("--team", "claude:claude-sonnet-5-5,codex:gpt-6.1-sol", "--token-budget", "4000000",
                    "--lockstep-gap", "250000")
    assert (env["RELEASE_LOCKSTEP_GAP"], env["RELEASE_LOCKSTEP_REF"], env["RELEASE_LOCKSTEP_STALL"]) == (
        "250000", "slowest", "3600")
    _, env = run_py("--cli", "codex", "--model", "gpt-6.1-sol", "--agents", "4", "--token-budget", "100",
                    "--lockstep-gap", "10")
    assert env["RELEASE_LOCKSTEP_REF"] == "slowest" and env["RELEASE_LOCKSTEP_STALL"] == "3600"
    base = run.parser().parse_args(["ale-bench", "--problem", "ahc014", "--cli", "codex", "--model", "m", "--agents", "2",
                                    "--timeout", "1000", "--limit-wait-max", "1"])
    stepped = run.parser().parse_args(["ale-bench", "--problem", "ahc014", "--cli", "codex", "--model", "m", "--agents", "2",
                                       "--timeout", "1000", "--limit-wait-max", "1", "--token-budget", "9",
                                       "--lockstep-gap", "5"])
    assert run.harbor_agent_timeout(base) == 1000 + 3600 + 1800
    assert run.harbor_agent_timeout(stepped) == 2 * (1000 + 3600) + 1800


@pytest.mark.parametrize("argv, message", [
    (("--cli", "codex", "--model", "m", "--agents", "2", "--lockstep-gap", "5"), "--lockstep-gap needs"),
    (("--cli", "codex", "--model", "m", "--token-budget", "9", "--lockstep-gap", "5"), "--lockstep-gap needs"),
    (("--cli", "codex", "--model", "m", "--agents", "2", "--token-budget", "9", "--lockstep-gap", "5",
      "--no-harness-guard"), "--lockstep-gap needs"),
])
def test_run_py_refuses(argv, message):
    with pytest.raises(ValueError, match=message):
        run_py(*argv)


def test_lockstep_off_leaves_no_trace(tmp_path):
    agent, calls = render_team(SONNET_SOL, {**LIMIT, **GUARD, **HOLD, "RELEASE_TOKEN_BUDGET": "4000000"}, tmp_path)
    assert all("team_pause" not in script and "lockstep" not in script for script in runners(agent).values())
    assert not any("mod-step.py" in call["command"] or "lockstep" in call["command"] for call in calls)


@pytest.mark.parametrize("hold", [False, True])
def test_lockstep_in_the_runner_and_the_keeper(tmp_path, hold):
    team = [*SONNET_SOL, {"cli": "codex", "model": "gpt-6.1-sol", "effort": "high"}]
    agent, calls = render_team(team, {**LIMIT, **GUARD, **ON, **(HOLD if hold else {}),
                                      "RELEASE_RESOURCES_TOOL": "1"}, tmp_path)
    for script in runners(agent).values():
        before = script.index("    team_pause || return 0\n")
        assert before < script.index("    attempt_started_at=$SECONDS\n", before)
        assert script.index('    if [ "$code" -ne 0 ] && team_paused; then\n') < script.index('if [ "$code" -eq 0 ]')
        assert ('      team_hold_request "$delay"\n' in script) is hold
        assert ('lockstep_held() {\n  [ -e "$limit_dir/team-hold.json" ]\n}' in script) is hold
        assert subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True).returncode == 0
    conf = json.loads(written(calls, "conf.json"))
    sessions = [unit["session_dir"] for unit in conf["units"]]
    for unit in conf["units"]:
        hold_watcher, limit = unit["watchers"]["hold"], unit["watchers"]["limit"]
        assert hold_watcher["module"] == limit["module"] == "mod-step.py" and limit["argv"][:2] == ["limit", "watch"]
        config = json.loads(hold_watcher["argv"][1])
        assert hold_watcher["argv"][0] == "watch" and config["hold"] is hold and config["gap"] == GAP
        assert [m["session_dir"] for m in config["team"]] == sessions and {m["budget"] for m in config["team"]} == {4 * M}
        assert config["binary"] == ("/mnt/release-claude/claude" if unit["index"] == 0 else "/mnt/release-codex/bin/codex")
        assert config["home_env"] == ("CLAUDE_CONFIG_DIR" if unit["index"] == 0 else "CODEX_HOME")
    assert written(calls, "mod-step.py") == lockstep.__loader__.get_data(lockstep.__file__).decode()
    assert "mod-hold.py" in next(call["command"] for call in calls if "> /usr/lib/hsv/hsvd" in call["command"])
    assert "waited_for_limits" in written(calls, "/usr/local/libexec/resources")
    assert max(len(call["command"].encode()) for call in calls) < 120000


def test_the_watcher_runs_as_the_keeper_runs_it(tmp_path):
    """harness_guard.py's run_watcher loads GUARD_DIR/mod-step.py, which finds team_hold.py and
    limit_resume.py next to it and uses the keeper's way of finding processes."""
    guard = tmp_path / "hsv"
    guard.mkdir()
    for name, source in (("mod-step.py", "lockstep.py"), ("mod-hold.py", "team_hold.py"),
                         ("mod-limit.py", "limit_resume.py")):
        (guard / name).write_text((ROOT / source).read_text())
    code = ("import sys, importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('hsv_hold', {str(guard / 'mod-step.py')!r})\n"
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
            "module.agent_processes = lambda env, home: ['keeper']\n"
            "print(module.team_hold.__file__.endswith('mod-hold.py'), module.limit_resume.__file__.endswith('mod-limit.py'))\n"
            "print(module.main(['nothing']))\n"
            "print(module.team_hold.agent_processes('E', 'H'), module.limit_resume.agent_processes('E', 'H'))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert out.stdout.split("\n")[:3] == ["True True", "2", "['keeper'] ['keeper']"], out.stderr


# ---- end to end: real CLIs on a mock API ----------------------------------------------------------------------

class LockstepRun:
    """A team (CLIS: each agent's CLI, in order) run as Harbor runs it, with the keeper, a token budget and
    lockstep; each CLI's agents answered by one mock from PLANS[cli]. Polls as in a run; the runner's lockstep
    steps are 2 s, as in a run."""

    def __init__(self, tmp: Path, clis: list[str], plans: dict, env: dict):
        self.tmp, self.clis = tmp, clis
        self.mocks = {}
        for cli in dict.fromkeys(clis):
            (tmp / f"mock-{cli}").mkdir(parents=True)
            self.mocks[cli] = Mock(tmp / f"mock-{cli}", plans[cli])
        home = tmp / "home"
        home.mkdir()
        base = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(home),
                "HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9", "NO_PROXY": "127.0.0.1,localhost",
                "META_DUO_CLAUDE_API_RETRY_BASE_SLEEP": "2", "META_DUO_CODEX_API_RETRY_BASE_SLEEP": "2"}
        if "claude" in self.mocks:
            base["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{self.mocks['claude'].port}"
        if "codex" in self.mocks:
            provider = {"name": "mock", "base_url": f"http://127.0.0.1:{self.mocks['codex'].port}/v1",
                        "wire_api": "responses", "env_key": "CODEX_API_KEY", "supports_websockets": "false",
                        "request_max_retries": "0", "stream_max_retries": "0"}
            args = ["-c", 'model_provider="mock"']
            for key, value in provider.items():
                args += ["-c", f"model_providers.mock.{key}={value if value in ('false', '0') else json.dumps(value)}"]
            hook = tmp / "hook.sh"
            hook.write_text('task_cli_args=()\nif [ "$1" = codex ]; then task_cli_args=('
                            + " ".join(f"'{arg}'" for arg in args) + "); fi\n")
            base["RELEASE_TASK_CLI_HOOK"] = str(hook)
        self.local = Local(tmp, base)
        patch = self.local.patch
        self.local.patch = lambda command: patch(command).replace("limit_wait 10;", "limit_wait 1;")
        self.env = {**team_env(agents=len(clis), extra={**LIMIT, **GUARD, **env}), "RELEASE_CONTINUATIONS": "1",
                    "MULTIAGENT_AGENT_EFFORT": "", "META_DUO_CONTAINER_CLAUDE_BINARY": str(CLAUDE_BINARY),
                    "META_DUO_CONTAINER_CODEX_BINARY": str(CODEX_BINARY), "CODEX_API_KEY": "sk-mock"}
        self.error = None
        self.thread = threading.Thread(target=self._main, daemon=True)
        self.thread.start()

    def _main(self) -> None:
        try:
            with mock.patch.dict(os.environ, self.env, clear=False):
                sys.modules.pop("agent", None)
                self.module = importlib.import_module("agent")
                team = [{"cli": cli, "model": "claude-sonnet-5-5" if cli == "claude" else "gpt-mock"} for cli in self.clis]
                agent = self.module.MixedTeamAgent(logs_dir=self.local.agent_dir, model_name="mixed", team=team)
                asyncio.run(agent.run(INSTRUCTION, self.local, SimpleNamespace()))
        except BaseException as error:  # noqa: BLE001 - the test reads it
            self.error = error

    def finish(self, timeout: float = 600) -> "LockstepRun":
        self.thread.join(timeout)
        alive = self.thread.is_alive()
        self.close()
        assert not alive, "the run did not end"
        assert self.error is None, repr(self.error)
        return self

    def close(self) -> None:
        for found in self.mocks.values():
            found.stop() if found.process.poll() is None else None
        state = harness_guard.read_json(str(self.local.guard_dir / "state" / "keeper.json")) or {}
        if state.get("pid") and harness_guard.alive(state["pid"], state.get("start")):
            kill(state["pid"])
        for index, cli in enumerate(self.clis):
            kill(harness_guard.tagged(f"HSV_UNIT={harness_guard.unit_tag(str(self.local.guard_dir), index)}"))
            for name in harness_guard.WATCHER_LABELS:
                kill(harness_guard.tagged(f"HSV_WATCH={harness_guard.watch_tag(str(self.local.guard_dir), name, index)}"))
            for session in self.local.agent_dir.glob(f"sessions-{index}-*"):
                kill(harness_guard.agent_processes("CLAUDE_CONFIG_DIR" if cli == "claude" else "CODEX_HOME",
                                                   str(session / f"{cli}-home")))

    def session(self, index: int) -> Path:
        return next(self.local.agent_dir.glob(f"sessions-{index}-*"))

    def events(self, index: int, event: str) -> list[dict]:
        found = []
        log = next(self.local.agent_dir.glob(f"agent-{index}-*-output.log"))
        for line in log.read_text(errors="replace").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict) and entry.get("type") == "environment.feedback" and entry["data"].get("event") == event:
                found.append(entry["data"])
        return found

    def holds(self, event: str | None = None, agent: str | None = None) -> list[dict]:
        path = self.local.agent_dir / team_hold.RECORD
        found = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return [r for r in found if (event is None or r["event"] == event) and (agent is None or r["agent"] == agent)]

    def tokens(self, index: int) -> float:
        return lockstep.spent(json.loads((self.session(index) / "token-usage.json").read_text()), "weighted")

    def in_flight(self, cli: str, when: float) -> list[dict]:
        """The mock's requests (with a delay) that were under way at WHEN: sent, not yet answered."""
        answered = {r["seq"]: r for r in self.mocks[cli].requests() if "answered" in r}
        return [r for r in self.mocks[cli].main() if r.get("seq") in answered
                and r["ts"] < when < answered[r["seq"]]["answered"]]

    def texts(self, cli: str) -> list[str]:
        return [text for request in self.mocks[cli].main() for text in user_texts(request)]


def tool(command: str, output: int, delay: float) -> dict:
    return {"type": "tool", "command": command, "output_tokens": output, "delay": delay}


def check_stops(run: LockstepRun, cli_of: dict) -> list[dict]:
    """Every lockstep stop of a running CLI came between its responses: none of its requests was under way."""
    ends = [r for r in run.holds("pause_end") if r.get("reason") == "lockstep"]
    for end in ends:
        if end["stopped_cli"]:
            assert end["stop"] == "between responses", end
            assert not run.in_flight(cli_of[end["agent"]], end["stopped_at"]), end
    return ends


@needs_claude
@needs_codex
def test_a_pair_in_lockstep_with_a_stall(tmp_path):
    """Claude Code spends 175 weighted tokens a response, Codex 625 (each response takes 0.6 s to come, then a
    1 s command), each with a budget of 12000: Codex is paused each time it is more than 1500 ahead, stopped
    while a command runs, and goes on once Claude Code has caught up. Claude Code's 25 s command (no tokens for
    25 s, the stall guard's 12 s) ends every pause; pausing starts again once Claude Code spends again."""
    claude = [tool("sleep 1; echo a", 150, 0.6) for _ in range(14)] + [tool("sleep 25; echo long", 150, 0.6)] \
        + [tool("sleep 1; echo a", 150, 0.6) for _ in range(90)]
    codex = [tool("sleep 1; echo c", 600, 0.6) for _ in range(40)]
    run = LockstepRun(tmp_path, ["claude", "codex"], {"claude": claude, "codex": codex},
                      {"RELEASE_TOKEN_BUDGET": "12000", "RELEASE_LOCKSTEP_GAP": "1500", "RELEASE_LOCKSTEP_REF": "slowest",
                       "RELEASE_LOCKSTEP_STALL": "12"}).finish()
    ends = check_stops(run, {"a1": "claude", "a2": "codex"})
    starts = run.holds("pause_start", "a2")
    assert len(starts) >= 2 and not run.holds("pause_start", "a1"), "the slower agent is never paused"
    for start in starts:
        assert start["reason"] == "lockstep" and set(start["tokens"]) == {"a1", "a2"}
        assert start["lead"] > 1500 and start["reference"] == start["tokens"]["a1"]
    released = [end["released"] for end in ends]
    assert "caught_up" in released and "stall" in released
    assert run.holds("lockstep_stall") and run.holds("lockstep_rearm")
    assert any(end["stopped_cli"] for end in ends)
    # Each stopped turn of Codex was resumed with one message: the note naming the command if Codex had started
    # it, else (stopped as it finished the response, before starting the call, which its resumed session drops)
    # the usual one. Nothing was sent twice.
    history = user_texts(run.mocks["codex"].main()[-1])
    resumed = [text for text in history if text.startswith((run.module.KILL_RESUME_NOTE, run.module.LIMIT_RESUME_NOTE))]
    assert len(resumed) == sum(end["stopped_cli"] and end["released"] != "agent_ended" for end in ends), history
    assert all("sleep 1; echo c" in text for text in resumed if text.startswith(run.module.KILL_RESUME_NOTE))
    assert run.events(1, "lockstep_pause") and run.events(1, "lockstep_pause_ended")
    # Both spent their budgets; the pauses counted as neither working time nor waiting.
    for index in (0, 1):
        assert (run.session(index) / "token-budget-exhausted.json").exists()
    state = [json.loads((run.session(index) / "limit-wait.json").read_text()) for index in (0, 1)]
    assert state[0]["lockstep_paused_seconds"] == 0 and state[1]["lockstep_paused_seconds"] > 5
    assert state[1]["waited_seconds"] == 0
    # Every stop, even one the agent's end overtook, is recorded as time not worked from the stop on.
    stopped = [end["stopped_at"] for end in ends if end["stopped_at"] is not None]
    recorded = [json.loads(line)["start"] for line in (run.session(1) / lockstep.PAUSES).read_text().splitlines()]
    assert recorded == pytest.approx(stopped, abs=0.01) and all(end["stop"] for end in ends if end["stopped_cli"])


@needs_claude
@needs_codex
def test_two_and_two_in_lockstep(tmp_path):
    """Two Claude Code agents and two Codex agents: the Codex agents are paused against the slower Claude Code
    agent, and every agent spends its budget."""
    claude = [tool("sleep 1; echo a", 150, 0.5) for _ in range(160)]
    codex = [tool("sleep 1; echo c", 600, 0.5) for _ in range(60)]
    run = LockstepRun(tmp_path, ["claude", "claude", "codex", "codex"], {"claude": claude, "codex": codex},
                      {"RELEASE_TOKEN_BUDGET": "6000", "RELEASE_LOCKSTEP_GAP": "1500", "RELEASE_LOCKSTEP_REF": "slowest",
                       "RELEASE_LOCKSTEP_STALL": "600"}).finish(900)
    check_stops(run, {"a1": "claude", "a2": "claude", "a3": "codex", "a4": "codex"})
    for agent in ("a3", "a4"):
        assert run.holds("pause_start", agent), agent
    for start in run.holds("pause_start"):
        assert set(start["tokens"]) == {"a1", "a2", "a3", "a4"} and start["working"] >= 2
        assert start["reference"] < start["tokens"][start["agent"]] - 1500
    for index in range(4):
        assert (run.session(index) / "token-budget-exhausted.json").exists()


@needs_claude
@needs_codex
def test_lockstep_and_the_team_limit_hold_together(tmp_path):
    """Claude Code starts 20 s late, then fakes a 15 s usage-limit wait (RELEASE_TEAM_HOLD_TEST): Codex, ahead
    by then and paused by lockstep, is held by the team hold as well, and goes on only once both are over; only
    the hold counts as its waiting."""
    claude = [tool("sleep 1; echo a", 150, 0.5) for _ in range(80)]
    codex = [tool("sleep 1; echo c", 600, 0.5) for _ in range(40)]
    run = LockstepRun(tmp_path, ["claude", "codex"], {"claude": claude, "codex": codex},
                      {**HOLD, "RELEASE_TEAM_HOLD_TEST": "20:15", "RELEASE_TOKEN_BUDGET": "6000",
                       "RELEASE_LOCKSTEP_GAP": "1500", "RELEASE_LOCKSTEP_REF": "slowest",
                       "RELEASE_LOCKSTEP_STALL": "0"}).finish()
    check_stops(run, {"a1": "claude", "a2": "codex"})
    assert not run.holds("pause_start", "a1")
    hold_start, hold_end = (next(r for r in run.holds(event, "a2") if r.get("reason") is None)
                            for event in ("pause_start", "pause_end"))
    lock_start, lock_end = (next(r for r in run.holds(event, "a2") if r.get("reason") == "lockstep")
                            for event in ("pause_start", "pause_end"))
    assert lock_start["epoch"] < hold_start["epoch"] < hold_end["epoch"] < lock_end["epoch"]
    assert lock_end["released"] == "caught_up"
    assert run.events(1, "lockstep_pause") and run.events(1, "team_limit_hold")
    state = json.loads((run.session(1) / "limit-wait.json").read_text())
    assert 0 < state["waited_seconds"] < 30 and state["lockstep_paused_seconds"] > 0


# ---- a deadlock from float rounding --------------------------------------------------------------------------

# token-usage.json counts whose weighted total, 2011174.72, gives the reference agent a lead of +2.3e-10 against
# a 4M budget (tokens - (tokens / budget) * budget), as 2065245.34 did in the real run.
ROUNDING = {"output_tokens": 329745, "input_tokens": 80515594, "cached_input_tokens": 80206386}


def spend_usage(found: Team, index: int, usage: dict) -> None:
    (found.sessions[index] / "token-usage.json").write_text(json.dumps({"usage": {**usage, "reasoning_output_tokens": 0}}))


def test_the_slowest_agent_is_not_kept_paused_by_float_rounding():
    tokens = lockstep.spent({"usage": ROUNDING}, "weighted")
    assert tokens - (tokens / (4 * M)) * (4 * M) > 0            # the rounding the rule must not count as a lead
    team = agents(tokens + GAP + 10_000, tokens)
    assert not lockstep.assess(team, 1, "slowest", GAP, True)["pause"]      # the slowest goes on
    assert lockstep.assess(team, 0, "slowest", GAP, False)["pause"]          # the other is still GAP ahead
    assert paused(agents(M + 1, M), now={0}) == {0}             # one real token ahead still counts


def test_the_pair_that_deadlocked_goes_on(team):
    """a2 ran ahead and paused; a1 passed it and got GAP ahead; a2, now the slowest, must go on (before the fix
    its lead was +2.3e-10, it stayed paused, a1 paused too, and the stall guard, which only watches unpaused
    agents, never fired)."""
    now = time.time()
    team.spend(0, 1_700_000)
    spend_usage(team, 1, ROUNDING)                              # a2: 2011174.72
    team.step(now)
    assert team.paused() == [1]                                 # a2 is 311k ahead
    team.spend(0, 2_011_174.72 + 100_000)                      # a1 passes it: a2 is now the slowest
    team.step(now + 30)
    assert team.paused() == []                                  # a2 goes on (it stayed paused before the fix)
    assert [r["released"] for r in team.records("pause_end")] == ["caught_up"]
    team.spend(0, 2_011_174.72 + GAP + 5_000)                   # a1 gets GAP ahead of the slowest
    team.step(now + 60)
    assert team.paused() == [0]                                 # a1 waits, a2 works


def test_a_team_with_every_agent_paused_releases_them(team, monkeypatch):
    # Whatever the leads say (here a rule that pauses everyone), a team with no unpaused agent cannot move.
    now = time.time()
    team.spend(0, 1_000_000)
    team.spend(1, 1_400_000)
    real = lockstep.assess
    monkeypatch.setattr(lockstep, "assess", lambda *a, **k: {**real(*a, **k), "pause": True, "lead": 5.0})
    team.step(now)
    assert team.paused() == [0, 1]
    team.step(now + 30)
    # The first watcher to step sees everyone paused and ends its pause; the team is never all paused again
    # for longer than one step.
    assert len(team.paused()) < 2
    assert [r["released"] for r in team.records("pause_end")][:1] == ["all_paused"]
