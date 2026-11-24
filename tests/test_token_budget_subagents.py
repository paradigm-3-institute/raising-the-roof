"""Codex sub-agents in the token count (token_budget.CodexReader).

A sub-agent of multi-agent v2 (spawn_agent; in Ultra mode the agent spawns them on its own) is a session
of its own, with a rollout in the agent's CLI home. One forked with its parent's history (Codex 0.162.1's
copied fork) opens its rollout with its own session_meta,
then a copy of the parent's session_meta and history up to a thread_settings_applied event. The reader used to
key a file by its last session_meta, so every forked sub-agent's totals fell under its parent's session and all
but the largest were lost (the smoke run: 20,891 weighted tokens counted of 64,792).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from token_budget import CodexReader, summed, weighted_tokens  # noqa: E402

ROOT_ID = "01a1413b-a0fa-7cd1-80f5-82bf7f82b7c4"


def meta(session: str, forked_from: str | None = None) -> dict:
    return {"timestamp": "2026-10-15T20:23:24.150Z", "type": "session_meta",
            "payload": {"id": session, "forked_from_id": forked_from, "source": "exec"}}


def count(input_tokens: int, cached: int, output: int) -> dict:
    return {"timestamp": "2026-10-15T20:23:30.653Z", "type": "event_msg", "payload": {
        "type": "token_count", "info": {"total_token_usage": {
            "input_tokens": input_tokens, "cached_input_tokens": cached, "output_tokens": output,
            "reasoning_output_tokens": 0}}}}


def event(kind: str) -> dict:
    return {"timestamp": "2026-10-15T20:23:24.159Z", "type": "event_msg", "payload": {"type": kind}}


def write(home: Path, name: str, lines: list[dict]) -> None:
    path = home / "sessions" / "2026" / "10" / "15" / f"rollout-{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def test_a_forked_sub_agent_counts_as_its_own_session(tmp_path):
    write(tmp_path, "root", [meta(ROOT_ID), count(11174, 0, 579), count(289986, 233472, 2093)])
    # The layout of a real sub-agent rollout: its session_meta, the parent's, the copied history, the boundary.
    write(tmp_path, "child", [meta("child-1", ROOT_ID), meta(ROOT_ID), event("task_started"),
                              event("thread_settings_applied"), event("task_started"),
                              count(11252, 9856, 171), count(240102, 196608, 1596)])
    sessions = CodexReader(tmp_path).poll()
    assert sorted(sessions) == sorted([ROOT_ID, "child-1"])
    assert sessions["child-1"]["output_tokens"] == 1596 and sessions[ROOT_ID]["output_tokens"] == 2093
    assert round(weighted_tokens(summed(sessions))) == 20891 + 16402


def test_the_parents_copied_totals_are_subtracted(tmp_path):
    # A sub-agent forked later copies its parent's totals, and its running total starts from them.
    write(tmp_path, "root", [meta(ROOT_ID), count(50000, 0, 1000), count(90000, 40000, 1500)])
    write(tmp_path, "child", [meta("child-2", ROOT_ID), meta(ROOT_ID), count(50000, 0, 1000),
                              event("thread_settings_applied"), count(62000, 45000, 1200), count(80000, 60000, 1400)])
    sessions = CodexReader(tmp_path).poll()
    assert sessions["child-2"] == {"input_tokens": 30000, "cached_input_tokens": 60000 - 0, "output_tokens": 400,
                                   "reasoning_output_tokens": 0}
    assert sessions[ROOT_ID]["output_tokens"] == 1500


def test_unforked_rollouts_are_read_as_before(tmp_path):
    # One session over a resumed second file (the same id, a repeated running total), and a sub-agent spawned
    # without history.
    write(tmp_path, "a", [meta("s1"), count(100, 0, 10), count(300, 100, 30)])
    write(tmp_path, "b", [meta("s1"), count(300, 100, 30), count(500, 300, 50)])
    write(tmp_path, "c", [meta("s2"), count(40, 0, 4)])
    sessions = CodexReader(tmp_path).poll()
    assert sessions["s1"]["output_tokens"] == 50 and sessions["s2"]["output_tokens"] == 4


def test_incremental_reads_keep_the_state(tmp_path):
    lines = [meta("child-3", ROOT_ID), meta(ROOT_ID), count(5000, 0, 100), event("thread_settings_applied"),
             count(6000, 4000, 150)]
    write(tmp_path, "child", lines[:3])
    reader = CodexReader(tmp_path)
    assert reader.poll() == {}
    write(tmp_path, "child", lines)
    assert reader.poll()["child-3"]["output_tokens"] == 50
