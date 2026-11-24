"""BrowseComp (browsecomp_task.py, benchmarks/browsecomp/).

Ported from Harbor PR #2355's tests/unit/adapters/test_browsecomp.py (decryption, the group(1) grader
fix, the pinned template hashes, upstream's seed-0 sample, task generation, task ids, the checksum
refusal, the oracle's response format, the audit); its one-shot reference agent and Harbor job-config
tests have no counterpart here (docs/browsecomp.md). Added: the real dataset (SHA-256, 1,266 rows,
every row decrypted; skipped until `python browsecomp_task.py fetch`), the answer log and its rule,
the agents' commands and the episode-over check, the verifier, host-side grading with a stubbed judge,
the judge's command, the reading for score-over-budget curves, run.py's commands, the CLI hook and the task's secrecy.
No Docker, no model.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import browsecomp_task as bc  # noqa: E402
from test_cli_agents import run_py  # noqa: E402

BENCH = ROOT / "benchmarks" / "browsecomp"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server_module = _load("browsecomp_answer_server", BENCH / "service" / "server.py")
client_module = _load("browsecomp_answerctl", BENCH / "client" / "answerctl.py")
finalize_module = _load("browsecomp_finalize", BENCH / "tests" / "finalize.py")


def _encrypt(plaintext: str, password: str) -> str:
    value = plaintext.encode()
    key = bc.derive_key(password, len(value))
    return base64.b64encode(bytes(a ^ b for a, b in zip(value, key, strict=True))).decode()


def _synthetic_csv(size: int = 4) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["problem", "answer", "problem_topic", "canary"])
    writer.writeheader()
    for index in range(1, size + 1):
        password = f"synthetic-password-{index}"
        writer.writerow({"problem": _encrypt(f"Synthetic question {index}?", password),
                         "answer": _encrypt(f"Synthetic answer {index}", password),
                         "problem_topic": f"topic-{index}", "canary": password})
    return output.getvalue().encode()


def _question(index: int = 1, answer: str = "Synthetic answer 1") -> bc.Question:
    return bc.Question(index, f"Synthetic question {index}?", answer, "topic")


# ----------------------------------------------------------------------------
# The pinned protocol (ported from the PR)


def test_official_decryption_round_trip():
    password = "synthetic-canary"
    plaintext = "A Unicode answer: café 東京"
    assert bc.decrypt(_encrypt(plaintext, password), password) == plaintext


def test_grader_parser_applies_upstream_capture_group_fix():
    assert bc.parse_grader_response("correct: yes") == "yes"
    assert bc.parse_grader_response("correct: no") == "no"
    assert bc.parse_grader_response("Correct: yes") == "no"
    assert bc.parse_grader_response("unparseable") == "no"
    # Upstream returns match.group(0), "correct: yes", and compares it with "yes": always wrong.
    import re
    upstream = re.search(r"correct: (yes|no)", "correct: yes").group(0)
    assert upstream != "yes" and bc.parse_grader_response("correct: yes") == "yes"


def test_official_prompts_match_pinned_openai_source():
    assert hashlib.sha256(bc.QUERY_TEMPLATE.encode()).hexdigest() == (
        "6990664d8e2ac329133e805515199d65f943c128aba9a3af9727fcf1dd32264f")
    assert hashlib.sha256(bc.GRADER_TEMPLATE.encode()).hexdigest() == (
        "0f0023ee579b8c134f1834ed8952778b9e01460e31d47c242ee3629da9d44835")
    assert bc.GRADER_PATTERN == r"correct: (yes|no)"


def test_pinned_grader_settings():
    assert (bc.OFFICIAL_GRADER_MODEL, bc.OFFICIAL_GRADER_TEMPERATURE, bc.OFFICIAL_GRADER_MAX_TOKENS) == (
        "gpt-4.1-2025-04-14", 0.5, 2048)
    assert bc.OFFICIAL_SYSTEM_MESSAGE == "You are a helpful assistant."
    assert bc.OFFICIAL_SOURCE_REVISION == "652c89d0ca9df547706735883097e9537d40dc47"
    assert bc.OFFICIAL_DATASET_SHA256 == "7b24471cd5b3eb2a46830a14802b5c029ea62f488ff75a0f88af7923d1454abf"
    assert bc.HARBOR_PR_COMMIT == "464e90c108bfc3d220288c8de9b3f2d80c572b02"


def test_official_examples_sampling_matches_random_seed_zero():
    rows = bc.read_rows(_synthetic_csv(), verify=False)
    # Upstream samples the row dicts; the ids it picks are those of a sample of ids.
    upstream = random.Random(0).sample(rows, 3)
    picked = bc.examples_sample(3, count=4)
    assert [rows[i - 1] for i in picked] == upstream
    assert picked == random.Random(bc.OFFICIAL_SAMPLE_SEED).sample(range(1, 5), 3)


def test_full_set_and_examples_selection():
    assert bc.select("all") == list(range(1, 1267))
    with pytest.raises(ValueError, match="needs --problem"):
        bc.select(None)
    with pytest.raises(ValueError, match="upstream's sample of the whole set"):
        bc.select("4", 3)
    assert bc.select("all", 5) == bc.examples_sample(5)
    with pytest.raises(ValueError, match="between 1 and 1266"):
        bc.examples_sample(0)
    # Upstream's `if num_examples:` reads --examples 0 as the whole set; here that is --problem all.
    with pytest.raises(SystemExit):
        run_py("browsecomp", "--examples", "0", "--cli", "codex", "--model", "m")


def test_official_thirty_example_subset_is_stable():
    assert bc.examples_sample(30) == [
        789, 862, 83, 531, 1048, 996, 830, 622, 977, 734, 1195, 448, 1034, 286, 578, 287, 195, 514, 1091,
        1233, 301, 636, 203, 152, 677, 967, 1147, 207, 725, 890]


def test_task_ids_accept_canonical_ids_and_reject_unknown():
    assert bc.parse_ids("openai/browsecomp__browsecomp-0002,browsecomp_004, 7,2") == [2, 4, 7]
    for bad in ("1267", "0", "browsecomp-x", ""):
        with pytest.raises(ValueError):
            bc.parse_ids(bad)


def test_adapter_refuses_unpinned_dataset_by_default(tmp_path):
    data = _synthetic_csv()
    with pytest.raises(ValueError, match="dataset checksum mismatch"):
        bc.read_rows(data)
    path = tmp_path / "browsecomp.csv"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="not the pinned BrowseComp test set"):
        bc.fetch_dataset(path)
    questions = bc.load_questions(path, verify=False)
    assert [(q.index, q.problem, q.answer) for q in questions[:2]] == [
        (1, "Synthetic question 1?", "Synthetic answer 1"), (2, "Synthetic question 2?", "Synthetic answer 2")]
    assert questions[0].task_id == "browsecomp-0001"


# ----------------------------------------------------------------------------
# The real dataset (downloaded by `python browsecomp_task.py fetch`)

real_dataset = pytest.mark.skipif(not bc.DATASET.is_file(), reason="run python browsecomp_task.py fetch first")


@real_dataset
def test_real_dataset_sha_rows_and_decryption():
    data = bc.DATASET.read_bytes()
    assert hashlib.sha256(data).hexdigest() == bc.OFFICIAL_DATASET_SHA256
    rows = bc.read_rows(data)
    assert len(rows) == bc.OFFICIAL_TASK_COUNT == 1266
    questions = bc.load_questions(bc.DATASET)
    assert len(questions) == 1266 and all(q.problem.strip() and q.answer.strip() for q in questions)
    assert len({q.problem_topic for q in questions}) == 10
    assert len(bc.canaries(bc.DATASET)) == 1
    # Upstream's --examples 30 picks these rows.
    upstream = random.Random(0).sample(rows, 30)
    assert [rows[i - 1] for i in bc.examples_sample(30)] == upstream


# ----------------------------------------------------------------------------
# The task


def test_instruction_is_the_official_query_then_the_harness():
    question = _question()
    query = bc.QUERY_TEMPLATE.format(Question="Synthetic question 1?")
    solo = bc.instruction(question, 1, "task_only")
    assert solo.startswith(query + "\n\n---\n\nHow this run works:\n")
    assert "You are one of" not in solo and "coordination protocol" not in solo
    team = bc.instruction(question, 4, "board")
    assert "You are one of 4 agents" in team and "coordination protocol" not in team
    for protocol, agents in (("antiherding_matched", 1), ("antiherding_matched", 4), ("antiherding", 2)):
        text = bc.instruction(question, agents, protocol)
        assert "written for tasks with builds and measured scores" in text
        assert "rule against internet access is about the shell" in text
    # antiherding without a solo prompt: one agent gets the bare task, so no note.
    assert "coordination protocol" not in bc.instruction(question, 1, "antiherding")
    for text in (solo, team, bc.instruction(question, 2, "antiherding_matched")):
        assert "browsecomp" not in text.lower().replace("synthetic", "")
        assert "submit-answer" in text and "the last submission is the answer that counts" in text


def test_build_task_files_and_secrecy(tmp_path, monkeypatch):
    monkeypatch.delenv("BROWSECOMP_NAME_PREFIX", raising=False)
    task = bc.build_task(_question(), tmp_path, agents=2, protocol="board", cpus=4, memory_mb=8192, cli="codex",
                         canary_values=["synthetic-canary-value"])
    assert task.path.name == "browsecomp-0001"
    config = tomllib.loads(task.task_toml)
    assert config["task"]["name"] == "openai/browsecomp__browsecomp-0001"
    assert config["agent"]["network_mode"] == "allowlist" and config["agent"]["allowed_hosts"] == []
    assert config["environment"]["allowed_hosts"] == list(bc.SETUP_HOSTS)
    assert (config["environment"]["cpus"], config["environment"]["memory_mb"]) == (4, 8192)
    assert config["metadata"]["browsecomp_index"] == 1
    assert set(config["verifier"]["env"]) == {"ANSWER_LOG_PORT", "ANSWER_LOG_FINALIZE_TOKEN"}
    assert "solution/solve.sh" not in task.files
    # No answer anywhere; the token only where the answer log and the verifier read it.
    for relative, data in task.files.items():
        assert b"Synthetic answer 1" not in data, relative
    token = config["verifier"]["env"]["ANSWER_LOG_FINALIZE_TOKEN"].encode()
    assert {relative for relative, data in task.files.items() if token in data} == {"task.toml", "service/config.json"}
    assert task.modes["service/config.json"] == 0o600
    # Agent-visible files name nothing of the benchmark (the image, the client mount, the verifier's tests).
    import re
    for relative, data in task.files.items():
        if relative.split("/")[0] in ("client", "tests") or relative == "environment/Dockerfile":
            assert not re.search(rb"browse[\s_-]?comp", data, re.IGNORECASE), relative
    compose = task.files["environment/docker-compose.yaml"].decode()
    assert "/opt/answer-log:ro" in compose and "/answers/state" in compose and "condition: service_healthy" in compose
    assert "RELEASE_TASK_CLI_HOOK" in compose


def test_secrecy_check_refuses_an_answer_in_container_files(tmp_path):
    # An answer that happens to be in the agents' Dockerfile would reach the container.
    with pytest.raises(ValueError, match="would carry the answer or a canary"):
        bc.build_task(_question(answer="ubuntu:24.04"), tmp_path, agents=1, protocol=None, cpus=2, memory_mb=4096,
                      cli="codex", canary_values=[])
    with pytest.raises(ValueError, match="would carry the answer or a canary"):
        bc.build_task(_question(), tmp_path, agents=1, protocol=None, cpus=2, memory_mb=4096, cli="codex",
                      canary_values=["python3"])
    # Short answers are ordinary words of the scripts, not leaks.
    bc.build_task(_question(answer="Art"), tmp_path / "short", agents=1, protocol=None, cpus=2, memory_mb=4096,
                  cli="codex", canary_values=[])


def test_container_files_do_not_depend_on_the_question(tmp_path):
    first = bc.build_task(_question(1, "Synthetic answer 1"), tmp_path / "a", agents=2, protocol="board", cpus=4,
                          memory_mb=8192, cli="codex", canary_values=[])
    second = bc.build_task(bc.Question(2, "Another question?", "Another answer", "topic-2"), tmp_path / "b", agents=2,
                           protocol="board", cpus=4, memory_mb=8192, cli="codex", canary_values=[])
    bound = [relative for relative in first.files if relative.split("/")[0] in ("environment", "client", "tests")
             and relative != "environment/docker-compose.yaml"]  # host paths, read by Docker on the host
    assert {"environment/Dockerfile", "client/answerctl.py", "client/task.json", "tests/finalize.py"} <= set(bound)
    for relative in bound:
        assert first.files[relative] == second.files[relative], relative


def test_name_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("BROWSECOMP_NAME_PREFIX", "adapter-browsecomp")
    task = bc.build_task(_question(42), tmp_path, agents=1, protocol=None, cpus=2, memory_mb=4096, cli="codex",
                         canary_values=[])
    assert task.path.name == "adapter-browsecomp-0042"
    monkeypatch.setenv("BROWSECOMP_NAME_PREFIX", "Bad Prefix")
    with pytest.raises(ValueError):
        bc.name_prefix()


def _oracle_solve(tmp_path, oracle: str) -> str:
    """Run the oracle's solve.sh with a submit-answer that records its input."""
    task = bc.build_task(_question(answer="Synthetic answer 1, 2024"), tmp_path / oracle, agents=1, protocol=None,
                         cpus=2, memory_mb=4096, cli=None, canary_values=[], oracle=oracle)
    bin_dir = tmp_path / f"bin-{oracle}"
    bin_dir.mkdir()
    captured = tmp_path / f"captured-{oracle}.txt"
    (bin_dir / "submit-answer").write_text(f"#!/bin/sh\ncat > {captured}\n")
    (bin_dir / "submit-answer").chmod(0o755)
    script = tmp_path / f"solve-{oracle}.sh"
    script.write_bytes(task.files["solution/solve.sh"])
    result = subprocess.run(["bash", str(script)], env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return captured.read_text()


def test_oracle_solution_writes_official_response_format(tmp_path):
    assert _oracle_solve(tmp_path, "right") == (
        "Explanation: This is the reference answer supplied by the benchmark oracle.\n"
        "Exact Answer: Synthetic answer 1, 2024\n"
        "Confidence: 100%\n")
    assert "Exact Answer: Synthetic answer 4, 5357\n" in _oracle_solve(tmp_path, "wrong")
    assert bc.wrong_answer("Ada Lovelace") == "Not Ada Lovelace"


def test_cli_hook_turns_on_live_search_for_codex_only(tmp_path):
    # The hook makes /work/$agent_id: run a copy that makes it under tmp_path instead.
    work = tmp_path / "work"
    copy = tmp_path / "hook.sh"
    copy.write_text((BENCH / "client" / "cli-hook.sh").read_text().replace("/work/", f"{work}/"))
    result = subprocess.run(["bash", "-c", f'agent_id=a1; set -- codex; . "{copy}" || exit 9; '
                                           'printf "%s\\n" "${task_cli_args[@]}"'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["-c", 'web_search="live"']
    assert (work / "a1").is_dir()
    refused = subprocess.run(["bash", "-c", f'agent_id=a1; set -- claude; . "{copy}" || exit 9'],
                             capture_output=True, text=True)
    assert refused.returncode == 9 and "built-in web search" in refused.stderr


def test_harness_config_keeps_search_off_for_every_other_task():
    # Search is turned on only by this task's CLI hook; agent.py's config is untouched.
    import agent
    assert agent.CODEX_CONFIG_TOML.splitlines()[0] == 'web_search = "disabled"'
    assert "web_search" not in agent.CODEX_HELPERS.replace("@codex_config_toml@", "")


# ----------------------------------------------------------------------------
# The answer log, its rule and the agents' commands


def _service(tmp_path, **overrides) -> "server_module.AnswerLog":
    config = {"port": 0, "state_dir": str(tmp_path / "state"), "finalize_token": "t0ken",
              "host_uid": os.getuid(), "host_gid": os.getgid(), "max_response_bytes": 2000,
              "max_submissions_per_agent": 3, **overrides}
    return server_module.AnswerLog(config)


def test_answer_log_rule_and_log(tmp_path):
    service = _service(tmp_path)
    assert service.submit("a1", "Explanation: x\nExact Answer: Paris\nConfidence: 80%")[0] == 200
    status, body = service.submit("a2", "**Exact Answer:** Lyon\nConfidence: 60%")
    assert status == 200 and body["sid"] == 2 and body["exact_answer"] == "Lyon" and body["format_ok"]
    status, body = service.submit("a1", "just Marseille")
    assert status == 200 and body["exact_answer"] is None and not body["format_ok"]
    listing = service.answers()
    assert listing["current"] == 3 and [r["agent_id"] for r in listing["submissions"]] == ["a1", "a2", "a1"]
    log = [json.loads(line) for line in (tmp_path / "state" / "submissions.jsonl").read_text().splitlines()]
    assert [r["sid"] for r in log] == [1, 2, 3]
    assert log[0]["sha256"] == hashlib.sha256(log[0]["response"].encode()).hexdigest()
    assert all(r["ts"].endswith("Z") for r in log)
    # Refusals: empty, too long, a bad agent id, the per-agent cap.
    assert service.submit("a1", "  \n")[0] == 400
    assert service.submit("a1", "x" * 2001)[0] == 413
    assert service.submit("a b", "x")[0] == 400
    assert service.submit("a1", "Exact Answer: Nice")[0] == 200
    assert service.submit("a1", "Exact Answer: Nice again")[0] == 429
    # Finalize: the team's answer is the last submission, then the log is closed.
    closed = service.finalize()
    assert closed["answer"]["sid"] == 4 and closed["answer"]["agent_id"] == "a1" and closed["submissions"] == 4
    assert service.submit("a2", "Exact Answer: late")[0] == 409
    assert service.finalize() == closed
    # A service started again goes on from its log.
    again = _service(tmp_path)
    assert len(again.records) == 4 and again.closed


@pytest.fixture
def live_log(tmp_path, monkeypatch):
    """The answer log over HTTP on a free port, and the client pointed at it."""
    service = _service(tmp_path)
    httpd = server_module.ThreadingHTTPServer(("127.0.0.1", 0), server_module.handler(service, "t0ken"))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    config = tmp_path / "task.json"
    config.write_text(json.dumps({"answer_urls": [f"http://127.0.0.1:{port}"], "work_root": "/work",
                                  "max_response_bytes": 2000}))
    monkeypatch.setattr(client_module, "CONFIG", config)
    yield SimpleNamespace(service=service, port=port, tmp=tmp_path)
    httpd.shutdown()


def test_agent_commands_over_http(live_log, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_ID", "a2")
    monkeypatch.setattr(sys, "stdin", io.StringIO("Explanation: e\nExact Answer: Rome\nConfidence: 70%\n"))
    assert client_module.submit([]) == 0
    assert "Submitted as #1" in capsys.readouterr().out
    response = live_log.tmp / "response.txt"
    response.write_text("Explanation: f\nExact Answer: Milan\nConfidence: 75%\n")
    assert client_module.submit([str(response)]) == 0
    assert client_module.answers([]) == 0
    out = capsys.readouterr().out
    assert "#1" in out and "#2" in out and "Milan  Confidence: 75%  <- current answer" in out
    bad = live_log.tmp / "bad.txt"
    bad.write_bytes(b"\xff\xfe")
    assert client_module.submit([str(bad)]) == 2
    assert live_log.service.records[-1]["agent_id"] == "a2"


def test_episode_over_waits_for_a_submission_and_the_end_of_the_turn(live_log, monkeypatch, tmp_path):
    # In a container the check reads every process; here only this test's own child, so that it never
    # reads the host's other processes.
    children: list[int] = []
    real = client_module.cli_running
    monkeypatch.setattr(client_module, "cli_running", lambda agent, pids=None: real(agent, pids=children))
    monkeypatch.setenv("AGENT_ID", "a1")
    assert client_module.episode_over([]) == 1  # nothing submitted yet
    live_log.service.submit("a1", "Exact Answer: Turin")
    # A CLI of this agent still running (a process called codex with its AGENT_ID): its turn goes on.
    fake = tmp_path / "codex"
    fake.symlink_to("/bin/sleep")
    process = subprocess.Popen([str(fake), "30"], env={**os.environ, "AGENT_ID": "a1"})
    children.append(process.pid)
    try:
        for _ in range(50):
            if real("a1", pids=children):
                break
            time.sleep(0.05)
        assert client_module.episode_over([]) == 1
        assert real("a2", pids=children) is False
    finally:
        process.kill()
        process.wait()
    assert client_module.episode_over([]) == 0
    monkeypatch.setenv("AGENT_ID", "a2")
    assert client_module.episode_over([]) == 1  # a2 has not submitted


def test_verifier_closes_the_log(live_log, monkeypatch, tmp_path):
    live_log.service.submit("a1", "Exact Answer: Genoa")
    monkeypatch.setattr(finalize_module, "LOGS", tmp_path / "verifier")
    monkeypatch.setenv("ANSWER_LOG_PORT", str(live_log.port))
    monkeypatch.setenv("ANSWER_LOG_FINALIZE_TOKEN", "t0ken")
    monkeypatch.setattr(finalize_module, "URLS", ("http://127.0.0.1:{port}",))
    assert finalize_module.main() == 0
    reward = json.loads((tmp_path / "verifier" / "reward.json").read_text())
    assert reward == {"reward": 1.0, "answered": 1.0, "submissions": 1.0}
    assert (tmp_path / "verifier" / "reward.txt").read_text() == "1.0"
    assert live_log.service.closed
    assert json.loads((live_log.tmp / "state" / "answer.json").read_text())["answer"]["sid"] == 1


def test_finalize_needs_the_token(live_log):
    import urllib.error
    import urllib.request
    request = urllib.request.Request(f"http://127.0.0.1:{live_log.port}/finalize", data=b"{}", method="POST",
                                     headers={"X-Finalize-Token": "wrong"})
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=5)
    assert caught.value.code == 403 and not live_log.service.closed


# ----------------------------------------------------------------------------
# Grading on the host (the judge stubbed), and reading trials


def _trial(tmp_path, responses: list[tuple[str, str]], index: int = 1) -> Path:
    """A finished trial: Harbor's config pointing at a task with browsecomp.json, and its answer log."""
    task_dir = tmp_path / "tasks" / f"browsecomp-{index:04d}"
    task_dir.mkdir(parents=True)
    (task_dir / "browsecomp.json").write_text(json.dumps({"index": index}))
    trial = tmp_path / "job" / f"browsecomp-{index:04d}__abc"
    (trial / "agent").mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps({"task": {"path": str(task_dir)}}))
    service = _service(trial, state_dir=str(trial / bc.STATE), max_submissions_per_agent=50)
    for agent_id, response in responses:
        service.submit(agent_id, response)
        time.sleep(0.002)
    return trial


def test_grade_trial_grades_every_response_once_and_takes_the_last(tmp_path, monkeypatch):
    calls = []

    def fake_judge(judge, prompt):
        calls.append(prompt)
        right = "Exact Answer: Synthetic answer 1" in prompt
        return {"verdict": "yes" if right else "no", "judge_response": "correct: " + ("yes" if right else "no"),
                "usage": {"input_tokens": 10}, "attempts": 1}

    monkeypatch.setattr(bc, "run_judge", fake_judge)
    trial = _trial(tmp_path, [("a1", "Exact Answer: Synthetic answer 1"), ("a2", "Exact Answer: Wrong"),
                              ("a1", "Exact Answer: Synthetic answer 1")])
    questions = [_question()]
    judge = bc.Judge(auth=tmp_path / "auth.json")
    final = bc.grade_trial(trial, questions, judge, workers=1)
    assert len(calls) == 2  # two distinct responses
    assert calls[0] == bc.grader_prompt("Synthetic question 1?", "Synthetic answer 1", "Exact Answer: Synthetic answer 1")
    assert (final["score"], final["verdict"], final["sid"], final["agent_id"]) == (1, "yes", 3, "a1")
    assert final["submissions"] == 3 and final["distinct_responses"] == 2 and final["any_correct"]
    # Cached: grading again calls nothing; --force grades again.
    bc.grade_trial(trial, questions, judge, workers=1)
    assert len(calls) == 2
    bc.grade_trial(trial, questions, judge, force=True, workers=1)
    assert len(calls) == 4
    records = bc.trial_records(trial)
    assert [(r["sid"], r["score"], r["status"]) for r in records] == [(1, 1, "done"), (2, 0, "done"), (3, 1, "done")]
    assert bc.trial_final(trial)["score"] == 1 and bc.is_trial(trial)


def test_grade_trial_without_an_answer_or_with_a_failed_judge(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "run_judge", lambda judge, prompt: {"verdict": None, "error": "exit 1", "usage": {},
                                                                "attempts": 3, "judge_response": None})
    judge = bc.Judge(auth=tmp_path / "auth.json")
    empty = _trial(tmp_path / "e", [])
    assert bc.grade_trial(empty, [_question()], judge)["score"] == 0
    failed = _trial(tmp_path / "f", [("a1", "Exact Answer: x")])
    final = bc.grade_trial(failed, [_question()], judge)
    assert final["score"] is None and final["graded"] == 0
    assert bc.trial_records(failed)[0]["status"] == "error"


def test_trial_question_from_its_task(tmp_path):
    trial = _trial(tmp_path, [], index=42)
    assert bc.trial_task(trial)["index"] == 42
    (trial / "config.json").write_text(json.dumps({"task": {"name": "adapter-browsecomp-0042"}}))
    assert bc.trial_task(trial)["index"] == 42


def test_accuracy_at_a_budget():
    steps = [(100.0, 0), (250.0, 1), (900.0, 0)]
    assert [bc.answer_at(steps, budget) for budget in (50, 100, 300, 1000)] == [0, 0, 1, 0]
    assert bc.wilson(9, 10) == (0.5958, 0.9821)


# ----------------------------------------------------------------------------
# The judge's command


def test_judge_runs_codex_without_tools_on_the_login(tmp_path):
    judge = bc.Judge(auth=tmp_path / "auth.json", codex=tmp_path / "codex-package")
    command = bc.judge_command(judge, "adapter-browsecomp-judge-x", tmp_path / "work")
    assert command[:4] == ["docker", "run", "--rm", "--name"]
    assert f"{(tmp_path / 'auth.json').resolve()}:/mnt/codex-auth/auth.json" in command
    assert f"{(tmp_path / 'codex-package').resolve()}:/opt/codex:ro" in command
    flags = command[command.index("judge") + 1:]
    assert flags[flags.index("-m") + 1] == "gpt-6-luna"
    assert 'model_reasoning_effort="high"' in flags and 'web_search="disabled"' in flags
    assert 'model_instructions_file="/judge/instructions.txt"' in flags and "--ephemeral" in flags
    disabled = {flags[i + 1] for i, flag in enumerate(flags) if flag == "--disable"}
    assert {"shell_tool", "unified_exec", "code_mode", "multi_agent", "multi_agent_v2",
            "standalone_web_search"} <= disabled
    assert bc.JUDGE_INSTRUCTIONS == "You are a helpful assistant.\n"
    assert judge.record["model"] == "gpt-6-luna" and judge.record["grader_template_sha256"].startswith("0f0023ee")


def test_judge_refuses_without_a_login(tmp_path):
    with pytest.raises(ValueError, match="no Codex login"):
        bc.run_judge(bc.Judge(auth=tmp_path / "missing.json"), "prompt")


# ----------------------------------------------------------------------------
# The audit (the PR's browsecomp-audit, extended)


def test_audit_flags_known_sources_without_echoing_content(tmp_path):
    trial = tmp_path / "trial"
    (trial / "agent" / "sessions-0-a1" / "codex-home").mkdir(parents=True)
    rollout = trial / "agent" / "sessions-0-a1" / "codex-home" / "rollout.jsonl"
    rollout.write_text("ordinary search result\n")
    assert bc.audit_trial(trial, ["canary-string-value"])["status"] == "clean"
    rollout.write_text('{"url": "https://example.test/browse_comp_test_set.csv", "secret": "must-not-be-echoed"}\n'
                       "line two mentions BrowseComp\nthird: canary-string-value\n")
    report = bc.audit_trial(trial, ["canary-string-value"])
    assert report["status"] == "suspected-contamination"
    found = {m["indicator"]: m["lines"] for m in report["matches"]}
    assert found["PR indicator browse_comp_test_set.csv"] == [1]
    assert found["benchmark name"] == [1, 2] and found["canary string"] == [3]
    assert "must-not-be-echoed" not in json.dumps(report)


def test_audit_weak_indicators_and_harness_files(tmp_path):
    trial = tmp_path / "trial"
    (trial / "agent" / "multiagent-collab").mkdir(parents=True)
    # Harness-written files are what the agents were given: not scanned.
    (trial / "agent" / "multiagent-collab" / "protocol.json").write_text("browsecomp")
    (trial / "agent" / "multiagent-collab" / "notes.md").write_text("see huggingface.co/datasets/foo")
    report = bc.audit_trial(trial, [])
    assert report["status"] == "review" and [m["strength"] for m in report["matches"]] == ["weak"]


def test_audit_reports_incomplete_scans(tmp_path, monkeypatch):
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)
    # Binary files (the CLIs' SQLite state) are scanned as bytes, across chunk boundaries.
    (trial / "agent" / "state.sqlite").write_bytes(b"a\x00b" + b"x" * ((1 << 20) - 5) + b"BrowseComp" + b"\x00")
    link = trial / "agent" / "auth.json"
    link.symlink_to("/mnt/release-codex-auth/auth.json")
    report = bc.audit_trial(trial, [])
    assert report["status"] == "suspected-contamination" and report["skipped"] == []
    assert report["matches"] == [{"path": "agent/state.sqlite", "indicator": "benchmark name", "strength": "strong",
                                  "occurrences": 1, "lines": [], "binary": True}]
    (trial / "agent" / "state.sqlite").write_bytes(b"a\x00b")
    assert bc.audit_trial(trial, [])["status"] == "clean"
    monkeypatch.setattr(bc, "AUDIT_MAX_FILE_BYTES", 2)
    report = bc.audit_trial(trial, [])
    assert report["status"] == "inconclusive"
    assert report["skipped"] == [{"path": "agent/state.sqlite", "reason": "file-too-large", "size_bytes": 3}]
    monkeypatch.setattr(bc, "AUDIT_MAX_FILE_BYTES", 1 << 30)
    (trial / "agent" / "found.txt").write_text("simple-evals")
    assert bc.audit_trial(trial, [])["status"] == "suspected-contamination"


# ----------------------------------------------------------------------------
# run.py


SOL = ("--cli", "codex", "--codex-binary", "/x/codex-0.162.1/bin/codex", "--model", "gpt-5.6-sol")


def test_run_py_four_arms():
    arms = {
        "solo": ("--effort", "max", "--protocol", "task_only", "--agents", "1", "--token-budget", "4000000"),
        "ultra": ("--effort", "max", "--protocol", "ultra", "--agents", "4", "--token-budget", "1000000"),
        "antiherding": ("--effort", "max", "--protocol", "antiherding_matched", "--agents", "4",
                        "--token-budget", "1000000"),
        "board": ("--effort", "max", "--protocol", "board", "--agents", "4", "--token-budget", "1000000"),
    }
    for name, options in arms.items():
        cmd, env, job = run_py("browsecomp", "--examples", "30", *SOL, *options, "--timeout", "7200",
                               "--name", f"bc-{name}")
        assert cmd[cmd.index("-p") + 1] == str(bc.TASKS / f"bc-{name}")
        hosts = [cmd[i + 1] for i, word in enumerate(cmd) if word == "--allow-agent-host"]
        assert hosts == ["chatgpt.com", "auth.openai.com"]
        assert "--override-cpus" not in cmd and "--override-memory-mb" not in cmd
        assert f"BROWSECOMP_QUESTIONS=examples:30" in cmd
        assert env["RELEASE_EPISODE_OVER_COMMAND"] == "python3 -I /opt/answer-log/answerctl.py episode-over"
        assert env["RELEASE_CONTINUATION"] == (BENCH / "continuation.txt").read_text().rstrip("\n")
        assert env["RELEASE_TASK"] == "browsecomp"
        assert env["MULTIAGENT_AGENT_EFFORT"] == "max"
        assert env["MULTIAGENT_PROTOCOL"].startswith(f"multiagent_protocols.{options[3]}:")
    cmd, _, _ = run_py("browsecomp", "--problem", "42", *SOL, "--protocol", "task_only", "--name", "bc-one")
    assert "BROWSECOMP_QUESTIONS=42" in cmd
    assert run_py("browsecomp", "--problem", "all", *SOL, "--protocol", "task_only")[2].name.startswith("browsecomp-all-")


def test_run_py_refusals():
    with pytest.raises(ValueError, match="--dry-run only"):
        import run
        run.prepare(run.parser().parse_args(["browsecomp", "--problem", "1", "--cli", "claude", "--model", "m"]))
    run_py("browsecomp", "--problem", "1", "--cli", "claude", "--model", "claude-opus-5-5", "--protocol", "task_only")
    with pytest.raises(ValueError, match="one agent with the question alone"):
        run_py("browsecomp", "--problem", "1", *SOL, "--protocol", "task_only", "--agents", "2")
    with pytest.raises(ValueError, match="task_only is BrowseComp's single agent"):
        run_py("ale-bench", "--problem", "ahc014", *SOL, "--protocol", "task_only")
    with pytest.raises(ValueError, match="--examples requires the browsecomp task"):
        run_py("ale-bench", "--problem", "ahc014", *SOL, "--examples", "3")
    with pytest.raises(ValueError, match="needs --problem"):
        run_py("browsecomp", *SOL)


def test_ultra_reaches_codex_unchanged():
    # The runner clamps a requested effort to the model's levels; gpt-5.6-sol lists ultra.
    import agent
    catalog = json.dumps({"models": [{"slug": "gpt-5.6-sol", "supported_reasoning_levels": [
        {"effort": level} for level in ("low", "medium", "high", "xhigh", "max", "ultra")]}]})
    result = subprocess.run([sys.executable, "-c", agent.CODEX_EFFORT_CLAMP_SOURCE, "gpt-5.6-sol", "ultra"],
                            input=catalog, capture_output=True, text=True)
    assert result.stdout.strip() == "ultra"
    luna = catalog.replace("gpt-5.6-sol", "gpt-6-luna").replace(', {"effort": "ultra"}', "")
    result = subprocess.run([sys.executable, "-c", agent.CODEX_EFFORT_CLAMP_SOURCE, "gpt-6-luna", "ultra"],
                            input=luna, capture_output=True, text=True)
    assert result.stdout.strip() == "max"
