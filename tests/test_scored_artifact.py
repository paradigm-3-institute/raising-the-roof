"""Checks for the scored-artifact layer (scored_artifact.py, benchmarks/scored-artifact)
and the performance take-home on it (perf_takehome_task.py, benchmarks/perf-takehome).

No Docker needed. A throwaway git repository stands in for the take-home where only
the files' names matter; the scorer itself is checked on the real take-home when
.cache/perf-takehome holds it (`python perf_takehome_task.py fetch`).
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
import threading
import tomllib
from pathlib import Path

import pytest
import yaml

from cli_render import ROOT
from test_cli_agents import run_py

sys.path.insert(0, str(ROOT))
import perf_takehome_task  # noqa: E402
import scored_artifact  # noqa: E402
from perf_takehome_variants import variants  # noqa: E402

LAYER = ROOT / "benchmarks" / "scored-artifact"
SCORER_DIR = LAYER / "scorer"
REAL_CLONE = perf_takehome_task.CLONE


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


server = load("scored_artifact_server", SCORER_DIR / "server.py")
scorectl = load("scored_artifact_scorectl", LAYER / "client" / "scorectl.py")


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def fake_takehome(tmp_path, monkeypatch):
    """A git repository with the take-home's file names, pinned like the real one."""
    origin = tmp_path / "takehome"
    origin.mkdir()
    git("init", "-q", "-b", "main", cwd=origin)
    contents = {name: f"fake {name}\n".encode() for name in perf_takehome_task.FILES}
    for name, data in contents.items():
        (origin / name).parent.mkdir(parents=True, exist_ok=True)
        (origin / name).write_bytes(data)
    git("add", ".", cwd=origin)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "fake", cwd=origin)
    commit = git("rev-parse", "HEAD", cwd=origin).strip()
    monkeypatch.setattr(perf_takehome_task, "COMMIT", commit)
    monkeypatch.setattr(perf_takehome_task, "FILES", {
        name: hashlib.sha256(data).hexdigest() for name, data in contents.items()})
    monkeypatch.setattr(perf_takehome_task.BENCHMARK, "clone", origin)
    return origin


# ----------------------------------------------------------------------------
# Task directories


def test_task_for_a_team(fake_takehome, tmp_path):
    task = scored_artifact.build_task(perf_takehome_task.BENCHMARK, tmp_path / "tasks", agents=2,
                                      cli="codex", codex_login=True).write()
    assert task.name == "perf-takehome"
    settings = tomllib.loads((task / "task.toml").read_text())
    # Two agents get twice one agent's CPUs and memory.
    assert (settings["environment"]["cpus"], settings["environment"]["memory_mb"]) == (4, 12288)
    assert settings["environment"]["network_mode"] == "allowlist"
    assert settings["environment"]["allowed_hosts"] == list(scored_artifact.SETUP_HOSTS)
    assert (settings["agent"]["network_mode"], settings["agent"]["allowed_hosts"]) == ("allowlist", [])
    token = settings["verifier"]["env"]["SCORER_TOKEN"]
    assert len(token) == 32
    from harbor.models.task.config import TaskConfig
    parsed = TaskConfig.model_validate_toml((task / "task.toml").read_text())
    assert parsed.agent.explicit_phase_policy().allowed_hosts == []

    compose = yaml.safe_load((task / "environment/docker-compose.yaml").read_text())
    main, scorer = compose["services"]["main"], compose["services"]["scorer"]
    assert f"{task / 'client'}:/opt/scored-artifact:ro" in main["volumes"]
    assert f"{task / 'client/bin/submit'}:/usr/local/bin/submit:ro" in main["volumes"]
    assert main["depends_on"]["scorer"]["condition"] == "service_healthy"
    # Agents never mount the scorer's code, configuration, data, or state.
    assert not any("/scorer" in volume for volume in main["volumes"])
    assert f"{task / 'scorer'}:/scorer/app:ro" in scorer["volumes"]
    assert scorer["volumes"][-1].endswith("/../scorer:/scorer/state")
    assert "HOST_VERIFIER_LOGS_PATH" in scorer["volumes"][-1]
    assert "network_mode" not in scorer and "networks" not in scorer  # shares the egress sidecar

    config = json.loads((task / "scorer/config.json").read_text())
    assert stat.S_IMODE((task / "scorer/config.json").stat().st_mode) == 0o600
    assert config["token"] == token and len(config["benchmark"]["seeds"]) == 9
    assert config["artifact"] == {"files": ["perf_takehome.py"], "max_bytes": 1_000_000}
    client = json.loads((task / "client/task.json").read_text())
    assert client["scorer_urls"][0] == "http://127.0.0.1:18090"
    assert "token" not in json.dumps(client) and "seeds" not in json.dumps(client)
    for name in perf_takehome_task.FILES:
        assert (task / "client/starter" / name).read_bytes() == (fake_takehome / name).read_bytes()
        assert (task / "data/upstream" / name).read_bytes() == (fake_takehome / name).read_bytes()
    assert (task / "scorer/benchmark/score.py").is_file() and (task / "scorer/server.py").is_file()
    assert (task / "client/bin/submit").stat().st_mode & 0o111
    instruction = (task / "instruction.md").read_text()
    assert "2 agents share this" in instruction and "fetch-submission" in instruction
    described = json.loads((task / "scored-artifact.json").read_text())
    assert described["commit"] == perf_takehome_task.COMMIT
    assert described["network"]["agent_phase"] == ["chatgpt.com", "auth.openai.com"]
    # Every job draws its own hidden inputs.
    again = scored_artifact.build_task(perf_takehome_task.BENCHMARK, tmp_path / "other")
    assert json.loads(again.files["scorer/config.json"])["benchmark"]["seeds"] != config["benchmark"]["seeds"]


def test_solo_instruction_is_the_team_text_minus_peers(fake_takehome, tmp_path):
    task = scored_artifact.build_task(perf_takehome_task.BENCHMARK, tmp_path, agents=1)
    solo = task.files["instruction.md"].decode()
    team = perf_takehome_task.TEAM_INSTRUCTION.replace("{n}", "1")
    assert "share this" not in solo and "teammate" not in solo and "of any agent" not in solo
    assert "- `scores` lists your graded submissions, best first." in solo
    assert len(solo.splitlines()) == len(team.splitlines()) - 2
    with pytest.raises(ValueError, match="no longer applies"):
        scored_artifact.solo_text("changed text", perf_takehome_task.SOLO_EDITS)


def test_resources_and_hosts(fake_takehome, tmp_path):
    task = scored_artifact.build_task(perf_takehome_task.BENCHMARK, tmp_path, agents=4, cpus=6,
                                      memory_mb=9000, cli="claude")
    settings = tomllib.loads(task.task_toml)
    assert (settings["environment"]["cpus"], settings["environment"]["memory_mb"]) == (6, 9000)
    assert task.agent_hosts == ["api.anthropic.com"]
    assert scored_artifact.model_api_hosts("codex", False) == ["api.openai.com"]


def test_pinned_files_are_checked(fake_takehome, monkeypatch):
    assert set(perf_takehome_task.upstream_files(fake_takehome)) == set(perf_takehome_task.FILES)
    monkeypatch.setitem(perf_takehome_task.FILES, "problem.py", "0" * 64)
    with pytest.raises(ValueError, match="problem.py .* does not match its pinned hash"):
        perf_takehome_task.upstream_files(fake_takehome)


def test_run_py_perf_takehome(fake_takehome, tmp_path):
    name = "pytest-perf-takehome-dry-run"
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    cmd, env, job = run_py("perf-takehome", "--cli", "codex", "--model", "gpt-6-luna", "--effort", "max",
                           "--protocol", "antiherding_matched", "--agents", "2", "--token-budget", "300000",
                           "--timeout", "3600", "--cpus", "4", "--memory-mb", "8192",
                           "--codex-auth", str(auth), "--name", name)
    task = scored_artifact.tasks_root("perf-takehome") / name / "perf-takehome"
    assert cmd[cmd.index("-p") + 1] == str(task)
    assert not task.exists()  # --dry-run writes nothing
    hosts = [cmd[i + 1] for i, token in enumerate(cmd) if token == "--allow-agent-host"]
    assert hosts == ["chatgpt.com", "auth.openai.com"]
    assert "--override-cpus" not in cmd and "--override-memory-mb" not in cmd
    # Harbor's timeout: --timeout, 48 h of usage-limit waiting and 30 min (run.py --limit-resume).
    assert cmd[cmd.index("--agent-timeout-multiplier") + 1] == str((3600 + 48 * 3600 + 1800) / 10800)
    verifier_env = [cmd[i + 1] for i, token in enumerate(cmd) if token == "--ve"]
    assert verifier_env == ["SCORED_ARTIFACT_BENCHMARK=perf-takehome",
                            f"PERF_TAKEHOME_COMMIT={perf_takehome_task.COMMIT}"]
    assert env["RELEASE_TASK"] == "perf-takehome" and env["RELEASE_TOKEN_BUDGET"] == "300000"
    assert env["RELEASE_CONTINUATION"] == (
        ROOT / "benchmarks/perf-takehome/continuation.txt").read_text().rstrip("\n")
    assert env["MULTIAGENT_PROTOCOL"] == "multiagent_protocols.antiherding_matched:MultiAgentProtocol"
    assert job == ROOT / "jobs" / name


def test_run_py_perf_takehome_errors(fake_takehome):
    with pytest.raises(ValueError, match="--cli and --model are required"):
        run_py("perf-takehome", "--model", "m")
    with pytest.raises(ValueError, match="requires the ale-bench or browsecomp task"):
        run_py("perf-takehome", "--cli", "codex", "--model", "m", "--problem", "3")


# ----------------------------------------------------------------------------
# The scorer service, with a stand-in benchmark


FAKE_SCORER = '''
def score(submission, sandbox, config):
    value = int((submission / "answer.txt").read_text())
    if value <= 0:
        return {"valid": False, "detail": "not positive"}
    return {"valid": True, "metric": value, "score": 100 / value, "detail": "ok", "info": {"secret": 1}}
'''


@pytest.fixture
def scorer_config(tmp_path):
    module = tmp_path / "fake_score.py"
    module.write_text(FAKE_SCORER)
    return {"benchmark_name": "fake", "module": str(module), "state_dir": str(tmp_path / "state"),
            "token": "tok", "sandbox_uid": 20000, "sandbox_root": str(tmp_path / "sandbox"), "workers": 1,
            "grade_timeout_sec": 60, "max_pending_per_agent": 2, "final_candidates": 3,
            "metric_name": "cycles", "artifact": {"files": ["answer.txt"], "max_bytes": 100},
            "public": {"metric": "cycles"}, "benchmark": {}}


def answer(value) -> dict:
    return {"answer.txt": base64.b64encode(str(value).encode()).decode()}


def drain(scorer) -> None:
    """Grade everything queued, in the scorer's round-robin order."""
    while True:
        with scorer.lock:
            sid = scorer._next()
            if sid is None:
                return
            scorer.running += 1
            scorer.records[sid]["status"] = "running"
        result = server.grade(scorer.config, scorer.state / "submissions" / sid / "files")
        with scorer.lock:
            scorer.records[sid].update(result, ts=server.now_iso())
            scorer.running -= 1
            scorer._append_log(scorer.records[sid])


def test_scorer_queue_log_and_final(scorer_config):
    scorer = server.Scorer(scorer_config)
    order = [scorer.submit("a1", answer(v))["sid"] for v in (50, 40)]
    order += [scorer.submit("a2", answer(v))["sid"] for v in (45, 0)]
    with pytest.raises(OverflowError, match="already has 2 submissions waiting"):
        scorer.submit("a1", answer(1))
    with pytest.raises(ValueError, match="exactly answer.txt"):
        scorer.submit("a1", {"other.txt": answer(1)["answer.txt"]})
    with pytest.raises(ValueError, match="limit is 100"):
        scorer.submit("a2", {"answer.txt": base64.b64encode(b"1" * 101).decode()})
    # Graded round robin, a1, a2, a1, a2, although a1 sent both of its first.
    drain(scorer)
    log = [json.loads(line) for line in (scorer.state / "submissions.jsonl").read_text().splitlines()]
    assert [r["sid"] for r in log] == [order[0], order[2], order[1], order[3]]
    assert [(r["agent_id"], r["valid"], r["metric"], r["score"]) for r in log] == [
        ("a1", True, 50, 2.0), ("a2", True, 45, 100 / 45), ("a1", True, 40, 2.5), ("a2", False, None, 0.0)]
    assert all(r["status"] == "done" and r["ts"] and r["submitted_ts"] for r in log)
    assert "secret" not in json.dumps(scorer.public(scorer.records[order[0]]))  # info stays host-side
    # The verifier's final copies: identical ones are not graded again.
    assert scorer.submit("a1", answer(40), final=True)["duplicate_of"] == order[1]
    final = scorer.finalize()
    assert (final["valid"], final["metric"], final["score"], final["sid"]) == (True, 40, 2.5, order[1])
    assert json.loads((scorer.state / "final.json").read_text()) == final
    with pytest.raises(PermissionError):
        scorer.submit("a1", answer(1))
    # A restarted scorer keeps the log and the final grade.
    again = server.Scorer(scorer_config)
    assert len(again.records) == 4 and again.closed and again.final == final


def test_a_restarted_scorer_queues_the_waiting_submissions_again(scorer_config):
    scorer = server.Scorer(scorer_config)
    graded = scorer.submit("a1", answer(50))["sid"]
    drain(scorer)
    waiting = [scorer.submit("a1", answer(40), note="second")["sid"], scorer.submit("a2", answer(45))["sid"]]
    # A directory left without submission.json (an older scorer's) is not queued, but its sid is not
    # given out again.
    (scorer.state / "submissions" / "9" / "files").mkdir(parents=True)
    again = server.Scorer(scorer_config)  # the restart: the two waiting ones were only in memory
    assert again.records[graded]["status"] == "done"
    assert [again.records[sid]["status"] for sid in waiting] == ["queued", "queued"]
    assert again.records[waiting[0]]["note"] == "second" and "9" not in again.records
    assert again.public(again.records[waiting[1]])["position"] == 1
    assert again.submit("a1", answer(30))["sid"] == "10"
    drain(again)
    log = [json.loads(line) for line in (again.state / "submissions.jsonl").read_text().splitlines()]
    assert [r["sid"] for r in log] == [graded, waiting[0], waiting[1], "10"]
    assert [r["metric"] for r in log] == [50, 40, 45, 30]


@pytest.mark.parametrize("deferred", [False, True])
def test_verifier_finalizes_or_postpones_the_final_grade(tmp_path, monkeypatch, deferred):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "finalize.py").write_bytes((LAYER / "tests" / "finalize.py").read_bytes())
    if deferred:
        (tests_dir / "defer_final.json").write_text(json.dumps({"why": "graded later on a quiet machine"}))
    work = tmp_path / "work"
    (work / "a1").mkdir(parents=True)
    (work / "a1" / "answer.txt").write_text("40")
    config = tmp_path / "task.json"
    config.write_text(json.dumps({"scorer_urls": ["http://scorer"], "work_root": str(work),
                                  "artifact_files": ["answer.txt"], "finalize_wait_sec": 5}))
    monkeypatch.setenv("SCORED_ARTIFACT_CONFIG", str(config))
    finalize = load(f"finalize_{deferred}", tests_dir / "finalize.py")
    monkeypatch.setattr(finalize, "LOGS", tmp_path / "logs")
    calls = []

    def request(urls, method, path, body=None, timeout=60):
        calls.append(path)
        if path == "/submit":
            return 200, {"sid": "7"}
        if path == "/finalize":
            return 200, {"score": 2.5, "metric": 40, "valid": True}
        return 200, {}

    monkeypatch.setattr(finalize, "request", request)
    assert finalize.main() == 0
    reward = json.loads((tmp_path / "logs" / "reward.json").read_text())
    final = json.loads((tmp_path / "logs" / "scorer_final.json").read_text())
    if deferred:
        assert calls == ["/health", "/submit"]  # the final copy is still submitted; nothing closes the scorer
        assert reward == {"reward": 0.0, "deferred": 1}
        assert final["deferred"] is True and final["final_copies"] == ["7"]
    else:
        assert calls == ["/health", "/submit", "/finalize"]
        assert reward == {"reward": 2.5, "metric": 40, "valid": 1}


def test_final_grade_falls_back_when_the_best_fails_its_regrade(scorer_config, tmp_path):
    scorer = server.Scorer(scorer_config)
    best = scorer.submit("a1", answer(10))["sid"]
    scorer.submit("a1", answer(20))
    drain(scorer)
    (scorer.state / "submissions" / best / "files" / "answer.txt").write_text("-1")
    final = scorer.finalize()
    assert (final["metric"], final["candidates_tried"]) == (20, 2)


def test_grading_crashes_and_timeouts(scorer_config, tmp_path):
    module = Path(scorer_config["module"])
    module.write_text("import time\ndef score(submission, sandbox, config):\n    time.sleep(30)\n")
    (tmp_path / "files").mkdir()
    result = server.grade({**scorer_config, "grade_timeout_sec": 1}, tmp_path / "files")
    assert (result["status"], result["valid"], result["score"]) == ("done", False, 0.0)
    module.write_text("def score(submission, sandbox, config):\n    raise RuntimeError('bug')\n")
    result = server.grade(scorer_config, tmp_path / "files")
    assert result["status"] == "error" and "RuntimeError: bug" in result["info"]["traceback"]


def test_client_against_a_live_scorer(scorer_config, tmp_path, monkeypatch, capsys):
    scorer = server.Scorer(scorer_config)
    threading.Thread(target=scorer.worker, args=(0,), daemon=True).start()
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.handler(scorer, "tok"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    starter = tmp_path / "starter"
    starter.mkdir()
    (starter / "answer.txt").write_text("80")
    config = tmp_path / "task.json"
    config.write_text(json.dumps({
        "benchmark": "fake", "starter": str(starter), "work_root": str(tmp_path / "work"),
        "artifact_files": ["answer.txt"], "scorer_urls": [f"http://127.0.0.1:{httpd.server_port}"],
        "metric_name": "cycles", "submit_wait_sec": 60}))
    monkeypatch.setattr(scorectl, "CONFIG", config)
    monkeypatch.setenv("AGENT_ID", "a2")
    try:
        assert scorectl.main(["workcopy"]) == 0
        work = tmp_path / "work" / "a2"
        assert (work / "answer.txt").read_text() == "80" and (work / ".git").is_dir()
        assert scorectl.main(["submit", "--note", "first"]) == 0
        (tmp_path / "better.txt").write_text("20")
        assert scorectl.main(["submit", str(tmp_path / "better.txt")]) == 0
        out = capsys.readouterr().out
        assert "valid cycles=80 score=1.2500" in out and "best so far: cycles=20 score=5.0000 by a2" in out
        assert scorectl.main(["scores"]) == 0
        assert scorectl.main(["fetch-submission", "best", str(tmp_path / "adopt")]) == 0
        assert (tmp_path / "adopt" / "answer.txt").read_text() == "20"
        assert scorectl.main(["submit", str(tmp_path / "missing")]) == 2
        status, reply = scorectl.request(scorectl.config(), "POST", "/finalize", {"token": "wrong"})
        assert status == 403
        status, reply = scorectl.request(scorectl.config(), "POST", "/finalize", {"token": "tok"})
        assert (status, reply["metric"]) == (200, 20)
        assert scorectl.main(["submit"]) == 3  # closed
    finally:
        httpd.shutdown()


# ----------------------------------------------------------------------------
# The take-home's scorer on the real take-home


needs_takehome = pytest.mark.skipif(
    not (REAL_CLONE / ".git").exists(), reason="needs the take-home in .cache/perf-takehome "
                                               "(python perf_takehome_task.py fetch)")


@needs_takehome
def test_takehome_scorer_on_variants(tmp_path):
    files = perf_takehome_task.upstream_files(REAL_CLONE, fetch_clone=False)
    upstream = tmp_path / "upstream"
    for name, data in files.items():
        (upstream / name).parent.mkdir(parents=True, exist_ok=True)
        (upstream / name).write_bytes(data)
    benchmark = {**perf_takehome_task.BENCHMARK.prepare(fetch=False).scorer_config,
                 "upstream_dir": str(upstream), "seeds": [1, 2, 3, 4], "fresh_inputs": 1}
    config = {"module": str(ROOT / "benchmarks/perf-takehome/scorer/score.py"), "sandbox_uid": 20000,
              "sandbox_root": str(tmp_path / "sandbox"), "grade_timeout_sec": 600, "benchmark": benchmark}
    results = {}
    for name, kernel in variants(files["perf_takehome.py"].decode()).items():
        directory = tmp_path / name
        directory.mkdir()
        (directory / "perf_takehome.py").write_text(kernel)
        results[name] = server.grade(config, directory)
    assert (results["starter"]["metric"], results["starter"]["score"]) == (147734, 1.0)
    assert results["packed"]["metric"] == 98583  # the take-home's tests/submission_tests.py agrees
    assert results["patcher"]["metric"] == 147734  # the upstream test reports 1
    branchy = results["branchy"]
    assert branchy["valid"] and branchy["metric"] == max(branchy["info"]["fixed_cycles"])
    assert set(branchy["info"]["fixed_cycles"]) <= {147738, 148038}
    for name in ("wrong", "slots"):
        assert results[name]["valid"] is False and results[name]["score"] == 0.0
    assert "incorrect output values" in results["wrong"]["detail"]
    assert "SLOT_LIMITS" in results["slots"]["detail"]
    assert all(r["status"] == "done" for r in results.values())


@needs_takehome
def test_takehome_scorer_rejects_bad_builds(tmp_path):
    files = perf_takehome_task.upstream_files(REAL_CLONE, fetch_clone=False)
    upstream = tmp_path / "upstream"
    for name, data in files.items():
        (upstream / name).parent.mkdir(parents=True, exist_ok=True)
        (upstream / name).write_bytes(data)
    config = {"module": str(ROOT / "benchmarks/perf-takehome/scorer/score.py"), "sandbox_uid": 20000,
              "sandbox_root": str(tmp_path / "sandbox"), "grade_timeout_sec": 120, "benchmark": {
                  "upstream_dir": str(upstream), "baseline_cycles": 147734, "forest_height": 10, "rounds": 16,
                  "batch_size": 256, "seeds": [1], "fresh_inputs": 0, "build_timeout_sec": 5,
                  "simulation_timeout_sec": 5}}
    cases = {
        "raises": "class KernelBuilder:\n    def build_kernel(self, *a):\n        raise ValueError('nope')\n",
        "slow": "import time\nclass KernelBuilder:\n    def build_kernel(self, *a):\n        time.sleep(60)\n",
        "engine": ("class KernelBuilder:\n    instrs = [{'gpu': [('x',)]}]\n"
                   "    def build_kernel(self, *a): pass\n    def debug_info(self): pass\n"),
        "object": ("class KernelBuilder:\n    instrs = [{'alu': [object()]}]\n"
                   "    def build_kernel(self, *a): pass\n    def debug_info(self): pass\n"),
        "loop": ("class KernelBuilder:\n    instrs = [{'flow': [('jump', 0)]}]\n"
                 "    def build_kernel(self, *a): pass\n    def debug_info(self): pass\n"),
    }
    details = {}
    for name, source in cases.items():
        directory = tmp_path / name
        directory.mkdir()
        (directory / "perf_takehome.py").write_text(source)
        result = server.grade(config, directory)
        assert (result["status"], result["valid"]) == ("done", False), (name, result)
        details[name] = result["detail"]
    assert "ValueError: nope" in details["raises"]
    assert "took longer than 5 s" in details["slow"]
    assert "engine 'gpu'" in details["engine"]
    assert "not plain data" in details["object"]
    assert "longer than 5 s" in details["loop"]


# ----------------------------------------------------------------------------
# Reading jobs back


def scored_job(root: Path, task_dir: Path) -> Path:
    job = root / "job"
    trial = job / "perf-takehome__abc1234"
    (trial / "agent").mkdir(parents=True)
    (trial / "scorer").mkdir()
    (job / "config.json").write_text(json.dumps({"tasks": [{"path": str(task_dir)}]}))
    records = [
        {"sid": "1", "agent_id": "a1", "status": "done", "valid": True, "metric": 2000, "score": 73.867,
         "ts": "2026-10-07T10:10:00.000Z", "sha256": "x1"},
        {"sid": "2", "agent_id": "a2", "status": "done", "valid": False, "metric": None, "score": 0.0,
         "ts": "2026-10-07T10:20:00.000Z", "sha256": "x2"},
        {"sid": "3", "agent_id": "a2", "status": "done", "valid": True, "metric": 1500, "score": 98.489,
         "ts": "2026-10-07T10:30:00.000Z", "sha256": "x3"},
    ]
    (trial / "scorer/submissions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (trial / "scorer/final.json").write_text(json.dumps(
        {"valid": True, "score": 98.489, "metric": 1500, "metric_name": "cycles", "sid": "3"}))
    # A forged agent-side log must not count.
    (trial / "agent/submissions.jsonl").write_text(json.dumps(
        {"status": "done", "score": 1e9, "ts": "2026-10-07T10:00:00Z", "agent_id": "a1"}) + "\n")
    (trial / "result.json").write_text(json.dumps({
        "agent_execution": {"started_at": "2026-10-07T10:00:00Z"},
        "verifier_result": {"rewards": {"reward": 5e9}}}))
    return job


def test_regrade_finds_the_best_submissions(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "scored-artifact.json").write_text(json.dumps({"benchmark": "perf-takehome"}))
    job = scored_job(tmp_path, task_dir)
    trial = next(p for p in job.iterdir() if p.is_dir())
    for sid in ("1", "3"):
        (trial / f"scorer/submissions/{sid}/files").mkdir(parents=True)
    found = scored_artifact.best_submissions(job, top=5)
    assert [(label.split("#")[1], record["metric"]) for label, record, _ in found] == [("3", 1500), ("1", 2000)]
