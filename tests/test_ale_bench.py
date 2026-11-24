"""ALE-Bench (ale_bench_task.py, benchmarks/ale-bench/): everything testable without agents,
Docker or the `ale_bench` package. The judge's grading with ALE-Bench's own code, the
parity with the package used directly, and runs through Harbor are checked by hand
(`python ale_bench_task.py direct|check`)."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import tomllib
import types
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ale_bench_task  # noqa: E402

BENCH = ROOT / "benchmarks" / "ale-bench"
LAYER = ROOT / "benchmarks" / "scored-artifact"


def load(name: str, path: Path, *extra: Path):
    """A module from a file, without leaving its directories on sys.path or its own
    imports (the layer's `server`, `sandbox`) in sys.modules for other tests' modules of the
    same names."""
    saved_path, saved_modules = list(sys.path), dict(sys.modules)
    for directory in (path.parent, *extra):
        sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path
        for added in set(sys.modules) - set(saved_modules) - {name}:
            if added in ("server", "sandbox"):
                del sys.modules[added]
    return module


judge = load("ale_bench_judge", BENCH / "scorer" / "judge.py", LAYER / "scorer")
alectl = load("ale_bench_alectl", BENCH / "client" / "alectl.py")
alelocal = load("ale_bench_alelocal", BENCH / "client" / "alelocal.py")


# ---------------------------------------------------------------------------
# Selection and feedback, as ALE-Bench's evaluation does them


def record(sid, score, valid=True, status="done", **extra):
    return {"sid": str(sid), "agent_id": "a1", "status": status, "valid": valid, "public_score": score, **extra}


def test_final_candidates_take_the_best_public_accepted_and_the_earliest_of_ties():
    records = [record(1, 100), record(2, 300, valid=False), record(3, 250), record(4, 250), record(5, None),
               record(6, 400, status="error")]
    assert [r["sid"] for r in judge.final_candidates(records, "maximize")] == ["3", "4", "1"]
    assert [r["sid"] for r in judge.final_candidates(records, "minimize")] == ["1", "3", "4"]
    assert judge.better(5, None, "maximize") and judge.better(5, 6, "minimize") and not judge.better(5, 5, "maximize")


def case(verdict, score=0, time_=0.5, memory=3 * 2**20, error="", message=""):
    return SimpleNamespace(judge_result=SimpleNamespace(value=verdict), absolute_score=score, execution_time=time_,
                           memory_usage=memory, error_str=error, message=message)


def test_feedback_is_ale_bench_evals_public_result_text():
    accepted = [case("ACCEPTED", 10, 1.25), case("ACCEPTED", 20, 1.5)]
    assert judge.feedback("ACCEPTED", 30, accepted, 2.0) == (
        "[Public test result]\nOverall judge result: ACCEPTED\nOverall absolute score: 30\n"
        "- Case 1: 10\n- Case 2: 20\nLongest execution time: 1.500 sec (case 2; time limit 2 sec)")
    failing = [case("ACCEPTED", 10), case("TIME_LIMIT_EXCEEDED", 0, 2.1, error="x" * 5000, message="Time limit "
                                                                                                    "exceeded.")]
    text = judge.feedback("TIME_LIMIT_EXCEEDED", 0, failing, 2.0)
    assert text.startswith("[Public test result]\nOverall judge result: TIME_LIMIT_EXCEEDED\n- Case 2:\n"
                           "    Absolute score: 0\n    Execution time: 2.100 sec\n    Memory usage: 3 MB\n")
    assert '    Message: "Time limit exceeded."' in text and "Overall absolute score" not in text
    assert len(text) < judge.STDERR_TAIL + 500  # a long stderr is cut


# ---------------------------------------------------------------------------
# The judge service, with a stand-in for ALE-Bench's evaluations


class FakeEvaluator:
    """Programs are 'score N' (public N; private 3N; performance 10N) or anything else
    (WRONG_ANSWER)."""

    def __init__(self) -> None:
        self.private_calls: list[tuple[str, bool]] = []
        self.gate = threading.Event()
        self.gate.set()

    @staticmethod
    def parse(code: str) -> int | None:
        parts = code.split()
        return int(parts[1]) if len(parts) == 2 and parts[0] == "score" else None

    def public(self, code: str):
        value = self.parse(code)
        if value is None:
            return ({"status": "done", "valid": False, "public_score": 0, "public_judge": "WRONG_ANSWER",
                     "detail": "wrong", "judge": {}}, {})
        return ({"status": "done", "valid": True, "public_score": value, "public_judge": "ACCEPTED",
                 "detail": f"score {value}", "judge": {}}, {"case_scores": [value]})

    def private(self, code: str, lane: bool = False) -> dict:
        self.gate.wait(10)
        self.private_calls.append((code, lane))
        value = self.parse(code) or 0
        return {"status": "done", "private_score": 3 * value, "private_absolute_score": 3 * value,
                "private_relative_score": 3 * value, "performance": 10 * value, "rank": 1000 - value,
                "private_judge": "ACCEPTED" if value else "COMPILATION_ERROR", "private_cases": {}, "judge": {}}


def judge_config(tmp_path: Path, score_type: str = "maximize", variant: str = "official") -> dict:
    return {"problem_id": "ahc999", "lite_version": False, "language": "cpp20", "judge_version": "202301",
            "variant": variant, "score_type": score_type, "state_dir": str(tmp_path / "state"), "token": "secret",
            "artifact": {"files": ["Main.cpp"], "max_bytes": 524288}, "metric_name": "public_score",
            "max_pending_per_agent": 10, "final_candidates": 3, "workers": 1, "public": {"benchmark": "ale-bench"}}


def files(code: str) -> dict:
    return {"Main.cpp": base64.b64encode(code.encode()).decode()}


def drain(service) -> None:
    """Grade everything queued with the service's own worker loop, then stop it."""
    worker = threading.Thread(target=service.worker, args=(0,), daemon=True)
    worker.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        with service.lock:
            if not any(service.queues.values()) and not service.running:
                return
        time.sleep(0.02)
    raise AssertionError("the queue did not drain")


def test_agents_see_public_results_only(tmp_path):
    service = judge.AleJudge(judge_config(tmp_path), FakeEvaluator())
    sid = service.submit("a1", files("score 7"))["sid"]
    drain(service)
    record_ = service.records[sid]
    record_["performance"] = 1234  # anything private on the record stays off the public view
    shown = service.public(record_)
    assert shown["public_score"] == 7 and shown["valid"] and shown["public_judge"] == "ACCEPTED"
    assert set(shown) <= set(judge.PUBLIC_FIELDS) and "performance" not in shown and "judge" not in shown
    logged = [json.loads(line) for line in (tmp_path / "state" / "submissions.jsonl").read_text().splitlines()]
    assert logged[0]["sid"] == sid and logged[0]["public_score"] == 7 and logged[0]["language"] == "cpp20"


def test_private_lane_grades_the_latest_final_candidate_and_finalize_the_best(tmp_path):
    evaluator = FakeEvaluator()
    service = judge.AleJudge(judge_config(tmp_path), evaluator)
    for code in ("score 100", "score 90", "wrong", "score 120"):
        service.submit("a1", files(code))
    drain(service)
    assert service.lane_best["sid"] == "4"  # 90 and the wrong one never become the candidate
    lane = threading.Thread(target=service.lane, name=judge.LANE_THREAD, daemon=True)
    lane.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        with service.lock:
            if "4" in service.lane_done:
                break
        time.sleep(0.02)
    entries = [json.loads(line) for line in (tmp_path / "state" / "private.jsonl").read_text().splitlines()]
    assert [(e["sid"], e["performance"], e["lane"]) for e in entries] == [("4", 1200, "private")]
    final = service.finalize()
    assert judge.LANE_CANCEL.is_set()
    judge.LANE_CANCEL.clear()
    assert final["valid"] and final["sid"] == "4" and final["score"] == 1200 and final["public_score"] == 120
    assert final["metric"] == 360 and final["metric_name"] == "private_score" and final["valid_submissions"] == 3
    assert evaluator.private_calls[-1] == ("score 120", False)  # re-graded, not reused from the lane
    assert json.loads((tmp_path / "state" / "final.json").read_text())["score"] == 1200
    with pytest.raises(PermissionError):
        service.submit("a1", files("score 999"))


def test_minimize_problems_pick_the_lowest_public_score(tmp_path):
    service = judge.AleJudge(judge_config(tmp_path, "minimize"), FakeEvaluator())
    for code in ("score 50", "score 20", "score 30"):
        service.submit("a1", files(code))
    drain(service)
    assert service.lane_best["sid"] == "2"
    assert service.finalize()["sid"] == "2"
    judge.LANE_CANCEL.clear()


def test_a_run_without_an_accepted_submission_gets_a_compile_errors_grade(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "ce_code", lambda language: "does not compile")
    evaluator = FakeEvaluator()
    service = judge.AleJudge(judge_config(tmp_path), evaluator)
    service.submit("a1", files("wrong"))
    drain(service)
    final = service.finalize()
    judge.LANE_CANCEL.clear()
    assert not final["valid"] and final["sid"] is None and final["score"] == 0
    assert evaluator.private_calls == [("does not compile", False)]
    assert final["detail"] == "no public-ACCEPTED submission"


def test_the_layers_http_api_and_the_agents_commands(tmp_path, monkeypatch):
    service = judge.AleJudge(judge_config(tmp_path), FakeEvaluator())
    threading.Thread(target=service.worker, args=(0,), daemon=True).start()
    http = ThreadingHTTPServer(("127.0.0.1", 0), judge.handler(service, "secret"))
    threading.Thread(target=http.serve_forever, daemon=True).start()
    task = {"scorer_urls": [f"http://127.0.0.1:{http.server_port}"], "work_root": str(tmp_path / "work"),
            "artifact_files": ["Main.cpp"], "score_type": "maximize", "agents": 2, "starter": str(tmp_path / "s")}
    monkeypatch.setenv("AGENT_ID", "a2")
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "problem.json").write_text("{}")
    assert alectl.cmd_workcopy(task, SimpleNamespace()) == 0
    (tmp_path / "work" / "a2" / "Main.cpp").write_text("score 41")
    assert alectl.cmd_submit(task, SimpleNamespace(path=None, note="first")) == 0
    status, listing = alectl.request(task, "GET", "/submissions")
    assert status == 200 and listing["submissions"][0]["public_score"] == 41
    assert "performance" not in json.dumps(listing) and "private" not in json.dumps(listing)
    assert alectl.best(listing["submissions"], task)["sid"] == "1"
    out = tmp_path / "fetched"
    assert alectl.cmd_fetch(task, SimpleNamespace(sid="best", dest=str(out))) == 0
    assert (out / "Main.cpp").read_text() == "score 41"
    assert alectl.request(task, "POST", "/finalize", {"token": "wrong"})[0] == 403
    assert alectl.request(task, "POST", "/submit", {"agent_id": "a2", "files": files("score 1"), "final": True,
                                                    "token": "wrong"})[0] == 403
    status, final = alectl.request(task, "POST", "/finalize", {"token": "secret"})
    judge.LANE_CANCEL.clear()
    assert status == 200 and final["score"] == 410
    http.shutdown()


# ---------------------------------------------------------------------------
# Timing: the judge's CPUs, one container each, machine-wide


def test_core_pool_hands_out_each_cpu_once_across_processes(tmp_path):
    holder = subprocess.Popen([sys.executable, "-c", (
        "import fcntl, os, sys, time\n"
        f"fd = os.open({str(tmp_path / 'ale-bench-cpu3.lock')!r}, os.O_RDWR | os.O_CREAT)\n"
        "fcntl.flock(fd, fcntl.LOCK_EX); print('held', flush=True); time.sleep(30)\n")],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        pool = judge.CorePool([3, 5], tmp_path)
        cpu, fd = pool.acquire()
        assert cpu == 5  # CPU 3 is held by another judge process
        got = []
        waiter = threading.Thread(target=lambda: got.append(pool.acquire()), daemon=True)
        waiter.start()
        time.sleep(0.3)
        assert not got  # both CPUs taken: the next container waits
        pool.release(cpu, fd)
        waiter.join(5)
        assert got and got[0][0] == 5
    finally:
        holder.kill()


def test_every_ale_bench_container_is_pinned_and_releases_its_cpu(tmp_path, monkeypatch):
    calls = []

    class Container:
        def remove(self, force=False):
            calls.append(("remove", force))

    class ContainerCollection:
        def run(self, image, command=None, stdout=True, stderr=False, remove=False, **kwargs):
            calls.append(("run", image, kwargs))
            return Container()

    fake = types.ModuleType("docker.models.containers")
    fake.ContainerCollection = ContainerCollection
    models, docker = types.ModuleType("docker.models"), types.ModuleType("docker")
    models.containers, docker.models = fake, models
    monkeypatch.setitem(sys.modules, "docker", docker)
    monkeypatch.setitem(sys.modules, "docker.models", models)
    monkeypatch.setitem(sys.modules, "docker.models.containers", fake)
    pool = judge.CorePool([7], tmp_path)
    judge.pin_containers(pool, {"ttc.ale-bench": "judge"}, "adapter-test")
    container = ContainerCollection().run("ale-bench:cpp20-202301", "true", cpu_quota=100000, network_disabled=True)
    _, image, kwargs = calls[0]
    assert kwargs["cpuset_cpus"] == "7" and kwargs["cpu_quota"] == 100000 and kwargs["network_disabled"]
    assert kwargs["labels"] == {"ttc.ale-bench": "judge"} and kwargs["name"].startswith("adapter-test-")
    assert pool.held == {7}
    container.remove(force=True)
    assert pool.held == set() and calls[-1] == ("remove", True)
    judge.LANE_CANCEL.set()
    errors = []

    def lane_run():
        try:
            ContainerCollection().run("x")
        except judge.LaneCancelled as exc:
            errors.append(exc)

    lane = threading.Thread(target=lane_run, name=judge.LANE_THREAD)
    lane.start()
    lane.join()
    judge.LANE_CANCEL.clear()
    assert errors and pool.held == set()


def test_an_empty_tester_log_is_read_again(tmp_path, monkeypatch):
    answers = iter([b"", b"", b"Score = 5\n"])

    class Container:
        def logs(self, stdout=True, stderr=True):
            return next(answers, b"")

        def remove(self, force=False):
            pass

    class ContainerCollection:
        def run(self, image, command=None, stdout=True, stderr=False, remove=False, **kwargs):
            return Container()

    fake = types.ModuleType("docker.models.containers")
    fake.ContainerCollection = ContainerCollection
    models, docker = types.ModuleType("docker.models"), types.ModuleType("docker")
    models.containers, docker.models = fake, models
    monkeypatch.setitem(sys.modules, "docker", docker)
    monkeypatch.setitem(sys.modules, "docker.models", models)
    monkeypatch.setitem(sys.modules, "docker.models.containers", fake)
    judge.LOG_STATS.clear()
    judge.pin_containers(judge.CorePool([0, 1], tmp_path), {}, "adapter-test")
    tester = ContainerCollection().run("ale-bench:cpp20-202301",
                                       ["/bin/bash", "-c", "/judge/target/release/tester ./a.out < in > out"])
    assert tester.logs(stdout=False, stderr=True) == b"Score = 5\n"
    assert judge.LOG_STATS == {"empty": 1, "recovered": 1}
    program = ContainerCollection().run("ale-bench:cpp20-202301", ["/bin/bash", "-c", "./a.out < in > out"])
    assert program.logs() == b""  # a program's own empty stderr is not waited for
    tester.remove()
    program.remove()
    monkeypatch.setattr(judge, "LOG_RETRY_SECONDS", 0.2)
    lost = ContainerCollection().run("rust:1.79.0-buster", "/judge/target/release/tester /tmp/i /tmp/o")
    with pytest.raises(judge.TesterLogLost):
        lost.logs()  # still empty: the case is run again (next test)
    lost.remove()


def test_a_case_whose_judging_raised_is_run_again(monkeypatch):
    calls = []

    def case_iter_func(index):
        calls.append(index)
        if len(calls) < 3:
            raise judge.TesterLogLost("lost")
        return f"verdict {index}"

    runner = types.ModuleType("ale_bench.tool_wrappers.case_runner")
    runner.case_iter_func = case_iter_func
    for name, module in (("ale_bench", types.ModuleType("ale_bench")),
                         ("ale_bench.tool_wrappers", types.ModuleType("ale_bench.tool_wrappers")),
                         ("ale_bench.tool_wrappers.case_runner", runner)):
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["ale_bench"].tool_wrappers = sys.modules["ale_bench.tool_wrappers"]
    sys.modules["ale_bench.tool_wrappers"].case_runner = runner
    judge.LOG_STATS.clear()
    judge.trace_internal_errors()
    assert runner.case_iter_func(7) == "verdict 7" and calls == [7, 7, 7]
    assert judge.LOG_STATS["case_reruns"] == 2 and "TesterLogLost" in judge.INTERNAL_TRACEBACKS[-1]["traceback"]

    def always(index):
        raise RuntimeError("judge broken")

    runner.case_iter_func = always
    judge.trace_internal_errors()
    with pytest.raises(RuntimeError):
        runner.case_iter_func(1)  # after CASE_RETRIES reruns the failure stands


def test_default_judge_cpus_are_distinct_physical_cores(monkeypatch):
    monkeypatch.setenv("ALE_BENCH_JUDGE_CPUS", "12,14")
    assert ale_bench_task.default_judge_cpus() == [12, 14]
    monkeypatch.delenv("ALE_BENCH_JUDGE_CPUS")
    cpus = ale_bench_task.default_judge_cpus()
    assert len(cpus) == min(2, len(cpus)) and len(set(cpus)) == len(cpus)
    assert ale_bench_task.parse_cpus("3,1-2") == [1, 2, 3]


# ---------------------------------------------------------------------------
# ale-local: upstream's run command and verdicts


def test_local_run_command_is_the_judges():
    problem = {"run_command": "./a.out"}
    batch = alelocal.run_command(problem, 2.0, None, Path("/t/in.txt"), Path("/t/out.txt"), Path("/t/p.json"))
    assert batch.startswith("timeout 3.2 prlimit --cpu=3.1 /usr/bin/time -f \"{")
    assert batch.endswith("-o /t/p.json ./a.out < /t/in.txt > /t/out.txt")
    reactive = alelocal.run_command(problem, 3.0, Path("/w/tester"), Path("/i"), Path("/o"), Path("/p"))
    assert reactive.startswith("timeout 4.2 prlimit --cpu=4.1 /w/tester /usr/bin/time")


def profile(elapsed=1.0, user=0.9, system=0.05, status=0, rss_kb=1024):
    return json.dumps({"exit_status": str(status), "elapsed_time_seconds": str(elapsed),
                       "user_cpu_seconds": str(user), "system_cpu_seconds": str(system),
                       "max_resident_set_size_kbytes": str(rss_kb)})


def test_local_verdicts_follow_case_runner_parse_profiles():
    gib = 2**30
    assert alelocal.parse_profiles(profile(), 1.1, 2.0, gib) == ("", 1.0, 1024 * 1024)
    assert alelocal.parse_profiles(profile(user=2.05, system=0.0), 2.1, 2.0, gib)[0] == "TIME_LIMIT_EXCEEDED"
    assert alelocal.parse_profiles("Command terminated by signal 9\n" + profile(), 3.0, 2.0, gib)[0] == \
        "TIME_LIMIT_EXCEEDED"
    assert alelocal.parse_profiles("Command exited with non-zero status 1\n" + profile(status=1), 0.2, 2.0, gib)[0] \
        == "RUNTIME_ERROR"
    assert alelocal.parse_profiles(profile(rss_kb=2 * 1024 * 1024), 1.0, 2.0, gib)[0] == "MEMORY_LIMIT_EXCEEDED"
    assert alelocal.parse_profiles("", 2.5, 2.0, gib)[0] == "TIME_LIMIT_EXCEEDED"
    assert alelocal.parse_profiles("", 0.1, 2.0, gib)[0] == "RUNTIME_ERROR"
    assert alelocal.tester_score("log\nScore = 1234\n") == 1234
    assert alelocal.tester_score("wrong answer: bad\nScore = 0") is None and alelocal.tester_score("") is None
    assert alelocal.seeds_from("0-3,7") == [0, 1, 2, 3, 7]


def test_agents_rank_by_the_problems_direction():
    rows = [record(1, 10), record(2, 30), record(3, 20), record(4, 5, valid=False)]
    assert alectl.best(rows, {"score_type": "maximize"})["sid"] == "2"
    assert alectl.best(rows, {"score_type": "minimize"})["sid"] == "1"


# ---------------------------------------------------------------------------
# The dataset reference, problems and task text


def test_reference_lists_the_40_problems_and_the_lite_set():
    ref = ale_bench_task.reference()
    assert ref["revision"] == ale_bench_task.REVISION and len(ref["all"]) == 40
    assert ref["lite"] == ["ahc008", "ahc011", "ahc015", "ahc016", "ahc024", "ahc025", "ahc026", "ahc027",
                           "ahc039", "ahc046"]
    assert set(ref["lite"]) <= set(ref["all"]) and set(ref["problems"]) == set(ref["all"])
    assert all(f"{pid}.zip" in ref["files"] for pid in ref["all"])
    ahc039 = ref["problems"]["ahc039"]
    assert (ahc039["problem_type"], ahc039["time_limit"], ahc039["n_public"], ahc039["n_private"]) == \
        ("batch", 2.0, 50, 150)
    assert ref["problems"]["ahc015"]["problem_type"] == "reactive"
    assert ref["problems"]["ahc025"]["n_private_lite"] == 500 and ref["problems"]["ahc025"]["score_type"] == "minimize"
    assert all("private_seeds" not in facts for facts in ref["problems"].values())


def test_problem_selection():
    assert ale_bench_task.check_problems("ahc039") == (["ahc039"], False)
    assert ale_bench_task.check_problems("lite")[1] and len(ale_bench_task.check_problems("all")[0]) == 40
    assert ale_bench_task.check_problems("ahc039, ahc015,ahc039") == (["ahc039", "ahc015"], False)
    with pytest.raises(ValueError, match="unknown ALE-Bench problem"):
        ale_bench_task.check_problems("ahc013")
    with pytest.raises(ValueError, match="lite cases exist only"):
        ale_bench_task.build_job(Path("/tmp/x"), ["ahc001"], lite=True, dry_run=True)
    with pytest.raises(ValueError, match="batch problems only"):
        ale_bench_task.build_job(Path("/tmp/x"), ["ahc015"], variant="outputs", dry_run=True)


PROFILES = {"cpp20": {"compile_command": "g++-12 -std=gnu++20 -O2 -o a.out Main.cpp", "run_command": "./a.out",
                      "submission_file": "Main.cpp", "object_file": "a.out"}}
NAMES = {"cpp20": "C++20 (gcc 12.2.0)"}
LIBRARIES = {"cpp20": "- AC Library@1.5.1"}


def facts(pid="ahc039", **extra):
    return {**ale_bench_task.reference()["problems"][pid], "problem_id": pid, **extra}


def test_solo_text_is_the_team_text_minus_peers():
    team = ale_bench_task.instruction(facts(), "cpp20", False, "official", 3, 6, PROFILES, NAMES, LIBRARIES)
    solo = ale_bench_task.instruction(facts(), "cpp20", False, "official", 1, 2, PROFILES, NAMES, LIBRARIES)
    assert "3 agents share this container" in team and "The 3 of you share its 6" in team
    assert "keep to 2 CPU-heavy processes" in team and "teammate" in team and "(of any agent)" in team
    for peer_text in ("agents share", "teammate", "of you share", "any agent", "{"):
        assert peer_text not in solo
    assert "You have its 2 CPUs." in solo and "your submission with" in solo
    assert "50 public cases (seeds 0-49" in solo and "150 hidden system-test cases" in solo
    assert "g++-12 -std=gnu++20" in solo and "Main.cpp" in solo and "Higher total scores are better." in solo
    lite = ale_bench_task.instruction(facts(), "cpp20", True, "official", 1, 2, PROFILES, NAMES, LIBRARIES)
    assert "5 public cases (seeds 0-4" in lite
    interactive = ale_bench_task.instruction(facts("ahc015"), "cpp20", False, "official", 1, 2, PROFILES, NAMES,
                                             LIBRARIES)
    assert "under the tester, which plays the judge's side" in interactive
    minimize = ale_bench_task.instruction(facts("ahc027"), "cpp20", True, "outputs", 2, 4, PROFILES, NAMES,
                                          LIBRARIES)
    assert "non-official variant" in minimize and "50 x 10^9 / total" in minimize and "{" not in minimize


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """A problem 'ahc999' with ALE-Bench's zip layout, prebuilt tools, and a reference."""
    data, tools = tmp_path / "data", tmp_path / "tools"
    data.mkdir()
    meta = {"metadata": {"problem_id": "ahc999", "title": "Test", "problem_type": "batch",
                         "score_type": "maximize", "contest_url": "https://atcoder.jp/contests/ahc999",
                         "start_at": "2025-01-01 19:00:00+09:00", "end_at": "2025-01-01 23:00:00+09:00"},
            "constraints": {"time_limit": 2.0, "memory_limit": 1073741824}, "image_files": {},
            "seeds": {"public": [0, 1], "public_lite": [0], "private": [123456789, 987654321],
                      "private_lite": [123456789]}}
    with zipfile.ZipFile(data / "ahc999.zip", "w") as archive:
        for name, content in {"data.json": json.dumps(meta), "statement_en.md": "Statement",
                              "statement_ja.md": "問題文", "images/a.png": "png", "example_input.txt": "1",
                              "example_output.txt": "2", "standings_scores.csv": "rank,score\n1,10\n2,0\n",
                              "standings_scores_lite.csv": "rank,score\n1,10\n2,0\n",
                              "performance.csv": "rank,performance\n1,3000\n2,100\n",
                              "relative_results.csv": "private_0\n5\n", "tools/README.md": "tools",
                              "tools/src/bin/gen.rs": "fn main() {}", "tools/seeds.txt": "0\n",
                              "visualization_server/index.html": "<html>"}.items():
            archive.writestr(f"ahc999/{name}", content)
    (tools / "ahc999" / "in").mkdir(parents=True)
    for tool in ("gen", "tester", "vis"):
        (tools / "ahc999" / tool).write_bytes(b"\x7fELF" + tool.encode())
    (tools / "ahc999" / "in" / "0000.txt").write_text("input 0")
    (tools / "ahc999" / "manifest.json").write_text(json.dumps({"binaries": ["gen", "tester", "vis"],
                                                                 "sources_sha256": "x"}))
    reference = {"revision": ale_bench_task.REVISION, "files": {}, "all": ["ahc999", "ahc998"], "lite": ["ahc999"],
                 "problems": {"ahc999": ale_bench_task.problem_facts("ahc999", data),
                              "ahc998": {**ale_bench_task.problem_facts("ahc999", data),
                                         "problem_type": "reactive"}}}
    monkeypatch.setattr(ale_bench_task, "reference", lambda: reference)
    monkeypatch.setattr(ale_bench_task, "DATA", data)
    monkeypatch.setattr(ale_bench_task, "TOOLS", tools)
    monkeypatch.setattr(ale_bench_task, "LOCK_DIR", tmp_path / "locks")
    return SimpleNamespace(data=data, tools=tools, tmp=tmp_path)


def build(synthetic, agents=2, **options):
    job = ale_bench_task.Job(synthetic.tmp / "job", [], {"problems": ["ahc999"], "lite_version": False,
                                                         "language": "cpp20", "variant": "official"},
                             ["chatgpt.com", "auth.openai.com"])
    task = ale_bench_task.build_task(
        "ahc999", job.path, agents=agents, cpus=2 * agents, memory_mb=6144 * agents, lite=False, language="cpp20",
        variant=options.get("variant", "official"), judge_cpus=[12, 14], private_lane=True,
        scorer="ttc-ale-bench-scorer:test", profiles=PROFILES, names=NAMES, libraries=LIBRARIES,
        data=synthetic.data, solution=options.get("solution"))
    job.tasks.append(task)
    job.write()
    return job, task.path


def test_task_keeps_private_data_and_the_docker_socket_away_from_agents(synthetic):
    job, task = build(synthetic)
    starter = sorted(str(p.relative_to(task / "client" / "starter")) for p in (task / "client" / "starter").rglob("*")
                     if p.is_file())
    assert starter == ["in/0000.txt", "problem.json", "problem/example_input.txt", "problem/example_output.txt",
                       "problem/images/a.png", "problem/statement.md", "problem/statement_ja.md",
                       "tools/README.md", "tools/bin/gen", "tools/bin/tester", "tools/bin/vis", "tools/seeds.txt",
                       "tools/src/bin/gen.rs"]
    client_text = "".join(p.read_bytes().decode(errors="replace") for p in (task / "client").rglob("*")
                          if p.is_file())
    assert "123456789" not in client_text and "987654321" not in client_text
    compose = __import__("yaml").safe_load((task / "environment" / "docker-compose.yaml").read_text())
    main_volumes = " ".join(compose["services"]["main"]["volumes"])
    assert "docker.sock" not in main_volumes and "/data" not in main_volumes and "scorer" not in main_volumes
    assert "scratch" not in main_volumes and "tools/" not in main_volumes.replace("client", "")
    scorer = compose["services"]["scorer"]
    assert "/var/run/docker.sock:/var/run/docker.sock" in scorer["volumes"]
    scratch = str(task / "scratch")
    assert f"{scratch}:{scratch}" in scorer["volumes"] and scorer["environment"]["TMPDIR"] == scratch
    assert (task / "scorer" / "config.json").stat().st_mode & 0o777 == 0o600
    config = json.loads((task / "scorer" / "config.json").read_text())
    assert config["judge_cpus"] == [12, 14] and config["artifact"]["files"] == ["Main.cpp"]
    assert config["host_uid"] == __import__("os").getuid() and config["private_lane"]
    toml = tomllib.loads((task / "task.toml").read_text())
    assert toml["environment"]["network_mode"] == "allowlist" and toml["environment"]["allowed_hosts"] == []
    assert toml["agent"]["network_mode"] == "allowlist" and toml["environment"]["cpus"] == 4
    assert toml["verifier"]["env"]["SCORED_ARTIFACT_CONFIG"] == "/opt/ale-bench/client/task.json"
    assert toml["verifier"]["timeout_sec"] >= 86400
    assert json.loads((task / "client" / "task.json").read_text())["finalize_wait_sec"] == toml["verifier"]["timeout_sec"]
    assert job.harbor_args[:4] == ["--allow-agent-host", "chatgpt.com", "--allow-agent-host", "auth.openai.com"]
    assert (task / "tests" / "finalize.py").read_bytes() == (LAYER / "tests" / "finalize.py").read_bytes()
    assert (task / "scorer" / "server.py").read_bytes() == (LAYER / "scorer" / "server.py").read_bytes()
    assert "FROM yimjk/ale-bench:cpp20-202301@sha256:" in (task / "environment" / "Dockerfile").read_text()
    problem = json.loads((task / "client" / "starter" / "problem.json").read_text())
    assert problem["public_seeds_text"] == "0-1" and problem["n_private_cases"] == 2
    assert (task / "client" / "starter" / "tools" / "bin" / "gen").stat().st_mode & 0o111


def test_oracle_solution_and_outputs_variant(synthetic):
    program = synthetic.tmp / "Main.cpp"
    program.write_text("int main() {}")
    _, task = build(synthetic, agents=1, solution=program)
    assert (task / "solution" / "Main.cpp").read_text() == "int main() {}"
    assert "submit" in (task / "solution" / "solve.sh").read_text()
    variant_job = ale_bench_task.Job(synthetic.tmp / "job2", [], {}, [])
    variant = ale_bench_task.build_task(
        "ahc999", variant_job.path, agents=1, cpus=2, memory_mb=6144, lite=False, language="cpp20",
        variant="outputs", judge_cpus=[3], private_lane=True, scorer="s", profiles=PROFILES, names=NAMES,
        libraries=LIBRARIES, data=synthetic.data)
    config = json.loads(variant.files["scorer/config.json"])
    assert config["artifact"]["files"][0] == "out/0000.txt" and len(config["artifact"]["files"]) == 50
    assert not config["private_lane"] and config["variant_seeds"] == list(range(50))
    with pytest.raises(ValueError, match="interactive"):
        ale_bench_task.build_task("ahc998", variant_job.path, agents=1, cpus=2, memory_mb=6144, lite=False,
                                  language="cpp20", variant="outputs", judge_cpus=[3], private_lane=True,
                                  scorer="s", profiles=PROFILES, names=NAMES, libraries=LIBRARIES,
                                  data=synthetic.data)


# ---------------------------------------------------------------------------
# Reading jobs back: the log for score-over-budget curves


def fake_trial(tmp_path: Path) -> Path:
    trial = tmp_path / "job" / "ahc999__abc"
    state = trial / "ale-bench"
    (trial / "agent").mkdir(parents=True)
    state.mkdir()
    rows = [record(1, 100, ts="2026-10-08T01:00:00.000Z", submitted_ts="2026-10-08T00:59:00.000Z"),
            record(2, 90, ts="2026-10-08T01:10:00.000Z"),
            record(3, 0, valid=False, ts="2026-10-08T01:20:00.000Z"),
            record(4, 130, ts="2026-10-08T01:30:00.000Z")]
    (state / "submissions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    lane = [{"sid": "1", "status": "done", "performance": 1500, "private_score": 300, "rank": 200,
             "ts": "2026-10-08T01:05:00.000Z"},
            {"sid": "4", "status": "done", "performance": 1700, "private_score": 390, "rank": 120,
             "ts": "2026-10-08T01:40:00.000Z"},
            {"sid": "9", "status": "cancelled"}]
    (state / "private.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lane))
    (state / "final.json").write_text(json.dumps({"score": 1690, "metric": 388, "rank": 125, "sid": "4",
                                                  "public_score": 130, "metric_name": "private_score"}))
    return trial


def test_trial_records_carry_the_private_lanes_performance(tmp_path):
    trial = fake_trial(tmp_path)
    assert ale_bench_task.is_trial(trial)
    records = ale_bench_task.trial_records(trial)
    assert [(r["sid"], r["score"], r["metric"]) for r in records] == [
        ("1", 1500, 300), ("2", None, None), ("3", None, None), ("4", 1700, 390)]
    assert records[0]["ts"] == "2026-10-08T01:00:00.000Z"  # the time the public result came back
    assert ale_bench_task.trial_final(trial)["score"] == 1690


def test_job_description_from_harbors_config(tmp_path):
    job = tmp_path / "jobs" / "j"
    (job / "t__1" / "agent").mkdir(parents=True)
    dataset = tmp_path / "tasks" / "j"
    dataset.mkdir(parents=True)
    (dataset / "ale-bench-job.json").write_text(json.dumps({"benchmark": "ale-bench", "problems": ["ahc039"],
                                                            "lite_version": True}))
    (job / "config.json").write_text(json.dumps({"datasets": [{"path": str(dataset)}]}))
    described = ale_bench_task.job_description(job)
    assert described["problems"] == ["ahc039"] and described["job_dir"] == str(dataset)
    assert ale_bench_task.job_description(job / "t__1")["lite_version"]
    (job / "config.json").write_text(json.dumps({"verifier": {"env": {"ALE_BENCH_PROBLEMS": "ahc015",
                                                                      "ALE_BENCH_LITE": "0"}}}))
    assert ale_bench_task.job_description(job)["problems"] == ["ahc015"]
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"tasks": [{"path": str(tmp_path)}]}))
    assert ale_bench_task.job_description(other) is None


# ---------------------------------------------------------------------------
# run.py


def test_run_py_dry_run_and_options():
    import run
    args = run.parser().parse_args([
        "ale-bench", "--problem", "ahc039", "--cli", "codex", "--model", "gpt-6-luna", "--effort", "max",
        "--protocol", "antiherding_matched", "--agents", "2", "--token-budget", "300000", "--ale-cases", "lite",
        "--dry-run", "--name", "ale-test-dry"])
    cmd, env, job = run.prepare(args)
    assert cmd[cmd.index("-p") + 1].endswith(".cache/ale-bench-tasks/ale-test-dry")
    assert "--override-cpus" not in cmd and "--override-memory-mb" not in cmd
    assert "ALE_BENCH_PROBLEMS=ahc039" in cmd and "ALE_BENCH_LITE=1" in cmd and "ALE_BENCH_VARIANT=official" in cmd
    hosts = [cmd[i + 1] for i, part in enumerate(cmd) if part == "--allow-agent-host"]
    assert set(hosts) <= {"chatgpt.com", "auth.openai.com", "api.openai.com"} and hosts
    assert env["RELEASE_CONTINUATION"] == ale_bench_task.Job(Path("/x"), [], {}).continuation.rstrip("\n")
    assert not (ale_bench_task.TASKS / "ale-test-dry").exists() and job.name == "ale-test-dry"
    lite = run.prepare(run.parser().parse_args(["ale-bench", "--problem", "lite", "--cli", "codex", "--model", "m",
                                                "--dry-run"]))[0]
    assert "ALE_BENCH_LITE=1" in lite and any("ahc008,ahc011" in part for part in lite)
    for argv, message in ((["perf-takehome", "--cli", "codex", "--model", "m", "--ale-cases", "lite"], "--ale-"),
                          (["ale-bench", "--cli", "codex", "--model", "m"], "needs --problem"),
                          (["ale-bench", "--cli", "codex", "--model", "m", "--problem", "ahc999"],
                           "unknown ALE-Bench problem"),
                          (["ale-bench", "--cli", "codex", "--model", "m", "--problem", "ahc015", "--ale-variant",
                            "outputs"], "batch problems only")):
        with pytest.raises(ValueError, match=message):
            run.prepare(run.parser().parse_args([*argv, "--dry-run"]))


# ---------------------------------------------------------------------------
# The pinned upstream (when the clone is there)


needs_clone = pytest.mark.skipif(not (ale_bench_task.CLONE / ".git").exists(),
                                 reason="the ALE-Bench clone is not in .cache (python ale_bench_task.py fetch)")


@needs_clone
def test_commands_and_texts_come_from_the_pinned_upstream():
    profiles = ale_bench_task.judge_profiles()
    assert set(profiles) == set(ale_bench_task.BASE_IMAGES) == set(ale_bench_task.SUBMISSION_NAMES)
    assert profiles["cpp20"]["compile_command"].startswith("g++-12 -std=gnu++20 -O2 -DONLINE_JUDGE -DATCODER")
    for language, profile in profiles.items():
        assert Path(profile["submission_file"]).name == ale_bench_task.SUBMISSION_NAMES[language]
    names, libraries = ale_bench_task.language_texts()
    assert names["cpp20"] == "C++20 (gcc 12.2.0)" and "AC Library@1.5.1" in libraries["cpp20"]
    constants: dict = {}
    exec(ale_bench_task.upstream_file("src/ale_bench/constants.py"), constants)
    assert constants["RUST_TOOL_DOCKER_IMAGE"] == ale_bench_task.TOOL_IMAGE
    assert constants["TIME_OUTPUT_FORMAT"] == alelocal.TIME_OUTPUT_FORMAT  # the same GNU time format
    runner = ale_bench_task.upstream_file("src/ale_bench/tool_wrappers/case_runner.py").decode()
    assert 'f"timeout {time_limit_ceil + 0.2} "' in runner and 'f"prlimit --cpu={time_limit_ceil + 0.1} {run_command}"' \
        in runner


@needs_clone
def test_scorer_requirements_are_upstreams_locked_versions():
    lock = tomllib.loads(ale_bench_task.upstream_file("uv.lock").decode())
    locked = {package["name"]: package.get("version") for package in lock["package"]}
    pins = [line.split("==") for line in ale_bench_task.SCORER_REQUIREMENTS.read_text().splitlines()
            if "==" in line and not line.startswith("#")]
    assert pins and all(locked.get(name) == version for name, version in pins)
    pyproject = tomllib.loads(ale_bench_task.upstream_file("pyproject.toml").decode())
    direct = {dep.split(">")[0].split("=")[0].strip() for dep in pyproject["project"]["dependencies"]}
    assert direct <= {name for name, _ in pins}


# Nothing a case does waits on the host's disk (2026-10-29)

def test_the_judge_drops_ale_benchs_trailing_sync_and_keeps_the_exit_status():
    # as case_runner.build_batch_run_command builds it at the pinned commit
    command = ('timeout 3.2 prlimit --cpu=3.1 /usr/bin/time -f "%e %M" -o /judge/profiles.txt ./a.out '
               "< /judge/input.txt > /judge/output.txt; sync")
    bash = ["/bin/bash", "--noprofile", "--norc", "-c", command]
    assert judge.without_sync(bash) == [*bash[:-1], command[: -len("; sync")] + "; true"]
    tester = ["/bin/bash", "--noprofile", "--norc", "-c", "/judge/target/release/tester in out > score"]
    assert judge.without_sync(tester) == tester and judge.without_sync(None) is None


def test_the_judge_uses_the_configured_docker_daemon_only_when_its_socket_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    judge.use_docker_host({"docker_host": f"unix://{tmp_path}/missing.sock"})
    assert "DOCKER_HOST" not in os.environ
    (tmp_path / "ale-docker.sock").touch()
    judge.use_docker_host({"docker_host": f"unix://{tmp_path}/ale-docker.sock"})
    assert os.environ["DOCKER_HOST"] == f"unix://{tmp_path}/ale-docker.sock"
