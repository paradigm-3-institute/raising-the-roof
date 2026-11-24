#!/usr/bin/env python3
"""Verifier of a scored-artifact task: hand the final grade over to the trusted scorer.

1. Each agent's final working copy (/work/<agent>/) is submitted, so the files a run
   leaves behind count; a copy identical to an earlier submission is not graded again.
2. /finalize closes the scorer to new submissions, waits for its queue, and re-grades
   the best valid submission (or the next best, if that one fails its re-grade).
3. The reward is that re-graded score (0 without a valid submission, or the benchmark's
   floor_score).

The scorer writes the authoritative final grade to <trial>/scorer/final.json on the
host; the reward files here are a copy for Harbor.

Postponed final grade: with defer_final.json next to this file, step 2 is
skipped. The final copies are still submitted (their files and submission.json stay in
the scorer's state on the host), but the scorer is not closed and grades nothing more:
the final grade is left to a later grading of the saved submissions on a quiet machine
(a final re-grade on a shared, loaded judge took hours and held up every other run's
feedback). The reward is then a placeholder 0.0 marked "deferred" in reward.json and
scorer_final.json; it is not a grade.
"""

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CONFIG = Path(os.environ.get("SCORED_ARTIFACT_CONFIG", "/opt/scored-artifact/task.json"))
TOKEN = os.environ.get("SCORER_TOKEN", "")
LOGS = Path("/logs/verifier")
DEFER = Path(__file__).with_name("defer_final.json")


def request(urls, method, path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    last = None
    for base in urls:
        req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as reply:
                return reply.status, json.loads(reply.read() or b"{}")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
    raise ConnectionError(f"cannot reach the scorer ({last})")


def write_reward(score: float, extra: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "reward.txt").write_text(str(score))
    numeric = {"reward": score}
    for key in ("metric", "valid", "submissions", "valid_submissions", "deferred"):
        value = extra.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric[key] = value
        elif isinstance(value, bool):
            numeric[key] = int(value)
    (LOGS / "reward.json").write_text(json.dumps(numeric, indent=2))
    (LOGS / "scorer_final.json").write_text(json.dumps(extra, indent=2))


def solo_agent(urls) -> str:
    """The single agent's id as its submissions name it (a1 with a protocol's solo prompt, single
    without one); a1 if it submitted nothing."""
    try:
        status, listing = request(urls, "GET", "/submissions")
    except ConnectionError:
        return "a1"
    agents = sorted({r.get("agent_id") for r in listing.get("submissions", []) if r.get("agent_id")})
    return agents[0] if status == 200 and len(agents) == 1 else "a1"


def final_copies(task: dict, work: Path):
    """(agent id, working copy, submission) of each agent's final working copy. Tree artifacts
    (task.json "tree"): a single agent's is the task's workspace, a team's are /work/<agent>, each
    packed as the client packs a submission (scorectl.py tree_archive)."""
    tree = task.get("tree")
    if tree:
        sys.path.insert(0, str(CONFIG.parent))
        import scorectl  # the agents' client, mounted next to task.json
        copies = ([(solo_agent(task["scorer_urls"]), Path(tree["workspace"]))] if not tree.get("team") else
                  [(p.name, p) for p in sorted(work.iterdir()) if p.is_dir() and not p.name.endswith(".partial")]
                  if work.is_dir() else [])
        for agent, directory in copies:
            try:
                yield agent, directory, {"archive": scorectl.tree_archive(task, directory)}
            except (OSError, FileNotFoundError) as exc:
                print(f"final working copy of {agent} not submitted: {exc}")
        return
    for directory in sorted(p for p in work.iterdir() if p.is_dir() and not p.name.endswith(".partial")) \
            if work.is_dir() else []:
        try:
            files = {name: base64.b64encode((directory / name).read_bytes()).decode()
                     for name in task["artifact_files"]}
        except OSError:
            continue
        yield directory.name, directory, {"files": files}


def main() -> int:
    task = json.loads(CONFIG.read_text())
    urls = task["scorer_urls"]
    for _ in range(90):
        try:
            if request(urls, "GET", "/health", timeout=5)[0] == 200:
                break
        except ConnectionError:
            pass
        time.sleep(2)
    else:
        print("ERROR: the scorer is not answering")
        write_reward(0.0, {"detail": "scorer unreachable"})
        return 0

    work = Path(task["work_root"])
    finals = []
    # task.json "final_copies" false: only the agents' own submissions count, as in a benchmark where only its
    # submit command promotes.
    for agent, directory, payload in (final_copies(task, work) if task.get("final_copies", True) else ()):
        status, reply = request(urls, "POST", "/submit", {"agent_id": agent[:64], **payload,
                                                          "final": True, "token": TOKEN,
                                                          "note": "final working copy"})
        print(f"final working copy of {agent}: {status} {reply}")
        if status == 200 and not reply.get("duplicate_of"):
            finals.append(reply["sid"])

    if DEFER.is_file():
        why = json.loads(DEFER.read_text() or "{}").get("why", "")
        print(f"Final grade postponed ({DEFER.name}): the scorer is not closed and grades nothing more. {why}")
        write_reward(0.0, {"deferred": True, "final_copies": finals,
                           "detail": "final grade postponed to a later grading of the saved submissions; "
                                     "the reward 0.0 is a placeholder, not a grade", "why": why})
        return 0
    print("Finalizing (waits for queued gradings, then re-grades the best submission)...", flush=True)
    status, final = request(urls, "POST", "/finalize", {"token": TOKEN},
                            timeout=float(task.get("finalize_wait_sec", 3600)))
    if status != 200:
        print(f"ERROR: finalize failed: {status} {final}")
        write_reward(0.0, {"detail": f"finalize failed: {final}"})
        return 0
    print(json.dumps(final, indent=2))
    write_reward(float(final.get("score") or 0.0), final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
