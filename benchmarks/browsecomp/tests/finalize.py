#!/usr/bin/env python3
"""The verifier: close the answer log.

It asks the answer log to record the team's answer (the last submission) and refuse further
submissions. It holds no answer key and grades nothing: the response is graded on the host after
the run, out of every container's reach (the trial's final.json). Harbor's reward is therefore 1
when the team submitted an answer and 0 otherwise; it is not the grade.
"""
# It runs in the agents' container (after them): it must not name the benchmark.

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

LOGS = Path("/logs/verifier")
URLS = ("http://127.0.0.1:{port}", "http://answers:{port}")


def finalize(port: int, token: str) -> dict:
    last = None
    for attempt in range(30):
        for base in URLS:
            request = urllib.request.Request(base.format(port=port) + "/finalize", data=b"{}", method="POST",
                                             headers={"X-Finalize-Token": token,
                                                      "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    return json.loads(response.read())
            except (urllib.error.URLError, OSError, ValueError) as error:
                last = error
        time.sleep(2)
    raise ConnectionError(f"the answer log is unreachable: {last}")


def write(reward: dict, detail: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "reward.txt").write_text(str(reward["reward"]))
    (LOGS / "reward.json").write_text(json.dumps(reward, indent=2) + "\n")
    (LOGS / "answer.json").write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n")


def main() -> int:
    port = int(os.environ.get("ANSWER_LOG_PORT") or 0)
    token = os.environ.get("ANSWER_LOG_FINALIZE_TOKEN") or ""
    try:
        closed = finalize(port, token)
    except ConnectionError as error:
        print(f"finalize: {error}", file=sys.stderr)
        write({"reward": 0.0, "answered": 0.0, "submissions": 0.0}, {"error": str(error)})
        return 0
    answered = 1.0 if closed.get("answer") else 0.0
    write({"reward": answered, "answered": answered, "submissions": float(closed.get("submissions") or 0)}, closed)
    print(json.dumps(closed, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
