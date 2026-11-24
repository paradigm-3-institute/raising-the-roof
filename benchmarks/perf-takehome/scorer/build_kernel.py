"""Untrusted stage of the take-home scorer: build the agent's kernel, save it as data.

Runs inside the scorer's sandbox (an unprivileged user, a fresh directory holding
only the submitted perf_takehome.py and the pristine problem.py) as

    python3 -I build_kernel.py DIRECTORY FOREST_HEIGHT N_NODES BATCH_SIZE ROUNDS

It calls KernelBuilder().build_kernel(...) with the shape tests/submission_tests.py
uses, and debug_info() as that test does, then writes kb.instrs to program.json.
Only plain data leaves this process: the trusted stage simulates the instructions
with its own frozen simulator, so nothing this code does (patching the simulator,
reading random state, editing files) can reach the score.
"""

import json
import sys
from pathlib import Path


def plain(value, path="instrs"):
    """The instruction list as JSON-ready data; anything else is an error."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [plain(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} has a non-string key {key!r}")
            result[str(key)] = plain(item, f"{path}[{key!r}]")
        return result
    raise TypeError(f"{path} is a {type(value).__name__}, not plain data (int, str, list, tuple, dict)")


def main() -> int:
    directory = Path(sys.argv[1]).resolve()
    shape = [int(value) for value in sys.argv[2:6]]
    sys.path.insert(0, str(directory))
    from perf_takehome import KernelBuilder  # the agent's code, with the pristine problem.py

    kb = KernelBuilder()
    kb.build_kernel(*shape)
    kb.debug_info()  # tests/submission_tests.py calls it too
    program = plain(kb.instrs)
    if not isinstance(program, list):
        raise TypeError("KernelBuilder.instrs must be a list of instruction bundles")
    with open(directory / "program.json", "w") as stream:
        json.dump(program, stream, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
