"""Hand-made variants of the take-home's starter kernel, for checking the scorer.

The take-home has no license, so this repository keeps no copy of it: each variant
is a patch applied to perf_takehome.py as fetched at the pinned commit.

- starter: unchanged (147,734 cycles).
- packed: build() packs consecutive slots into VLIW bundles greedily, in order,
  respecting slot limits and dependencies (fewer cycles, still correct).
- branchy: the starter plus a block of extra bundles that runs only when the first
  input value is odd, so cycles depend on the input (the worst-case rule).
- wrong: the branch choice inverted (incorrect output values).
- patcher: the starter plus code that patches the frozen simulator's Machine.run to
  report 1 cycle when perf_takehome.py is imported after it, as tests/submission_tests.py
  does. The upstream test reports 1 cycle; the scorer must not.
- slots: packed, plus bundles of 3 loads (over the load engine's limit of 2). It
  passes only with an edited tests/frozen_problem.py.
"""

from __future__ import annotations

import ast
import re

# The body of KernelBuilder.build, packed; it replaces the starter's body under the same signature.
PACKED_BODY = '''\
        # Greedy in-order packing: a slot joins the current bundle unless that
        # bundle writes something it reads or writes, its engine is full, or it
        # touches memory a store in the bundle touches.
        def accesses(engine, slot):
            op = slot[0]
            if engine == "alu":
                return {slot[2], slot[3]}, {slot[1]}, False, False
            if engine == "load":
                return ({slot[2]} if op == "load" else set()), {slot[1]}, op == "load", False
            if engine == "store":
                return {slot[1], slot[2]}, set(), False, True
            if engine == "flow":
                return {slot[2], slot[3], slot[4]}, {slot[1]}, False, False
            raise NotImplementedError(engine)

        instrs, bundle, written, mem_read, mem_write = [], {}, set(), False, False
        for engine, slot in slots:
            if engine == "debug":
                continue
            reads, writes, loads, stores = accesses(engine, slot)
            conflict = (reads & written or writes & written
                        or len(bundle.get(engine, [])) >= SLOT_LIMITS[engine]
                        or (loads and mem_write) or (stores and (mem_read or mem_write)))
            if conflict:
                instrs.append(bundle)
                bundle, written, mem_read, mem_write = {}, set(), False, False
            bundle.setdefault(engine, []).append(slot)
            written |= writes
            mem_read |= loads
            mem_write |= stores
        if bundle:
            instrs.append(bundle)
        return instrs
'''

# Inserted before build_kernel's first pause.
BRANCH_BLOCK = '''        # Extra work, only when the first input value is odd.
        first = self.alloc_scratch("first")
        self.add("load", ("load", first, self.scratch["inp_values_p"]))
        self.add("alu", ("%", first, first, two_const))
        self.add("alu", ("==", first, first, zero_const))
        self.add("flow", ("cond_jump_rel", first, 300))
        for _ in range(300):
            self.add("alu", ("+", tmp3, tmp3, zero_const))
'''

PATCHER = '''

# Patch the frozen simulator when the test has already imported it.
import sys as _sys
_frozen = _sys.modules.get("frozen_problem")
if _frozen is not None:
    _original_run = _frozen.Machine.run

    def _patched_run(self):
        _original_run(self)
        self.cycle = 1

    _frozen.Machine.run = _patched_run
'''

# Appended to build_kernel.
SLOTS_TAIL = '''
        # Three loads in one bundle: over the load engine's limit of 2, unless
        # tests/frozen_problem.py is edited.
        spare = self.alloc_scratch("spare", 3)
        self.instrs.insert(0, {"load": [("const", spare + k, k) for k in range(3)]})
'''

# The branch choice in the starter's loop: the first select, whose last two operands the "wrong" variant swaps.
SELECT = re.compile(r'\("select", (\w+), (\w+), (\w+), (\w+)\)')


def _method(text: str, name: str) -> ast.FunctionDef:
    """KernelBuilder.NAME in TEXT."""
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == "KernelBuilder":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise ValueError(f"perf_takehome.py changed; no KernelBuilder.{name}")


def _splice(text: str, start: int, end: int, new: str) -> str:
    """TEXT with its lines start..end (0-based, end excluded) replaced by NEW."""
    lines = text.splitlines(keepends=True)
    return "".join(lines[:start]) + new + "".join(lines[end:])


def _packed(starter: str) -> str:
    build = _method(starter, "build")
    return _splice(starter, build.lineno, build.end_lineno, PACKED_BODY)


def _branchy(starter: str) -> str:
    kernel = _method(starter, "build_kernel")
    pause = next((statement for statement in kernel.body
                  if any(isinstance(node, ast.Constant) and node.value == "pause" for node in ast.walk(statement))),
                 None)
    if pause is None:
        raise ValueError("perf_takehome.py changed; build_kernel has no pause")
    return _splice(starter, pause.lineno - 1, pause.lineno - 1, BRANCH_BLOCK)


def _wrong(starter: str) -> str:
    first = SELECT.search(starter)
    if first is None:
        raise ValueError("perf_takehome.py changed; no select")
    swapped = f'("select", {first[1]}, {first[2]}, {first[4]}, {first[3]})'
    return starter[:first.start()] + swapped + starter[first.end():]


def _slots(packed: str) -> str:
    kernel = _method(packed, "build_kernel")
    return _splice(packed, kernel.end_lineno, kernel.end_lineno, SLOTS_TAIL)


def _replace(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"perf_takehome.py changed; cannot find {old[:60]!r}")
    return text.replace(old, new)


def variants(starter: str) -> dict[str, str]:
    packed = _packed(starter)
    return {
        "starter": starter,
        "packed": packed,
        "branchy": _branchy(starter),
        "wrong": _wrong(starter),
        "patcher": starter + PATCHER,
        "slots": _slots(packed),
    }


def edited_frozen_problem(frozen: str) -> str:
    """tests/frozen_problem.py with the load limit raised: the test-editing hack."""
    return _replace(frozen, '"load": 2,', '"load": 3,')
