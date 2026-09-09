#!/usr/bin/env python3
# coding: utf-8
"""The result store keeps every finished answer and settles duplicates.

The JSONL is an append-only log, so its guarantees are behavioural, not
structural: a torn final line must not cost the run, a mid-file bad line must
not be silently skipped, and when one question id appears several times the
row that counts must be chosen by one rule shared with resume. Each of those
is a decision this test pins down.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochrono.results.store import ResultStore  # noqa: E402

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:52} {detail}")
    if not passed:
        failures.append(name)


def fresh_store() -> ResultStore:
    return ResultStore(Path(tempfile.mkdtemp()) / "model" / "scenario" / "dim.jsonl")


print("1. completed means answered without error")
s = fresh_store()
s.append([{"id": "a", "model_output": "x"},
          {"id": "b", "error": "boom"},
          {"id": "c", "model_output": "", }])
check("success counts", "a" in s.completed_ids())
check("error does not", "b" not in s.completed_ids())
check("empty output does not", "c" not in s.completed_ids())

print("2. final_rows picks one row per id")
s = fresh_store()
s.append([{"id": "a", "error": "first try failed"},
          {"id": "a", "model_output": "retry worked"}])
s.append([{"id": "b", "model_output": "good"},
          {"id": "b", "error": "stray error appended after success"}])
s.append([{"id": "c", "error": "failed once"},
          {"id": "c", "error": "failed twice"}])
final = {r["id"]: r for r in s.final_rows()}
check("retry supersedes failure", final["a"].get("model_output") == "retry worked")
check("success survives a later error row",
      final["b"].get("model_output") == "good", str(final["b"]))
check("all-failed keeps the last failure", final["c"].get("error") == "failed twice")
check("one row per id", len(s.final_rows()) == 3)

print("3. a torn final line is tolerated; a torn middle line is not")
s = fresh_store()
s.append([{"id": "a", "model_output": "x"}])
with s.path.open("a", encoding="utf-8") as h:
    h.write('{"id": "b", "model_out')          # killed mid-write
check("rows survive the torn tail", [r["id"] for r in s.rows()] == ["a"])
check("resume sees the torn question as pending", s.completed_ids() == {"a"})

s = fresh_store()
s.path.parent.mkdir(parents=True)
with s.path.open("a", encoding="utf-8") as h:
    h.write('{"id": "a", "model_out\n')        # bad line with rows after it
    h.write('{"id": "b", "model_output": "x"}\n')
try:
    list(s.rows())
    check("mid-file corruption raises", False, "no exception")
except Exception:
    check("mid-file corruption raises", True)

print("4. displace backs up instead of deleting, one generation deep")
s = fresh_store()
s.append([{"id": "a", "model_output": "x"}, {"id": "b", "model_output": "y"}])
moved = s.displace()
backup = s.path.with_name(s.path.name + ".bak")
check("row count returned", moved == 2, str(moved))
check("data moved to .jsonl.bak", backup.exists() and not s.path.exists())
s.append([{"id": "a", "model_output": "z"}])
s.displace()
check("second displace keeps one backup", backup.read_text().count("\n") == 1)
check("empty store displaces nothing", fresh_store().displace() == 0)

print("5. the summary lands beside the log")
s = fresh_store()
s.write_summary({"total": 1})
check("summary path", s.summary_path.name == "dim.summary.json",
      s.summary_path.name)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
