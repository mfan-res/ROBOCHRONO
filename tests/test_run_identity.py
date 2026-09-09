#!/usr/bin/env python3
# coding: utf-8
"""A run's identity: same experiment resumes, changed experiment separates.

The fingerprint must move when — and only when — something that changes
results moves. Each component is toggled one at a time; an unrelated file is
edited to show it does not participate. Directory resolution is then checked
against the three situations that motivated the scheme: rerun after an
interruption, rerun with changed settings, and an explicit fresh start.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochrono.results import runid  # noqa: E402

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:52} {detail}")
    if not passed:
        failures.append(name)


# A synthetic configuration, so the test does not move when the real one does.
base = Path(tempfile.mkdtemp())
(base / "protocol.json").write_text('{"temperature": 0}')
(base / "suite.json").write_text('{"scenarios": ["a"]}')
(base / "model_x.json").write_text('{"generation": 1}')
(base / "model_y.json").write_text('{"generation": 2}')
(base / "runtime.json").write_text('{"api_concurrency": 4}')
CODE = {"git": "abc123", "dirty": False}


def fingerprint(**overrides):
    kwargs = dict(protocol_path=base / "protocol.json",
                  suite_path=base / "suite.json",
                  model_paths=[base / "model_x.json", base / "model_y.json"],
                  dataset_digest="73b4d26419bb",
                  code=CODE)
    kwargs.update(overrides)
    return runid.config_fingerprint(**kwargs)


print("1. the fingerprint moves with what changes results, and only that")
first = fingerprint()
check("stable across calls", fingerprint() == first)
check("12 hex chars", len(first) == 12 and all(c in "0123456789abcdef" for c in first))

(base / "protocol.json").write_text('{"temperature": 1}')
moved = fingerprint()
check("protocol edit moves it", moved != first)
(base / "protocol.json").write_text('{"temperature": 0}')
check("reverting restores it", fingerprint() == first)

(base / "model_x.json").write_text('{"generation": 99}')
check("model config edit moves it", fingerprint() != first)
(base / "model_x.json").write_text('{"generation": 1}')

check("model set moves it",
      fingerprint(model_paths=[base / "model_x.json"]) != first)
check("model order does not",
      fingerprint(model_paths=[base / "model_y.json", base / "model_x.json"]) == first)
check("dataset moves it", fingerprint(dataset_digest="other") != first)
check("commit moves it", fingerprint(code={"git": "def456", "dirty": False}) != first)
check("a dirty tree moves it",
      fingerprint(code={"git": "abc123", "dirty": True, "diff_sha": "aa"}) != first)

(base / "runtime.json").write_text('{"api_concurrency": 64}')
check("a runtime edit does not", fingerprint() == first)

print("2. directory resolution")
results = base / "results"
fp = first
created = runid.resolve_run_dir(results, fp, today="2026-08-27")
check("created with the date", created.run_id == f"2026-08-27_{fp}")
check("not a resume", not created.resumed)

resumed = runid.resolve_run_dir(results, fp, today="2026-09-01")
check("later rerun finds the same directory", resumed.path == created.path)
check("and reports a resume", resumed.resumed)
check("the date does not move", "2026-08-27" in resumed.run_id)

other = runid.resolve_run_dir(results, "eeeeeeeeeeee", today="2026-09-01")
check("a different fingerprint separates", other.path != created.path)

fresh = runid.resolve_run_dir(results, fp, fresh=True, today="2026-09-02")
check("fresh gets an ordinal", fresh.run_id == f"2026-09-02_{fp}-2", fresh.run_id)
after = runid.resolve_run_dir(results, fp, today="2026-09-03")
check("resume prefers the newest sibling", after.path == fresh.path)

print("3. run.json is written once and defends the directory")
record = runid.write_run_record(created, {"suite": "v1",
                                          "dataset": {"version": "2.0"}})
check("snapshot written", (created.path / "run.json").exists())
check("started stamped", bool(record["started"]))
check("not yet finished", record["finished"] is None)

again = runid.write_run_record(created, {"suite": "SOMETHING ELSE"})
check("resume keeps the original snapshot", again["suite"] == "v1")

runid.mark_finished(created)
check("finished stamped", runid.read_run_record(created.path)["finished"] is not None)

renamed = runid.RunDir(path=created.path, run_id=created.run_id,
                       fingerprint="eeeeeeeeeeee", resumed=True)
try:
    runid.write_run_record(renamed, {})
    check("a moved directory is refused", False, "no exception")
except ValueError as e:
    check("a moved directory is refused", "refusing" in str(e))

try:
    runid.read_run_record(base)
    check("a directory without run.json is not a run", False, "no exception")
except FileNotFoundError:
    check("a directory without run.json is not a run", True)

print("4. the code version reports what the fingerprint consumes")
code = runid.code_version(ROOT)
check("commit known", bool(code.get("git")), str(code.get("git")))
check("dirty carries a diff digest",
      not code.get("dirty") or bool(code.get("diff_sha")))

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
