#!/usr/bin/env python3
# coding: utf-8
"""Reporting: one table, two kinds of flag, and a hard line on provenance.

Built entirely from synthetic run directories. The floors come from the real
``configs/protocol.json`` so this also pins the shape agreement between the
protocol file and ``floor_breach``.
"""
from __future__ import annotations

import json
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochrono.results import report as report_mod  # noqa: E402

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:52} {detail}")
    if not passed:
        failures.append(name)


FLOORS = json.loads((ROOT / "configs/protocol.json").read_text())["degenerate_floor"]
DATASET = {"digest": "d1d1d1d1d1d1",
           "scenarios": {"make_tea_tianji": "a" * 64}}


def make_run(root: Path, run_id: str, dataset: dict, summaries: dict) -> Path:
    """summaries: {(model, scenario, dimension): summary_dict}"""
    run = root / run_id
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({
        "run_id": run_id, "fingerprint": run_id.split("_")[-1],
        "dataset": dataset, "protocol": {"version": "1.0", "fingerprint": "p1"},
        "generation_by_model": {"modelA": {"thinking": "on"},
                                "modelB": {"thinking": "off"}},
    }))
    for (model, scenario, dimension), summary in summaries.items():
        d = run / model / scenario
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{dimension}.summary.json").write_text(json.dumps(summary))
        (d / f"{dimension}.jsonl").write_text('{"id": "q1"}\n')
        (d / f"{dimension}.jsonl.bak").write_text('{"id": "old"}\n')
    return run


root = Path(tempfile.mkdtemp())
run = make_run(root, "2026-08-27_aaaaaaaaaaaa", DATASET, {
    ("modelA", "make_tea_tianji", "current_action"):
        {"total": 120, "answered": 120, "errors": 0,
         "accuracy": 0.62, "parse_failure_rate": 0.01},
    ("modelA", "make_tea_tianji", "action_time"):
        {"total": 117, "answered": 117, "errors": 0,
         "tIoU@0.5": 0.31, "mean_tIoU": 0.35, "parse_failure_rate": 0.0},
    ("modelB", "make_tea_tianji", "current_action"):
        {"total": 120, "answered": 120, "errors": 0,
         "accuracy": 0.15, "parse_failure_rate": 0.02},     # below the 0.25 floor
    ("modelB", "make_tea_tianji", "action_time"):
        {"total": 117, "answered": 0, "errors": 117,
         "tIoU@0.5": None, "parse_failure_rate": 0.0},      # never executed
})

print("1. collection")
rep = report_mod.collect([run], FLOORS)
check("one row per summary", len(rep.rows) == 4, str(len(rep.rows)))
by = {(r["model"], r["dimension"]): r for r in rep.rows}
check("model and scenario read from the path",
      by[("modelA", "current_action")]["scenario"] == "make_tea_tianji")
check("headline metric per dimension",
      by[("modelA", "action_time")]["metric"] == "tIoU@0.5")
check("healthy cell has no flags",
      not by[("modelA", "current_action")]["floor"]
      and not by[("modelA", "current_action")]["fault"])
check("below-floor cell flagged",
      bool(by[("modelB", "current_action")]["floor"]))
check("never-executed cell is a fault, not a floor",
      bool(by[("modelB", "action_time")]["fault"])
      and not by[("modelB", "action_time")]["floor"])

print("2. markdown")
md = report_mod.to_markdown(rep)
# collect() recomputes the digest from the union of scenario pins rather
# than trusting any single record, so the header carries the recomputed one.
check("the pins digest in the header", rep.dataset["digest"] in md)
check("run id in the header", run.name in md)
check("fault section present", "did not execute properly" in md)
check("floor section present", "degenerate floor" in md)
check("fault marked in the table", "✗" in md and "⚠" in md)
check("execution settings listed", "Execution settings" in md and "thinking" in md)

print("3. csv")
csv_path = root / "report.csv"
report_mod.to_csv(rep, csv_path)
lines = csv_path.read_text().strip().splitlines()
check("header plus one line per row", len(lines) == 5, str(len(lines)))

print("4. different datasets refuse to merge")
other = make_run(root, "2026-08-27_bbbbbbbbbbbb",
                 {"digest": "f2f2f2f2f2f2",
                  "scenarios": {"make_tea_tianji": "f" * 64}},
                 {("modelA", "make_tea_tianji", "current_action"):
                      {"total": 10, "answered": 10, "errors": 0, "accuracy": 0.5,
                       "parse_failure_rate": 0.0}})
try:
    report_mod.collect([run, other], FLOORS)
    check("mixed datasets raise", False, "no exception")
except ValueError as e:
    check("mixed datasets raise", "refusing" in str(e))
    check("the error names both runs",
          run.name in str(e) and other.name in str(e))

same = make_run(root, "2026-08-28_cccccccccccc", DATASET,
                {("modelC", "make_tea_tianji", "current_action"):
                     {"total": 10, "answered": 10, "errors": 0, "accuracy": 0.5,
                      "parse_failure_rate": 0.0}})
merged = report_mod.collect([run, same], FLOORS)
check("same dataset merges", len(merged.rows) == 5)

print("5. packing")
(run / "report.md").write_text(report_mod.to_markdown(rep))
report_mod.to_csv(rep, run / "report.csv")
default = report_mod.pack(run, root / "default.tar.gz")
with tarfile.open(root / "default.tar.gz") as tar:
    names = tar.getnames()
check("default carries run.json, reports, summaries",
      any(n.endswith("run.json") for n in names)
      and any(n.endswith(".summary.json") for n in names)
      and any(n.endswith("report.md") for n in names))
check("default carries no jsonl", not any(n.endswith(".jsonl") for n in names))

report_mod.pack(run, root / "full.tar.gz", full=True)
with tarfile.open(root / "full.tar.gz") as tar:
    names = tar.getnames()
check("full adds the jsonl", any(n.endswith(".jsonl") for n in names))
check("backups are never packed", not any(n.endswith(".bak") for n in names))
check("paths are rooted at the run id",
      all(n.startswith(run.name) for n in names))

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
