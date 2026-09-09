#!/usr/bin/env python3
# coding: utf-8
"""Audit a run directory for completeness before its results are merged.

Runs arrive from many machines and many operators. Every row a run contains
is trustworthy — the harness wrote it — but a run can silently contain *less*
than the suite promises: a forgotten --limit flag, an interrupted final
model, a combination that never executed. This audit answers one question:
**does this directory hold everything the suite says it should**, so that
merging it cannot quietly understate a model.

    python3 tools/audit_run.py <run_dir> --data-root scenarios [--suite official-v1]

Exit code 0 means safe to merge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robochrono.config.suites import load_suite, pins_digest  # noqa: E402
from robochrono.dataset.loader import load_questions  # noqa: E402
from robochrono.dataset.render import load_question_bank  # noqa: E402
from robochrono.results.runid import read_run_record  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--suite", default=None,
                    help="default: the suite named in run.json")
    args = ap.parse_args()

    problems: list[str] = []
    record = read_run_record(args.run_dir)
    suite = load_suite(args.suite or record.get("suite"), "configs/suites")

    recorded = (record.get("dataset") or {}).get("digest")
    if recorded != pins_digest(suite.pins):
        problems.append(
            f"dataset mismatch: run recorded pins digest {recorded}, the "
            f"suite gives {pins_digest(suite.pins)} — audit against the "
            f"suite this run actually used")
    models = record.get("models") or []
    invocation = record.get("invocation")
    label = record.get("label")
    print(f"run       {record.get('run_id')}  (code {record.get('code', {}).get('git')})"
          + (f"  [{label}]" if label else ""))
    print(f"models    {', '.join(models)}")
    print(f"invocation {' '.join(invocation) if invocation else '(not recorded — pre-audit-era run)'}")

    bank = load_question_bank(args.data_root)
    expected: dict[tuple[str, str], int] = {}
    for scenario in suite.scenarios:
        for dimension in suite.dimensions:
            expected[(scenario, dimension)] = len(
                load_questions(args.data_root, scenario, dimension, bank=bank))

    total_err = total_q = 0
    for model in models:
        missing, short, errors = [], [], 0
        for (scenario, dimension), want in expected.items():
            path = (args.run_dir / model / scenario /
                    f"{dimension}.summary.json")
            if not path.exists():
                if want:                       # an empty combo needs no summary
                    missing.append(f"{scenario}/{dimension}")
                continue
            s = json.loads(path.read_text())
            total_q += s["total"]; errors += s["errors"]
            if s["total"] != want:
                short.append(f"{scenario}/{dimension}: {s['total']} of {want}")
        total_err += errors
        state = "OK" if not (missing or short) else "INCOMPLETE"
        print(f"  {model:32} {state}  errors={errors}"
              + (f"  missing={missing[:3]}" if missing else "")
              + (f"  short={short[:3]}" if short else ""))
        for item in missing:
            problems.append(f"{model}: no summary for {item}")
        for item in short:
            problems.append(f"{model}: question count mismatch — {item}")

    print(f"TOTAL {total_q} questions, {total_err} error rows")
    if problems:
        print(f"\nNOT SAFE TO MERGE — {len(problems)} problem(s):")
        for p in problems[:20]:
            print(f"  {p}")
        return 1
    print("safe to merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
