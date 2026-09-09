#!/usr/bin/env python3
# coding: utf-8
"""The five commands, exercised as real subprocesses.

A replay model in a temporary --models-dir runs `eval` end to end through the
actual entry point — the same path a real model takes, minus the GPU. The
child-command contract that dispatch generates is parsed back through the
CLI's own parser, so the two cannot drift apart silently.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochrono import dimensions  # noqa: E402
from robochrono.config.protocol import load_protocol  # noqa: E402
from robochrono.dataset.loader import load_questions  # noqa: E402
from robochrono.dataset.render import load_question_bank  # noqa: E402

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:52} {detail}")
    if not passed:
        failures.append(name)


ENV = {**os.environ,
       "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}


def cli(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "robochrono", *argv],
                          cwd=ROOT, env=ENV, capture_output=True, text=True)


DATA = ROOT / "scenarios"
if not (DATA / "make_tea_tianji" / "manifest.json").exists():
    print("skip: dataset not present")
    sys.exit(0)

print("1. the surface behaves")
check("top-level help", cli("--help").returncode == 0)
check("unknown command fails", cli("frobnicate").returncode != 0)
check("unknown flag fails", cli("report", "--frobnicate").returncode != 0)
for command in ("preflight", "validate-data", "eval", "report", "pack"):
    check(f"{command} --help", cli(command, "--help").returncode == 0)

print("2. validate-data")
result = cli("validate-data", "--data-root", "scenarios")
check("real pool passes", result.returncode == 0, result.stdout[-200:].strip())
check("headline numbers reported", "34713" in result.stdout)
broken = Path(tempfile.mkdtemp())
result = cli("validate-data", "--data-root", str(broken))
check("an empty root fails", result.returncode == 1)

print("3. a replay model runs eval end to end")
work = Path(tempfile.mkdtemp())
models_dir = work / "models" / "local"
models_dir.mkdir(parents=True)
scenario, dims = "make_tea_tianji", ["current_action", "next_action"]

protocol = load_protocol(ROOT / "configs/protocol.json")
bank = load_question_bank(DATA, [scenario])
table: dict[str, str] = {}
for dimension in dims:
    dim = dimensions.build(dimension, strip_reasoning=protocol.strip_reasoning)
    for unit in dim.units(load_questions(DATA, scenario, dimension, bank=bank)):
        table[unit.key] = json.dumps({"choice": unit.items[0]["answer"]})
table_path = work / "table.json"
table_path.write_text(json.dumps(table))
(models_dir / "replay-2b.json").write_text(json.dumps({
    "name": "Replay-2B", "adapter": "replay", "weights": "replay",
    "environment": "transformers-4x", "api": {"table": str(table_path)},
}))

results_root = work / "results"
eval_args = ("eval", "--data-root", "scenarios", "--models-dir", str(work / "models"),
             "--results-root", str(results_root),
             "--scenarios", scenario, "--dimensions", *dims, "--no-dispatch")
result = cli(*eval_args)
check("eval exits cleanly", result.returncode == 0, result.stdout[-300:].strip())
runs = [p for p in results_root.iterdir() if (p / "run.json").exists()]
check("one run directory", len(runs) == 1)
run = runs[0]
record = json.loads((run / "run.json").read_text())
check("identity recorded", len(record["dataset"]["digest"]) == 12
      and record["dataset"]["scenarios"][scenario]
      and record["models"] == ["replay-2b"])
check("finished stamped", record["finished"] is not None)
summaries = sorted(run.glob("replay-2b/*/*.summary.json"))
check("both summaries written", len(summaries) == 2)
check("perfect scores against the dataset's own answers",
      all(json.loads(p.read_text())["accuracy"] == 1.0 for p in summaries))
check("report written into the run", (run / "report.md").exists()
      and record["dataset"]["digest"] in (run / "report.md").read_text())

print("4. the same command resumes instead of redoing")
before = {p: p.read_text() for p in run.glob("replay-2b/*/*.jsonl")}
result = cli(*eval_args)
check("second eval exits cleanly", result.returncode == 0)
check("resuming is announced", "resuming" in result.stdout)
check("no rows rewritten", all(p.read_text() == before[p] for p in before))

print("5. report and pack consume the run")
result = cli("report", "--results-root", str(results_root))
check("report finds the latest run", result.returncode == 0
      and "report.md" in result.stdout)
result = cli("pack", "--results-root", str(results_root))
check("pack produces a bundle", result.returncode == 0)
bundle = results_root / f"{run.name}.tar.gz"
check("bundle is small and beside the runs", bundle.exists()
      and bundle.stat().st_size < 200_000)

print("6. dry run prices the work and touches nothing")
dry_root = work / "dry-results"
result = cli("eval", "--data-root", "scenarios", "--models-dir", str(work / "models"),
             "--results-root", str(dry_root),
             "--scenarios", scenario, "--dimensions", *dims, "--dry-run")
check("dry run exits cleanly", result.returncode == 0, result.stderr[-200:])
check("counts reported", "2 (model × scenario × dimension)" in result.stdout
      and "media touched" in result.stdout)
check("nothing written", not dry_root.exists())

print("7. preflight verdicts through the CLI")
result = cli("preflight", "--data-root", "scenarios", "--models-dir", str(work / "models"),
             "--scenarios", scenario, "--dimensions", *dims)
check("missing weights fail preflight", result.returncode == 1
      and "replay-2b weights" in result.stdout and "FAIL" in result.stdout)
check("data checks passed inside the same report",
      "sampled media" in result.stdout)

print("8. the dispatch child contract parses")
from robochrono.cli import build_parser  # noqa: E402
from robochrono.config.models import load_models  # noqa: E402
from robochrono.config.environments import load_environments  # noqa: E402
from robochrono.orchestrate.dispatch import build_commands  # noqa: E402

models = sorted(load_models(ROOT / "configs/models").values(), key=lambda m: m.slug)
envs = load_environments(ROOT / "configs/environments.json")
commands = build_commands(models, environments=envs, repo_root=ROOT,
                          run_dir="/r/2026-08-27_abc",
                          passthrough=["--suite", "smoke", "--limit-items", "5"])
parser = build_parser()
for env_name, cmd in commands:
    parsed = parser.parse_args(cmd[3:])   # after interpreter -m robochrono
    check(f"{env_name} child command parses",
          parsed.command == "eval" and parsed.run_dir == "/r/2026-08-27_abc"
          and parsed.no_dispatch and parsed.limit_items == 5)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
