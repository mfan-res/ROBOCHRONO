#!/usr/bin/env python3
# coding: utf-8
"""The whole chain, on the real dataset, with a replayed model.

prepare_run establishes identity, execute runs the matrix serially, and the
results layer reports it — everything a real evaluation does except the GPU.
The replay table is built from the dataset's own answers, so every score has
a known expected value: exactly 1.0, anything else is a chain defect.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochrono import dimensions  # noqa: E402
from robochrono.config.models import load_model  # noqa: E402
from robochrono.config.protocol import load_protocol  # noqa: E402
from robochrono.config.suites import load_suite, pins_digest  # noqa: E402
from robochrono.dataset.loader import load_questions  # noqa: E402
from robochrono.dataset.render import load_question_bank  # noqa: E402
from robochrono.orchestrate.execute import execute, prepare_run  # noqa: E402
from robochrono.orchestrate.matrix import expand  # noqa: E402
from robochrono.results import report  # noqa: E402

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:52} {detail}")
    if not passed:
        failures.append(name)


DATA = ROOT / "scenarios"
if not (DATA / "make_tea_tianji" / "manifest.json").exists():
    print("skip: dataset not present")
    sys.exit(0)

PROTOCOL = load_protocol(ROOT / "configs/protocol.json")
SUITE = load_suite("official-v1", ROOT / "configs/suites")
MODEL_PATH = ROOT / "configs/models/local/qwen3-vl-2b-instruct.json"
MODEL = dataclasses.replace(load_model(MODEL_PATH), adapter="replay")

specs, _ = expand([MODEL], SUITE, DATA,
                  only_scenarios=["make_tea_tianji", "pack_airpods_tianji"],
                  only_dimensions=["current_action", "action_time"])
check("four specs to run", len(specs) == 4, str(len(specs)))

# The replay table answers every question correctly, from the dataset itself.
bank = load_question_bank(DATA)
table: dict[str, str] = {}
for spec in specs:
    dim = dimensions.build(spec.dimension, strip_reasoning=PROTOCOL.strip_reasoning)
    items = load_questions(DATA, spec.scenario, spec.dimension, bank=bank)
    for unit in dim.units(items):
        item = unit.items[0]
        if spec.dimension == "action_time":
            sec = item["answer_seconds"]
            table[unit.key] = json.dumps({"start": sec["start"], "end": sec["end"]})
        else:
            table[unit.key] = json.dumps({"choice": item["answer"]})

results_root = Path(tempfile.mkdtemp())

print("1. identity is established before anything runs")
run = prepare_run(results_root, protocol_path=ROOT / "configs/protocol.json",
                  suite_path=ROOT / "configs/suites/official-v1.json",
                  suite_name="official-v1",
                  models=[MODEL], model_paths=[MODEL_PATH],
                  protocol=PROTOCOL, data_root=DATA, repo_root=ROOT)
record = json.loads((run.path / "run.json").read_text())
check("run.json snapshot written", record["suite"] == "official-v1")
check("the suite's pins digest recorded",
      record["dataset"]["digest"] == pins_digest(SUITE.pins))
check("per-scenario pins recorded",
      record["dataset"]["scenarios"]["make_tea_tianji"] == SUITE.pins["make_tea_tianji"])
check("generation snapshot per model",
      record["generation_by_model"]["qwen3-vl-2b-instruct"]["max_new_tokens"] == 4096)

print("2. the matrix executes and scores perfectly against its own answers")
summaries = execute(specs, models=[MODEL], protocol=PROTOCOL, data_root=DATA,
                    run_dir=run, adapter_runtime={"replay_table": table})
check("all four specs summarized", len(summaries) == 4)
for spec in specs:
    s = summaries[spec.key]
    metric = "tIoU@0.5" if spec.dimension == "action_time" else "accuracy"
    check(f"{spec.scenario}/{spec.dimension} scores 1.0",
          s.get(metric) == 1.0, f"{metric}={s.get(metric)}")
    check(f"{spec.scenario}/{spec.dimension} files on disk",
          spec.store_path(run.path).exists()
          and spec.store_path(run.path).with_suffix(".summary.json").exists())

print("3. re-running the same command is a no-op")
before = {s.key: s.store_path(run.path).read_text() for s in specs}
execute(specs, models=[MODEL], protocol=PROTOCOL, data_root=DATA,
        run_dir=run, adapter_runtime={"replay_table": table})
check("no rows appended on resume",
      all(s.store_path(run.path).read_text() == before[s.key] for s in specs))
run2 = prepare_run(results_root, protocol_path=ROOT / "configs/protocol.json",
                   suite_path=ROOT / "configs/suites/official-v1.json",
                   suite_name="official-v1",
                   models=[MODEL], model_paths=[MODEL_PATH],
                   protocol=PROTOCOL, data_root=DATA, repo_root=ROOT)
check("identity resolves to the same directory", run2.path == run.path and run2.resumed)

print("4. the run reports")
floors = json.loads((ROOT / "configs/protocol.json").read_text())["degenerate_floor"]
rep = report.collect([run.path], floors)
check("all specs collected", len(rep.rows) == 4)
check("nothing flagged", not any(r["floor"] or r["fault"] for r in rep.rows))
md = report.to_markdown(rep)
check("the pins digest in the header", pins_digest(SUITE.pins) in md)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
