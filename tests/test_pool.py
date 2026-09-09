#!/usr/bin/env python3
# coding: utf-8
"""The worker pool, exercised with real spawned processes and no GPU.

The replay adapter needs no inference stack, so actual spawn workers can run
the full pool machinery — queue distribution, single-writer collection,
startup-failure reporting, resume — as an ordinary test. GPU indices here
only shape CUDA_VISIBLE_DEVICES strings nothing reads.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Spawned workers import the package afresh; they inherit the environment,
# not this process's sys.path.
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

from robochrono import dimensions  # noqa: E402
from robochrono.config.models import load_model  # noqa: E402
from robochrono.config.protocol import load_protocol  # noqa: E402
from robochrono.dataset.loader import load_questions  # noqa: E402
from robochrono.dataset.render import load_question_bank  # noqa: E402
from robochrono.orchestrate.matrix import RunSpec  # noqa: E402
from robochrono.orchestrate.pool import run_model_pool  # noqa: E402
from robochrono.results.store import ResultStore  # noqa: E402

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:52} {detail}")
    if not passed:
        failures.append(name)


def check_card_selection() -> None:
    """Which cards a run is given — a count, or the exact ones named.

    Naming cards matters on a shared machine, where the free ones are rarely
    the first ones. `CUDA_VISIBLE_DEVICES` cannot serve here: workers set it
    from these indices before importing torch, so an inherited value is
    overwritten and the run lands on the wrong cards without saying so.
    """
    from robochrono.orchestrate.pool import visible_gpus
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            print("  (no multi-GPU machine; card selection not exercised)")
            return
        count = torch.cuda.device_count()
    except ImportError:
        print("  (torch absent; card selection not exercised)")
        return

    check("no argument takes every card", visible_gpus(None) == list(range(count)))
    check("a count takes the first N", visible_gpus(2) == [0, 1])
    check("a count as a string means the same", visible_gpus("2") == [0, 1])
    last = count - 1
    check("a list takes exactly those cards", visible_gpus(f"1,{last}") == [1, last])
    try:
        visible_gpus(f"1,{count}")
        check("a card this machine lacks is refused", False, "no error raised")
    except ValueError as exc:
        check("a card this machine lacks is refused", str(count) in str(exc))


def main() -> None:
    print("0. card selection")
    check_card_selection()

    DATA = ROOT / "scenarios"
    if not (DATA / "make_tea_tianji" / "manifest.json").exists():
        print("skip: dataset not present")
        sys.exit(0)

    PROTOCOL_PATH = ROOT / "configs/protocol.json"
    PROTOCOL = load_protocol(PROTOCOL_PATH)

    # A replay model the worker can load from disk.
    base = json.loads((ROOT / "configs/models/local/qwen3-vl-2b-instruct.json").read_text())
    base["adapter"] = "replay"
    model_dir = Path(tempfile.mkdtemp()) / "local"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "replay-model.json"
    model_path.write_text(json.dumps(base))
    MODEL = load_model(model_path)

    SPECS = [RunSpec(model=MODEL.slug, scenario="make_tea_tianji", dimension=d)
             for d in ("current_action", "next_action")]
    LIMIT = 12

    bank = load_question_bank(DATA)
    table: dict[str, str] = {}
    for spec in SPECS:
        dim = dimensions.build(spec.dimension, strip_reasoning=PROTOCOL.strip_reasoning)
        items = load_questions(DATA, spec.scenario, spec.dimension, bank=bank)
        for unit in dim.units(items)[:LIMIT]:
            table[unit.key] = json.dumps({"choice": unit.items[0]["answer"]})

    run_dir = Path(tempfile.mkdtemp())

    print("1. two spawned workers drain the queue, main process writes")
    summaries = run_model_pool(
        MODEL, SPECS, protocol=PROTOCOL, data_root=DATA, run_dir=run_dir,
        adapter_runtime={"replay_table": table}, gpus=[0, 1],
        limit_items=LIMIT, model_path=model_path, protocol_path=PROTOCOL_PATH)
    for spec in SPECS:
        check(f"{spec.dimension} scores 1.0",
              summaries[spec.key].get("accuracy") == 1.0,
              str(summaries[spec.key].get("accuracy")))
    rows = list(ResultStore(SPECS[0].store_path(run_dir)).rows())
    check("rows written by the main process", len(rows) == LIMIT, str(len(rows)))
    check("rows carry a worker id",
          all(r.get("timing", {}).get("worker") in (0, 1) for r in rows))

    print("2. a second pool run finds nothing pending")
    before = {s.key: s.store_path(run_dir).read_text() for s in SPECS}
    run_model_pool(MODEL, SPECS, protocol=PROTOCOL, data_root=DATA, run_dir=run_dir,
                   adapter_runtime={"replay_table": table}, gpus=[0, 1],
                   limit_items=LIMIT, model_path=model_path, protocol_path=PROTOCOL_PATH)
    check("files unchanged on resume",
          all(s.store_path(run_dir).read_text() == before[s.key] for s in SPECS))

    print("3. a worker that cannot start reports instead of hanging")
    broken = model_dir / "broken.json"
    bad = dict(base)
    bad["adapter"] = "no_such_adapter"
    broken.write_text(json.dumps(bad))
    result_dir = Path(tempfile.mkdtemp())
    summaries = run_model_pool(
        MODEL, [SPECS[0]], protocol=PROTOCOL, data_root=DATA, run_dir=result_dir,
        adapter_runtime={}, gpus=[0], limit_items=2,
        model_path=broken, protocol_path=PROTOCOL_PATH)
    check("pool returns with an empty summary rather than hanging",
          summaries[SPECS[0].key].get("total") == 0,
          str(summaries[SPECS[0].key].get("total")))

    print("4. a missing recording becomes an error row, not a dead worker")
    gap_dir = Path(tempfile.mkdtemp())
    partial = dict(list(table.items())[:LIMIT - 2])   # drop two recordings
    summaries = run_model_pool(
        MODEL, [SPECS[0]], protocol=PROTOCOL, data_root=DATA, run_dir=gap_dir,
        adapter_runtime={"replay_table": partial}, gpus=[0, 1],
        limit_items=LIMIT, model_path=model_path, protocol_path=PROTOCOL_PATH)
    s = summaries[SPECS[0].key]
    check("answered plus errors covers the limit",
          (s.get("answered") or 0) + (s.get("errors") or 0) == LIMIT,
          f"answered={s.get('answered')} errors={s.get('errors')}")
    check("the errors are the two missing recordings", s.get("errors") == 2)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
