#!/usr/bin/env python3
# coding: utf-8
"""The engine's promises, exercised without a GPU.

The replay adapter stands in for a model, which lets everything the engine is
responsible for — resumption, the circuit breaker, error recording, limits,
the single-writer rule — run as a plain test. What a real model would add is
only the text; the machinery around it is identical.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochrono import adapters, dimensions, engine  # noqa: E402
from robochrono.adapters.base import Adapter, AdapterResult  # noqa: E402
from robochrono.config.models import load_model  # noqa: E402
from robochrono.config.protocol import load_protocol  # noqa: E402
from robochrono.results.store import ResultStore  # noqa: E402

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:52} {detail}")
    if not passed:
        failures.append(name)


PROTOCOL = load_protocol(ROOT / "configs/protocol.json")
MODEL = dataclasses.replace(
    load_model(ROOT / "configs/models/local/qwen3-vl-2b-instruct.json"),
    adapter="replay")
DIMENSION = dimensions.build("current_action", strip_reasoning=PROTOCOL.strip_reasoning)


def make_items(n: int) -> list[dict]:
    return [{
        "id": f"scenario/file-000/s{i:02d}@current_action",
        "type": "current_action",
        "subtask": f"act_{i}",
        "question": f"Q{i}?\nOptions:\nA. yes\nB. no",
        "options": [{"id": "A", "text": "yes"}, {"id": "B", "text": "no"}],
        "answer": "A",
        "input": {"clip_path": f"media/clips/scenario/file-000@main@{i}.mp4"},
    } for i in range(n)]


def make_engine_run(items, table, store, **kwargs):
    adapter = adapters.build(MODEL, PROTOCOL, {"replay_table": table})
    return engine.run(DIMENSION, items, adapter, store,
                      data_root="/data", **kwargs)


def fresh_store() -> ResultStore:
    return ResultStore(Path(tempfile.mkdtemp()) / "m" / "s" / "current_action.jsonl")


GOOD = '{"choice": "A"}'
BAD = '{"choice": "B"}'

print("1. a full run scores, persists, and summarizes")
items = make_items(4)
table = {items[0]["id"]: GOOD, items[1]["id"]: GOOD,
         items[2]["id"]: BAD, items[3]["id"]: GOOD}
store = fresh_store()
summary = make_engine_run(items, table, store)
check("all answered", summary.get("answered") == 4, str(summary.get("answered")))
check("accuracy computed", abs(summary.get("accuracy") - 0.75) < 1e-9,
      str(summary.get("accuracy")))
check("summary written beside the log", store.summary_path.exists())
rows = list(store.rows())
check("one row per question, timing attached",
      len(rows) == 4 and all("timing" in r for r in rows))
check("prompt recorded in full",
      all(r.get("prompt", "").startswith("You are answering") for r in rows))

print("2. rerunning the same command does nothing twice")
before = store.path.read_text()
summary = make_engine_run(items, table, store)
check("nothing pending", store.path.read_text() == before)
check("summary unchanged", summary.get("answered") == 4)

print("3. failures become error rows, and a retry heals them")
items = make_items(3)
store = fresh_store()
partial = {items[0]["id"]: GOOD}          # 1 recorded, 2 will fail
summary = make_engine_run(items, partial, store)
check("failed calls recorded as errors", summary.get("errors") == 2,
      str(summary.get("errors")))
summary = make_engine_run(items, {i["id"]: GOOD for i in items}, store)
check("retry reruns only the failed", summary.get("answered") == 3)
check("errors cleared after retry", summary.get("errors") == 0, str(summary.get("errors")))
final = {r["id"]: r for r in store.final_rows()}
check("healed rows supersede error rows",
      all(not r.get("error") for r in final.values()))

print("4. the circuit breaker stops a run that keeps failing")
items = make_items(10)
store = fresh_store()
summary = make_engine_run(items, {}, store, circuit_breaker=5)
check("aborted", "circuit breaker" in str(summary.get("aborted")))
appended = sum(1 for _ in store.rows())
check("stopped at the threshold", appended == 5, str(appended))

print("5. limits")
items = make_items(6)
store = fresh_store()
make_engine_run(items, {i["id"]: GOOD for i in items}, store, limit_items=2)
check("limit_items caps questions", sum(1 for _ in store.rows()) == 2)
store = fresh_store()
make_engine_run(items, {i["id"]: GOOD for i in items}, store, limit_groups=3)
check("limit_groups caps calls", sum(1 for _ in store.rows()) == 3)

print("6. overwrite displaces instead of deleting")
items = make_items(2)
store = fresh_store()
make_engine_run(items, {i["id"]: GOOD for i in items}, store)
make_engine_run(items, {i["id"]: BAD for i in items}, store, overwrite=True)
backup = store.path.with_name(store.path.name + ".bak")
check("old rows preserved in .bak", backup.exists())
check("new rows written fresh",
      all(json.loads(l)["model_output"] == BAD
          for l in store.path.read_text().splitlines()))

print("7. media paths are resolved once, in the engine")
resolved = engine.resolve_parts(
    [{"type": "video", "path": "media/clips/x.mp4"},
     {"type": "text", "text": "hi"},
     {"type": "image", "path": "/absolute/kept.jpg"}], "/data/root")
check("relative becomes absolute", resolved[0]["path"] == "/data/root/media/clips/x.mp4")
check("text untouched", resolved[1] == {"type": "text", "text": "hi"})
check("absolute kept", resolved[2]["path"] == "/absolute/kept.jpg")

print("8. a failure after the model answered keeps the model's text")


class AnswersThenBreaks(Adapter):
    def call(self, parts, *, frames, key=""):
        return AdapterResult(text="the model said this")


class BreakingDimension:
    name = "current_action"
    def units(self, items): return DIMENSION.units(items)
    def parts(self, unit): return DIMENSION.parts(unit)
    def rows(self, unit, text, ctx): raise ValueError("scoring exploded")
    def error_rows(self, unit, error):
        return [{"id": unit.key, "model_output": None, "error": error}]
    def summarize(self, rows, elapsed): return {"total": len(rows)}


items = make_items(1)
store = fresh_store()
engine.run(BreakingDimension(), items, AnswersThenBreaks(MODEL, PROTOCOL), store,
           data_root="/data")
row = next(store.rows())
check("model text backfilled into the error row",
      row.get("model_output") == "the model said this", str(row.get("model_output")))
check("the error is still an error", "scoring exploded" in str(row.get("error")))

print("9. concurrency: many threads, one writer, consistent results")
items = make_items(30)
store = fresh_store()
summary = make_engine_run(items, {i["id"]: GOOD for i in items}, store, concurrency=8)
check("all questions answered", summary.get("answered") == 30)
lines = store.path.read_text().splitlines()
check("no interleaved lines", all(json.loads(l) for l in lines) and len(lines) == 30)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
