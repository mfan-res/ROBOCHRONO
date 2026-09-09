#!/usr/bin/env python3
# coding: utf-8
"""Matrix expansion: dense, ordered, sharded, and loud about typos."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochrono.config.models import load_models  # noqa: E402
from robochrono.config.suites import load_suite  # noqa: E402
from robochrono.orchestrate.matrix import expand, shard_of  # noqa: E402

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:52} {detail}")
    if not passed:
        failures.append(name)


MODELS = sorted(load_models(ROOT / "configs/models").values(), key=lambda m: m.slug)
SUITE = load_suite("official-v1", ROOT / "configs/suites")
DATA = ROOT / "scenarios"

print("1. the suite expands densely")
selected, skipped = expand(MODELS, SUITE, DATA)
expected = len(MODELS) * len(SUITE.scenarios) * len(SUITE.dimensions)
check("every combination selected", len(selected) == expected,
      f"{len(selected)} of {expected}")
check("nothing skipped on the real dataset", not skipped, str(skipped[:2]))

print("2. model-major order")
models_seen = [s.model for s in selected]
check("each model's work is consecutive",
      models_seen == sorted(models_seen))

print("3. filters narrow, typos raise")
one, _ = expand(MODELS, SUITE, DATA, only_models=["qwen3-vl-2b-instruct"],
                only_scenarios=["make_tea_tianji"], only_dimensions=["action_time"])
check("narrowed to one spec", len(one) == 1 and one[0].key ==
      "qwen3-vl-2b-instruct__make_tea_tianji__action_time")
for kwargs in ({"only_models": ["qwen3-vl-2b"]},
               {"only_scenarios": ["make_tea"]},
               {"only_dimensions": ["tiem"]}):
    try:
        expand(MODELS, SUITE, DATA, **kwargs)
        check(f"typo raises {kwargs}", False, "no exception")
    except ValueError as e:
        check(f"typo raises ({next(iter(kwargs))})", "unknown" in str(e))
api_only, _ = expand(MODELS, SUITE, DATA, only_kind="api")
api_models = {m.slug for m in MODELS if m.kind == "api"}
check("kind filter applies",
      {s.model for s in api_only} == api_models
      and len(api_only) == len(api_models) * len(SUITE.scenarios) * len(SUITE.dimensions))

print("4. shards are disjoint and complete")
shards = [expand(MODELS, SUITE, DATA, shard=(i, 3))[0] for i in (1, 2, 3)]
keys = [set(s.key for s in part) for part in shards]
check("disjoint", not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]))
check("complete", set().union(*keys) == {s.key for s in selected})
check("stable", shard_of("a__b__c", 4) == shard_of("a__b__c", 4))

print("5. a missing question file is skipped with a reason")
import tempfile, shutil
tmp = Path(tempfile.mkdtemp())
partial = tmp / "make_tea_tianji" / "qa"
partial.mkdir(parents=True)
shutil.copy(DATA / "make_tea_tianji/qa/action_time.json", partial / "action_time.json")
got, missing = expand(MODELS[:1], SUITE, tmp)
check("existing file selected", len(got) == 1)
check("the rest skipped with reasons",
      len(missing) == len(SUITE.scenarios) * len(SUITE.dimensions) - 1
      and all(reason == "question file missing" for _, reason in missing))

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
