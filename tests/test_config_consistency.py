#!/usr/bin/env python3
# coding: utf-8
"""Cross-file consistency of the configuration.

A JSON schema can check structure — required fields, types, misspelled keys.
It cannot check that two files agree with each other, and that is where the
failures that matter live: a suite naming a scenario the dataset does not have,
a model mapped to an environment that does not exist, a dimension with no frame
sampling declared. None of those raise on their own; they produce a run that
quietly covers less than it claims.

What is validated is that a decision was made and that the decisions are
mutually consistent — not whether any individual value is the right one.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:44} {detail}")
    if not passed:
        failures.append(name)


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


protocol = load("configs/protocol.json")
runtime = load("configs/runtime.json")
environments = load("configs/environments.json")
suites = {p.stem: json.loads(p.read_text(encoding="utf-8"))
          for p in sorted((ROOT / "configs/suites").glob("*.json"))}
models = {p.stem: json.loads(p.read_text(encoding="utf-8"))
          for p in sorted((ROOT / "configs/models").rglob("*.json"))}

# The scenario pool: one manifest per scenario directory.
pool = {}
for path in sorted((ROOT / "scenarios").glob("*/manifest.json")):
    m = json.loads(path.read_text(encoding="utf-8"))
    pool[m["scenario_id"]] = m
DIMENSIONS = ("current_action", "next_action", "next_action_with_goal",
              "frame_match", "view_match", "frame_order", "action_time")

print("1. model -> environment")
env_names = set(environments["envs"])
for slug, m in models.items():
    check(f"{slug} environment exists", m.get("environment") in env_names,
          m.get("environment", "(missing)"))

print("2. declared transformers requirement vs the environment provided")


def satisfies(spec: str | None, actual: str) -> bool | None:
    """Supports ==X and >=X. None means no requirement is declared."""
    if not spec:
        return None
    m = re.fullmatch(r"(==|>=)\s*([\d.]+)", spec.strip())
    if not m:
        return None
    op, want = m.group(1), tuple(int(x) for x in m.group(2).split("."))
    got = tuple(int(x) for x in actual.split("."))
    want += (0,) * (len(got) - len(want))
    want = want[: len(got)]
    # A pinned version is read as a lower bound: the environments ship a later
    # patch release that satisfies every model mapped to them.
    return got >= want


for slug, m in models.items():
    tf = (m.get("official") or {}).get("transformers")
    # A missing declaration is an error. "The documentation does not say" and
    # "nobody filled this in" must be distinguishable, so the former is written
    # explicitly as source: "none".
    if not isinstance(tf, dict) or "source" not in tf:
        check(f"{slug} declares official.transformers.source", False,
              "missing — write source: \"none\" when the docs say nothing")
        continue
    spec, src = tf.get("value"), tf["source"]
    env = environments["envs"].get(m.get("environment"), {})
    actual = env.get("transformers", "")
    ok = satisfies(spec, actual) if actual else None
    if ok is None:
        check(f"{slug}", True, f"no version declared (source={src})")
    else:
        check(f"{slug}", ok, f"requires {spec}, environment has {actual}")

print("3. suites are present, well-formed, and pinned to the pool")
# An empty glob must fail, not fall through: with zero suite files the loop
# below checks nothing, and a cleared directory would read as all-clear.
# That silent pass happened once, in B1-a, and this line is what stops it.
check("at least one suite file exists", bool(suites), str(sorted(suites)))
for sname, sdef in suites.items():
    missing_keys = [k for k in ("name", "scenarios", "dimensions") if k not in sdef]
    check(f"{sname} has the schema keys", not missing_keys, str(missing_keys))
    if missing_keys:
        continue
    pins = sdef["scenarios"]
    bad_hash = [sc for sc, h in pins.items()
                if not (isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h))]
    check(f"{sname} pins full hashes", not bad_hash, str(bad_hash[:3]))
    dims = sdef["dimensions"]
    if dims is not None:
        unknown_dim = set(dims) - set(DIMENSIONS)
        check(f"{sname} dimensions exist", not unknown_dim, f"unknown: {sorted(unknown_dim)}")
    if pool:
        unknown_sc = set(pins) - set(pool)
        check(f"{sname} scenarios exist in the pool", not unknown_sc,
              "" if not unknown_sc else
              f"not in the local pool: {sorted(unknown_sc)[:3]}"
              f"{' …' if len(unknown_sc) > 3 else ''} — a partial download, "
              f"not a model problem; tools/download_data.py --all --qa-only "
              f"fetches every manifest (~50 MB)")
        stale = [sc for sc in set(pins) & set(pool)
                 if pins[sc] != pool[sc]["hash"]]
        check(f"{sname} pins match the pool", not stale, f"stale: {sorted(stale)[:3]}")
if pool and "official-v1" in suites:
    pinned = set(suites["official-v1"]["scenarios"])
    if pinned <= set(pool):
        total = sum(pool[sc]["questions"] for sc in pinned)
        check("official-v1 covers 34,713 questions", total == 34713, str(total))
    else:
        # With scenarios absent the count can only be wrong, and the existence
        # check above has already failed and said why — repeating the fault as
        # a second FAIL would read as a new problem.
        print(f"  ({len(pinned - set(pool))} pinned scenario(s) not in the "
              f"local pool — question-count check skipped)")
if not pool:
    print("  (scenario pool not downloaded — pin checks against it skipped)")

print("4. frame sampling covers every dimension")
declared = set(protocol["frames"]["by_dimension"])
known = set(DIMENSIONS)
check("all dimensions declared", not (known - declared), f"missing: {sorted(known - declared)}")
check("no extra dimensions declared", not (declared - known), f"extra: {sorted(declared - known)}")

print("5. degenerate floors cover every dimension")
floors = protocol["degenerate_floor"]
for dim in DIMENSIONS:
    kind = "interval" if dim == "action_time" else "choice"
    check(f"{dim} has a floor", kind in floors, f"uses {kind}")

print("6. weights")
for slug, m in models.items():
    w = m.get("weights", "")
    check(f"{slug} declares weights", bool(w), w)

print("7. operational settings stay out of the protocol")
leaked = [k for k in ("api_concurrency", "proxy", "media_cache_dir", "gpus_per_worker")
          if k in protocol or k in protocol.get("generation", {})]
check("no machine-specific settings", not leaked, f"leaked: {leaked}")

print("8. scenario and dimension names are current everywhere")
# Names appear in prose and in docstrings, not only in the dataset, and a
# checked-in example that still uses a superseded name reads as a second set of
# data rather than as a stale line.
# Two generations of superseded names. The first are the working names the
# dataset was built under; the second are scenario names from dataset 1.0 that
# 2.0 renamed when scenarios recorded on other embodiments joined.
# `understanding`, `time` and `planning` were also working names, but they are
# ordinary English words and would match prose, so they are not scanned for.
# `stack_cubes_tianji` left this list with dataset 2.2: the 1.0 name was
# retired as mislabeled, and 2.2 re-issued it for the genuine tianji
# recording (QAGen's canonical mapping: raw `stack_cubes` -> this name).
superseded = {
    "airpods", "gift_inhand", "pen_inbox", "stack_cubes", "tea", "wash",
    "planning_2", "left_right", "image_in_video", "step_order",
    "hand_gift_tianji", "box_pen_tianji",
}
pattern = re.compile(r"\b(" + "|".join(sorted(superseded)) + r")\b")
# This file is the one place the superseded names may appear: the list above.
skip = {"tests/test_config_consistency.py"}
stale = []
for f in sorted(ROOT.rglob("*")):
    rel = f.relative_to(ROOT).as_posix()
    if f.suffix not in {".md", ".py", ".json", ".toml"} or rel in skip:
        continue
    # Several scenario names are also ordinary words, so a line-oriented scan
    # over the dataset would read them out of a goal description and report a
    # name that is not there. The dataset's own naming is covered by
    # tests/test_dataset_contract.py, which checks fields, not lines.
    # `.venvs/` holds third-party source, and `dev/` and `internal/` the
    # untracked internal ledgers; all legitimately contain the old names and
    # none is this project's published text. They used to be symlinks, which
    # rglob does not follow — so this only started mattering once an
    # environment was rebuilt in place.
    if rel.startswith(("data/", "scenarios/", ".git/", "results/", ".venvs/",
                       "dev/", "internal/", "models/")):
        continue
    for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        if pattern.search(line):
            stale.append(f"{rel}:{n}")
check("no superseded names in tracked files", not stale, f"{len(stale)}: {stale[:3]}")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
