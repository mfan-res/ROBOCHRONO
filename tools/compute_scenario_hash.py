#!/usr/bin/env python3
# coding: utf-8
"""The scenario hash and manifest: one implementation, used by everyone.

A scenario directory is self-contained: ``qa/`` holds the seven dimension
files and the three render files split down to this scenario; ``media/``
holds what the questions refer to. The hash pins **what a model is asked** —
the question items, the subtask wording, the goal, the question templates —
and nothing else. Metadata (camera, counts, transition statistics) and media
bytes deliberately stay out: reformatting a JSON file or re-encoding a video
must not move the pin, and B-0 measured that superseded scenario copies
differ from their revisions in exactly those provenance fields.

Canonicalisation is what makes the hash portable: parse, then re-serialise
with sorted keys, compact separators and raw UTF-8. Array order is kept —
item order comes from the data producer and is part of the content.

    compute_scenario_hash.py <scenario_dir>              print the hash
    compute_scenario_hash.py <scenario_dir> --manifest   print the manifest JSON
    compute_scenario_hash.py <scenario_dir> --write      write manifest.json
    compute_scenario_hash.py <scenario_dir> --verify     compare with manifest.json

Exit codes: 0 success or verified, 1 verification mismatch, 2 the directory
is not a valid scenario (missing file, missing entry, embodiment mismatch).

This script is deliberately standalone — standard library only, no imports
from the package — so the data publishing side can run it on a bare checkout
of one scenario directory and get the same numbers the evaluator checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SPEC_VERSION = 1
SCHEMA_VERSION = 1

# Canonical benchmark order, used for the manifest's dimensions table.
# The hash itself sorts keys, so this order is cosmetic there.
DIMENSIONS = ("current_action", "next_action", "next_action_with_goal",
              "frame_match", "view_match", "frame_order", "action_time")

EMBODIMENTS = ("gim", "tianji", "tianjihand", "hand")


class ScenarioError(Exception):
    """The directory does not hold a valid scenario."""


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise ScenarioError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _items(raw: Any, path: Path) -> list:
    items = raw.get("items", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ScenarioError(f"{path} holds no item list")
    return items


def load_scenario(scenario_dir: Path) -> dict[str, Any]:
    """Everything the hash and the manifest need, validated on the way in."""
    scenario_dir = Path(scenario_dir)
    scenario_id = scenario_dir.name
    qa = scenario_dir / "qa"

    dimensions = {dim: _items(_read_json(qa / f"{dim}.json"), qa / f"{dim}.json")
                  for dim in DIMENSIONS}

    subtasks_file = _read_json(qa / "subtasks.json")
    if scenario_id not in subtasks_file:
        raise ScenarioError(f"qa/subtasks.json has no entry for {scenario_id!r}")
    subtasks = subtasks_file[scenario_id]

    scenarios_file = _read_json(qa / "scenarios.json")
    if scenario_id not in scenarios_file:
        raise ScenarioError(f"qa/scenarios.json has no entry for {scenario_id!r}")
    meta = scenarios_file[scenario_id]
    if "goal" not in meta:
        raise ScenarioError(f"scenarios.json entry for {scenario_id!r} has no goal")

    templates = _read_json(qa / "dimensions.json").get("templates")
    if templates is None:
        raise ScenarioError("qa/dimensions.json has no templates")

    embodiment = meta.get("embodiment")
    suffix = scenario_id.rsplit("_", 1)[-1]
    if embodiment not in EMBODIMENTS:
        raise ScenarioError(f"embodiment {embodiment!r} not one of {EMBODIMENTS}")
    if embodiment != suffix:
        raise ScenarioError(f"embodiment {embodiment!r} does not match the "
                            f"name suffix _{suffix} — one of them is wrong")

    return {"scenario_id": scenario_id, "dimensions": dimensions,
            "subtasks": subtasks, "meta": meta, "templates": templates,
            "dir": scenario_dir}


def scenario_hash(loaded: dict[str, Any]) -> str:
    """sha256 over the canonicalised model-visible content. Spec: dev notes."""
    obj = {
        "spec": SPEC_VERSION,
        "scenario": loaded["scenario_id"],
        "dimensions": loaded["dimensions"],
        # Only what rendering reads: id and text. Provenance fields on a
        # subtask (evidence, reviewer, generation trail) change nothing a
        # model sees and stay out.
        "subtasks": [{"id": s["id"], "text": s["text"]} for s in loaded["subtasks"]],
        "goal": loaded["meta"]["goal"],
        "templates": loaded["templates"],
    }
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def media_inventory(scenario_dir: Path) -> dict[str, Any]:
    """The weak check: every file under media/, path and size, no hashes."""
    media = Path(scenario_dir) / "media"
    entries = []
    if media.exists():
        for f in sorted(media.rglob("*")):
            if f.is_file():
                entries.append({"path": f.relative_to(scenario_dir).as_posix(),
                                "bytes": f.stat().st_size})
    return {"files": len(entries),
            "bytes": sum(e["bytes"] for e in entries),
            "entries": entries}


def build_manifest(scenario_dir: Path) -> dict[str, Any]:
    loaded = load_scenario(scenario_dir)
    meta = loaded["meta"]
    counts = {dim: len(items) for dim, items in loaded["dimensions"].items()}
    manifest = {
        "schema": SCHEMA_VERSION,
        "scenario_id": loaded["scenario_id"],
        "embodiment": meta["embodiment"],
        "hash": scenario_hash(loaded),
        "questions": sum(counts.values()),
        "dimensions": {dim: counts[dim] for dim in DIMENSIONS},
        "episodes": (meta.get("counts") or {}).get("episodes"),
        # Metadata rides along, outside the hash: reformatting or restating
        # it must not move the pin, but it must not be lost either.
        "camera": meta.get("camera"),
        "counts": meta.get("counts"),
        "next_action": meta.get("next_action"),
        "media": media_inventory(scenario_dir),
    }
    return manifest


def _dump(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scenario_dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--manifest", action="store_true",
                      help="print the full manifest instead of the bare hash")
    mode.add_argument("--write", action="store_true",
                      help="write <scenario_dir>/manifest.json")
    mode.add_argument("--verify", action="store_true",
                      help="recompute and compare against manifest.json")
    parser.add_argument("--force", action="store_true",
                        help="with --write: replace a manifest that differs")
    args = parser.parse_args(argv)

    try:
        manifest = build_manifest(args.scenario_dir)
    except (ScenarioError, json.JSONDecodeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    target = args.scenario_dir / "manifest.json"

    if args.write:
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing == manifest:
                print(f"unchanged: {target}")
                return 0
            if not args.force:
                print(f"refusing to overwrite {target} — it differs from the "
                      f"recomputed manifest; pass --force to replace it",
                      file=sys.stderr)
                return 1
        target.write_text(_dump(manifest), encoding="utf-8")
        print(f"wrote {target} ({manifest['hash'][:12]}, "
              f"{manifest['questions']} questions)")
        return 0

    if args.verify:
        if not target.exists():
            print(f"error: {target} does not exist", file=sys.stderr)
            return 1
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing == manifest:
            print(f"ok: {manifest['scenario_id']} {manifest['hash'][:12]} "
                  f"({manifest['questions']} questions)")
            return 0
        fields = [k for k in set(existing) | set(manifest)
                  if existing.get(k) != manifest.get(k)]
        print(f"MISMATCH: {manifest['scenario_id']} differs in {sorted(fields)}",
              file=sys.stderr)
        return 1

    if args.manifest:
        print(_dump(manifest), end="")
    else:
        print(manifest["hash"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
