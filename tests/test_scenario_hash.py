#!/usr/bin/env python3
# coding: utf-8
"""The scenario hash pins what a model is asked, and nothing else.

The properties under test are the contract the data side relies on: the same
content always hashes the same, however it is formatted; touching anything a
model sees moves the hash; touching metadata, statistics or provenance does
not. A hash that shifted on reformatting would make every re-upload look like
a new dataset; one that ignored a changed answer would let two different
exams share a pin.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "compute_scenario_hash.py"

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:56} {detail}")
    if not passed:
        failures.append(name)


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                          capture_output=True, text=True)


DIMENSIONS = ("current_action", "next_action", "next_action_with_goal",
              "frame_match", "view_match", "frame_order", "action_time")


def make_scenario(root: Path, name: str = "fold_towel_gim") -> Path:
    """A minimal but structurally faithful scenario directory."""
    sc = root / name
    qa = sc / "qa"
    qa.mkdir(parents=True)
    item = {"id": f"{name}/file-000/s00@current_action", "type": "current_action",
            "answer": "B", "segment": "0", "subtask": "pick_towel",
            "options": [{"id": "A", "subtask": "place_towel"},
                        {"id": "B", "subtask": "pick_towel"}],
            "input": {"clip_path": f"media/clips/{name}/file-000@main@000-010.mp4"}}
    interval = {"id": f"{name}/file-000@action_time", "type": "action_time",
                "answer": [1.0, 2.5], "subtask": "pick_towel",
                "input": {"video_path": f"media/episodes/{name}/file-000@main.mp4"}}
    for dim in DIMENSIONS:
        if dim == "current_action":
            items = [item]
        elif dim == "action_time":
            items = [interval]
        else:
            items = []          # empty dimensions exist as files, and count
        (qa / f"{dim}.json").write_text(json.dumps({"items": items}), "utf-8")
    (qa / "subtasks.json").write_text(json.dumps({name: [
        {"id": "pick_towel", "text": "Pick up the towel.",
         "evidence": "annotator", "review_required": False},
        {"id": "place_towel", "text": "Place the towel on the rack.",
         "evidence": "annotator", "review_required": True},
    ]}), "utf-8")
    (qa / "scenarios.json").write_text(json.dumps({name: {
        "embodiment": "gim", "goal": "folding a towel and racking it",
        "camera": {"resolution": [640, 480], "fps": 30.0, "views": ["main"]},
        "counts": {"episodes": 2, "segments": 4, "subtasks": 2},
        "next_action": {"transitions": 2, "conditional_entropy": 0.0},
    }}), "utf-8")
    (qa / "dimensions.json").write_text(json.dumps({"templates": {
        "current_action": "What is happening right now?\n{options}",
        "action_time": "When does the robot {subtask}?",
    }}), "utf-8")
    media = sc / "media" / "clips" / name
    media.mkdir(parents=True)
    (media / "file-000@main@000-010.mp4").write_bytes(b"\x00" * 64)
    return sc


def edit_json(path: Path, mutate) -> None:
    data = json.loads(path.read_text("utf-8"))
    mutate(data)
    path.write_text(json.dumps(data), "utf-8")


with tempfile.TemporaryDirectory() as tmp_:
    tmp = Path(tmp_)
    sc = make_scenario(tmp)

    print("1. determinism and canonicalisation")
    first = run(sc).stdout.strip()
    second = run(sc).stdout.strip()
    check("same input twice, same hash", first == second and len(first) == 64, first[:12])

    # Re-serialise one dimension file with indentation and shuffled keys:
    # content identical, bytes different.
    raw = json.loads((sc / "qa/current_action.json").read_text("utf-8"))
    (sc / "qa/current_action.json").write_text(
        json.dumps(raw, indent=4, sort_keys=True), "utf-8")
    check("reformatting changes nothing", run(sc).stdout.strip() == first)

    print("2. model-visible edits move the hash")
    variants = {}

    other = make_scenario(tmp / "a"); edit_json(other / "qa/current_action.json",
        lambda d: d["items"][0].__setitem__("answer", "A"))
    variants["an item's answer"] = other

    other = make_scenario(tmp / "b"); edit_json(other / "qa/subtasks.json",
        lambda d: d["fold_towel_gim"][0].__setitem__("text", "Grab the towel."))
    variants["a subtask's wording"] = other

    other = make_scenario(tmp / "c"); edit_json(other / "qa/scenarios.json",
        lambda d: d["fold_towel_gim"].__setitem__("goal", "tidying the rack"))
    variants["the goal"] = other

    other = make_scenario(tmp / "d"); edit_json(other / "qa/dimensions.json",
        lambda d: d["templates"].__setitem__("current_action", "Now what?\n{options}"))
    variants["a question template"] = other

    other = make_scenario(tmp / "e"); edit_json(other / "qa/frame_order.json",
        lambda d: d["items"].append({"id": "fold_towel_gim/x@frame_order",
                                     "type": "frame_order", "answer": "A"}))
    variants["an empty dimension gaining an item"] = other

    for what, v in variants.items():
        check(f"{what} changes the hash", run(v).stdout.strip() != first)

    print("3. metadata and provenance edits do not")
    for what, path, mutate in [
        ("camera fps", "qa/scenarios.json",
         lambda d: d["fold_towel_gim"]["camera"].__setitem__("fps", 60.0)),
        ("episode counts", "qa/scenarios.json",
         lambda d: d["fold_towel_gim"]["counts"].__setitem__("episodes", 99)),
        ("transition statistics", "qa/scenarios.json",
         lambda d: d["fold_towel_gim"]["next_action"].__setitem__("transitions", 7)),
        ("subtask provenance", "qa/subtasks.json",
         lambda d: d["fold_towel_gim"][0].__setitem__("evidence", "revised")),
    ]:
        v = make_scenario(tmp / f"m_{what.replace(' ', '_')}")
        edit_json(v / path, mutate)
        check(f"{what} leaves the hash alone", run(v).stdout.strip() == first)

    v = make_scenario(tmp / "m_media")
    (v / "media/clips/fold_towel_gim/extra.mp4").write_bytes(b"\x01" * 128)
    check("media bytes leave the hash alone", run(v).stdout.strip() == first)

    print("4. invalid scenarios are refused, not guessed at")
    v = make_scenario(tmp / "bad_missing")
    (v / "qa/view_match.json").unlink()
    check("a missing dimension file is exit 2", run(v).returncode == 2)

    v = make_scenario(tmp / "bad_embodiment")
    edit_json(v / "qa/scenarios.json",
              lambda d: d["fold_towel_gim"].__setitem__("embodiment", "tianji"))
    r = run(v)
    check("embodiment vs suffix mismatch is exit 2", r.returncode == 2,
          r.stderr.strip()[:60])

    print("5. manifest write and verify round-trip")
    sc2 = make_scenario(tmp / "roundtrip")
    manifest = json.loads(run(sc2, "--manifest").stdout)
    check("manifest counts the questions", manifest["questions"] == 2)
    check("empty dimensions are recorded as zero",
          manifest["dimensions"]["view_match"] == 0)
    check("metadata rides along", manifest["camera"]["fps"] == 30.0
          and manifest["counts"]["episodes"] == 2)
    check("media inventory lists path and size",
          manifest["media"]["files"] == 1 and manifest["media"]["bytes"] == 64)

    check("--write succeeds", run(sc2, "--write").returncode == 0)
    check("--verify passes on the written manifest",
          run(sc2, "--verify").returncode == 0)
    check("--write again reports unchanged",
          "unchanged" in run(sc2, "--write").stdout)

    edit_json(sc2 / "manifest.json", lambda d: d.__setitem__("questions", 3))
    check("--verify catches a tampered manifest",
          run(sc2, "--verify").returncode == 1)
    check("--write refuses to silently repair without --force",
          run(sc2, "--write").returncode == 1)
    check("--write --force repairs it",
          run(sc2, "--write", "--force").returncode == 0
          and run(sc2, "--verify").returncode == 0)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
