#!/usr/bin/env python3
# coding: utf-8
"""Reading scenario manifests — each scenario's self-description.

Every scenario directory carries a ``manifest.json`` written by
``tools/compute_scenario_hash.py``: its content hash, question counts,
embodiment, and the metadata that rides along outside the hash (camera,
episode counts, transition statistics, the media inventory). The hash is what
suites pin; results carry it so that scores computed on different content can
never be merged silently.

The manifest deliberately does not hold **judgements about the data**, such
as the degenerate-performance floor a score is compared against. Those live
in ``configs/protocol.json``, because the same data can reasonably be judged
against different floors — a floor is a choice about evaluation, not a
property of the dataset.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EMBODIMENTS = ("gim", "tianji", "tianjihand", "hand")
_HASH = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ScenarioManifest:
    path: Path
    scenario_id: str
    embodiment: str
    hash: str
    questions: int
    dimensions: dict[str, int]     # dimension -> question count (0 kept)
    episodes: int | None
    camera: dict[str, Any] | None
    counts: dict[str, Any] | None
    next_action: dict[str, Any] | None
    media: dict[str, Any]          # {files, bytes, entries: [{path, bytes}]}

    def dimension_names(self) -> list[str]:
        return sorted(self.dimensions)


_REQUIRED = ("schema", "scenario_id", "embodiment", "hash", "questions",
             "dimensions", "media")


def load_scenario_manifest(data_root: Any, scenario: str) -> ScenarioManifest:
    """Read and validate one scenario's manifest. A missing field is a hard
    error — filling in a default would make the code assert something about
    the data, and what the data contains is not something code should infer."""
    path = Path(data_root) / scenario / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — the scenario is not downloaded, or the data "
            f"root is wrong. See the Data section of README.md, or run "
            f"`robochrono validate-data`.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}")
    if raw["scenario_id"] != scenario:
        raise ValueError(f"{path} names scenario {raw['scenario_id']!r} but "
                         f"lives under {scenario!r} — the directory was "
                         f"renamed or the manifest copied")
    if raw["embodiment"] not in EMBODIMENTS:
        raise ValueError(f"{path}: embodiment {raw['embodiment']!r} is not "
                         f"one of {EMBODIMENTS}")
    if not (isinstance(raw["hash"], str) and _HASH.match(raw["hash"])):
        raise ValueError(f"{path}: hash is not a full sha256 hex digest")
    for name, count in raw["dimensions"].items():
        if not isinstance(count, int):
            raise ValueError(f"{path}: dimensions.{name} must be a question "
                             f"count, got {type(count).__name__}")

    return ScenarioManifest(
        path=path, scenario_id=raw["scenario_id"], embodiment=raw["embodiment"],
        hash=raw["hash"], questions=raw["questions"],
        dimensions=raw["dimensions"], episodes=raw.get("episodes"),
        camera=raw.get("camera"), counts=raw.get("counts"),
        next_action=raw.get("next_action"), media=raw["media"],
    )


def list_scenarios(data_root: Any) -> list[str]:
    """The scenarios present under a pool root: directories with a manifest.

    Presence means "downloaded", not "valid" — validation is
    ``validate_dataset``'s job, and preflight checks the hashes a suite pins.
    """
    root = Path(data_root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / "manifest.json").exists())
