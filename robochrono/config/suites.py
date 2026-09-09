#!/usr/bin/env python3
# coding: utf-8
"""Reading ``configs/suites/*.json`` — a frozen set of scenarios, pinned by hash.

A suite pins exactly which questions a published number covers: each scenario
carries the hash of its model-visible content (computed by
``tools/compute_scenario_hash.py``, the one authoritative implementation), so
"the score on this suite" cannot silently drift when a scenario is revised —
a revision changes the hash, and the mismatch fails loudly in preflight
rather than folding into the table.

``dimensions`` may be ``null``, which means all seven: the common case is a
suite that selects scenarios and asks everything about them, and spelling the
seven out in every file would make a future eighth dimension a silent
inconsistency instead of one edit.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..dimensions import ALL_DIMENSIONS

_HASH = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Suite:
    name: str
    pins: dict[str, str]           # scenario -> sha256 of its content
    dimensions: tuple[str, ...]    # concrete: null in the file expands here

    @property
    def scenarios(self) -> tuple[str, ...]:
        return tuple(sorted(self.pins))


def load_suite(name_or_path: Any, root: Any = "configs/suites") -> Suite:
    path = Path(name_or_path)
    if not path.exists():
        path = Path(root) / f"{name_or_path}.json"
    if not path.exists():
        available = sorted(p.stem for p in Path(root).glob("*.json"))
        raise FileNotFoundError(
            f"no suite named {name_or_path!r} under {root} — pass the suite "
            f"name without an extension; available: {available}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    for k in ("name", "scenarios", "dimensions"):
        if k not in raw:
            raise ValueError(f"{path} is missing {k}")

    pins = raw["scenarios"]
    if not isinstance(pins, dict) or not pins:
        raise ValueError(f"{path}: scenarios must map scenario ids to hashes")
    bad = [s for s, h in pins.items()
           if not (isinstance(h, str) and _HASH.match(h))]
    if bad:
        raise ValueError(f"{path}: not a full sha256 hex hash for: {sorted(bad)}")

    dims = raw["dimensions"]
    if dims is None:
        dimensions = ALL_DIMENSIONS
    else:
        unknown = [d for d in dims if d not in ALL_DIMENSIONS]
        if unknown:
            raise ValueError(f"{path}: unknown dimensions {unknown}; "
                             f"known: {list(ALL_DIMENSIONS)}")
        if len(set(dims)) != len(dims):
            raise ValueError(f"{path}: dimensions listed twice")
        dimensions = tuple(dims)

    return Suite(name=raw["name"], pins=dict(pins), dimensions=dimensions)


def pins_digest(pins: dict[str, str]) -> str:
    """The ordered digest of a suite's data identity.

    This is the fingerprint's dataset component: it moves exactly when the
    pinned content moves — a scenario added, removed, or revised — and stays
    put across machines, paths and label changes.
    """
    payload = json.dumps(sorted(pins.items())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]
