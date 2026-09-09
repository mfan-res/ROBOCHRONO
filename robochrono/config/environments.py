#!/usr/bin/env python3
# coding: utf-8
"""Reading ``configs/environments.json`` — which Python environment each model needs.

More than one environment is required, and this is an upstream constraint rather
than a preference: some model families only load under transformers 4.x, others
only under 5.x, and the two requirements are mutually exclusive.

This module imports no inference libraries. The orchestrator must be able to
start from any environment and dispatch each model to the interpreter it needs;
``tests/test_orchestrator_is_light.py`` enforces that.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Environment:
    name: str
    python: str
    extra: str
    transformers: str


def load_environments(path: Any = "configs/environments.json") -> dict[str, Environment]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        name: Environment(name=name, python=e["python"], extra=e["extra"],
                          transformers=e["transformers"])
        for name, e in raw["envs"].items()
    }


def satisfies(spec: str | None, actual: str) -> bool | None:
    """Check a declared requirement against the version an environment provides.

    Returns ``None`` when the model's documentation states no requirement.

    A pinned requirement such as ``==4.57.0`` is treated as a lower bound: the
    environments ship a later patch release that satisfies every model mapped to
    them. The environments themselves are declared in
    ``configs/environments.json``.
    """
    if not spec:
        return None
    m = re.fullmatch(r"(==|>=)\s*([\d.]+)", str(spec).strip())
    if not m:
        return None
    want = tuple(int(x) for x in m.group(2).split("."))
    got = tuple(int(x) for x in actual.split("."))
    want = (want + (0,) * len(got))[: len(got)]
    return got >= want
