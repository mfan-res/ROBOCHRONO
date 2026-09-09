#!/usr/bin/env python3
# coding: utf-8
"""Expanding a suite × a model set into the list of work to run.

A full evaluation is models × scenarios × dimensions — over a hundred
combinations per model. Nobody types those; they are expanded from the suite
(which pins scenarios and dimensions) and the model configurations (which are
the model roster — there is no separate plan file).

Three properties matter here:

- **Selection errors are loud.** A misspelled model or dimension raises; the
  silent alternative is an empty expansion that looks exactly like missing
  data.
- **Model-major order.** All of one model's work runs consecutively, so local
  weights load once per model instead of once per combination.
- **Stable sharding.** Machines split the matrix by a content hash of each
  key, so any machine computes the same split — disjoint, and jointly
  complete — without coordination.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.models import ModelConfig
from ..config.suites import Suite
from ..dataset.loader import qa_path


@dataclass(frozen=True)
class RunSpec:
    """One (model, scenario, dimension) — the unit the engine executes."""

    model: str            # model slug
    scenario: str
    dimension: str

    @property
    def key(self) -> str:
        return f"{self.model}__{self.scenario}__{self.dimension}"

    def store_path(self, run_dir: Path) -> Path:
        return Path(run_dir) / self.model / self.scenario / f"{self.dimension}.jsonl"


def shard_of(key: str, shards: int) -> int:
    """Stable hash sharding: the same key lands on the same shard anywhere."""
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % shards


def _validate(chosen: list[str] | None, known: list[str], what: str) -> set[str] | None:
    if not chosen:
        return None
    wanted = set(chosen)
    unknown = wanted - set(known)
    if unknown:
        raise ValueError(f"unknown {what}: {sorted(unknown)}; known: {sorted(known)}")
    return wanted


def expand(
    models: list[ModelConfig],
    suite: Suite,
    data_root: Any,
    *,
    shard: tuple[int, int] | None = None,
    only_kind: str | None = None,
    only_models: list[str] | None = None,
    only_scenarios: list[str] | None = None,
    only_dimensions: list[str] | None = None,
) -> tuple[list[RunSpec], list[tuple[str, str]]]:
    """Expand the matrix. Returns (selected, [(key, reason skipped)]).

    ``shard`` is (index, total) with 1-based index: (1, 4) is the first of
    four machines. Filters narrow the expansion and reject names that do not
    exist — an empty result must mean "nothing to do", never "you typo'd".
    """
    wanted_models = _validate(only_models, [m.slug for m in models], "models")
    wanted_scenarios = _validate(only_scenarios, list(suite.scenarios), "scenarios")
    wanted_dimensions = _validate(only_dimensions, list(suite.dimensions), "dimensions")

    selected: list[RunSpec] = []
    skipped: list[tuple[str, str]] = []
    for model in models:
        if only_kind and model.kind != only_kind:
            continue
        if wanted_models and model.slug not in wanted_models:
            continue
        for scenario in suite.scenarios:
            if wanted_scenarios and scenario not in wanted_scenarios:
                continue
            for dimension in suite.dimensions:
                if wanted_dimensions and dimension not in wanted_dimensions:
                    continue
                spec = RunSpec(model=model.slug, scenario=scenario, dimension=dimension)
                if not qa_path(data_root, scenario, dimension).exists():
                    skipped.append((spec.key, "question file missing"))
                    continue
                if shard is not None:
                    index, total = shard
                    if shard_of(spec.key, total) != index - 1:
                        continue
                selected.append(spec)

    selected.sort(key=lambda s: (s.model, s.scenario, s.dimension))
    return selected, skipped
