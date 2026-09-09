#!/usr/bin/env python3
# coding: utf-8
"""Reading ``configs/models/**`` — one file per model.

Every value taken from a model's official documentation carries a ``source``
grade recording how strong that evidence is. The grade is **per field, not per
block**: a model card may give an explicit generation budget while saying
nothing about library versions, and a single block-level grade cannot express
that.

``source`` is required. "The official documentation does not say" is itself a
decision and must be written as ``source: "none"``. What is validated is that
someone made the call, not whether the value is correct — there is no
machine-checkable standard for the latter.

Grade definitions are in ``configs/README.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Graded:
    """A value together with the grade of evidence behind it."""
    value: Any
    source: str
    note: str = ""

    @property
    def is_official(self) -> bool:
        return self.source in ("L1", "L2", "L3", "L4")


@dataclass(frozen=True)
class ModelConfig:
    slug: str
    name: str
    adapter: str
    weights: str
    environment: str
    kind: str                               # local | api
    official: dict[str, Graded] = field(default_factory=dict)
    generation: dict[str, Graded] = field(default_factory=dict)
    media: dict[str, Any] = field(default_factory=dict)
    # API models only: endpoint, served model id, auth env var, dialect quirks.
    # Keys never appear here — `key_env` names the environment variable.
    api: dict[str, Any] = field(default_factory=dict)
    # What it takes to run: gpus_per_worker for models too large for one card.
    # Declared on the model because it is a property of the weights, not of
    # the machine — a global flag would run every small model at the big
    # model's parallelism, or the big model into a certain OOM.
    resources: dict[str, Any] = field(default_factory=dict)

    def max_new_tokens(self) -> int | None:
        g = self.generation.get("max_new_tokens")
        return None if g is None else int(g.value)

    def thinking(self, protocol_default: str) -> Graded:
        """Fall back to the protocol value when the model declares nothing.

        That value is written explicitly in ``protocol.json``; it is not a
        default hidden in code.
        """
        return self.generation.get("thinking", Graded(protocol_default, "protocol"))


def _graded(raw: Any, where: str) -> Graded:
    if not isinstance(raw, dict) or "source" not in raw:
        raise ValueError(
            f"{where} has no source. Even \"the official docs do not say\" must be "
            f"written explicitly as source: \"none\". See configs/README.md.")
    return Graded(value=raw.get("value"), source=raw["source"], note=raw.get("note", ""))


def load_model(path: Any) -> ModelConfig:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    for k in ("name", "adapter", "weights", "environment"):
        if k not in raw:
            raise ValueError(f"{path} is missing {k}")
    official = {k: _graded(v, f"{path}:official.{k}")
                for k, v in (raw.get("official") or {}).items()
                if not k.startswith("_") and k != "url"}
    generation = {k: _graded(v, f"{path}:generation.{k}")
                  for k, v in (raw.get("generation") or {}).items()
                  if not k.startswith("_")}
    return ModelConfig(
        slug=path.stem, name=raw["name"], adapter=raw["adapter"],
        weights=raw["weights"], environment=raw["environment"],
        kind=path.parent.name, official=official, generation=generation,
        media={k: v for k, v in (raw.get("media") or {}).items() if not k.startswith("_")},
        api={k: v for k, v in (raw.get("api") or {}).items() if not k.startswith("_")},
        resources={k: v for k, v in (raw.get("resources") or {}).items()
                   if not k.startswith("_")},
    )


def load_models(root: Any = "configs/models") -> dict[str, ModelConfig]:
    """The files in the directory *are* the model list. There is no separate roster."""
    out = {}
    for p in sorted(Path(root).rglob("*.json")):
        m = load_model(p)
        if m.name in out:
            raise ValueError(f"duplicate model name {m.name!r} (from {p})")
        out[m.name] = m
    return out
