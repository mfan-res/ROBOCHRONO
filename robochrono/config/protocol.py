#!/usr/bin/env python3
# coding: utf-8
"""Reading ``configs/protocol.json`` — the evaluation protocol.

Changing the protocol changes the experiment: results produced under a
different protocol are not comparable.

Every field is required. There are no silent fallbacks — a missing field raises
rather than being filled in from a default. A default that quietly applies when
a value is absent is indistinguishable, after the fact, from a value someone
chose on purpose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Floor:
    """A score a degenerate strategy reaches without looking at the video."""
    metric: str
    value: float
    label: str


@dataclass(frozen=True)
class Protocol:
    version: str
    temperature: float
    max_new_tokens: int
    thinking: str
    timeout: int
    max_retries: int
    max_request_bytes: int
    system_prompt: str
    strip_reasoning: bool
    frames_by_dimension: dict[str, dict[str, Any]]
    min_frames: int
    frame_factor: int
    max_frames: int
    floors: dict[str, Floor]

    # -- frame sampling --------------------------------------------------
    def frames_for(self, dimension: str) -> dict[str, Any]:
        spec = self.frames_by_dimension.get(dimension)
        if spec is None:
            raise KeyError(
                f"protocol.frames.by_dimension has no entry for {dimension!r}. "
                f"Adding a dimension requires declaring its frame sampling; "
                f"there is no default.")
        return spec

    def frames_from_fps(self, duration_seconds: float, fps: float) -> int:
        """Convert a sampling rate to a frame count.

        Rounds to a multiple of ``frame_factor`` and clamps to the configured
        bounds, matching the rounding used by ``qwen_vl_utils``.

        **Every adapter calls this one function.** If each adapter derived its
        own frame count, the same declared fps could yield different numbers of
        frames per model — a difference that would not surface as an error, only
        as scores that are quietly not measuring the same thing.
        """
        if fps <= 0 or duration_seconds <= 0:
            return self.min_frames
        n = round(duration_seconds * fps / self.frame_factor) * self.frame_factor
        return int(max(self.min_frames, min(self.max_frames, n)))

    # -- merging with per-model configuration ----------------------------
    def effective_max_new_tokens(self, model_value: int | None) -> int:
        """A model may raise the baseline budget, never lower it.

        Raise-only keeps models that fit comfortably inside the baseline
        unaffected: they never reach the ceiling, so the value they run under is
        the protocol value regardless of what any other model declares.

        Reasoning models are the case that needs more room — their output is
        long enough that a budget sized for direct answers truncates a
        substantial fraction of it, and a truncated chain of thought contains no
        answer to recover.
        """
        if model_value is None:
            return self.max_new_tokens
        return max(self.max_new_tokens, int(model_value))

    def floor_for(self, dimension: str) -> Floor | None:
        return self.floors.get("interval" if dimension == "action_time" else "choice")


_REQ_GEN = ("temperature", "max_new_tokens", "thinking",
            "timeout", "max_retries", "max_request_bytes")


def load_protocol(path: Any = "configs/protocol.json") -> Protocol:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    def need(container: dict, key: str, where: str):
        if key not in container:
            raise ValueError(
                f"{path} is missing {where}{key} — protocol fields are required, "
                f"there is no silent fallback")
        return container[key]

    gen = need(raw, "generation", "")
    for k in _REQ_GEN:
        need(gen, k, "generation.")
    frames = need(raw, "frames", "")
    by_dim = need(frames, "by_dimension", "frames.")
    parsing = need(raw, "parsing", "")
    floors_raw = need(raw, "degenerate_floor", "")

    floors = {}
    for kind, f in floors_raw.items():
        if kind.startswith("_"):
            continue
        floors[kind] = Floor(metric=f["metric"], value=float(f["value"]), label=f["label"])

    return Protocol(
        version=need(raw, "version", ""),
        temperature=float(gen["temperature"]),
        max_new_tokens=int(gen["max_new_tokens"]),
        thinking=str(gen["thinking"]),
        timeout=int(gen["timeout"]),
        max_retries=int(gen["max_retries"]),
        max_request_bytes=int(gen["max_request_bytes"]),
        system_prompt=need(raw, "system_prompt", ""),
        strip_reasoning=bool(need(parsing, "strip_reasoning", "parsing.")),
        frames_by_dimension={k: v for k, v in by_dim.items() if not k.startswith("_")},
        min_frames=int(need(frames, "min_frames", "frames.")),
        frame_factor=int(need(frames, "frame_factor", "frames.")),
        max_frames=int(need(frames, "max_frames", "frames.")),
        floors=floors,
    )
