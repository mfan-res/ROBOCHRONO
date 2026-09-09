#!/usr/bin/env python3
# coding: utf-8
"""The Dimension protocol.

What differs between dimensions reduces to four hooks: pick the media, build the
prompt, parse the output, score it. Everything else — batching, retries,
resumption, persistence — is handled once by the engine.

``Unit`` is what makes that possible. One Unit is one model call and produces one
or more result rows, which lets a dimension that asks several questions in a
single call sit behind the same interface as one that asks a single question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class Unit:
    """One model call.

    key    identity used for resumption
    items  the questions this call covers
    """

    key: str
    items: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CallContext:
    """Runtime information handed to a dimension, recorded alongside results."""

    frames_used: dict[str, int] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    # Transformations applied to media to fit a request-size budget. Recorded
    # because otherwise there is no way to audit what the model actually saw.
    media_transforms: list[dict[str, Any]] = field(default_factory=list)


class Dimension(Protocol):
    """One evaluation dimension.

    These six members are the entire contract between a dimension and the code
    that runs it. Declaring them as a Protocol — rather than passing dimensions
    around as ``Any`` — means a new dimension that forgets one is reported when
    it is written, not when a run fails partway through.
    """

    name: str

    def units(self, items: list[dict[str, Any]]) -> list[Unit]:
        """Group questions into model calls."""
        ...

    def parts(self, unit: Unit) -> list[dict[str, Any]]:
        """Assemble what is sent to the model: text, image and video parts."""
        ...

    def rows(self, unit: Unit, text: str, ctx: CallContext) -> list[dict[str, Any]]:
        """Parse and score the output, producing one row per question."""
        ...

    def error_rows(self, unit: Unit, error: str) -> list[dict[str, Any]]:
        """Placeholder rows for a failed call, """
        ...

    def summarize(self, rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
        """Aggregate metrics over the rows of one (model, scenario, dimension)."""
        ...


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def load_items(path: Path) -> list[dict[str, Any]]:
    """Read a question file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"Input must be a list or contain an `items` list: {path}")
    return [item for item in items if isinstance(item, dict)]


def one_item_per_unit(items: list[dict[str, Any]]) -> list[Unit]:
    """Default grouping: one question per call."""
    return [Unit(key=str(item.get("id")), items=[item]) for item in items]


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def image_part(path: Any) -> dict[str, Any]:
    return {"type": "image", "path": str(path)}


def video_part(path: Any) -> dict[str, Any]:
    return {"type": "video", "path": str(path)}


def base_row(item: dict[str, Any], prompt: str, text: str | None, ctx: CallContext | None) -> dict[str, Any]:
    """Fields common to every result row.

    The original question is not copied in; rows reference it by ``id``. Copying
    it would multiply the size of every result file for information already on
    disk.

    ``frames_used`` and ``usage`` are recorded because how many frames a model
    actually received cannot be reconstructed afterwards from the score alone.
    """
    row: dict[str, Any] = {
        "id": str(item.get("id")),
        "prompt": prompt,
        "model_output": text,
    }
    if ctx is not None:
        if ctx.frames_used:
            row["frames_used"] = dict(ctx.frames_used)
        if ctx.usage:
            row["usage"] = dict(ctx.usage)
        if ctx.media_transforms:
            row["media_transforms"] = list(ctx.media_transforms)
    return row


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Accuracy grouped by one field, used for the per-action and per-choice breakdowns."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "unknown"), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for name, group in sorted(groups.items()):
        count = len(group)
        out[name] = {
            "total": count,
            "accuracy": sum(bool(r.get("correct")) for r in group) / count if count else 0.0,
        }
    return out
