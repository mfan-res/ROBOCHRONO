#!/usr/bin/env python3
# coding: utf-8
"""The seven evaluation dimensions.

A dimension is a *way of asking* — the same footage is queried seven different
ways. It is not a scoring axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import choice, time_eqa
from .base import load_items

# Headline metric per dimension.
#
# `action_time` reports tIoU@0.5 rather than mean_tIoU. mean_tIoU has a floor around
# 0.13 that a strategy of "always answer: the whole video" reaches without
# watching anything, so it cannot separate a model from that strategy.
# The threshold is 0.5 rather than 0.3 or 0.7: at 0.3 some degenerate strategies
# still pass, and at 0.7 a reasonable answer that trims 20% off each end scores
# zero — while segment boundaries are handoff points between actions, not
# precise onsets, so demanding that precision asks for accuracy the annotation
# does not carry. Expressed as tolerance: for a segment of length D, a pure
# shift of s satisfies (D-s)/(D+s) >= 0.5 when s <= D/3, which for the median
# segment is roughly two seconds.
PRIMARY_METRIC: dict[str, str] = {
    "current_action": "accuracy",
    "next_action": "accuracy",
    "next_action_with_goal": "accuracy",
    "view_match": "accuracy",
    "frame_match": "accuracy",
    "frame_order": "accuracy",
    "action_time": "tIoU@0.5",
}

ALL_DIMENSIONS: tuple[str, ...] = tuple(PRIMARY_METRIC)


# The pool layout rule lives in dataset.loader.scenario_root; these are the
# loader's own helpers, imported rather than redefined, so the rule cannot
# fork between the two halves of the system.
from ..dataset.loader import qa_path, resolve_media  # noqa: E402,F401


def load_for_dimension(data_root: Any, scenario: str, dimension: str) -> list[dict[str, Any]]:
    return load_items(qa_path(data_root, scenario, dimension))


def build(name: str, *, strip_reasoning: bool):
    """Construct a dimension by name.

    ``strip_reasoning`` is required rather than defaulted; it is declared in
    ``configs/protocol.json`` and passed through explicitly.
    """
    if name == "action_time":
        return time_eqa.build()
    if name in choice.SPECS:
        return choice.build(name, strip_reasoning=strip_reasoning)
    raise ValueError(f"unknown dimension {name!r}; known: {list(ALL_DIMENSIONS)}")


# --------------------------------------------------------------------------
# Degenerate floors and execution faults
# --------------------------------------------------------------------------


def floor_breach(dimension: str, summary: dict[str, Any],
                 floors: dict[str, Any]) -> str | None:
    """Report whether a score is at or below what a degenerate strategy reaches.

    A degenerate floor is the score obtainable **without looking at the video**:
    guessing uniformly for multiple choice, or answering "the whole clip" for
    temporal grounding. It is a fault detector, not a pass mark. A score below
    it says something is wrong, but not what — the model, the parsing, or the
    question measuring something other than intended.

    ``floors`` comes from ``configs/protocol.json``.

    Both checks require that questions were actually answered. A run where
    nothing was answered is a different problem — missing media or unparseable
    output — and is reported by ``answered`` and ``parse_failure_rate``. Folding
    it in here would drain this signal of meaning.
    """
    if not summary.get("answered"):
        return None

    floor = floors.get("interval" if dimension == "action_time" else "choice")
    if floor:
        value = summary.get(floor["metric"])
        if isinstance(value, (int, float)) and value < floor["value"]:
            return (f"below {floor['label']} "
                    f"({floor['metric']} {value:.3g} < {floor['value']:g})")

    # A headline metric of exactly zero is worth flagging separately: degenerate
    # strategies also score zero on tIoU@0.5, so zero does not mean "worse than
    # them", it means "indistinguishable from them".
    primary = PRIMARY_METRIC.get(dimension)
    if primary and summary.get(primary) == 0:
        return (f"{primary} is 0 — indistinguishable from a degenerate strategy "
                f"({summary['answered']} answered)")
    return None


# Above this share of unparseable answers, the run did not execute properly;
# it is not a model that answered badly.
PARSE_FAILURE_FAULT = 0.5
# Above this share of failed calls (OOM, timeouts, API errors), likewise —
# and the flag must say the calls failed, not blame the parser.
CALL_FAILURE_FAULT = 0.5


def execution_fault(summary: dict[str, Any]) -> str | None:
    """Report whether a run failed to execute, as distinct from scoring poorly.

    ``floor_breach`` says "it answered, but no better than guessing".
    This says "it never answered at all", which is a fault rather than a result.
    """
    total = summary.get("total") or 0
    if not total:
        return None
    if summary.get("aborted"):
        return "aborted by the failure circuit breaker"
    errors = summary.get("errors") or 0
    if errors / total >= CALL_FAILURE_FAULT:
        return (f"{errors / total:.0%} of calls failed — the error rows "
                f"record why (out of memory, timeouts, API errors)")
    answered = summary.get("answered") or 0
    if answered == 0:
        return f"nothing was answered ({total} questions)"
    rate = summary.get("parse_failure_rate")
    if isinstance(rate, (int, float)) and rate >= PARSE_FAILURE_FAULT:
        return f"{rate:.0%} of answers could not be parsed (the calls succeeded)"
    return None
