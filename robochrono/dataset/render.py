#!/usr/bin/env python3
# coding: utf-8
"""Rendering a question from the ids it stores.

A question file records which action a question is about, not the sentence that
describes it. Everything needed to turn one into the other sits beside it, in
each scenario's own qa/ directory:

    <scenario>/qa/subtasks.json    this scenario's actions: id, description
    <scenario>/qa/scenarios.json   what this scenario is doing overall
    <scenario>/qa/dimensions.json  dimension -> how it words its question
                                   (identical in every scenario, checked on load)

The reason for storing ids is not brevity. Spelling the description out
would copy it into the stem, into the options array, and again into whichever
option is correct — several copies per question across the whole dataset. Any
rewording would then have to hit every copy, and a missed one would leave a
dataset that still loads, still scores, and disagrees with itself. Storing the
id makes that unrepresentable.

Action ids are unique within a scenario, not across the dataset — two
scenarios that both pick up a pen both call it ``pick_pen``. Every question
names only actions from its own scenario, so lookups here are scoped to the
question's scenario. The scoping is also a check: a question that referenced
another scenario's action would fail loudly instead of silently borrowing
whatever text a global table happened to hold.

The reason all three files live inside ``qa/`` is that a question bank has to
be readable on its own. The manifest fingerprint answers "is this the same set
of questions", and it hashes questions as they are asked, so it has to be
computable from the dataset alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QuestionBank:
    """Everything a stored question is rendered from."""

    subtask_text: dict[str, dict[str, str]]   # scenario -> action id -> "Pick the pen."
    scenario_goal: dict[str, str]             # scenario -> "putting a pen into its box"
    templates: dict[str, str]                 # dimension -> question template


def load_question_bank(data_root: Any,
                       scenarios: list[str] | None = None) -> QuestionBank:
    """One bank over the pool: each scenario's qa/ carries its own render
    files, merged here. Every scenario ships the same dimensions.json — a
    copy that disagreed would mean two scenarios asking differently worded
    questions under one template name, so disagreement is a hard error, not
    a merge."""
    root = Path(data_root)
    if scenarios is None:
        scenarios = sorted(p.name for p in root.iterdir()
                           if p.is_dir() and (p / "qa").is_dir())
    if not scenarios:
        raise FileNotFoundError(
            f"no scenario directories with qa/ under {root} — the dataset is "
            f"not downloaded, or the data root is wrong")

    subtask_text: dict[str, dict[str, str]] = {}
    scenario_goal: dict[str, str] = {}
    templates: dict[str, str] | None = None
    for scenario in scenarios:
        qa = root / scenario / "qa"
        read = lambda name: json.loads((qa / name).read_text(encoding="utf-8"))
        for name, actions in read("subtasks.json").items():
            subtask_text[name] = {s["id"]: s["text"] for s in actions}
        for name, entry in read("scenarios.json").items():
            scenario_goal[name] = entry["goal"]
        here = {k: v for k, v in read("dimensions.json")["templates"].items()
                if not k.startswith("_")}
        if templates is None:
            templates = here
        elif templates != here:
            raise ValueError(
                f"{qa / 'dimensions.json'} words its questions differently "
                f"from the other scenarios' copies — the pool is inconsistent")
    return QuestionBank(subtask_text=subtask_text, scenario_goal=scenario_goal,
                        templates=templates or {})


def option_text(option: dict[str, Any], scenario: str, bank: QuestionBank) -> str | None:
    """The text of one option, or None where the option is an image.

    Options name what they offer rather than spelling it out: an action by id,
    an ordering as the sequence of image numbers it puts them in.
    """
    if option.get("subtask"):
        return bank.subtask_text[scenario][option["subtask"]]
    if option.get("order"):
        return " -> ".join(f"Image {n}" for n in option["order"])
    return None


def render_options(item: dict[str, Any], scenario: str, bank: QuestionBank) -> None:
    """Fill in each option's text, in place. Image options are left alone."""
    for option in item.get("options") or []:
        text = option_text(option, scenario, bank)
        if text is not None:
            option["text"] = text


def render_question(item: dict[str, Any], dimension: str, scenario: str,
                    bank: QuestionBank) -> str:
    """The question as the model is shown it.

    Call after `render_options`: the option block is built from the rendered
    option text, in the order the options are stored, which is what fixes which
    letter goes with which action.
    """
    template = bank.templates[dimension]
    fields: dict[str, str] = {}

    if "{subtask}" in template:
        # The stem quotes the action inside a sentence of its own, so the full
        # stop that ends the standalone description would land mid-sentence.
        fields["subtask"] = bank.subtask_text[scenario][item["subtask"]].rstrip(".")
    if "{options}" in template:
        fields["options"] = "\n".join(
            f"{o['id']}. {o['text']}" for o in item["options"])
    if "{side}" in template:
        fields["side"] = "left" if item["target_camera"] == "wrist_left" else "right"
    if "{goal}" in template:
        fields["goal"] = bank.scenario_goal[scenario]

    return template.format(**fields)


def render(item: dict[str, Any], dimension: str, scenario: str,
           bank: QuestionBank) -> dict[str, Any]:
    """Materialise `question` and option text on one question, in place."""
    render_options(item, scenario, bank)
    item["question"] = render_question(item, dimension, scenario, bank)
    return item
