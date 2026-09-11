#!/usr/bin/env python3
# coding: utf-8
"""Loading questions and resolving media paths.

The data root is a scenario pool: one self-contained directory per scenario,
``<root>/<scenario>/{manifest.json, qa/, media/}``. Media paths inside a
question file are relative to the scenario directory and keep the scenario
segment they were produced with, so a path reads the same no matter which
file it appears in::

    {"input": {"clip_path": "media/clips/pack_airpods_tianji/file-000@main@000100-000210.mp4"}}

resolves under ``<root>/pack_airpods_tianji/``. Resolution takes the
scenario explicitly rather than parsing it back out of the path — every
caller knows which scenario it is working on, and an explicit argument
cannot be fooled by a path that names another scenario's file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .render import QuestionBank, load_question_bank, render


def scenario_root(data_root: Any, scenario: str) -> Path:
    """The one place the pool layout rule lives: a scenario's files sit under
    ``<data-root>/<scenario>/``. Everything that touches a scenario's qa or
    media — the loaders, the dimension helpers, both execution paths — goes
    through here, so a layout change is one edit instead of a synchronized
    hunt across the codebase."""
    return Path(data_root) / scenario


def qa_path(data_root: Any, scenario: str, dimension: str) -> Path:
    return scenario_root(data_root, scenario) / "qa" / f"{dimension}.json"


def resolve_media(data_root: Any, scenario: str, relative_path: str) -> Path:
    return scenario_root(data_root, scenario) / str(relative_path)


def load_questions(data_root: Any, scenario: str, dimension: str, *,
                   bank: QuestionBank | None = None,
                   keep: set[str] | None = None) -> list[dict[str, Any]]:
    """Load every question for one (scenario, dimension), rendered.

    A stored question names the action it is about; the sentence a model reads
    is produced here, from the question bank beside it. Rendering on load rather
    than at each use means the whole evaluation sees one rendering, produced
    once. Pass `bank` to load it once across many calls.

    ``preflight`` and the evaluation run share this function. If self-check and
    execution load data by different paths, the check validates something other
    than what actually runs — it can warn on data that works, and stay silent
    when something is genuinely wrong.

    ``keep`` restricts the result to questions whose id is listed, and is how a
    run reproduces the question set of an earlier one. It filters here, at the
    single point where questions enter the evaluation, so every caller —
    preflight, the dry run, both execution paths — narrows identically. The
    files on disk are untouched, so scenario hashes and suite pins do not move
    and the run still merges with the full run it is a subset of.
    """
    path = qa_path(data_root, scenario, dimension)
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"{path} must be a list, or contain an `items` list")
    questions = [q for q in items if isinstance(q, dict)]
    if keep is not None:
        questions = [q for q in questions if str(q.get("id")) in keep]
    if bank is None:
        bank = load_question_bank(data_root)
    for question in questions:
        render(question, dimension, scenario, bank)
    return questions


def read_item_ids(path: Any) -> set[str]:
    """Read a question-id list: one id per line, ``#`` comments and blanks
    ignored.

    A plain text file rather than a flag repeated a thousand times, because the
    lists that matter are long and belong in version control next to whatever
    produced them.
    """
    text = Path(path).read_text(encoding="utf-8")
    ids = {line.strip() for line in text.splitlines()
           if line.strip() and not line.lstrip().startswith("#")}
    if not ids:
        raise ValueError(f"{path} lists no question ids")
    return ids


_MEDIA_KEYS = ("clip_path", "video_path", "image_path")


def media_paths(question: dict[str, Any]) -> list[str]:
    """List the media this question actually sends to the model.

    Provenance fields that are never sent are excluded: counting them would
    both overstate download size and produce false "missing file" reports.
    """
    out: list[str] = []
    data = question.get("input") or {}
    for key in _MEDIA_KEYS:
        if data.get(key):
            out.append(str(data[key]))
    for p in data.get("image_paths") or []:
        out.append(str(p))
    for option in question.get("options") or []:
        if isinstance(option, dict) and option.get("image_path"):
            out.append(str(option["image_path"]))
    return out
