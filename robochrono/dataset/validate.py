#!/usr/bin/env python3
# coding: utf-8
"""Validating that the pool on disk delivers what its manifests declare.

Everything downstream — suites, run identity, result merging — trusts the
scenario manifests, and nothing enforces that trust by itself: a question
that fails to render, a media path that resolves to nothing, or a count that
drifted would each surface mid-evaluation at best. This sweep loads every
question the way the evaluation does — through ``load_questions``, rendered —
so what it validates is what actually runs.

The content hash and the media inventory are re-verified through
``tools/compute_scenario_hash.py``, the same authoritative implementation
that wrote the manifests; a second implementation here would drift from the
first exactly where it matters. One sweep serves both
``robochrono validate-data`` and the test suite, for the same reason.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .loader import load_questions, media_paths, resolve_media
from .manifest import ScenarioManifest, list_scenarios, load_scenario_manifest
from .render import load_question_bank

HASH_TOOL = Path("tools/compute_scenario_hash.py")


@dataclass
class ValidationReport:
    manifests: dict[str, ScenarioManifest] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    total: int = 0
    dimension_counts: Counter = field(default_factory=Counter)
    scenario_counts: Counter = field(default_factory=Counter)
    media_references: int = 0
    media_missing: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems


def _check_question(q: dict[str, Any], dimension: str, problems: list[str]) -> None:
    if not q.get("question", "").strip():
        problems.append(f"{q.get('id')}: empty question")
    options = q.get("options") or []
    if options:
        letters = [o.get("id") for o in options]
        if letters != [chr(ord("A") + i) for i in range(len(options))]:
            problems.append(f"{q.get('id')}: option letters {letters}")
        if q.get("answer") not in letters:
            problems.append(f"{q.get('id')}: answer {q.get('answer')!r} not an option")
        for o in options:
            if not o.get("text") and not o.get("image_path"):
                problems.append(f"{q.get('id')}: option {o.get('id')} has neither "
                                f"text nor image")
    if dimension == "action_time":
        sec = q.get("answer_seconds") or {}
        if options:
            problems.append(f"{q.get('id')}: action_time question has options")
        if not (isinstance(sec.get("start"), (int, float))
                and isinstance(sec.get("end"), (int, float))
                and sec["start"] < sec["end"]):
            problems.append(f"{q.get('id')}: answer_seconds {sec!r}")
    elif len(options) != 4:
        problems.append(f"{q.get('id')}: {len(options)} options")


def _verify_hash(data_root: Path, scenario: str, problems: list[str]) -> None:
    """Content hash and media inventory, through the authoritative tool."""
    if not HASH_TOOL.exists():
        problems.append(f"{scenario}: {HASH_TOOL} not found — run from the "
                        f"repository root to verify hashes")
        return
    proc = subprocess.run(
        [sys.executable, str(HASH_TOOL), str(Path(data_root) / scenario),
         "--verify"], capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        problems.append(f"{scenario}: {detail[-1] if detail else 'hash verify failed'}")


def validate_dataset(data_root: Any) -> ValidationReport:
    report = ValidationReport()
    data_root = Path(data_root)
    scenarios = list_scenarios(data_root)
    if not scenarios:
        report.problems.append(
            f"no scenario directories with manifest.json under {data_root} — "
            f"the dataset is not downloaded, or the path is wrong")
        return report

    try:
        bank = load_question_bank(data_root, scenarios)
    except Exception as exc:  # noqa: BLE001 — any load failure is the finding
        report.problems.append(str(exc))
        return report

    for scenario in scenarios:
        try:
            manifest = load_scenario_manifest(data_root, scenario)
        except Exception as exc:  # noqa: BLE001
            report.problems.append(f"{scenario}: {exc}")
            continue
        report.manifests[scenario] = manifest
        _verify_hash(data_root, scenario, report.problems)

        media_refs: set[str] = set()
        for dimension in manifest.dimension_names():
            try:
                questions = load_questions(data_root, scenario, dimension, bank=bank)
            except Exception as exc:  # noqa: BLE001
                report.problems.append(f"{scenario}/{dimension}: {exc!r}")
                continue
            for q in questions:
                _check_question(q, dimension, report.problems)
                media_refs.update(media_paths(q))
            if len(questions) != manifest.dimensions[dimension]:
                report.problems.append(
                    f"{scenario}/{dimension}: {len(questions)} loaded vs "
                    f"{manifest.dimensions[dimension]} declared")
            report.dimension_counts[dimension] += len(questions)
            report.scenario_counts[scenario] += len(questions)
            report.total += len(questions)

        if report.scenario_counts[scenario] != manifest.questions:
            report.problems.append(
                f"{scenario}: {report.scenario_counts[scenario]} questions "
                f"loaded vs {manifest.questions} declared")

        report.media_references += len(media_refs)
        missing = sorted(p for p in media_refs
                         if not resolve_media(data_root, scenario, p).exists())
        report.media_missing += len(missing)
        for path in missing[:5]:
            report.problems.append(f"{scenario}: missing media: {path}")
        if len(missing) > 5:
            report.problems.append(f"{scenario}: ... and {len(missing) - 5} "
                                   f"more missing media files")
        # The manifest inventories the media that ships; the questions
        # reference some set of files. Equal (with none missing) means nothing
        # in the download is dead weight and nothing referenced is absent.
        if len(media_refs) != manifest.media["files"]:
            report.problems.append(
                f"{scenario}: {len(media_refs)} unique media references vs "
                f"{manifest.media['files']} files declared")
    return report


def format_report(report: ValidationReport) -> str:
    lines = []
    if report.manifests:
        lines.append(f"scenario pool: {len(report.manifests)} scenarios")
        lines.append(f"  {report.total} questions across "
                     f"{len(report.dimension_counts)} dimensions")
        lines.append(f"  {report.media_references} media references, "
                     f"{report.media_missing} missing")
    if report.ok:
        lines.append("all checks passed")
    else:
        lines.append(f"{len(report.problems)} problem(s):")
        lines.extend(f"  {p}" for p in report.problems[:50])
        if len(report.problems) > 50:
            lines.append(f"  ... and {len(report.problems) - 50} more")
    return "\n".join(lines)
