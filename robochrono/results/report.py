#!/usr/bin/env python3
# coding: utf-8
"""Turning summaries into one comparison table — when they may be compared.

Every summary this module aggregates was produced against pinned scenario
content, recorded per scenario in its run's ``run.json``. Aggregation is
where that provenance either gets enforced or silently lost, so the rule
lives here: two runs merge exactly when every scenario they share carries
the same content hash — a subset run combines with the full run it came
from, and a revised scenario under the same name is **refused**, not
footnoted. A table quietly mixing two versions looks exactly like a table
that means something.

Two kinds of flag ride along with the scores, pointing at two different
places to go digging:

- ``fault`` (✗) — the run did not execute properly; the number in that cell is
  not a score. Go look at the framework: prompts, batching, parsing, media.
- ``floor`` (⚠) — it executed, and scored no better than a strategy that never
  watches the video. Go look at the model, or at the questions.

Collapsing the two into one mark makes a 100%-errored run indistinguishable
from a model that honestly scored zero.
"""

from __future__ import annotations

import csv
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.suites import pins_digest
from ..dimensions import ALL_DIMENSIONS, PRIMARY_METRIC, execution_fault, floor_breach
from .runid import read_run_record


# -- collection -------------------------------------------------------------

@dataclass(frozen=True)
class Report:
    rows: list[dict[str, Any]]
    runs: list[dict[str, Any]]     # one run.json record per run directory
    dataset: dict[str, Any]        # the single dataset all rows share


def collect(run_paths: list[Any], floors: dict[str, Any]) -> Report:
    """Gather every summary under the given run directories.

    ``floors`` is ``degenerate_floor`` from the protocol; judging scores
    against a floor is an evaluation choice, so it arrives as an argument
    rather than being read from disk here.
    """
    records = [read_run_record(p) for p in run_paths]

    # Runs combine exactly when every scenario they share carries identical
    # content. Comparing per scenario rather than whole-identity lets a
    # subset run merge with the full run it came from, while a revised
    # scenario under the same name is still refused.
    seen: dict[str, tuple[str, Any]] = {}
    for record in records:
        pins = (record.get("dataset") or {}).get("scenarios") or {}
        for scenario, pin in pins.items():
            if scenario in seen and seen[scenario][0] != pin:
                raise ValueError(
                    f"refusing to merge: {scenario} has different content in "
                    f"{seen[scenario][1]} and {record.get('run_id')} — a "
                    f"single table would present the two as comparable")
            seen.setdefault(scenario, (pin, record.get("run_id")))

    rows: list[dict[str, Any]] = []
    for run_path, record in zip(run_paths, records):
        for summary_path in sorted(Path(run_path).rglob("*.summary.json")):
            dimension = summary_path.name[: -len(".summary.json")]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metric = PRIMARY_METRIC.get(dimension, "accuracy")
            rows.append({
                "run_id": record.get("run_id"),
                "model": summary_path.parents[1].name,
                "scenario": summary_path.parents[0].name,
                "dimension": dimension,
                "metric": metric,
                "value": summary.get(metric),
                "total": summary.get("total"),
                "answered": summary.get("answered"),
                "errors": summary.get("errors"),
                "parse_failure_rate": summary.get("parse_failure_rate"),
                "aborted": summary.get("aborted"),
                "floor": floor_breach(dimension, summary, floors),
                "fault": execution_fault(summary),
            })
    union = {scenario: pin for scenario, (pin, _) in sorted(seen.items())}
    return Report(rows=rows, runs=records,
                  dataset={"digest": pins_digest(union), "scenarios": union})


# -- markdown ---------------------------------------------------------------

def to_markdown(report: Report) -> str:
    """One model × dimension table per scenario, then the flags spelled out."""
    rows = report.rows
    out: list[str] = ["# Results", ""]

    dataset = report.dataset
    out.append(f"Dataset: {len(dataset.get('scenarios') or {})} scenario(s), "
               f"pins digest `{dataset.get('digest')}`.")
    out.append("Runs: " + ", ".join(f"`{r.get('run_id')}`" for r in report.runs) + ".")
    protocols = {(r.get("protocol") or {}).get("fingerprint") for r in report.runs}
    if len(protocols) > 1:
        out.append("")
        out.append("⚠ **These runs used different protocols** — the scores were "
                   "produced under different rules and are not directly comparable.")
    out.append("")

    if not rows:
        out.append("No summaries found.")
        return "\n".join(out)

    for scenario in sorted({r["scenario"] for r in rows}):
        subset = [r for r in rows if r["scenario"] == scenario]
        dimensions = [d for d in ALL_DIMENSIONS if any(x["dimension"] == d for x in subset)]
        models = sorted({r["model"] for r in subset})
        index = {(r["model"], r["dimension"]): r for r in subset}

        out.append(f"### {scenario}\n")
        out.append("| model | " + " | ".join(dimensions) + " |")
        out.append("| --- | " + " | ".join("---:" for _ in dimensions) + " |")
        for model in models:
            cells = []
            for dimension in dimensions:
                row = index.get((model, dimension))
                if row is None:
                    cells.append("—")
                elif not row.get("total"):
                    # The dataset asked nothing here (a partial scenario);
                    # 0.00 would read as a score, and it is not one.
                    cells.append("n/a")
                elif row.get("value") is None:
                    cells.append("n/a")
                else:
                    mark = "✗" if row.get("fault") else ""
                    mark += "⚠" if row.get("floor") else ""
                    cells.append(f"{row['value']:.4g}{mark}")
            out.append(f"| {model} | " + " | ".join(cells) + " |")
        out.append("")

    out.append("Headline metrics: accuracy for the choice dimensions, tIoU@0.5 for "
               "`action_time` — mean tIoU has a ~0.13 floor that answering "
               "\"the whole video\" reaches without watching anything.")

    empty = sum(1 for r in rows if not r.get("total"))
    if empty:
        out.append("")
        out.append(f"{empty} cell(s) read `n/a`: the dataset asks no questions "
                   "there — a dimension can be empty for a scenario (a single "
                   "camera rules out `view_match`; too few segments per episode "
                   "starve the others) — so there was nothing to run. Not a "
                   "gap in coverage.")

    # Faults come first: they say "that cell is not a score", and a reader has
    # to know which cells not to compare before comparing any.
    faults = [r for r in rows if r.get("fault")]
    if faults:
        out.append("")
        out.append(f"✗ **{len(faults)} run(s) did not execute properly:**")
        out.append("")
        out.append("| model | scenario | dimension | answered | errors | what happened |")
        out.append("| --- | --- | --- | ---: | ---: | --- |")
        for r in sorted(faults, key=lambda x: (x["model"], x["scenario"], x["dimension"])):
            out.append(f"| {r['model']} | {r['scenario']} | {r['dimension']} | "
                       f"{r.get('answered')}/{r.get('total')} | {r.get('errors')} | {r['fault']} |")
        out.append("")
        out.append("These cells are \"not measured\", not \"scored zero\". Fix the "
                   "framework before comparing them across models.")

    breaches = [r for r in rows if r.get("floor")]
    if breaches:
        out.append("")
        out.append(f"⚠ **{len(breaches)} cell(s) at or below the degenerate floor:**")
        out.append("")
        out.append("| model | scenario | dimension | detail |")
        out.append("| --- | --- | --- | --- |")
        for r in sorted(breaches, key=lambda x: (x["model"], x["scenario"], x["dimension"])):
            out.append(f"| {r['model']} | {r['scenario']} | {r['dimension']} | {r['floor']} |")
        out.append("")
        out.append("Below the floor does not necessarily mean the model is weak — "
                   "wrong units, a parser mismatch, or a question measuring "
                   "something else look exactly the same. Check these before "
                   "reading anything else.")

    out.extend(_execution_settings(report))
    return "\n".join(out) + "\n"


def _execution_settings(report: Report) -> list[str]:
    """What each model ran under. Read before comparing across models.

    Generation settings — the thinking toggle above all — are part of what is
    being measured, and their effect is large and dimension-dependent. Two
    models under different settings are different experiments sharing a table.
    """
    settings: dict[str, set[str]] = {}
    for record in report.runs:
        for model, gen in (record.get("generation_by_model") or {}).items():
            shown = json.dumps(gen, sort_keys=True) if gen else "(not recorded)"
            settings.setdefault(model, set()).add(shown)
    if not settings:
        return []

    out = ["", "### Execution settings", "",
           "Generation settings are part of the experiment; models under "
           "different settings are not directly comparable.", "",
           "| model | generation |", "| --- | --- |"]
    for model in sorted(settings):
        for shown in sorted(settings[model]):
            out.append(f"| {model} | `{shown}` |")
    mixed = sorted(m for m, v in settings.items() if len(v) > 1)
    if mixed:
        out.append("")
        out.append(f"⚠ **{', '.join(mixed)} appear(s) under more than one setting** — "
                   "those rows are different experiments, not one.")
    return out


# -- csv --------------------------------------------------------------------

_CSV_FIELDS = ["run_id", "model", "scenario", "dimension", "metric", "value",
               "total", "answered", "errors", "parse_failure_rate", "aborted",
               "floor", "fault"]


def to_csv(report: Report, path: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in report.rows:
            writer.writerow({k: row.get(k) for k in _CSV_FIELDS})


# -- packing ----------------------------------------------------------------

def pack(run_path: Any, output: Any, *, full: bool = False) -> dict[str, Any]:
    """Bundle one run for delivery.

    The default bundle — ``run.json``, the summaries, the reports — is
    kilobytes, and is what a leaderboard submission or a paper appendix needs.
    ``full`` adds the per-question ``.jsonl``; those are what every diagnosis
    runs on, so keep them locally even when they are not shipped. Backups
    (``.jsonl.bak``) are never packed.
    """
    run_path, output = Path(run_path), Path(output)
    patterns = ["run.json", "report.md", "report.csv", "**/*.summary.json"]
    if full:
        patterns.append("**/*.jsonl")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(p for p in sorted(run_path.glob(pattern)) if p.is_file())

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=str(Path(run_path.name) / path.relative_to(run_path)))
    return {"files": len(files), "bytes": output.stat().st_size, "output": str(output)}
