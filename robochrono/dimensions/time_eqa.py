#!/usr/bin/env python3
# coding: utf-8
"""Temporal grounding: locating when an action happens.

The only dimension that sends a full episode rather than a clip, and the only
one whose answer is an interval rather than a letter — so parsing falls back
through several shapes before giving up.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from .base import CallContext, Unit, one_item_per_unit, base_row, text_part, video_part


# --------------------------------------------------------------------------
# Parsing time values and intervals
# --------------------------------------------------------------------------


def seconds_to_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, rem = divmod(milliseconds, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def parse_time_value(value: str) -> float:
    value = value.strip()
    # A bare number may carry a seconds suffix ("119.0s", "43 sec") — models
    # write units even when the prompt shows none. Only seconds are accepted:
    # a minutes suffix would silently mean a 60x error, better left loud.
    value = re.sub(r"(?i)\s*(?:s|sec|secs|seconds)$", "", value)
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return float(value)
    match = re.fullmatch(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", value)
    if not match:
        raise ValueError(f"Cannot parse time value: {value!r}")
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


_NUM = r"\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?"


def _normalize_dashes(text: str) -> str:
    # Range separators a model may emit instead of a plain hyphen, including
    # the CJK word for "to".
    return text.replace("–", "-").replace("—", "-").replace("到", "-")


def parse_interval_text(text: str) -> tuple[float, float]:
    cleaned = _normalize_dashes(text.strip())
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            if "start" in data and "end" in data:
                return parse_time_value(str(data["start"])), parse_time_value(str(data["end"]))
            if "start_time" in data and "end_time" in data:
                return parse_time_value(str(data["start_time"])), parse_time_value(str(data["end_time"]))
            if "answer" in data:
                return parse_interval_text(str(data["answer"]))
    except json.JSONDecodeError:
        pass
    match = re.search(rf"({_NUM})\s*(?:-|,|to)\s*({_NUM})", cleaned, flags=re.I)
    if not match:
        raise ValueError(f"Cannot parse interval from model output: {text!r}")
    return parse_time_value(match.group(1)), parse_time_value(match.group(2))


def parse_interval_row(row: Any) -> tuple[float, float]:
    if isinstance(row, dict):
        if "start" in row and "end" in row:
            return parse_time_value(str(row["start"])), parse_time_value(str(row["end"]))
        if "start_time" in row and "end_time" in row:
            return parse_time_value(str(row["start_time"])), parse_time_value(str(row["end_time"]))
        for key in ("answer", "interval", "timestamp"):
            if key in row:
                value = row[key]
                # The field may hold the interval as an object rather than a
                # string ({"answer": {"start": 1.2, "end": 3.4}}). Stringifying
                # that yields a Python repr the interval pattern cannot match,
                # so the answer would be lost although it is stated exactly.
                if isinstance(value, dict):
                    return parse_interval_row(value)
                return parse_interval_text(str(value))
    if isinstance(row, str):
        return parse_interval_text(row)
    raise ValueError(f"Cannot parse interval row: {row!r}")


def row_id(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    for key in ("id", "question_id", "item_id"):
        if row.get(key) is not None:
            return str(row[key])
    return None


def rows_from_multi_json(data: Any) -> list[Any] | dict[str, Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("answers", "results", "items", "predictions"):
            if isinstance(data.get(key), list):
                return data[key]
        # A flat single-answer object: {"index":1,"id":...,"start":...,"end":...}.
        # One question per call makes this a natural shape to return, and the
        # {"answers":[...]} envelope is not required.
        #
        # Failing to recognise it would not surface as a format error but as an
        # empty result: the lookup below treats the id as a key, whereas here it
        # is a value, so nothing matches and the row is reported as a missing
        # answer — indistinguishable from a model that said nothing.
        #
        # Wrapping it in a single-element list routes it through the per-row
        # branch below, which already tolerates answers addressed by index
        # rather than by id.
        if any(k in data for k in ("start", "end", "answer", "interval")):
            return [data]
        return data
    raise ValueError(f"Cannot parse multi-answer JSON: {data!r}")


def _video_seconds(items: list[dict[str, Any]]) -> float | None:
    """Duration in seconds of the video these questions refer to.

    Read from ``input.video_seconds``. Taking the first question is sufficient:
    every question in a call refers to the same video.
    Returns None when absent, in which case the prompt omits the sentence
    stating the duration rather than inventing a number.
    """
    for item in items:
        data = item.get("input")
        if isinstance(data, dict) and data.get("video_seconds"):
            try:
                return float(data["video_seconds"])
            except (TypeError, ValueError):
                return None
    return None


# One ffprobe per distinct video file, not per question: an episode carries
# many questions, and workers keep a dimension instance across units.
_DURATION_CACHE: dict[str, float] = {}


def ensure_video_seconds(items: list[dict[str, Any]], data_root: Any) -> float:
    """The duration the prompt states, measured off the media when the data
    does not carry it.

    ``input.video_seconds`` is the duration's home, but no shipped question
    file has it (the field was lost in a pipeline rewrite), so the fallback
    measures the episode with ffprobe and writes the value back into the
    in-memory items — ``rows`` rebuilds the prompt later and must state the
    same number. The question files on disk are never touched, so scenario
    hashes do not move.

    Failure is loud by design: a missing video or an unreadable duration
    raises, which the engine records as an error row. Silently omitting the
    sentence would put some questions back in the no-duration condition and
    make the run internally inconsistent.
    """
    declared = _video_seconds(items)
    if declared is not None:
        return declared
    from pathlib import Path

    from ..media_prep import video_duration

    relative = video_path_for_item(items[0])
    path = Path(relative)
    if not path.is_absolute():
        if data_root is None:
            raise ValueError(
                f"question {items[0].get('id')}: no input.video_seconds and no "
                f"data_root to resolve {relative!r} against — cannot state the "
                f"video duration the prompt requires")
        path = Path(data_root) / path
    key = str(path)
    seconds = _DURATION_CACHE.get(key)
    if seconds is None:
        if not path.exists():
            raise ValueError(
                f"question {items[0].get('id')}: video not found at {path} — "
                f"cannot measure the duration the prompt requires")
        measured = video_duration(path)
        if not measured or measured <= 0:
            raise ValueError(
                f"question {items[0].get('id')}: ffprobe could not read a "
                f"duration from {path} — cannot state the duration the "
                f"prompt requires")
        seconds = _DURATION_CACHE[key] = round(float(measured), 3)
    for item in items:
        data = item.get("input")
        if isinstance(data, dict):
            data["video_seconds"] = seconds
    return seconds


def parse_multi_interval_text(text: str, question_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Split a multi-answer response back into per-question predictions.

    Four tiers: parse the whole response as JSON, then look for JSON embedded in
    prose, then read an interval stated in prose, then fail.

    The second tier exists because a model may emit perfectly valid JSON
    preceded by prose ("Got it, let's find when the robot picks up the case"),
    so that the response as a whole is not JSON and a whole-response parse
    fails.

    That is a parsing gap, not a wrong answer. Scoring it as wrong would measure
    whether a model speaks in the requested format rather than whether it can
    locate an action.

    The tier is monotonic: any input that the whole-response parse accepts never
    reaches it, so it can only turn a failure into a parse, never change a
    result that already parsed.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    predictions: dict[str, dict[str, Any]] = {}
    try:
        rows = rows_from_multi_json(json.loads(cleaned))
        if isinstance(rows, dict):
            for item_id in question_ids:
                if item_id not in rows:
                    continue
                start, end = parse_interval_row(rows[item_id])
                predictions[item_id] = {"pred_start": start, "pred_end": end, "model_answer": rows[item_id]}
            return predictions
        for index, row in enumerate(rows):
            item_id = row_id(row)
            # The prompt addresses questions by `index`, so that is the
            # expected key. `row_id` is tried first only because a model may
            # label its answer `id` while still numbering it, and positional
            # order is the last resort when it does neither.
            if item_id not in question_ids:
                hint = row.get("index") if isinstance(row, dict) else None
                if hint is None and isinstance(item_id, str) and item_id.isdigit():
                    hint = item_id
                try:
                    pos = int(hint) - 1                 # the prompt numbers from 1
                except (TypeError, ValueError):
                    pos = index
                item_id = question_ids[pos] if 0 <= pos < len(question_ids) else None
            if item_id not in question_ids:
                continue
            try:
                start, end = parse_interval_row(row)
            except ValueError:
                # One unreadable row must not discard the rest of the response,
                # nor the tiers below.
                continue
            predictions[item_id] = {"pred_start": start, "pred_end": end, "model_answer": row}
        # Only return here if something matched. Valid JSON that answers none of
        # the asked questions — a model numbering its single answer `index: 10`,
        # say — otherwise ends the parse with an empty result, and the prose tier
        # that exists precisely for that case never runs. The tiers are meant to
        # be monotonic: each one only ever adds.
        if predictions:
            return predictions
    except json.JSONDecodeError:
        pass

    # -- tier 2: the response is not JSON, but contains a JSON object -------
    if not predictions:
        from ..parsing import iter_json_objects
        for data in iter_json_objects(cleaned):
            try:
                rows = rows_from_multi_json(data)
            except ValueError:
                continue
            found: dict[str, dict[str, Any]] = {}
            pairs = rows.items() if isinstance(rows, dict) else list(enumerate(rows))
            for key, row in pairs:
                item_id = key if isinstance(key, str) else row_id(row)
                if item_id not in question_ids:
                    # Same fallback as tier 1: answers may be addressed by index
                    if isinstance(key, int) and 0 <= key < len(question_ids):
                        item_id = question_ids[key]
                    else:
                        continue
                try:
                    start, end = parse_interval_row(row)
                except ValueError:
                    # Same rule as tier 1: skip the row, keep the tier, and
                    # leave the prose tier reachable. A malformed time value
                    # inside otherwise-fine JSON used to abort the whole parse.
                    continue
                found[item_id] = {"pred_start": start, "pred_end": end,
                                  "model_answer": row}
            if found:
                return found

    # -- tier 3: an interval stated in prose ---------------------------------
    # With one question per call, an interval anywhere in the response can only
    # be about that question, so it needs no anchor to be attributed. The last
    # match wins: a model that reasons before answering states its conclusion
    # last, and the previous id-anchored scan resolved repeats the same way.
    if not predictions and len(question_ids) == 1:
        matches = list(re.finditer(rf"({_NUM})\s*(?:-|,|to)\s*({_NUM})",
                                   _normalize_dashes(cleaned), flags=re.I))
        if matches:
            match = matches[-1]
            predictions[question_ids[0]] = {
                "pred_start": parse_time_value(match.group(1)),
                "pred_end": parse_time_value(match.group(2)),
                "model_answer": match.group(0),
            }
    if predictions:
        return predictions
    raise ValueError(f"Cannot parse multi-answer model output: {text!r}")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def temporal_metrics(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> dict[str, Any]:
    if pred_end < pred_start:
        pred_start, pred_end = pred_end, pred_start

    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    gt_duration = max(0.0, gt_end - gt_start)
    pred_center = (pred_start + pred_end) / 2

    return {
        "tIoU": intersection / union if union > 0 else 0.0,
        "center_inside": gt_start <= pred_center <= gt_end,
        "overlap_recall": intersection / gt_duration if gt_duration > 0 else 0.0,
        "pointing": intersection > 0,
        "start_error": pred_start - gt_start,
        "end_error": pred_end - gt_end,
        "abs_start_error": abs(pred_start - gt_start),
        "abs_end_error": abs(pred_end - gt_end),
        "pred_center": pred_center,
        "intersection": intersection,
    }


_EMPTY_METRICS = {
    "pred_start": None,
    "pred_end": None,
    "predicted_answer": None,
    "tIoU": 0.0,
    "center_inside": False,
    "overlap_recall": 0.0,
    "pointing": False,
    "start_error": None,
    "end_error": None,
    "abs_start_error": None,
    "abs_end_error": None,
    "pred_center": None,
    "intersection": 0.0,
}


# --------------------------------------------------------------------------
# Dimension
# --------------------------------------------------------------------------


def build_prompt(items: list[dict[str, Any]]) -> str:
    # The question id is deliberately not sent. It encodes the start of the
    # answer interval as a frame number at the episode's native rate, so a model
    # that divides by the frame rate reads the answer off the prompt instead of
    # the video. The id is only ever needed to attribute answers when one call
    # carries several questions, and this dimension sends one (see
    # ``TimeEqa.units``), so ``index`` alone is enough to map the answer back.
    question_lines = []
    for index, item in enumerate(items, 1):
        question_lines.append(
            json.dumps(
                {
                    "index": index,
                    "question": str(item.get("Q") or item.get("question")),
                },
                ensure_ascii=False,
            )
        )
    questions = "\n".join(question_lines)
    # The prompt must state the video's duration. Without it, "answer in
    # seconds" is not an executable instruction: the model sees a set of sampled
    # frames and cannot tell whether they span 9 seconds or 90. Models given no
    # duration answer with normalised fractions of the video instead, which
    # scores zero everywhere — not because the localisation is poor but because
    # the units are wrong.
    seconds = _video_seconds(items)
    duration_line = (f"\nThe video is {seconds:.1f} seconds long; "
                     f"answers must fall inside [0, {seconds:.1f}]."
                     if seconds else "")
    return f"""You are answering temporal grounding questions about synchronized robot manipulation videos.

For each question below, find the full time interval where the robot performs the queried action.
Return the full action segment, not only the instant of contact, grasp, or release.
Use seconds from the start of the video.{duration_line}

Questions, one JSON object per line:
{questions}

Output JSON only. Do not use Markdown.
Return one answer for every question index.
Required schema:
{{
  "answers": [
    {{
      "index": <question index>,
      "start": <start time in seconds>,
      "end": <end time in seconds>,
      "answer": "<HH:MM:SS.mmm-HH:MM:SS.mmm>",
      "reason": "<brief visual reason>"
    }}
  ]
}}
"""


def video_path_for_item(item: dict[str, Any]) -> str:
    """Path to the full episode this question refers to."""
    data = item.get("input") or {}
    path = data.get("video_path")
    if not path:
        raise KeyError(f"question {item.get('id')} has no video_path")
    return str(path)


class TimeEqaTask:
    """One question per model call."""

    name = "action_time"

    def __init__(self) -> None:
        # `strip_reasoning` does not apply here: this dimension does not parse an
        # option letter, it falls back through interval-shaped patterns.
        pass

    def units(self, items: list[dict[str, Any]]) -> list[Unit]:
        """One question per call.

        Batching every question about an episode into a single call would save
        media transfer, but it also makes "produce N answers in one response" a
        precondition for taking the test — and that is not one of the abilities
        this benchmark sets out to measure. A model that answers one question
        well but returns nothing when asked for several at once would score
        near zero for a reason unrelated to temporal grounding.

        The cost of not batching is small, because per-call overhead is
        dominated by generation rather than setup. Batching per model would be
        worse still: a model given "this clip contains exactly N actions" gains
        an elimination cue that others do not have.
        """
        return one_item_per_unit(items)

    def parts(self, unit: Unit, data_root: Any = None) -> list[dict[str, Any]]:
        # The duration line is non-negotiable for this dimension; measure it
        # off the media when the question does not carry it. Writing the value
        # into the items here is what makes rows() rebuild the same prompt.
        ensure_video_seconds(unit.items, data_root)
        video_path = video_path_for_item(unit.items[0])
        return [video_part(video_path), text_part(build_prompt(unit.items))]

    def rows(self, unit: Unit, text: str, ctx: CallContext) -> list[dict[str, Any]]:
        prompt = build_prompt(unit.items)
        question_ids = [str(i["id"]) for i in unit.items]
        video_path = video_path_for_item(unit.items[0])
        # An unreadable answer is a wrong answer, not an execution failure.
        # Raising here would turn it into an error row: excluded from
        # "completed", re-run on every resume (deterministically, to the same
        # text), and counted beside genuine call failures. The choice
        # dimensions already score unparseable output as zero with
        # parse_ok=False; the same convention applies here.
        try:
            predictions = parse_multi_interval_text(text, question_ids)
        except ValueError:
            predictions = {}

        out: list[dict[str, Any]] = []
        for item in unit.items:
            item_id = str(item["id"])
            row = base_row(item, prompt, text, ctx)
            row["video_path"] = video_path
            prediction = predictions.get(item_id)

            if prediction is None:
                row.update(_EMPTY_METRICS)
                row["parse_ok"] = False
                out.append(row)
                continue

            try:
                pred_start = parse_time_value(str(prediction["pred_start"]))
                pred_end = parse_time_value(str(prediction["pred_end"]))
            except ValueError:
                row.update(_EMPTY_METRICS)
                row["parse_ok"] = False
                out.append(row)
                continue
            row["parse_ok"] = True
            gt_start = float(item["answer_seconds"]["start"])
            gt_end = float(item["answer_seconds"]["end"])
            row["pred_start"] = pred_start
            row["pred_end"] = pred_end
            row["predicted_answer"] = f"{seconds_to_timestamp(pred_start)}-{seconds_to_timestamp(pred_end)}"
            row["model_answer"] = prediction.get("model_answer")
            row.update(temporal_metrics(pred_start, pred_end, gt_start, gt_end))
            out.append(row)
        return out

    def error_rows(self, unit: Unit, error: str) -> list[dict[str, Any]]:
        prompt = build_prompt(unit.items)
        out = []
        for item in unit.items:
            row = base_row(item, prompt, None, None)
            row.update(_EMPTY_METRICS)
            row["error"] = error
            out.append(row)
        return out

    def summarize(self, rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
        values = [float(r.get("tIoU", 0.0)) for r in rows]
        # Rows whose call failed carry `error`; they are counted there, not
        # as parse failures — a model cannot be unparseable when it was
        # never heard from.
        succeeded = [r for r in rows if not r.get("error")]
        answered = [
            r for r in rows
            if isinstance(r.get("pred_start"), (int, float)) and isinstance(r.get("pred_end"), (int, float))
        ]
        overlap_recalls = [float(r.get("overlap_recall", 0.0)) for r in answered]
        abs_start = [float(r.get("abs_start_error", 0.0)) for r in answered]
        abs_end = [float(r.get("abs_end_error", 0.0)) for r in answered]

        def mean(xs: list[float]) -> float:
            # math.fsum, not sum. Floating-point addition is not associative,
            # and rows complete in a nondeterministic order under concurrency —
            # so identical per-row values could aggregate to different totals
            # depending on timing. A metric that varies between two runs of the
            # same experiment cannot support a claim of reproducibility.
            # fsum returns the correctly rounded exact sum, independent of order.
            return math.fsum(xs) / len(xs) if xs else 0.0

        return {
            "total": len(rows),
            "answered": len(answered),
            "errors": sum(1 for r in rows if r.get("error")),
            "mean_tIoU": mean(values),
            "tIoU@0.3": mean([float(v >= 0.3) for v in values]) if values else 0.0,
            "tIoU@0.5": mean([float(v >= 0.5) for v in values]) if values else 0.0,
            "tIoU@0.7": mean([float(v >= 0.7) for v in values]) if values else 0.0,
            # The denominator is part of the name. Metrics computed over
            # answered rows and metrics computed over all rows are equal
            # whenever nothing failed, which is most of the time — so a name
            # that hides which denominator it used can disagree with its
            # neighbours in exactly the rare cases that matter.
            "center_inside_acc": (
                sum(bool(r.get("center_inside")) for r in rows) / len(rows) if rows else 0.0
            ),
            "center_inside_acc_answered": (
                sum(bool(r.get("center_inside")) for r in answered) / len(answered) if answered else 0.0
            ),
            "pointing_acc": (
                sum(bool(r.get("pointing")) for r in rows) / len(rows) if rows else 0.0
            ),
            "pointing_acc_answered": (
                sum(bool(r.get("pointing")) for r in answered) / len(answered) if answered else 0.0
            ),
            "mean_overlap_recall": mean(overlap_recalls),
            "mean_abs_start_error": mean(abs_start),
            "mean_abs_end_error": mean(abs_end),
            "elapsed_seconds": round(elapsed, 3),
            # A parse failure here means the call succeeded and still no
            # interval could be read. Failed calls are `errors`, not this.
            "parse_failure_rate": (
                sum(1 for r in succeeded
                    if not (isinstance(r.get("pred_start"), (int, float))
                            and isinstance(r.get("pred_end"), (int, float))))
                / len(succeeded) if succeeded else 0.0
            ),
        }


def build() -> TimeEqaTask:
    return TimeEqaTask()
