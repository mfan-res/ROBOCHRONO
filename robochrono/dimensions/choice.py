#!/usr/bin/env python3
# coding: utf-8
"""The six multiple-choice dimensions.

current_action / next_action / next_action_with_goal / view_match /
frame_match / frame_order

They differ in only four places — which media to send, how to phrase the
question, which ``expected_*`` fields belong in a result row, and which
breakdowns to include in the summary. ``ChoiceSpec`` captures those four; the
rest is shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .. import parsing
from .base import CallContext, Unit, base_row, count_by, image_part, one_item_per_unit, text_part, video_part

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


def _question(item: dict[str, Any]) -> str:
    return str(item.get("Q") or item.get("question") or "")


def _question_head(item: dict[str, Any]) -> str:
    """For image-option dimensions, keep only the part before ``Options:``."""
    return _question(item).split("\nOptions:", 1)[0].strip()


def prompt_current_action(item: dict[str, Any]) -> str:
    return f"""You are answering a multiple-choice visual understanding question about an egocentric robot manipulation video clip.

The clips show the video up to the current moment. Choose the option that best matches what is happening right now.
Choose exactly one option letter from the provided options. Do not invent a new action.

Question:
{_question(item)}

Output JSON only. Do not use Markdown.
Required schema:
{{
  "choice": "<one option letter, e.g. A>",
  "reason": "<brief visual reason>"
}}
"""


def prompt_next_action(item: dict[str, Any]) -> str:
    return f"""You are observing a robot manipulation task and need to predict the next action.

Look at the current video clip and choose the next action the robot should take.
Choose exactly one option from the provided option letters. Do not invent a new action.

Question:
{_question(item)}

Output JSON only. Do not use Markdown.
Required schema:
{{
  "choice": "<one option letter, e.g. A>",
  "reason": "<brief visual reason>"
}}
"""


def prompt_next_action_with_goal(item: dict[str, Any]) -> str:
    return f"""You are observing a robot manipulation task and need to predict the next action.

Look at the current video clip and choose the next action the robot should take.
Choose exactly one option from the provided option letters. Do not invent a new action.

Question:
{_question(item)}

Output JSON only. Do not use Markdown.
Required schema:
{{
  "choice": "<one option letter, e.g. A>",
  "reason": "<brief visual reason>"
}}
"""


def prompt_view_match(item: dict[str, Any]) -> str:
    option_ids = ", ".join(sorted(parsing.options_from_item(item)))
    return f"""You are answering a visual matching question for a robot manipulation episode.

The first image is from the head camera. The following labeled option images are candidate gripper-camera views at the same moment or distractors.
Choose the single option letter that shows the requested gripper camera's view for the head-camera moment.

Question:
{_question_head(item)}

Valid option letters: {option_ids}

Output JSON only. Do not use Markdown.
Required schema:
{{
  "choice": "<one option letter>",
  "reason": "<brief visual reason>"
}}
"""


def prompt_frame_match(item: dict[str, Any]) -> str:
    option_ids = ", ".join(sorted(parsing.options_from_item(item)))
    return f"""You are answering a visual matching question for a robot manipulation video clip.

First inspect the left-eye video clip, then inspect each labeled option image.
Choose the single option letter whose image appears in the video clip.

Question:
{_question_head(item)}

Valid option letters: {option_ids}

Output JSON only. Do not use Markdown.
Required schema:
{{
  "choice": "<one option letter>",
  "reason": "<brief visual reason>"
}}
"""


def prompt_frame_order(item: dict[str, Any]) -> str:
    question = _question(item).strip()
    choices = parsing.choices_from_item(item)
    if choices and "Options:" not in question:
        option_lines = "\n".join(f"{label}. {text}" for label, text in sorted(choices.items()))
        question = f"{question}\nOptions:\n{option_lines}"
    return f"""You are solving a robot manipulation step-order VQA task.

You will receive several still images from the same episode, each labeled
"Image 1", "Image 2", ... They are presented in random order, not in time order.

Choose the option that lists the images in the correct chronological order.
Choose exactly one option letter from the provided options. Do not invent a new option.

Question:
{question}

Output JSON only. Do not use Markdown.
Required schema:
{{
  "choice": "<one option letter, e.g. A>",
  "reason": "<brief visual reason>"
}}
"""


# --------------------------------------------------------------------------
# Media selection
# --------------------------------------------------------------------------


def media_video(item: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    """One video clip, then the prompt."""
    data = item.get("input", {})
    for key in ("clip_path", "video_path"):
        if data.get(key):
            return [video_part(data[key]), text_part(prompt)]
    raise KeyError(f"question {item.get('id')} has no clip_path or video_path")


def media_image(item: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    """Images first, then the prompt; falls back to video when no images are given."""
    data = item.get("input", {})
    paths = data.get("image_paths")
    if isinstance(paths, list) and paths:
        return [image_part(p) for p in paths] + [text_part(prompt)]

    if data.get("image_path"):
        return [image_part(data["image_path"]), text_part(prompt)]

    # next_action and next_action_with_goal send the same clip, so the only
    # difference between
    # them is the sentence naming the overall task. Keeping the media identical
    # is what makes that comparison clean — varying both the wording and the
    # media would leave two variables inside one difference.
    return media_video(item, prompt)


def _allow_missing_media(item: dict[str, Any], field: str) -> None:
    """Decide whether missing media means "omit deliberately" or "the data is broken".

    A blind baseline — measuring what a model scores without seeing the media —
    must travel this exact code path, otherwise it measures a second harness
    rather than this one. But presence of a field alone cannot distinguish a
    blind item from a corrupt one: a ``frame_match`` question that lost its
    clip would be sent as an image-only question and quietly score low instead
    of failing.

    Blind items therefore carry ``blind: true`` and only they may omit media.
    Anything else raises.
    """
    if not item.get("blind"):
        raise ValueError(
            f"question {item.get('id')} has no input.{field}."
            "\nA deliberately blind item must carry `blind: true`;"
            "\notherwise the data is broken. Omitting it silently would yield a"
            "\nlow score with no visible cause."
        )


def media_head_and_options(item: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    """The reference frame, then each labelled option image."""
    data = item.get("input", {})
    head = data.get("image_path") or (data.get("head_image") or {}).get("image_path")
    parts: list[dict[str, Any]] = []
    if head:
        parts += [text_part("Head camera image:"), image_part(head)]
    else:
        _allow_missing_media(item, "image_path")
    parts.append(text_part("Candidate options:"))
    parts.extend(_option_image_parts(item))
    parts.append(text_part(prompt))
    return parts


def media_clip_and_options(item: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    """The clip, then each labelled option image."""
    data = item.get("input", {})
    clip = data.get("clip_path") or data.get("video_path")
    parts: list[dict[str, Any]] = []
    if clip:
        parts += [text_part("Left-eye video clip:"), video_part(clip)]
    else:
        _allow_missing_media(item, "clip_path")
    parts.append(text_part("Candidate option images:"))
    parts.extend(_option_image_parts(item))
    parts.append(text_part(prompt))
    return parts


def _option_image_parts(item: dict[str, Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for option in item.get("options", []):
        option_id = str(option["id"]).upper()
        if option.get("is_none_option") or option.get("type") == "none":
            parts.append(text_part(f"Option {option_id}: {option.get('text')}"))
            continue
        image_path = option.get("image_path")
        if not image_path:
            raise ValueError(f"option {option_id} in {item.get('id')} has no image_path")
        parts.append(text_part(f"Option {option_id} image:"))
        parts.append(image_part(image_path))
    return parts


def media_frame_order(item: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    """Each frame sent separately, preceded by its label.

    The label is text (``Image 1:``) rather than rendered into the pixels of a
    montage, so it matches how the question refers to the frames and can be
    checked without decoding an image.

    When no images are present the parts simply contain none, rather than
    raising: a blind baseline relies on travelling this same code path, and for
    frame_order it asks whether a model prefers ``1 -> 2 -> 3`` on format alone.
    """
    data = item.get("input", {}) if isinstance(item.get("input"), dict) else {}
    parts: list[dict[str, Any]] = []
    for i, path in enumerate(data.get("image_paths") or []):
        parts.append(text_part(f"Image {i + 1}:"))
        parts.append(image_part(str(path)))
    parts.append(text_part(prompt))
    return parts



# --------------------------------------------------------------------------
# Dimension definitions
# --------------------------------------------------------------------------


@dataclass
class ChoiceSpec:
    name: str
    prompt: Callable[[dict[str, Any]], str]
    media: Callable[[dict[str, Any], str], list[dict[str, Any]]]
    # Result field name -> key to read from the question (a tuple means try in order)
    extra_fields: dict[str, Any] = field(default_factory=dict)
    summary_groups: tuple[str, ...] = ()
    distractor_counts: bool = False
    use_choices: bool = False


SPECS: dict[str, ChoiceSpec] = {
    "current_action": ChoiceSpec(
        name="current_action",
        prompt=prompt_current_action,
        media=media_video,
        extra_fields={
            "expected_answer_text": "answer_text",
            "expected_action": "answer_action",
            "expected_subject": "answer_subject",
            "expected_target": "answer_target",
        },
        summary_groups=("expected_action", "expected_choice"),
    ),
    "next_action": ChoiceSpec(
        name="next_action",
        prompt=prompt_next_action,
        media=media_video,
        extra_fields={
            "expected_answer_text": "answer_text",
            "expected_action": "answer_action",
            "expected_subject": "answer_subject",
            "expected_target": "answer_target",
        },
        summary_groups=("expected_action", "expected_choice"),
    ),
    "next_action_with_goal": ChoiceSpec(
        name="next_action_with_goal",
        prompt=prompt_next_action_with_goal,
        media=media_image,
        extra_fields={
            "expected_answer_text": "answer_text",
            "expected_action": "answer_action",
            "expected_subject": "answer_subject",
            "expected_target": "answer_target",
        },
        summary_groups=("expected_action", "expected_choice"),
    ),
    "view_match": ChoiceSpec(
        name="view_match",
        prompt=prompt_view_match,
        media=media_head_and_options,
        extra_fields={
            "expected_answer_text": "answer_text",
            "expected_target_side": "target_side",
        },
        summary_groups=("expected_target_side", "expected_choice"),
        distractor_counts=True,
    ),
    "frame_match": ChoiceSpec(
        name="frame_match",
        prompt=prompt_frame_match,
        media=media_clip_and_options,
        extra_fields={
            "expected_answer_text": "answer_text",
            "expected_category": "answer_category",
        },
        distractor_counts=True,
    ),
    "frame_order": ChoiceSpec(
        name="frame_order",
        prompt=prompt_frame_order,
        media=media_frame_order,
        extra_fields={"expected_answer_order": ("answer_order", "answer_text")},
        use_choices=True,
    ),
}

# Public names for the by_* breakdowns in a summary
_GROUP_LABEL = {
    "expected_action": "by_action",
    "expected_choice": "by_choice",
    "expected_target_side": "by_side",
}


class ChoiceTask:
    """Shared implementation for the six multiple-choice dimensions."""

    def __init__(self, spec: ChoiceSpec, *, strip_reasoning: bool) -> None:
        self.spec = spec
        self.name = spec.name
        # Declared in configs/protocol.json and passed through explicitly;
        # parsing behaviour is part of the protocol, not a code default.
        self.strip_reasoning = strip_reasoning

    # -- grouping and assembly --------------------------------------------

    def units(self, items: list[dict[str, Any]]) -> list[Unit]:
        return one_item_per_unit(items)

    def parts(self, unit: Unit, data_root: Any = None) -> list[dict[str, Any]]:
        item = unit.items[0]
        return self.spec.media(item, self.spec.prompt(item))

    # -- parsing and scoring ----------------------------------------------

    def _options(self, item: dict[str, Any]) -> dict[str, str]:
        if self.spec.use_choices:
            return parsing.choices_from_item(item)
        return parsing.options_from_item(item)

    def rows(self, unit: Unit, text: str, ctx: CallContext) -> list[dict[str, Any]]:
        item = unit.items[0]
        prompt = self.spec.prompt(item)
        prediction = parsing.parse_choice_answer(
            text,
            self._options(item),
            keep_hyphen=self.spec.use_choices,
            strip_reasoning=self.strip_reasoning,
        )
        row = base_row(item, prompt, text, ctx)
        row["model_prediction"] = prediction.get("parsed")
        row.update(self._score(item, prediction.get("choice")))
        # Reported alongside accuracy; the definition of `correct` is unchanged.
        row["parse_ok"] = bool(prediction.get("parse_ok"))
        row["parse_recovered"] = bool(prediction.get("parse_recovered"))
        return [row]

    def error_rows(self, unit: Unit, error: str) -> list[dict[str, Any]]:
        item = unit.items[0]
        row = base_row(item, self.spec.prompt(item), None, None)
        row["model_prediction"] = None
        row.update(self._score(item, None))
        row["correct"] = False
        row["error"] = error
        row["parse_ok"] = False
        row["parse_recovered"] = False
        return [row]

    def _score(self, item: dict[str, Any], pred_choice: str | None) -> dict[str, Any]:
        expected = str(item.get("answer") or item.get("A") or "").upper()
        scored: dict[str, Any] = {"expected_choice": expected}
        for out_key, src in self.spec.extra_fields.items():
            keys = src if isinstance(src, tuple) else (src,)
            value = None
            for key in keys:
                value = item.get(key)
                if value:
                    break
            scored[out_key] = value
        scored["pred_choice"] = pred_choice
        scored["correct"] = pred_choice == expected
        if self.spec.distractor_counts:
            # Recorded at scoring time, because result rows reference the
            # question by id rather than copying its options.
            #
            # The default here must not be "correct". Questions are not required
            # to label their distractors, and treating an absent label as "the
            # model picked the right answer" turns "unknown" into "correct" —
            # producing a breakdown that says every choice was correct while the
            # accuracy in the same summary says otherwise, with nothing to flag
            # the contradiction.
            #
            # So: use the declared type when the question gives one; record
            # `correct` only when the answer actually was correct, decided by
            # `correct` rather than by a missing field; `unknown` otherwise.
            scored["chosen_distractor_type"] = None
            for option in item.get("options", []):
                if str(option.get("id")).upper() != str(pred_choice or "").upper():
                    continue
                declared = option.get("distractor_type")
                if declared:
                    scored["chosen_distractor_type"] = str(declared)
                elif scored["correct"]:
                    scored["chosen_distractor_type"] = "correct"
                else:
                    # Wrong answer, and the question declares no type for it.
                    scored["chosen_distractor_type"] = "unknown"
                break
        return scored

    # -- summary -----------------------------------------------------------

    def summarize(self, rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
        total = len(rows)
        answered = [r for r in rows if r.get("model_output")]
        summary: dict[str, Any] = {
            "total": total,
            "answered": len(answered),
            "errors": sum(1 for r in rows if r.get("error")),
            "accuracy": sum(bool(r.get("correct")) for r in rows) / total if total else 0.0,
            "elapsed_seconds": round(elapsed, 3),
        }

        if self.spec.distractor_counts:
            summary["chosen_option_type_counts"] = self._distractor_counts(rows)

        for group_key in self.spec.summary_groups:
            summary[_GROUP_LABEL[group_key]] = count_by(rows, group_key)

        # Parse failure is reported separately from a wrong answer.
        # `accuracy` counts unparsed answers as wrong; `accuracy_answered` does not.
        # A failed call is neither: parse_failure_rate reads only rows whose
        # call succeeded, so it means what the report says it means — the
        # model spoke and no parser could read it. Failed calls are `errors`.
        parsed = [r for r in rows if r.get("parse_ok")]
        succeeded = [r for r in rows if not r.get("error")]
        unparsed = sum(1 for r in succeeded if not r.get("parse_ok"))
        summary["parse_failure_rate"] = unparsed / len(succeeded) if succeeded else 0.0
        summary["accuracy_answered"] = (
            sum(bool(r.get("correct")) for r in parsed) / len(parsed) if parsed else 0.0
        )
        summary["parse_recovered"] = sum(1 for r in rows if r.get("parse_recovered"))
        return summary

    @staticmethod
    def _distractor_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        """Which kind of distractor the model picked.

        Recorded during scoring (see ``chosen_distractor_type``) because result
        rows do not carry the original options.
        """
        counts: dict[str, int] = {}
        for row in rows:
            key = row.get("chosen_distractor_type")
            if key:
                counts[str(key)] = counts.get(str(key), 0) + 1
        return counts


def build(name: str, *, strip_reasoning: bool) -> ChoiceTask:
    """``strip_reasoning`` is required; it is declared in configs/protocol.json."""
    return ChoiceTask(SPECS[name], strip_reasoning=strip_reasoning)
