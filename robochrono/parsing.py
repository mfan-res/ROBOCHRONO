#!/usr/bin/env python3
# coding: utf-8
"""Parsing model output into an answer.

Parsing chain
-------------
The baseline always runs first, regardless of ``strip_reasoning``::

    1. strip_json_fence      remove a ```json … ``` fence
    2. json.loads            on success, read choice / answer / option / letter
    3. extract_choice        a. the whole string is a single letter
                             b. the first isolated capital letter \\b([A-Z])\\b
                             c. an option's text appears in the answer

With ``strip_reasoning`` enabled, two more tiers run **only when the baseline
returns None**::

    4. strip_reasoning_blocks  drop <think>/<thinking>/<reasoning>/<thought>/<analysis>
    5. re-run the whole baseline on the stripped text
    6. last_json_object        scan for the *last* JSON object anywhere in the text

Tier 6 is the one that does the work. Measured on 1,903 baseline failures from a
reasoning model::

    4+5  strip tags, re-run baseline      20  ( 1.1%)
    6    scan for the last JSON object 1,438  (75.6%)
         still unparsed                  445  (23.4%)   all output truncation

The reason tier 4 contributes so little: models often emit a bare closing
``</think>`` with no opening tag, because the opening tag is written into the
prompt by the chat template rather than generated. ``_REASONING_BLOCK`` requires
a matching pair and ``_UNCLOSED_REASONING`` only matches an opening tag at the
start, so neither handles that shape. Tier 6 does not depend on tags at all.

Tier 3c is a no-op for image-based options, whose ``text`` is absent — that is
why those dimensions are more fragile to format deviations.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator

# Reasoning models may emit their chain of thought inside the response content
# even when thinking is nominally disabled.
_REASONING_BLOCK = re.compile(
    r"<\s*(think|thinking|reasoning|thought|analysis)\s*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_UNCLOSED_REASONING = re.compile(
    r"^\s*<\s*(?:think|thinking|reasoning|thought|analysis)\s*>",
    re.IGNORECASE,
)

_CHOICE_KEYS = ("choice", "answer", "option", "letter")


def strip_json_fence(text: str) -> str:
    """Remove a surrounding markdown code fence."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def strip_reasoning_blocks(text: str) -> str:
    """Remove ``<think>…</think>`` style blocks."""
    cleaned = _REASONING_BLOCK.sub(" ", text)
    if _UNCLOSED_REASONING.match(cleaned):
        # Opening tag with no closing tag: drop the tag, keep what follows.
        cleaned = _UNCLOSED_REASONING.sub("", cleaned, count=1)
    return cleaned.strip()


def iter_json_objects(text: str) -> Iterator[dict[str, Any]]:
    """Yield every decodable top-level JSON object, in order of appearance."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            yield data


def last_json_object(text: str) -> dict[str, Any]:
    """Return the last decodable JSON object.

    Reasoning traces often contain draft JSON that looks like an answer; the
    final answer is normally at the end, so take the last one rather than the
    first.
    """
    cleaned = strip_json_fence(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    found: dict[str, Any] = {}
    for data in iter_json_objects(cleaned):
        found = data
    return found


def normalize_text(text: str, keep_hyphen: bool = False) -> str:
    """Lowercase, drop punctuation, collapse whitespace.

    ``keep_hyphen`` is for ``frame_order``, whose options are sequences such as
    ``1-3-2-4`` where the hyphen is meaningful.
    """
    text = text.lower().strip()
    pattern = r"[^a-z0-9\s-]+" if keep_hyphen else r"[^a-z0-9\s]+"
    text = re.sub(pattern, " ", text)
    return " ".join(text.split())


def options_from_item(item: dict[str, Any]) -> dict[str, str]:
    """Build ``{letter: option text}`` from ``item.options``.

    An option whose ``text`` is null contributes an empty string and therefore
    does not participate in text matching. Image-based options carry no text at
    all, so tier 3c never fires for them.
    """
    options: dict[str, str] = {}
    for option in item.get("options", []):
        if not isinstance(option, dict) or option.get("id") is None:
            continue
        raw = option.get("text", "")
        options[str(option["id"]).upper()] = "" if raw is None else str(raw)
    return options


def choices_from_item(item: dict[str, Any]) -> dict[str, str]:
    """``frame_order`` variant: prefer ``item.choices``, fall back to ``options``."""
    choices = item.get("choices")
    if isinstance(choices, dict):
        return {str(k).upper(): str(v) for k, v in choices.items()}
    return options_from_item(item)


def extract_choice(text: str, options: dict[str, str], keep_hyphen: bool = False) -> str | None:
    """Pull a single option letter out of free text. Three tiers; see module docstring."""
    valid_ids = set(options)
    normalized = str(text).strip().upper()
    if normalized in valid_ids:
        return normalized

    match = re.search(r"\b([A-Z])\b", normalized)
    if match and match.group(1) in valid_ids:
        return match.group(1)

    normalized_text = normalize_text(str(text), keep_hyphen)
    for option_id, option_text in options.items():
        candidate = normalize_text(option_text, keep_hyphen)
        if candidate and candidate in normalized_text:
            return option_id
    return None


def _choice_from_payload(data: dict[str, Any], options: dict[str, str],
                         keep_hyphen: bool) -> str | None:
    for key in _CHOICE_KEYS:
        value = data.get(key)
        if value:
            return extract_choice(str(value), options, keep_hyphen)
    # None of the expected keys is present, so fall back to the whole object.
    # A model may state its answer under a name of its own — `{"label": "Zip
    # the pouch."}` — and the option text is matched anyway. Reading only the
    # keys we thought of scores a correct answer as unparseable, which measures
    # our key list rather than the model.
    return extract_choice(json.dumps(data, ensure_ascii=False), options, keep_hyphen)


def _parse_choice_baseline(text: str, options: dict[str, str],
                           keep_hyphen: bool) -> dict[str, Any]:
    cleaned = strip_json_fence(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return {"choice": _choice_from_payload(data, options, keep_hyphen), "parsed": data}
    except json.JSONDecodeError:
        pass
    return {"choice": extract_choice(cleaned, options, keep_hyphen), "parsed": cleaned}


def parse_choice_answer(
    text: str,
    options: dict[str, str],
    *,
    strip_reasoning: bool,
    keep_hyphen: bool = False,
) -> dict[str, Any]:
    """Parse a multiple-choice answer.

    ``strip_reasoning`` is **required**, not defaulted. It is declared in
    ``configs/protocol.json`` and callers must pass it explicitly, so forgetting
    it raises ``TypeError`` instead of silently changing how answers are read.

    The fallback tiers are strictly additive: they run only when the baseline
    fails, and any row they rescue is flagged with ``parse_recovered``.
    """
    baseline = _parse_choice_baseline(text, options, keep_hyphen)
    if baseline["choice"] is not None or not strip_reasoning:
        baseline["parse_recovered"] = False
        baseline["parse_ok"] = baseline["choice"] is not None
        return baseline

    stripped = strip_reasoning_blocks(text)
    recovered = _parse_choice_baseline(stripped, options, keep_hyphen)
    if recovered["choice"] is None:
        payload = last_json_object(stripped)
        if payload:
            recovered = {
                "choice": _choice_from_payload(payload, options, keep_hyphen),
                "parsed": payload,
            }

    recovered["parse_ok"] = recovered["choice"] is not None
    recovered["parse_recovered"] = recovered["choice"] is not None
    return recovered
