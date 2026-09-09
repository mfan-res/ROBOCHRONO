#!/usr/bin/env python3
# coding: utf-8
"""action_time scoring semantics: unreadable answers are wrong, not broken.

Pinned after the stress run surfaced three ways a readable-looking response
still failed to score: a missing per-item entry, a seconds suffix the parser
refused, and a response no tier could read. All three must land as zero-scored
rows with parse_ok=False — the convention the choice dimensions already use —
never as error rows that resume re-runs forever and summaries count beside
genuine call failures.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochrono.dimensions.base import CallContext, Unit  # noqa: E402
from robochrono.dimensions.time_eqa import (  # noqa: E402
    build, parse_multi_interval_text, parse_time_value)

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:52} {detail}")
    if not passed:
        failures.append(name)


print("1. time values models actually write")
for raw, want in (("119.0s", 119.0), ("43 sec", 43.0), ("12.5 SECONDS", 12.5),
                  ("00:01:23.5", 83.5), ("75", 75.0)):
    check(f"parse {raw!r}", parse_time_value(raw) == want)
try:
    parse_time_value("3 minutes")
    check("a minutes suffix stays loud", False, "no exception")
except ValueError:
    check("a minutes suffix stays loud", True)

DIM = build()
ITEM = {"id": "s/e/s00@action_time", "subtask": "x",
        "question": "When did it happen?",
        "input": {"video_path": "media/episodes/s/e.mp4"},
        "answer_seconds": {"start": 10.0, "end": 20.0}}
UNIT = Unit(key=ITEM["id"], items=[ITEM])
CTX = CallContext()

print("2. a correct answer scores and is marked parsed")
rows = DIM.rows(UNIT, '{"start": "10.0s", "end": "20.0s"}', CTX)
check("parsed", rows[0]["parse_ok"] is True)
check("perfect tIoU", rows[0]["tIoU"] == 1.0)
check("no error field", "error" not in rows[0])

print("3. unreadable answers are wrong answers, not failures")
for label, text in (("prose with no interval", "I cannot tell from the video."),
                    ("unparseable time values", '{"start": "ten", "end": "20"}')):
    rows = DIM.rows(UNIT, text, CTX)
    r = rows[0]
    check(f"{label}: zero-scored", r["tIoU"] == 0.0)
    check(f"{label}: parse_ok False", r["parse_ok"] is False)
    check(f"{label}: not an error row", "error" not in r)
    check(f"{label}: output kept for the post-mortem", r["model_output"] == text)

print("4. the summary separates the three counts")
ok = DIM.rows(UNIT, '{"start": 10, "end": 20}', CTX)
bad = DIM.rows(UNIT, "no idea", CTX)
err = DIM.error_rows(UNIT, "ConnectionError: boom")
summary = DIM.summarize(ok + bad + err, elapsed=1.0)
check("total counts every row", summary["total"] == 3)
check("answered counts only parsed", summary["answered"] == 1)
check("errors counts only real failures", summary["errors"] == 1)
# The error row is a failed call, not an unreadable answer: the rate's
# denominator is the two calls that succeeded, of which one was unreadable.
check("parse failure rate covers the unreadable, not the failed",
      abs(summary["parse_failure_rate"] - 1 / 2) < 1e-9)
check("zero rows stay in the tIoU denominator",
      abs(summary["mean_tIoU"] - 1 / 3) < 1e-9)

print("5. an interval the model did state is not thrown away")
# Each case below is a real response that scored zero because the parser
# stopped early, not because the model failed to answer. The interval is
# present and unambiguous in every one of them.
QID = ["scenario/file-000/s00@action_time"]


def interval(text):
    """The interval the parser recovers, or None when it recovers nothing.

    A response it cannot read raises, and the caller turns that into a
    zero-scored row — so None here means exactly "would be marked unparsed".
    """
    try:
        got = parse_multi_interval_text(text, QID).get(QID[0])
    except ValueError:
        return None
    return None if got is None else (got["pred_start"], got["pred_end"])


check("an answer field holding an object, not a string",
      interval('{"answers":[{"index":1,"answer":{"start":0.01,"end":0.06}}]}') == (0.01, 0.06))
check("valid JSON numbering its only answer out of range still reaches the prose tier",
      interval('{"answers":[{"index":10,"start":"35.0","end":"40.0",'
               '"answer":"00:35.0-00:40.0"}]}') == (35.0, 40.0))
check("one unreadable row does not discard the rest of the response",
      interval('```json\n[\n{"answers": [\n{"index": 1, "start":":00:05", '
               '"end":":00:08", "answer": "00:00:05.000-00:00:08.000"}\n}\n]\n```')
      == (5.0, 8.0))
check("a model that declines is still unparsed — nothing is invented",
      interval('{"answers":[{"index":1,"start":null,"end":null,"answer":null}]}') is None)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
