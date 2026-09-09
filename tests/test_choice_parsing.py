#!/usr/bin/env python3
# coding: utf-8
"""Multiple-choice answers: read what the model meant, invent nothing.

A model states its choice in whatever shape it likes — a bare letter, a JSON
object under a key we expected, or one under a name of its own. What must not
happen is scoring a correct answer as unreadable because the key it used was
not on our list: that measures the key list, not the model.

The refusals matter as much as the recoveries. A response that names no option
has to stay unparsed; a parser that guesses turns a silent failure into a wrong
number nobody can spot.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochrono import parsing  # noqa: E402

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:56} {detail}")
    if not passed:
        failures.append(name)


OPTIONS = {"A": "Close the pouch.", "B": "Take the pouch out of the box.",
           "C": "Zip the pouch.", "D": "Put the pouch in the box."}


def choice(text: str):
    return parsing.parse_choice_answer(text, OPTIONS, strip_reasoning=True)["choice"]


print("1. the shapes models actually answer in")
check("a bare letter", choice("B") == "B")
check("the expected key", choice('{"answer":"B"}') == "B")
check("a fenced object", choice('```json\n{"choice":"C"}\n```') == "C")
check("option text rather than its letter", choice("Zip the pouch.") == "C")

print("2. an answer under a key we did not think of")
# Gemini Robotics-ER answers in its pointing format: the option is stated in
# full under `label`, beside a coordinate that has nothing to do with the
# question. Reading only our own key list scored this as unreadable.
check("pointing format with the option under `label`",
      choice('```json\n{"point": [150, 895], '
             '"label": "Take the pouch out of the box."}\n```') == "B")
check("any other key carrying the option text",
      choice('{"selection":"Zip the pouch."}') == "C")

print("3. what must stay unreadable")
check("an object naming no option", choice('{"foo":"bar","n":3}') is None)
check("text matching no option", choice('{"label":"Something else entirely."}') is None)
check("an empty response", choice("") is None)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
