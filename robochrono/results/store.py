#!/usr/bin/env python3
# coding: utf-8
"""Per-question result storage: one append-only JSONL per (model, scenario, dimension).

Append-only is what makes two promises hold at once:

- **Nothing finished is ever lost.** A row is written the moment its question
  completes; killing the process costs at most the question in flight.
- **Resuming needs no bookkeeping.** The completed set is recomputed from the
  file itself, so "rerun the same command" is the whole recovery procedure.

The price is that the file is a log, not a table: the same question id can
appear more than once, and readers must go through :meth:`ResultStore.final_rows`
rather than consuming lines directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class ResultStore:
    """Results for one (model, scenario, dimension)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.summary_path = self.path.with_suffix(".summary.json")

    # -- writing -----------------------------------------------------------

    def displace(self) -> int:
        """Move existing results to ``<name>.jsonl.bak``; return the row count.

        A forced rerun means "run again", not "destroy": these rows are GPU
        hours, and renaming costs kilobytes where deleting costs a rerun.
        Only one generation of backup is kept — more would be an unbounded
        pile nobody clears. The suffix is ``.jsonl.bak`` rather than
        ``.bak.jsonl`` so that anything collecting ``*.jsonl`` files —
        packing, reporting — never picks a backup up.
        """
        if not self.path.exists():
            return 0
        rows = sum(1 for line in self.path.read_text(encoding="utf-8").splitlines()
                   if line.strip())
        backup = self.path.with_name(self.path.name + ".bak")
        backup.unlink(missing_ok=True)
        self.path.rename(backup)
        return rows

    def append(self, rows: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- reading -----------------------------------------------------------

    def rows(self) -> Iterator[dict[str, Any]]:
        """Yield rows, tolerating a torn **final** line only.

        A kill or power loss can leave the last line half-written; refusing
        the whole file over it would turn half a line into a lost run. A bad
        line **mid-file** is a different animal — two processes appending to
        one file, say — and must be raised, not skipped.
        """
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle]
        for index, line in enumerate(lines):
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    print(f"  {self.path.name}: dropping a torn final line "
                          f"(the previous run was interrupted)", flush=True)
                    return
                raise

    def completed_ids(self) -> set[str]:
        """Question ids that need no rerun: answered, and without error."""
        done: set[str] = set()
        for row in self.rows():
            if row.get("model_output") and not row.get("error"):
                done.add(str(row.get("id")))
        return done

    def final_rows(self) -> list[dict[str, Any]]:
        """One row per question id. Summaries and exports must use this.

        The log can hold several rows for one id: a failed attempt, then the
        retry that succeeded. The rule mirrors :meth:`completed_ids`, so that
        "counts as done" and "counts in the score" are one criterion:

        - any successful row present → the **last successful** row wins
          (a retry supersedes what it retried)
        - only failures → the last failure (the question genuinely failed)

        Plain last-row-wins would be wrong: a misconfigured process can append
        an error row *after* the success it failed to notice, and then the
        newest row is exactly the one to ignore.
        """
        best: dict[str, dict[str, Any]] = {}
        for row in self.rows():
            key = str(row.get("id"))
            ok = bool(row.get("model_output")) and not row.get("error")
            previous = best.get(key)
            if previous is None:
                best[key] = row
                continue
            previous_ok = bool(previous.get("model_output")) and not previous.get("error")
            if ok or not previous_ok:
                best[key] = row
        return list(best.values())
