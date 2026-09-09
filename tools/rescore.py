#!/usr/bin/env python3
# coding: utf-8
"""Re-score finished runs from the model output already on disk.

A parser fix changes what a stored response means, not what the model said.
Every row keeps its raw ``model_output``, so the answers can be re-read without
calling a single model again — which is the whole reason the raw text is kept.

Only ``action_time`` is touched: it is the dimension whose parser changed. The
choice dimensions are left exactly as they are.

Runs in place unless ``--dry-run`` is given; the JSONL rows and the per-cell
``*.summary.json`` are both rewritten so ``report`` picks the change up.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochrono.dimensions import time_eqa  # noqa: E402


def ground_truth(data_root: Path) -> dict[str, tuple[float, float]]:
    """Every action_time question's true interval, by question id.

    Read from the dataset rather than from the rows: a row that failed to parse
    carries no prediction, so its error fields are empty and the truth cannot be
    reconstructed from them — which is exactly the row this tool exists to
    rescue.
    """
    out: dict[str, tuple[float, float]] = {}
    for qa in sorted(Path(data_root).glob("qa/*/action_time.json")):
        for item in json.loads(qa.read_text(encoding="utf-8")).get("items", []):
            a = item.get("answer_seconds")
            if a:
                out[str(item["id"])] = (float(a["start"]), float(a["end"]))
    return out


def rescore_cell(jsonl: Path, gt_by_id: dict) -> dict:
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    before = sum(1 for r in rows if r.get("parse_ok"))
    out = []
    for row in rows:
        # An error row never reached a parser; leave it untouched.
        if row.get("error"):
            out.append(row)
            continue
        text = row.get("model_output") or ""
        item_id = str(row.get("id"))
        try:
            preds = time_eqa.parse_multi_interval_text(text, [item_id])
        except ValueError:
            preds = {}
        pred = preds.get(item_id)
        if pred is None:
            out.append(row)          # still unreadable — unchanged
            continue
        try:
            ps = time_eqa.parse_time_value(str(pred["pred_start"]))
            pe = time_eqa.parse_time_value(str(pred["pred_end"]))
        except ValueError:
            out.append(row)
            continue
        truth = gt_by_id.get(item_id)
        if truth is None:
            out.append(row)          # not a question of this dataset
            continue
        gs, ge = truth
        row = dict(row)
        row["parse_ok"] = True
        row["pred_start"], row["pred_end"] = ps, pe
        row["predicted_answer"] = (f"{time_eqa.seconds_to_timestamp(ps)}-"
                                   f"{time_eqa.seconds_to_timestamp(pe)}")
        row["model_answer"] = pred.get("model_answer")
        row.update(time_eqa.temporal_metrics(ps, pe, gs, ge))
        out.append(row)
    after = sum(1 for r in out if r.get("parse_ok"))
    return {"rows": out, "before": before, "after": after, "total": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--data-root", required=True, type=Path,
                    help="the dataset these runs were produced against")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gt_by_id = ground_truth(args.data_root)
    if not gt_by_id:
        print(f"no action_time ground truth under {args.data_root}", file=sys.stderr)
        return 1

    task = time_eqa.build()
    grand = [0, 0, 0]
    for run_dir in args.run_dirs:
        gained_run = 0
        for jsonl in sorted(Path(run_dir).rglob("action_time.jsonl")):
            res = rescore_cell(jsonl, gt_by_id)
            gained = res["after"] - res["before"]
            grand[0] += res["total"]; grand[1] += res["before"]; grand[2] += res["after"]
            if not gained:
                continue
            gained_run += gained
            if args.dry_run:
                continue
            jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                     for r in res["rows"]), encoding="utf-8")
            summary_path = jsonl.with_name("action_time.summary.json")
            old = json.loads(summary_path.read_text(encoding="utf-8"))
            new = task.summarize(res["rows"], old.get("elapsed_seconds", 0.0))
            # elapsed came from the original run; recomputing would report the
            # re-scoring time as if the models had been called again.
            new["elapsed_seconds"] = old.get("elapsed_seconds", 0.0)
            summary_path.write_text(json.dumps(new, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
        print(f"{run_dir.name}: +{gained_run} answers recovered")
    verb = "would recover" if args.dry_run else "recovered"
    print(f"\n{grand[0]} action_time rows: {grand[1]} parsed before, "
          f"{grand[2]} after — {verb} {grand[2]-grand[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
