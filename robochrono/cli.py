#!/usr/bin/env python3
# coding: utf-8
"""The command line: five commands, no logic of their own.

    robochrono preflight        can this machine run this selection?
    robochrono validate-data    is the dataset what its manifest declares?
    robochrono eval             run (--dry-run: what would run, what it costs)
    robochrono report           one comparison table from a run
    robochrono pack             bundle a run for delivery

Everything here is wiring: parsing arguments and handing them to the layers
that do the work. ``eval`` orchestrates in two roles — the parent establishes
the run's identity and dispatches one child per environment; a child (marked
by ``--run-dir``) executes its model group inside the directory it was given
and never re-derives identity from its own subset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .config.environments import load_environments
from .config.models import ModelConfig, load_models
from .config.protocol import load_protocol
from .config.runtime import load_runtime
from .config.suites import load_suite


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", default="scenarios",
                        help="the scenario pool (default: scenarios/)")
    common.add_argument("--results-root", default="results",
                        help="where run directories live (default: results/)")
    common.add_argument("--models-dir", default="configs/models",
                        help="the model configuration tree (default: configs/models)")

    selection = argparse.ArgumentParser(add_help=False)
    selection.add_argument("--suite", default="official-v1",
                           help="suite name under configs/suites, or a path to "
                                "a suite file (default: official-v1)")
    selection.add_argument("--models", nargs="+", default=None,
                           help="model slugs; default: every configured model")
    selection.add_argument("--scenarios", nargs="+", default=None,
                           help="narrow to these scenarios (the suite's "
                                "completeness check still covers every pin)")
    selection.add_argument("--dimensions", nargs="+", default=None,
                           help="narrow to these dimensions")
    selection.add_argument("--only", choices=["local", "api"], default=None,
                           help="restrict the selection to one model kind")
    selection.add_argument("--shard", default=None,
                           help="i/n — this machine's share of the matrix")

    parser = argparse.ArgumentParser(prog="robochrono")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight", parents=[common, selection],
                   help="check environment, weights, keys, and data before running")

    sub.add_parser("validate-data", parents=[common],
                   help="verify the dataset delivers what its manifest declares")

    run = sub.add_parser("eval", parents=[common, selection],
                         help="run the evaluation")
    run.add_argument("--dry-run", action="store_true",
                     help="show what would run and what it costs; run nothing")
    run.add_argument("--fresh", action="store_true",
                     help="start a new run directory even if this configuration ran before")
    run.add_argument("--overwrite", action="store_true",
                     help="rerun completed questions (existing rows go to .bak)")
    run.add_argument("--items-file", default=None,
                     help="a file of question ids, one per line; run only "
                          "these — reproduces an earlier run's question set "
                          "without touching the data")
    run.add_argument("--limit-items", type=int, default=None,
                     help="cap questions per (scenario, dimension) — "
                          "sampling for a quick check")
    run.add_argument("--limit-groups", type=int, default=None,
                     help="cap units (model calls) per (scenario, dimension)")
    run.add_argument("--gpus", default=None,
                     help="N for the first N cards, or a list like 2,5 for "
                          "those cards; default: all visible")
    run.add_argument("--gpus-per-worker", type=int, default=None,
                     help="cards per model copy (default: the model's own "
                          "declaration; raise it when card memory is tight)")
    run.add_argument("--api-concurrency", type=int, default=None,
                     help="parallel in-flight API calls (default: runtime.json)")
    run.add_argument("--label", default=None,
                     help="a name for this run, recorded in run.json; does not "
                          "affect the run's identity")
    run.add_argument("--run-dir", default=None,
                     help="(internal) execute inside this run directory")
    run.add_argument("--no-dispatch", action="store_true",
                     help="run in this interpreter instead of dispatching per environment")

    rep = sub.add_parser("report", parents=[common],
                         help="aggregate a run's summaries into one table")
    rep.add_argument("run_dirs", nargs="*", type=Path,
                     help="default: the most recent run")
    rep.add_argument("--out", default=None,
                     help="directory for report.md/report.csv; default: the run directory")

    pack = sub.add_parser("pack", parents=[common],
                          help="bundle a run for delivery")
    pack.add_argument("run_dir", nargs="?", type=Path, default=None,
                      help="default: the most recent run")
    pack.add_argument("--full", action="store_true",
                      help="include per-question records")
    pack.add_argument("-o", "--output", default=None,
                      help="archive path (default: results/<run_id>.tar.gz)")
    return parser


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------

def _item_filter(args, specs) -> set[str] | None:
    """The ``--items-file`` id set, checked against the selection.

    An id that matches nothing is a typo, a stale list, or a scenario left out
    of the selection — and all three would otherwise show up as a quietly
    smaller run that still reports success. Refuse instead, and name what was
    not found.
    """
    if not getattr(args, "items_file", None):
        return None
    from .dataset.loader import load_questions, read_item_ids
    from .dataset.render import load_question_bank

    keep = read_item_ids(args.items_file)
    bank = load_question_bank(args.data_root, sorted({s.scenario for s in specs}))
    found: set[str] = set()
    for scenario, dimension in sorted({(s.scenario, s.dimension) for s in specs}):
        for question in load_questions(args.data_root, scenario, dimension,
                                       bank=bank, keep=keep):
            found.add(str(question.get("id")))
    missing = sorted(keep - found)
    if missing:
        shown = ", ".join(missing[:5])
        raise SystemExit(
            f"{args.items_file}: {len(missing)} of {len(keep)} question id(s) "
            f"match nothing in this selection — e.g. {shown}"
            f"{' …' if len(missing) > 5 else ''}\n"
            f"The list, the --scenarios/--dimensions selection and the data "
            f"root must agree before a subset run means anything.")
    print(f"--items-file: {len(keep)} question(s) selected from "
          f"{args.items_file}")
    return keep


def _print_skipped(skipped: list) -> None:
    by_reason: dict[str, set[str]] = {}
    for key, reason in skipped:
        by_reason.setdefault(reason, set()).add(key.split("__")[1])
    print(f"{len(skipped)} (model × scenario × dimension) "
          f"combination(s) not runnable here and excluded:")
    for reason, scenarios in sorted(by_reason.items()):
        names = ", ".join(sorted(scenarios)[:6])
        more = f" (+{len(scenarios) - 6} more)" if len(scenarios) > 6 else ""
        print(f"  {reason} — {len(scenarios)} scenario(s): {names}{more}")


def _selected_models(args) -> list[ModelConfig]:
    """The models this invocation names — read from configuration, not from
    which combinations survive expansion. An empty data root must not
    disguise a local selection as an API-only one."""
    models = sorted(load_models(args.models_dir).values(), key=lambda m: m.slug)
    if args.only:
        models = [m for m in models if m.kind == args.only]
    if args.models:
        known = {m.slug for m in models}
        unknown = sorted(set(args.models) - known)
        if unknown:
            raise SystemExit(f"unknown models: {unknown}; known: {sorted(known)}")
        models = [m for m in models if m.slug in set(args.models)]
    return models


def _selection(args) -> tuple[list[ModelConfig], Any, list, list]:
    """The models and specs this invocation covers."""
    from .orchestrate.matrix import expand

    models = sorted(load_models(args.models_dir).values(), key=lambda m: m.slug)
    suite = load_suite(args.suite, "configs/suites")
    shard = None
    if args.shard:
        index, _, total = args.shard.partition("/")
        shard = (int(index), int(total))
    specs, skipped = expand(models, suite, args.data_root, shard=shard,
                            only_kind=args.only, only_models=args.models,
                            only_scenarios=args.scenarios,
                            only_dimensions=args.dimensions)
    order: list[str] = []
    for spec in specs:
        if spec.model not in order:
            order.append(spec.model)
    by_slug = {m.slug: m for m in models}
    return [by_slug[s] for s in order], suite, specs, skipped


def _apply_proxy(setting: str) -> None:
    """runtime.json's proxy: "system" leaves the environment alone, "bypass"
    strips proxy variables, anything else routes HTTP(S) through it."""
    names = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
             "ALL_PROXY", "all_proxy")
    if setting == "system":
        return
    if setting == "bypass":
        for name in names:
            os.environ.pop(name, None)
        return
    os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = setting


def _run_path(arg: Any, results_root: Any) -> Path:
    """A run named by path or by bare run id (resolved under results/)."""
    path = Path(arg)
    if (path / "run.json").exists():
        return path
    candidate = Path(results_root) / path.name
    if (candidate / "run.json").exists():
        return candidate
    raise SystemExit(f"{arg} is not a run directory (no run.json found, "
                     f"also looked under {results_root})")


def _latest_run(results_root: Any) -> Path:
    candidates = [p for p in Path(results_root).iterdir()
                  if p.is_dir() and (p / "run.json").exists()]
    if not candidates:
        raise SystemExit(f"no runs under {results_root}")
    return max(candidates, key=lambda p: (p / "run.json").stat().st_mtime)


def _floors() -> dict[str, Any]:
    return json.loads(Path("configs/protocol.json").read_text())["degenerate_floor"]


def _write_report(run_dirs: list[Path], out_dir: Path) -> int:
    from .results import report as report_mod

    try:
        rep = report_mod.collect([Path(p) for p in run_dirs], _floors())
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(report_mod.to_markdown(rep), encoding="utf-8")
    report_mod.to_csv(rep, out_dir / "report.csv")
    faults = sum(1 for r in rep.rows if r.get("fault"))
    floors = sum(1 for r in rep.rows if r.get("floor"))
    print(f"{len(rep.rows)} cell(s) -> {out_dir / 'report.md'}")
    if faults:
        print(f"  ✗ {faults} did not execute properly — those cells are not scores")
    if floors:
        print(f"  ⚠ {floors} at or below the degenerate floor")
    return 0


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_preflight(args) -> int:
    from .preflight import format_checks, has_failures, run_preflight

    _, suite, specs, skipped = _selection(args)
    # What the user selected, not what survived expansion: with the data
    # absent, every combination is skipped, and deriving the model list from
    # the survivors made a local selection look API-only — torch, GPU and
    # weights checks silently vanished exactly when a newcomer runs
    # preflight first.
    models = _selected_models(args)
    if skipped:
        _print_skipped(skipped)
    checks = run_preflight(specs, models, suite=suite,
                           protocol=load_protocol(),
                           environments=load_environments(),
                           data_root=args.data_root, repo_root=".")
    print(format_checks(checks))
    return 1 if has_failures(checks) else 0


def cmd_validate_data(args) -> int:
    from .dataset.validate import format_report, validate_dataset

    report = validate_dataset(args.data_root)
    print(format_report(report))
    return 0 if report.ok else 1


def _dry_run(args, models, suite, specs, skipped, keep) -> int:
    from .dataset.loader import load_questions, media_paths, resolve_media
    from .dataset.render import load_question_bank
    from .orchestrate.dispatch import dispatch

    total = len(specs) + len(skipped)
    print(f"suite {suite.name}: {len(specs)} of {total} (model × scenario × "
          f"dimension) combination(s) runnable, {len(skipped)} skipped")
    for key, reason in skipped[:10]:
        print(f"  skipped {key}: {reason}")

    bank = load_question_bank(args.data_root, sorted({s.scenario for s in specs}))
    combos = sorted({(s.scenario, s.dimension) for s in specs})
    questions_by_combo: dict[tuple, int] = {}
    media: set[tuple[str, str]] = set()      # (scenario, relative path)
    for scenario, dimension in combos:
        items = load_questions(args.data_root, scenario, dimension,
                               bank=bank, keep=keep)
        questions_by_combo[(scenario, dimension)] = len(items)
        for q in items:
            media.update((scenario, p) for p in media_paths(q))
    per_model = sum(questions_by_combo.values())
    media_bytes = sum(resolve_media(args.data_root, sc, p).stat().st_size
                      for sc, p in media
                      if resolve_media(args.data_root, sc, p).exists())
    print(f"per model: {per_model} question(s) over {len(combos)} file(s)")
    print(f"total calls: {per_model * len(models)} across {len(models)} model(s)")
    print(f"media touched: {len(media)} file(s), {media_bytes / 1e9:.2f} GB "
          f"(sent once per API model)")
    print()
    dispatch(models, environments=load_environments(), repo_root=".",
             run_dir="<run-dir>", passthrough=["--suite", args.suite],
             dry_run=True)
    return 0


def _execute(args, models, specs, run_dir, keep) -> None:
    from .orchestrate.execute import execute
    from .orchestrate.pool import visible_gpus

    runtime = load_runtime()
    protocol = load_protocol()
    gpus = []
    if any(m.kind == "local" for m in models):
        gpus = visible_gpus(args.gpus)
    execute(specs, models=models, protocol=protocol, data_root=args.data_root,
            run_dir=run_dir,
            adapter_runtime={"media_cache_dir": runtime.media_cache_dir},
            api_concurrency=args.api_concurrency or runtime.api_concurrency,
            api_rate_limit=runtime.api_rate_limit,
            gpus=gpus,
            gpus_per_worker=args.gpus_per_worker or runtime.gpus_per_worker,
            limit_items=args.limit_items, limit_groups=args.limit_groups,
            keep=keep,
            overwrite=args.overwrite,
            models_dir=args.models_dir, protocol_path="configs/protocol.json")


def cmd_eval(args) -> int:
    from .orchestrate.dispatch import dispatch
    from .orchestrate.execute import prepare_run
    from .results import runid

    models, suite, specs, skipped = _selection(args)
    if not specs:
        print("nothing to run:")
        _print_skipped(skipped)
        return 1
    # Before anything is created: a bad id list must not leave a run
    # directory behind, and the dispatched children must not each discover
    # the same fault separately.
    keep = _item_filter(args, specs)
    if args.dry_run:
        return _dry_run(args, models, suite, specs, skipped, keep)

    if not args.run_dir:
        # The suite pins content; running with any of it missing would
        # produce results indistinguishable from a complete run. Refuse,
        # by name, before anything starts. Children skip the recheck — the
        # parent that dispatched them already paid for it.
        from .preflight import verify_suite_pins

        pins = verify_suite_pins(suite, args.data_root)
        if not pins.ok:
            print(f"refusing to run: suite {suite.name!r} pins scenarios "
                  f"that are not usable under {args.data_root}")
            for label, names in (("missing", pins.absent),
                                 ("content differs from the pin", pins.mismatched),
                                 ("manifest out of date (regenerate with "
                                  "tools/compute_scenario_hash.py --write)", pins.stale),
                                 ("unreadable", pins.unreadable)):
                if names:
                    print(f"  {label}: {names}")
            if pins.absent:
                print("fetch the missing scenarios first:")
                print(f"  .venvs/tf4/bin/python tools/download_data.py "
                      f"{' '.join(pins.absent)}")
            print("or pin the subset you do have as its own suite "
                  "(docs/DATA_FORMAT.md, \"Suites\").")
            return 1

    _apply_proxy(load_runtime().proxy)
    protocol = load_protocol()
    environments = load_environments()

    if args.run_dir:
        # Child: identity was established by the parent; never re-derive it.
        record = runid.read_run_record(args.run_dir)
        run = runid.RunDir(path=Path(args.run_dir), run_id=Path(args.run_dir).name,
                           fingerprint=str(record.get("fingerprint")), resumed=True)
        _execute(args, models, specs, run, keep)
        return 0

    model_paths = [Path(args.models_dir) / m.kind / f"{m.slug}.json" for m in models]
    # --suite accepts a path as well as a name (load_suite resolves both);
    # the fingerprint must hash the same file either way.
    suite_arg = Path(args.suite)
    suite_path = (suite_arg if suite_arg.exists()
                  else Path("configs/suites") / f"{args.suite}.json")
    run = prepare_run(
        args.results_root, protocol_path="configs/protocol.json",
        suite_path=suite_path,
        suite_name=args.suite, models=models, model_paths=model_paths,
        protocol=protocol, data_root=args.data_root,
        environments={name: env.transformers for name, env in environments.items()},
        repo_root=".", fresh=args.fresh, label=args.label)
    print(f"run {run.run_id}" + (" (resuming)" if run.resumed else " (new)"))

    if args.no_dispatch:
        _execute(args, models, specs, run, keep)
        failures = 0
    else:
        passthrough = ["--suite", args.suite,
                       "--data-root", args.data_root,
                       "--results-root", args.results_root,
                       "--models-dir", args.models_dir]
        for flag, value in (("--scenarios", args.scenarios),
                            ("--dimensions", args.dimensions)):
            if value:
                passthrough += [flag, *value]
        if args.only:
            passthrough += ["--only", args.only]
        if args.shard:
            passthrough += ["--shard", args.shard]
        if args.items_file:
            passthrough += ["--items-file", args.items_file]
        for flag, value in (("--limit-items", args.limit_items),
                            ("--limit-groups", args.limit_groups),
                            ("--gpus", args.gpus),
                            ("--gpus-per-worker", args.gpus_per_worker),
                            ("--api-concurrency", args.api_concurrency),
                            ("--label", args.label)):
            if value is not None:
                passthrough += [flag, str(value)]
        if args.overwrite:
            passthrough.append("--overwrite")
        failures = dispatch(models, environments=environments, repo_root=".",
                            run_dir=run.path, passthrough=passthrough)

    if failures:
        print(f"{failures} environment group(s) failed; the run can be resumed "
              f"with the same command")
        return failures
    runid.mark_finished(run)
    return _write_report([run.path], run.path)


def cmd_report(args) -> int:
    run_dirs = ([_run_path(p, args.results_root) for p in args.run_dirs]
                or [_latest_run(args.results_root)])
    out = Path(args.out) if args.out else run_dirs[0]
    return _write_report(run_dirs, out)


def cmd_pack(args) -> int:
    from .results.report import pack

    run_dir = (_run_path(args.run_dir, args.results_root) if args.run_dir
               else _latest_run(args.results_root))
    suffix = "-full" if args.full else ""
    output = Path(args.output) if args.output else (
        Path(args.results_root) / f"{run_dir.name}{suffix}.tar.gz")
    result = pack(run_dir, output, full=args.full)
    print(f"{result['files']} file(s), {result['bytes'] / 1e6:.1f} MB -> "
          f"{result['output']}")
    if not args.full:
        print("per-question records not included; add --full to bundle them")
    return 0


_COMMANDS = {
    "preflight": cmd_preflight,
    "validate-data": cmd_validate_data,
    "eval": cmd_eval,
    "report": cmd_report,
    "pack": cmd_pack,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _COMMANDS[args.command](args)
