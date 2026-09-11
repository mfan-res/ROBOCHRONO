#!/usr/bin/env python3
# coding: utf-8
"""Driving an evaluation: identity first, then one model at a time.

The run's identity is established **once, over the full selection**, before
anything executes. Dispatch later splits the models across environments and
sharding splits the matrix across machines, but both work inside the same run
directory — a child process that recomputed identity from its own subset
would derive a different fingerprint and quietly file its results elsewhere.

Execution is model-major: every spec of one model runs before the next model
starts, so local weights load once. Within a model, the serial path hands
each spec to the engine directly; the pooled path (local models on GPUs)
spreads units across workers instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import adapters, dimensions, engine
from ..config.models import ModelConfig
from ..config.protocol import Protocol
from ..config.suites import load_suite, pins_digest
from ..dataset.loader import load_questions, scenario_root
from ..dataset.render import load_question_bank
from ..results import runid
from ..results.store import ResultStore
from .matrix import RunSpec


def prepare_run(
    results_root: Any,
    *,
    protocol_path: Any,
    suite_path: Any,
    suite_name: str,
    models: list[ModelConfig],
    model_paths: list[Any],
    protocol: Protocol,
    data_root: Any,
    environments: dict[str, Any] | None = None,
    repo_root: Any = ".",
    fresh: bool = False,
    label: str | None = None,
) -> runid.RunDir:
    """Establish the run's identity and write its configuration snapshot.

    The data identity comes from the suite's pins, not from whatever sits at
    the data root: the suite file declares which content this experiment is
    about, and preflight is what checks the disk against that declaration.
    """
    suite = load_suite(suite_path)
    digest = pins_digest(suite.pins)
    code = runid.code_version(repo_root)
    fingerprint = runid.config_fingerprint(
        protocol_path=protocol_path, suite_path=suite_path,
        model_paths=sorted(str(p) for p in model_paths),
        dataset_digest=digest, code=code)
    run_dir = runid.resolve_run_dir(results_root, fingerprint, fresh=fresh)

    generation = {m.slug: adapters.build(m, protocol).generation_settings()
                  for m in models}
    import sys
    runid.write_run_record(run_dir, {
        # The exact command line, so a run produced on any machine by any
        # operator carries what was actually asked of it — limits included.
        "invocation": list(sys.argv),
        # The full pin map, not just the digest: merging can then check
        # per-scenario agreement, so runs of overlapping suites combine
        # exactly when their shared scenarios carry identical content.
        "dataset": {"digest": digest, "scenarios": dict(sorted(suite.pins.items()))},
        "protocol": {"version": protocol.version},
        "suite": suite_name,
        # Whose run this is, in words. Deliberately outside the fingerprint:
        # a label describes an operator's intent, and two machines running the
        # same experiment must stay the same experiment however they label it.
        **({"label": label} if label else {}),
        "models": sorted(m.slug for m in models),
        "generation_by_model": generation,
        "code": {k: code[k] for k in ("git", "dirty", "diff_sha")
                 if k in code},
        "environment": environments or {},
    })
    return run_dir


def execute_model(
    model: ModelConfig,
    specs: list[RunSpec],
    *,
    protocol: Protocol,
    data_root: Any,
    run_dir: Path,
    adapter_runtime: dict[str, Any] | None = None,
    concurrency: int = 1,
    rate_limit: float = 0.0,
    limit_items: int | None = None,
    limit_groups: int | None = None,
    keep: set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run one model's specs serially; return summaries by spec key."""
    adapter = adapters.build(model, protocol, adapter_runtime)
    bank = load_question_bank(data_root)
    summaries: dict[str, dict[str, Any]] = {}
    for spec in specs:
        print(f"[{model.slug}] {spec.scenario}/{spec.dimension}", flush=True)
        dimension = dimensions.build(spec.dimension,
                                     strip_reasoning=protocol.strip_reasoning)
        items = load_questions(data_root, spec.scenario, spec.dimension,
                               bank=bank, keep=keep)
        store = ResultStore(spec.store_path(run_dir))
        summaries[spec.key] = engine.run(
            dimension, items, adapter, store,
            # Media paths are relative to the scenario directory; the engine
            # resolves against whatever root it is handed, so hand it the
            # scenario's.
            data_root=scenario_root(data_root, spec.scenario),
            frames=protocol.frames_for(spec.dimension),
            concurrency=concurrency, rate_limit=rate_limit,
            limit_items=limit_items, limit_groups=limit_groups,
            overwrite=overwrite)
    return summaries


def execute(
    specs: list[RunSpec],
    *,
    models: list[ModelConfig],
    protocol: Protocol,
    data_root: Any,
    run_dir: runid.RunDir,
    adapter_runtime: dict[str, Any] | None = None,
    api_concurrency: int = 1,
    api_rate_limit: float = 0.0,
    gpus: list[int] | None = None,
    gpus_per_worker: int = 1,
    limit_items: int | None = None,
    limit_groups: int | None = None,
    keep: set[str] | None = None,
    overwrite: bool = False,
    models_dir: Any = "configs/models",
    protocol_path: Any = "configs/protocol.json",
) -> dict[str, dict[str, Any]]:
    """Run every spec, model-major. Returns summaries by spec key.

    Local models with GPUs available go through the worker pool; everything
    else runs serially — API models get thread concurrency inside the engine
    instead, which is the right kind of parallelism for a network-bound call.
    """
    by_slug = {m.slug: m for m in models}
    summaries: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for spec in specs:
        if spec.model not in order:
            order.append(spec.model)

    for slug in order:
        model = by_slug[slug]
        model_specs = [s for s in specs if s.model == slug]
        if model.kind == "local" and gpus:
            from .pool import run_model_pool
            # A model may declare how many cards one replica needs; the
            # invocation-level value is only a default for those that do not.
            per_worker = int(model.resources.get("gpus_per_worker", gpus_per_worker))
            summaries.update(run_model_pool(
                model, model_specs, protocol=protocol, data_root=data_root,
                run_dir=run_dir.path, adapter_runtime=adapter_runtime,
                gpus=gpus, gpus_per_worker=per_worker,
                limit_items=limit_items, limit_groups=limit_groups, keep=keep,
                overwrite=overwrite,
                model_path=Path(models_dir) / model.kind / f"{model.slug}.json",
                protocol_path=protocol_path))
        else:
            is_api = model.kind == "api"
            summaries.update(execute_model(
                model, model_specs, protocol=protocol, data_root=data_root,
                run_dir=run_dir.path, adapter_runtime=adapter_runtime,
                concurrency=api_concurrency if is_api else 1,
                rate_limit=api_rate_limit if is_api else 0.0,
                limit_items=limit_items, limit_groups=limit_groups, keep=keep,
                overwrite=overwrite))
    return summaries
