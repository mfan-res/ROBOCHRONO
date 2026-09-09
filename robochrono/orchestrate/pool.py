#!/usr/bin/env python3
# coding: utf-8
"""The GPU worker pool: one model at a time, every card busy.

Loading weights per (scenario, dimension) would load one model dozens of
times; loading serially would leave all but one card idle. The pool does
neither::

    main process
      ├─ worker-0  CUDA_VISIBLE_DEVICES=0  ┐ each worker loads one copy of the
      ├─ worker-1  CUDA_VISIBLE_DEVICES=1  │ current model, draws units from a
      └─ worker-N  CUDA_VISIBLE_DEVICES=N  ┘ shared queue, sends rows back

All cards run the **same model** at once; its whole matrix is spread across
workers, then the next model starts. Weights load once per worker, every card
stays busy.

Hard-won constraints, kept on purpose:

- ``CUDA_VISIBLE_DEVICES`` is set **before any torch import** — hence spawn
  workers whose first act is setting it. Every torch import in this package
  is lazy, which is the precondition.
- ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``: the caching allocator
  manages fixed-size segments, and variable-length episode inputs fragment
  them until a request fails with memory technically free. Measured: 12/12
  out-of-memory failures on episode-length inputs dropped to 0 with
  expandable segments. Set with ``setdefault`` so an explicit choice wins.
- **The main process is the only writer.** Workers send rows back; nothing
  else touches a JSONL.
- Liveness and slowness are separate questions. A five-second poll notices
  dead workers promptly; a slow unit is waited on for as long as the worker
  stays alive, with a one-time warning past ten minutes. One shared timeout
  cannot serve both — tightening it misreads slow as dead, loosening it
  waits ten minutes to notice a crash.
- Cleanup runs in ``finally``, terminate before join. Orphaned workers hold
  their GPU memory until killed by hand.
- Cards are grouped ``gpus_per_worker`` at a time for models too large for
  one card; a leftover partial group is dropped loudly rather than launched
  into a certain OOM.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.protocol import Protocol
from ..config.models import ModelConfig
from ..results.store import ResultStore
from .matrix import RunSpec

STOP = "__STOP__"

POLL_SECONDS = 5.0
STALL_WARN_SECONDS = 600.0
JOIN_TIMEOUT_SECONDS = 30.0


@dataclass
class WorkItem:
    spec_key: str
    scenario: str
    dimension: str
    unit_key: str
    items: list[dict[str, Any]]
    frames: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkResult:
    spec_key: str
    unit_key: str
    rows: list[dict[str, Any]]
    seconds: float
    worker: int
    error: str = ""


def _worker(
    gpu_indices: tuple[int, ...],
    work_queue: Any,
    result_queue: Any,
    model_path: str,
    protocol_path: str,
    adapter_runtime: dict[str, Any],
    strip_reasoning: bool,
    data_root: str,
) -> None:
    """Worker entry: set cards, build the adapter, loop on the queue.

    Arguments are paths and plain dicts — everything a spawn start must
    pickle. The adapter is built once and loads weights on first call;
    dimensions are cached by name.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    worker_id = gpu_indices[0]

    try:
        from .. import adapters, dimensions, engine
        from ..config.models import load_model
        from ..config.protocol import load_protocol
        from ..dataset.loader import scenario_root

        protocol = load_protocol(protocol_path)
        model = load_model(model_path)
        runtime = dict(adapter_runtime)
        # Card layout is the one thing only the worker knows: one card keeps
        # the single-device placement, several switch to layer splitting.
        runtime["device_map"] = {"": 0} if len(gpu_indices) == 1 else "auto"
        adapter = adapters.build(model, protocol, runtime)
    except Exception:  # noqa: BLE001 — startup failure must reach the main process
        result_queue.put(WorkResult("", "", [], 0.0, worker_id, traceback.format_exc()))
        return

    dimension_cache: dict[str, Any] = {}
    while True:
        try:
            item = work_queue.get(timeout=5)
        except queue.Empty:
            continue
        if item == STOP:
            break

        started = time.perf_counter()
        dim = dimension_cache.get(item.dimension)
        if dim is None:
            dim = dimension_cache[item.dimension] = dimensions.build(
                item.dimension, strip_reasoning=strip_reasoning)

        from ..dimensions.base import Unit
        unit = Unit(key=item.unit_key, items=item.items)
        # Media paths are relative to the scenario directory; scope the root
        # per item, through the same helper as the serial path.
        rows, error = engine._run_unit(dim, unit, adapter, item.frames,
                                       scenario_root(data_root, item.scenario))
        seconds = round(time.perf_counter() - started, 3)
        for row in rows:
            row["timing"] = {"seconds": seconds, "worker": worker_id}
        result_queue.put(WorkResult(item.spec_key, item.unit_key, rows,
                                    seconds, worker_id, error))


def _gpu_groups(gpus: list[int], per_worker: int) -> list[tuple[int, ...]]:
    if per_worker <= 1:
        return [(g,) for g in gpus]
    groups = [tuple(gpus[i:i + per_worker]) for i in range(0, len(gpus), per_worker)]
    full = [g for g in groups if len(g) == per_worker]
    if len(full) != len(groups):
        left = len(gpus) - len(full) * per_worker
        print(f"  ⚠ {len(gpus)} cards in groups of {per_worker}: {left} left unused — "
              f"a partial group cannot hold the model and would OOM", flush=True)
    if not full:
        print(f"❌ {len(gpus)} card(s) cannot form one group of {per_worker}", flush=True)
    return full


def run_pool(
    work: list[WorkItem],
    *,
    gpus: list[int],
    gpus_per_worker: int = 1,
    model_path: Any,
    protocol_path: Any,
    adapter_runtime: dict[str, Any] | None = None,
    strip_reasoning: bool,
    data_root: Any,
    on_result: Any = None,
) -> dict[str, int]:
    """Spread work across GPU workers; hand results to ``on_result`` as they land."""
    if not work:
        return {"done": 0, "errors": 0}

    ctx = mp.get_context("spawn")
    work_queue: Any = ctx.Queue()
    result_queue: Any = ctx.Queue()

    groups = _gpu_groups(gpus, gpus_per_worker)
    if not groups:
        return {"done": 0, "errors": 0}

    for item in work:
        work_queue.put(item)
    for _ in groups:
        work_queue.put(STOP)

    procs = [ctx.Process(target=_worker,
                         args=(group, work_queue, result_queue, str(model_path),
                               str(protocol_path), adapter_runtime or {},
                               strip_reasoning, str(data_root)),
                         daemon=True)
             for group in groups]
    for proc in procs:
        proc.start()

    done = errors = 0
    try:
        started = time.perf_counter()
        last_result = started
        warned = False
        while done < len(work):
            try:
                result: WorkResult = result_queue.get(timeout=POLL_SECONDS)
            except queue.Empty:
                alive = [p for p in procs if p.is_alive()]
                if not alive:
                    print(f"  all workers exited with {len(work) - done} unit(s) "
                          f"unfinished", flush=True)
                    break
                idle = time.perf_counter() - last_result
                if idle > STALL_WARN_SECONDS and not warned:
                    print(f"  ⚠ no results for {idle / 60:.0f} min; {len(alive)} "
                          f"worker(s) still alive. A slow unit, or a stall?", flush=True)
                    warned = True
                continue
            last_result = time.perf_counter()
            warned = False

            if not result.spec_key and result.error:
                print(f"  worker-{result.worker} failed to start:\n{result.error}",
                      flush=True)
                errors += 1
                break

            done += 1
            if result.error:
                errors += 1
            if on_result is not None:
                on_result(result)
            if done % 20 == 0 or done == len(work):
                rate = done / max(1e-6, time.perf_counter() - started)
                print(f"    [{done}/{len(work)}] {rate * 60:.1f} unit/min  "
                      f"errors={errors}", flush=True)
    finally:
        _stop_workers(procs)

    return {"done": done, "errors": errors}


def _stop_workers(procs: list[Any]) -> None:
    """Terminate first, join second — a worker mid-generate would otherwise
    hold the join for its full timeout, once per worker."""
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
    for proc in procs:
        proc.join(timeout=JOIN_TIMEOUT_SECONDS)
        if proc.is_alive():
            proc.kill()


def visible_gpus(requested: str | int | None = None) -> list[int]:
    """Which cards to run on: a count, an explicit list, or everything.

    ``4`` means the first four. ``"2,5"`` means those two and no others — for
    sharing a machine, where the first N are the wrong N.

    Note that ``CUDA_VISIBLE_DEVICES`` is not the way to do this here: workers
    set it themselves from these indices before importing torch, so a value
    inherited from the environment is overwritten rather than honoured. It
    would look like it worked and quietly run on the wrong cards.
    """
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    count = torch.cuda.device_count()
    if requested is None:
        return list(range(count))
    if isinstance(requested, str) and "," in requested:
        picked = [int(part) for part in requested.split(",") if part.strip()]
        outside = [i for i in picked if not 0 <= i < count]
        if outside:
            raise ValueError(
                f"--gpus names card(s) {outside} but this machine has {count} "
                f"(0-{count - 1})")
        return picked
    return list(range(min(int(requested), count)))


# --------------------------------------------------------------------------
# Driving one model's whole matrix through the pool
# --------------------------------------------------------------------------

def run_model_pool(
    model: ModelConfig,
    specs: list[RunSpec],
    *,
    protocol: Protocol,
    data_root: Any,
    run_dir: Path,
    adapter_runtime: dict[str, Any] | None = None,
    gpus: list[int],
    gpus_per_worker: int = 1,
    limit_items: int | None = None,
    limit_groups: int | None = None,
    overwrite: bool = False,
    model_path: Any = None,
    protocol_path: Any = "configs/protocol.json",
) -> dict[str, dict[str, Any]]:
    """Pool-run one model's specs; write rows and summaries like the serial path.

    Pending units from every spec go into one queue, so the tail of one
    (scenario, dimension) does not leave cards idle while the next waits.
    """
    from .. import dimensions, engine
    from ..dataset.loader import load_questions
    from ..dataset.render import load_question_bank

    if model_path is None:
        model_path = Path("configs/models") / model.kind / f"{model.slug}.json"

    bank = load_question_bank(data_root)
    stores: dict[str, ResultStore] = {}
    dims: dict[str, Any] = {}
    elapsed_start = time.perf_counter()
    work: list[WorkItem] = []

    for spec in specs:
        dim = dims[spec.key] = dimensions.build(
            spec.dimension, strip_reasoning=protocol.strip_reasoning)
        items = load_questions(data_root, spec.scenario, spec.dimension, bank=bank)
        store = stores[spec.key] = ResultStore(spec.store_path(run_dir))
        if overwrite:
            moved = store.displace()
            if moved:
                print(f"  [overwrite] {moved} rows moved to {store.path.name}.bak",
                      flush=True)
        units = engine.limit_units(dim.units(items), limit_items, limit_groups)
        done = store.completed_ids()
        frames = protocol.frames_for(spec.dimension)
        for unit in units:
            if all(str(i.get("id")) in done for i in unit.items):
                continue
            work.append(WorkItem(spec_key=spec.key, scenario=spec.scenario,
                                 dimension=spec.dimension,
                                 unit_key=unit.key, items=unit.items, frames=frames))

    print(f"[{model.slug}] pool: {len(work)} pending unit(s) on {len(gpus)} card(s)",
          flush=True)

    def on_result(result: WorkResult) -> None:
        stores[result.spec_key].append(result.rows)

    run_pool(work, gpus=gpus, gpus_per_worker=gpus_per_worker,
             model_path=model_path, protocol_path=protocol_path,
             adapter_runtime=adapter_runtime, strip_reasoning=protocol.strip_reasoning,
             data_root=data_root, on_result=on_result)

    elapsed = time.perf_counter() - elapsed_start
    summaries: dict[str, dict[str, Any]] = {}
    for spec in specs:
        store = stores[spec.key]
        summary = dims[spec.key].summarize(store.final_rows(), elapsed)
        store.write_summary(summary)
        summaries[spec.key] = summary
    return summaries
