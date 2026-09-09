#!/usr/bin/env python3
# coding: utf-8
"""The execution engine: one loop for every (model, scenario, dimension).

What differs between dimensions went into the Dimension protocol; what differs
between models went into adapters. What is left is the same for everyone —
resumption, error handling, persistence, a circuit breaker — and lives here
exactly once.

The engine never imports an adapter module: it receives a constructed adapter.
That is what keeps this module importable from any environment, which the
whole environment-switching scheme rests on.

Behavioural decisions worth naming:

- **The main thread is the only writer.** Concurrent calls return their rows
  to it; nothing else touches the JSONL, so the append-only log can never
  interleave.
- **A run that keeps failing is stopped**, not left to burn hours on a broken
  configuration. The breaker counts consecutive failures.
- **A failed call still records whatever text the model produced.** The bug
  hunt that motivates per-question records usually starts from exactly the
  output that accompanied the failure.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .adapters.base import Adapter, AdapterResult
from .dimensions.base import CallContext, Dimension, Unit
from .results.store import ResultStore

DEFAULT_CIRCUIT_BREAKER = 20


class RateLimiter:
    """Token bucket. ``rate`` is requests per second; ``0`` means unlimited.

    Capacity defaults to one second's worth, so tokens saved up during a lull
    let the next burst go out immediately instead of being spread thin.
    """

    def __init__(self, rate: float, burst: float | None = None) -> None:
        self.rate = float(rate)
        self.capacity = float(burst if burst is not None else max(1.0, rate))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.rate <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity,
                                   self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
            # sleep outside the lock, or other threads cannot even refill
            time.sleep(min(wait, 1.0))


def resolve_parts(parts: list[dict[str, Any]], data_root: Any) -> list[dict[str, Any]]:
    """Make media paths absolute, in one place.

    Question files store paths relative to the dataset root; adapters must not
    each invent their own resolution rule.
    """
    root = Path(data_root)
    out = []
    for part in parts:
        if part.get("type") in {"image", "video"} and part.get("path"):
            path = Path(part["path"])
            if not path.is_absolute():
                part = {**part, "path": str(root / path)}
        out.append(part)
    return out


def limit_units(units: list[Unit], limit_items: int | None,
                limit_groups: int | None) -> list[Unit]:
    """Two explicit limits instead of one ambiguous ``--limit``.

    A unit can cover several questions, so "limit the calls" and "limit the
    questions" genuinely differ, and a smoke run that means one but applies
    the other samples the wrong amount.
    """
    if limit_groups is not None:
        if limit_groups < 0:
            raise ValueError("limit_groups must be non-negative")
        units = units[:limit_groups]
    if limit_items is not None:
        if limit_items < 0:
            raise ValueError("limit_items must be non-negative")
        kept: list[Unit] = []
        seen = 0
        for unit in units:
            if seen >= limit_items:
                break
            kept.append(unit)
            seen += len(unit.items)
        units = kept
    return units


def _run_unit(dimension: Dimension, unit: Unit, adapter: Adapter,
              frames: dict[str, Any], data_root: Any) -> tuple[list[dict[str, Any]], str]:
    """Run one unit; return (rows, error). An empty error string means success.

    Never writes to disk — persistence is the caller's, so the concurrent path
    can reuse this function without two threads appending to one file.
    """
    unit_started = time.perf_counter()
    text: str | None = None
    try:
        result: AdapterResult = adapter.call(
            resolve_parts(dimension.parts(unit), data_root),
            frames=frames, key=unit.key)
        text = result.text
        ctx = CallContext(frames_used=result.frames_used, usage=result.usage,
                          media_transforms=result.media_transforms)
        rows = dimension.rows(unit, text, ctx)
        error = ""
    except Exception as exc:  # noqa: BLE001 — any failure becomes an error row
        error = f"{type(exc).__name__}: {exc}"
        rows = dimension.error_rows(unit, error)
        # Keep whatever the model actually said. `error_rows` only sees the
        # error string, so `model_output` would be None — and the first
        # question in any post-mortem is "what did the model say".
        if text is not None:
            for row in rows:
                if row.get("model_output") in (None, ""):
                    row["model_output"] = text
        print(f"    error: {error}", flush=True)

    seconds = round(time.perf_counter() - unit_started, 3)
    for row in rows:
        row["timing"] = {"seconds": seconds}
    return rows, error


def run(
    dimension: Dimension,
    items: list[dict[str, Any]],
    adapter: Adapter,
    store: ResultStore,
    *,
    data_root: Any,
    frames: dict[str, Any] | None = None,
    limit_items: int | None = None,
    limit_groups: int | None = None,
    overwrite: bool = False,
    circuit_breaker: int = DEFAULT_CIRCUIT_BREAKER,
    on_row: Callable[[dict[str, Any]], None] | None = None,
    concurrency: int = 1,
    rate_limit: float = 0.0,
) -> dict[str, Any]:
    """Run one (model, scenario, dimension) to completion; return its summary.

    ``concurrency > 1`` fans calls out over threads — meaningful only for API
    adapters. Local models are parallelised by the GPU worker pool one level
    up; the two must not be stacked.
    """
    frames = frames or {}
    units = limit_units(dimension.units(items), limit_items, limit_groups)

    if overwrite:
        moved = store.displace()
        if moved:
            print(f"  [overwrite] {moved} rows moved to {store.path.name}.bak", flush=True)

    done = set() if overwrite else store.completed_ids()
    pending = [u for u in units
               if not all(str(i.get("id")) in done for i in u.items)]

    total_items = sum(len(u.items) for u in units)
    print(f"  units={len(units)} pending={len(pending)} items={total_items}", flush=True)

    started = time.perf_counter()
    consecutive_failures = 0
    aborted: str | None = None

    def collect(index: int, unit_key: str, rows: list[dict[str, Any]],
                failed: bool) -> bool:
        """Single point of persistence, main thread only. False means stop."""
        nonlocal consecutive_failures, aborted
        consecutive_failures = consecutive_failures + 1 if failed else 0
        for row in rows:
            if on_row is not None:
                on_row(row)
        store.append(rows)

        if index % 10 == 0 or index == len(pending):
            elapsed = time.perf_counter() - started
            print(f"    [{index}/{len(pending)}] "
                  f"{index / max(1e-6, elapsed) * 60:.1f}/min  {unit_key}", flush=True)

        if circuit_breaker and consecutive_failures >= circuit_breaker:
            aborted = f"circuit breaker: {consecutive_failures} consecutive failures"
            print(f"  ABORTED — {aborted}", flush=True)
            return False
        return True

    if concurrency > 1 and pending:
        limiter = RateLimiter(rate_limit)

        def submit(unit: Unit) -> tuple[Unit, list[dict[str, Any]], bool]:
            limiter.acquire()
            rows, error = _run_unit(dimension, unit, adapter, frames, data_root)
            return unit, rows, bool(error)

        print(f"  concurrency {concurrency}"
              + (f", rate limit {rate_limit:g} req/s" if rate_limit > 0
                 else ", unlimited rate"), flush=True)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(submit, u) for u in pending]
            for index, future in enumerate(as_completed(futures), 1):
                unit, rows, failed = future.result()
                if not collect(index, unit.key, rows, failed):
                    for remaining in futures:
                        remaining.cancel()
                    break
    else:
        for index, unit in enumerate(pending, 1):
            rows, error = _run_unit(dimension, unit, adapter, frames, data_root)
            if not collect(index, unit.key, rows, bool(error)):
                break

    all_rows = store.final_rows()   # dedup by id: an error row and its retry must not both count
    summary = dimension.summarize(all_rows, time.perf_counter() - started)
    if aborted:
        summary["aborted"] = aborted
    store.write_summary(summary)
    return summary
