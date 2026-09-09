#!/usr/bin/env python3
# coding: utf-8
"""Replay adapter: answers from a recorded table instead of a model.

Two jobs. In tests it stands in for a model, which is what lets the engine's
behaviour — resumption, the circuit breaker, error handling — be exercised on
a machine with no GPU. Offline, it re-scores historical outputs after a metric
changes definition, without paying for a single new call.

The table maps unit keys to recorded output text; asking for a key that was
never recorded is an error, not an empty answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import Adapter, AdapterResult


class ReplayAdapter(Adapter):
    def __init__(self, model, protocol, runtime=None) -> None:
        super().__init__(model, protocol, runtime)
        self.table: dict[str, str] = dict(self.runtime.get("replay_table") or {})
        # With no in-memory table, the model configuration may point at one on
        # disk (api.table) — which is what lets a replay model run through the
        # command line exactly like a real one.
        if not self.table and model.api.get("table"):
            self.table = json.loads(Path(model.api["table"]).read_text(encoding="utf-8"))

    def call(self, parts: list[dict[str, Any]], *, frames: dict[str, Any],
             key: str = "") -> AdapterResult:
        if key not in self.table:
            raise KeyError(f"replay: no recorded output for key {key!r}")
        return AdapterResult(text=self.table[key], raw={"_replay": True})


def build(model, protocol, runtime=None) -> ReplayAdapter:
    return ReplayAdapter(model, protocol, runtime)
