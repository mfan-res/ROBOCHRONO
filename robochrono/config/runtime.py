#!/usr/bin/env python3
# coding: utf-8
"""Reading ``configs/runtime.json`` — operational settings.

These vary by machine and deliberately stay out of results and out of the run
fingerprint: one host sustains 24 concurrent requests where another cannot,
and recording that would make two runs look different when they measured the
same thing. Anything that *does* change output — timeouts, retries, request
size budgets — lives in ``protocol.json`` instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Runtime:
    gpus_per_worker: int
    api_concurrency: int
    api_rate_limit: float
    media_cache_dir: str
    proxy: str


_REQUIRED = ("gpus_per_worker", "api_concurrency", "api_rate_limit",
             "media_cache_dir", "proxy")


def load_runtime(path: Any = "configs/runtime.json") -> Runtime:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}")
    return Runtime(
        gpus_per_worker=int(raw["gpus_per_worker"]),
        api_concurrency=int(raw["api_concurrency"]),
        api_rate_limit=float(raw["api_rate_limit"]),
        media_cache_dir=str(raw["media_cache_dir"]),
        proxy=str(raw["proxy"]),
    )
