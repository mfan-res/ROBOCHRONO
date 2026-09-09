#!/usr/bin/env python3
# coding: utf-8
"""One adapter per model family, found by the ``adapter`` field of a model's
configuration. The registry imports lazily so that naming an adapter a machine
cannot run does not stop the ones it can."""

from __future__ import annotations

from typing import Any

from ..config.models import ModelConfig
from ..config.protocol import Protocol
from .base import Adapter, AdapterResult

_ADAPTERS = ("qwen3_vl", "rynnbrain", "internvl", "cosmos3_edge",
             "openai_compat", "gemini", "replay")


def build(model: ModelConfig, protocol: Protocol,
          runtime: dict[str, Any] | None = None) -> Adapter:
    name = model.adapter
    if name not in _ADAPTERS:
        raise ValueError(f"model {model.slug!r} names unknown adapter {name!r}; "
                         f"known: {list(_ADAPTERS)}")
    import importlib
    module = importlib.import_module(f".{name}", __package__)
    return module.build(model, protocol, runtime)
