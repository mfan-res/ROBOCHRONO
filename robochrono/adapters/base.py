#!/usr/bin/env python3
# coding: utf-8
"""The adapter contract, and the helpers every adapter shares.

An adapter is the only kind of code in this package allowed to import an
inference stack — and even then only lazily, inside methods. Environment
switching works by starting each model under the interpreter its environment
declares, which requires that *importing* any module here works everywhere;
only *calling* into a model needs the heavy libraries. A top-level
``import torch`` in any adapter would break that silently, so the constraint
is enforced by ``tests/test_orchestrator_is_light.py``.

Construction is cheap and resolves the per-model generation settings once,
from the protocol and the model's own configuration. ``load()`` does the
expensive part — weights, processors — and is called only inside worker
processes. ``call()`` takes the dimension-assembled parts and returns what
the model said, along with what can be observed about the call: how many
frames were actually sampled, the server-reported token usage, and any
transformations applied to media. Those observations go into result rows
because none of them can be reconstructed afterwards from a score.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.models import ModelConfig
from ..config.protocol import Protocol


@dataclass
class AdapterResult:
    """What one model call produced and what was observable about it."""

    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    frames_used: dict[str, int] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    media_transforms: list[dict[str, Any]] = field(default_factory=list)


class Adapter:
    """Base adapter: light to construct, heavy only on ``load()``."""

    def __init__(self, model: ModelConfig, protocol: Protocol,
                 runtime: dict[str, Any] | None = None) -> None:
        self.model = model
        self.protocol = protocol
        self.runtime = runtime or {}
        # Per-model generation, resolved once. The protocol sets the baseline;
        # a model may raise the token budget, never lower it, and declares its
        # real thinking state when it cannot match the protocol's.
        self.temperature = protocol.temperature
        self.max_new_tokens = protocol.effective_max_new_tokens(model.max_new_tokens())
        self.thinking = str(model.thinking(protocol.thinking).value)
        self.system_prompt = protocol.system_prompt

    def generation_settings(self) -> dict[str, Any]:
        """What this model runs under — recorded in ``run.json`` so that two
        models under different settings are never silently compared."""
        return {"temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "thinking": self.thinking}

    def load(self) -> None:
        """Load weights and processors. A no-op for API adapters."""

    def call(self, parts: list[dict[str, Any]], *, frames: dict[str, Any],
             key: str = "") -> AdapterResult:
        """Run one model call.

        ``parts``   text/image/video parts, media paths already absolute
        ``frames``  the dimension's frame-sampling spec from the protocol
        ``key``     the unit key; used by the replay adapter, ignored elsewhere
        """
        raise NotImplementedError


# --------------------------------------------------------------------------
# Shared helpers (standard library + requests only)
# --------------------------------------------------------------------------

def media_mime(path: Path, media_type: str) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    return "video/mp4" if media_type == "video" else "image/jpeg"


def encode_file(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def data_url(path: Path, media_type: str) -> str:
    return f"data:{media_mime(path, media_type)};base64,{encode_file(path)}"


def backoff(attempt: int, retry_after: str | None = None) -> float:
    """How long to wait before a retry.

    A server-sent ``Retry-After`` wins — it knows better than we do. Otherwise
    exponential backoff **with jitter**: concurrent threads tend to hit a 429
    together, and without jitter they retry in lockstep and hit it together
    again, which is no backoff at all.
    """
    if retry_after:
        try:
            # delta-seconds form only; the rare HTTP-date form falls through
            return max(0.0, min(120.0, float(retry_after)))
        except ValueError:
            pass
    return min(60.0, 2 ** attempt) * random.uniform(0.5, 1.5)


def post_with_retries(url: str, *, headers: dict[str, str], payload: dict[str, Any],
                      timeout: int, max_retries: int,
                      proxy: str | None = None) -> dict[str, Any]:
    """POST JSON, retrying transient failures, raising rich errors otherwise.

    ``proxy`` routes this call through an explicit proxy URL — endpoints
    reachable only through one declare it in their model configuration, so
    the route is a recorded fact instead of ambient environment state.
    """
    import requests

    proxies = {"http": proxy, "https": proxy} if proxy else None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload,
                                     timeout=timeout, proxies=proxies)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                wait = backoff(attempt, response.headers.get("Retry-After"))
                print(f"  HTTP {response.status_code}, retrying in {wait:.1f}s", flush=True)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == max_retries:
                raise
            wait = backoff(attempt)
            print(f"  connection error (attempt {attempt}/{max_retries}), "
                  f"retrying in {wait:.1f}s: {exc}", flush=True)
            time.sleep(wait)
        except requests.HTTPError as exc:
            try:
                detail = json.dumps(response.json(), ensure_ascii=False, indent=2)
            except ValueError:
                detail = response.text
            raise RuntimeError(
                f"API request failed: {response.status_code} {response.reason}\n{detail}"
            ) from exc
    raise RuntimeError("API request failed without a response")


def chat_completion_text(response: dict[str, Any]) -> str:
    """The assistant text out of a chat/completions-shaped response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected chat/completions response: {response}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def weights_path(declared: str) -> str:
    """A local directory if it exists, otherwise the id as declared."""
    path = Path(declared).expanduser()
    return str(path) if path.exists() else declared


def generation_kwargs(temperature: float, max_new_tokens: int,
                      eos_token_id: Any) -> dict[str, Any]:
    """Shared ``generate()`` arguments for the local transformers adapters.

    ``do_sample`` follows the temperature; a temperature of zero means greedy
    decoding, and passing ``temperature=0`` alongside sampling is an error in
    some stacks. ``pad_token_id`` is set explicitly — transformers otherwise
    logs a warning per call, which at benchmark scale buries the log lines
    that carry information.
    """
    do_sample = temperature > 0
    kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        kwargs["temperature"] = temperature
    if eos_token_id is not None:
        kwargs["pad_token_id"] = eos_token_id
    return kwargs


def max_memory_map(spec: dict[str, Any], device_count: int) -> dict[int, str]:
    """Per-card weight budgets for accelerate, from a model's declaration.

    ``device_map="auto"`` packs weights by what fits, blind to activations —
    measured on an eight-card split, it filled every card to the brim and the
    card hosting the vision tower then failed a ~5 GiB activation allocation
    on a third of the calls. A model that needs headroom declares it::

        "resources": {"max_memory": {"0": "55GiB", "default": "68GiB"}}

    Keys are visible-device indices (after CUDA_VISIBLE_DEVICES), "default"
    covers the rest. Values pass through to accelerate verbatim.
    """
    default = spec.get("default")
    out: dict[int, str] = {}
    for index in range(device_count):
        value = spec.get(str(index), default)
        if value is not None:
            out[index] = str(value)
    return out
