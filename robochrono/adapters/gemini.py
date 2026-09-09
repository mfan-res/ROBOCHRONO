#!/usr/bin/env python3
# coding: utf-8
"""Adapter for the Gemini generateContent API.

The ``api`` block supplies ``url`` (with an optional ``{model}`` placeholder),
``model``, and ``key_env``; media goes inline as base64. The generation
budget is sent as ``maxOutputTokens`` — the same must-actually-send rule as
the OpenAI-compatible adapter, which historically missed this exact field on
this exact dialect. ``generation_config`` from the configuration is merged
into ``generationConfig`` key by key: replacing the whole object would silently
drop the temperature.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import Adapter, AdapterResult, encode_file, media_mime


def gemini_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for part in parts:
        kind = part.get("type")
        if kind == "text":
            output.append({"text": str(part.get("text", ""))})
        elif kind in {"image", "video"}:
            path = Path(str(part["path"]))
            output.append({"inline_data": {"mime_type": media_mime(path, kind),
                                           "data": encode_file(path)}})
        else:
            raise ValueError(f"unsupported content part type: {kind}")
    return output


def gemini_text(response: dict[str, Any]) -> str:
    try:
        parts = response["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected Gemini response: {response}") from exc
    return "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))


class GeminiAdapter(Adapter):
    def __init__(self, model, protocol, runtime=None) -> None:
        super().__init__(model, protocol, runtime)
        api = model.api
        for required in ("url", "model"):
            if not api.get(required):
                raise ValueError(f"model {model.slug!r}: api.{required} is required "
                                 f"for the gemini adapter")
        self.served_model = str(api["model"])
        self.url = str(api["url"]).format(model=self.served_model)
        self.min_video_seconds = float(api.get("min_video_seconds") or 0.0)
        self.max_request_bytes = int(api.get("max_request_bytes",
                                             protocol.max_request_bytes))
        self.proxy = api.get("proxy")
        # The key is looked up at construction but only *required* at call
        # time: a missing credential should stop a call, not a dry run or a
        # configuration check. Preflight reports it as its own line item.
        self.key_env = api.get("key_env")
        self.api_key = os.environ.get(str(self.key_env), "") if self.key_env else ""

    def payload(self, parts: list[dict[str, Any]]) -> dict[str, Any]:
        api = self.model.api
        gen_cfg: dict[str, Any] = {"temperature": self.temperature,
                                   "maxOutputTokens": self.max_new_tokens}
        # key-by-key, never a whole-object replace
        gen_cfg.update(api.get("generation_config") or {})
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": self.system_prompt}]},
            "contents": [{"role": "user", "parts": gemini_parts(parts)}],
            "generationConfig": gen_cfg,
        }
        for k, v in (api.get("extra_payload") or {}).items():
            if k == "generationConfig" and isinstance(v, dict):
                payload["generationConfig"].update(v)
            else:
                payload[k] = v
        return payload

    def call(self, parts: list[dict[str, Any]], *, frames: dict[str, Any],
             key: str = "") -> AdapterResult:
        if self.key_env and not self.api_key:
            raise RuntimeError(f"{self.model.slug}: environment variable "
                               f"{self.key_env} is not set — see configs/README.md "
                               f"on keys, or run `robochrono preflight`")
        from .base import post_with_retries
        from ..media_prep import prepare_parts

        transforms: list[dict[str, Any]] = []
        budget = self.max_request_bytes
        if budget > 0 or self.min_video_seconds > 0:
            cache_dir = Path(self.runtime.get("media_cache_dir") or ".cache/media")
            parts, transforms = prepare_parts(
                parts, budget or (1 << 62), cache_dir,
                min_video_seconds=self.min_video_seconds)

        url = self.url
        if self.api_key and "key=" not in url:
            url = f"{url}{'&' if '?' in url else '?'}key={self.api_key}"
        raw = post_with_retries(
            url, headers={"Content-Type": "application/json",
                          **(self.model.api.get("extra_headers") or {})},
            payload=self.payload(parts),
            timeout=self.protocol.timeout, max_retries=self.protocol.max_retries,
            proxy=self.proxy)
        usage = raw.get("usageMetadata") if isinstance(raw.get("usageMetadata"), dict) else {}
        return AdapterResult(text=gemini_text(raw), raw=raw,
                             usage=usage, media_transforms=transforms)


def build(model, protocol, runtime=None) -> GeminiAdapter:
    return GeminiAdapter(model, protocol, runtime)
