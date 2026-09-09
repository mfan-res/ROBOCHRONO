#!/usr/bin/env python3
# coding: utf-8
"""Adapter for OpenAI-compatible chat/completions endpoints.

The model's ``api`` configuration block supplies what varies by endpoint:

    url               chat/completions endpoint (required)
    model             the served model id (required)
    key_env           name of the environment variable holding the key;
                      keys themselves never appear in configuration files
    media_url_format  "data_url" (default) or bare "base64"
    extra_payload     endpoint dialect, merged last so it can override anything
    extra_headers     additional HTTP headers
    thinking_param    name of a boolean thinking toggle (DashScope-style
                      `enable_thinking` and friends)
    send_thinking     send `thinking: {"type": ...}` (GLM-style dialect)
    min_video_seconds server-side minimum video duration, if the endpoint
                      enforces one

Two hard-won rules are load-bearing here. ``max_tokens`` must actually be
sent — a budget that lives only in result metadata leaves the server free to
apply its own. And the thinking toggle must actually be sent, in whichever
dialect the endpoint speaks: a configuration that *says* thinking is disabled
while the server happily reasons at length is worse than no setting, because
the results claim a uniformity that does not exist. ``max_tokens`` does not
cap reasoning tokens on the measured endpoints, so cost control comes only
from the toggle itself.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import Adapter, AdapterResult, chat_completion_text, data_url, encode_file


def openai_content(parts: list[dict[str, Any]],
                   media_url_format: str = "data_url") -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in parts:
        kind = part.get("type")
        if kind == "text":
            content.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind in {"image", "video"}:
            path = Path(str(part["path"]))
            if media_url_format == "data_url":
                url = data_url(path, kind)
            elif media_url_format == "base64":
                url = encode_file(path)
            else:
                raise ValueError(f"unsupported media_url_format: {media_url_format!r}")
            url_key = "image_url" if kind == "image" else "video_url"
            content.append({"type": url_key, url_key: {"url": url}})
        else:
            raise ValueError(f"unsupported content part type: {kind}")
    return content


class OpenAICompatAdapter(Adapter):
    def __init__(self, model, protocol, runtime=None) -> None:
        super().__init__(model, protocol, runtime)
        api = model.api
        for required in ("url", "model"):
            if not api.get(required):
                raise ValueError(f"model {model.slug!r}: api.{required} is required "
                                 f"for the openai_compat adapter")
        self.url = str(api["url"])
        self.served_model = str(api["model"])
        self.media_url_format = str(api.get("media_url_format", "data_url"))
        self.extra_payload = dict(api.get("extra_payload") or {})
        self.extra_headers = dict(api.get("extra_headers") or {})
        self.min_video_seconds = float(api.get("min_video_seconds") or 0.0)
        # The request-body limit is a property of the serving endpoint, not of
        # the protocol; the protocol value is only the fallback.
        self.max_request_bytes = int(api.get("max_request_bytes",
                                             protocol.max_request_bytes))
        self.proxy = api.get("proxy")
        # The key is looked up at construction but only *required* at call
        # time: a missing credential should stop a call, not a dry run or a
        # configuration check. Preflight reports it as its own line item.
        self.key_env = api.get("key_env")
        self.api_key = os.environ.get(str(self.key_env), "") if self.key_env else ""

    # -- payload -----------------------------------------------------------

    def payload(self, parts: list[dict[str, Any]]) -> dict[str, Any]:
        """The request body. Separate from ``call`` so tests can pin it."""
        api = self.model.api
        payload: dict[str, Any] = {
            "model": self.served_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",
                 "content": openai_content(parts, self.media_url_format)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            # dialect escape hatch: merged last, may override anything above
            **self.extra_payload,
        }
        if api.get("send_thinking"):
            payload.setdefault("thinking", {"type": self.thinking})
        param = api.get("thinking_param")
        if param:
            payload.setdefault(str(param), self.thinking == "enabled")
        return payload

    # -- call --------------------------------------------------------------

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

        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        raw = post_with_retries(self.url, headers=headers,
                                payload=self.payload(parts),
                                timeout=self.protocol.timeout,
                                max_retries=self.protocol.max_retries,
                                proxy=self.proxy)
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        return AdapterResult(text=chat_completion_text(raw), raw=raw,
                             usage=usage, media_transforms=transforms)


def build(model, protocol, runtime=None) -> OpenAICompatAdapter:
    return OpenAICompatAdapter(model, protocol, runtime)
