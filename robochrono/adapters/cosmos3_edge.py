#!/usr/bin/env python3
# coding: utf-8
"""Adapter for Cosmos3-Edge through its transformers processor.

The frame spec goes to the processor through ``processor_kwargs`` — its
formally declared channel. Left unset, the processor applies its own default
of fps=2, and two declared sampling tiers then produce bit-identical results
without an error anywhere; only comparing the numbers reveals it. In this
processor ``num_frames`` and ``fps`` are mutually exclusive, and the built-in
fps must be nulled out for ``num_frames`` to take effect at all.

The frames actually used are read off ``video_grid_thw`` — its time dimension
is the sampled frame count, no reconstruction needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Adapter, AdapterResult, generation_kwargs, weights_path


def cosmos_frame_kwargs(frames: dict[str, Any]) -> dict[str, Any]:
    mode = str(frames.get("mode") or "")
    value = frames.get("value")
    if mode == "fps" and value:
        return {"fps": float(value)}
    if mode == "uniform" and value:
        return {"num_frames": int(value), "fps": None}
    return {}


def cosmos_content(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in parts:
        kind = part.get("type")
        if kind == "text":
            content.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind in {"image", "video"}:
            path = Path(str(part["path"])).expanduser()
            resolved = path if path.is_absolute() else path.resolve()
            content.append({"type": str(kind), "url": str(resolved)})
        else:
            raise ValueError(f"unsupported content part type: {kind}")
    return content


class Cosmos3EdgeAdapter(Adapter):
    def __init__(self, model, protocol, runtime=None) -> None:
        super().__init__(model, protocol, runtime)
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        weights = weights_path(self.model.weights)
        print(f"Loading {self.model.name} from {weights}", flush=True)
        self._processor = AutoProcessor.from_pretrained(weights, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            weights,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self._model.eval()

    def messages(self, parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": cosmos_content(parts)},
        ]

    def call(self, parts: list[dict[str, Any]], *, frames: dict[str, Any],
             key: str = "") -> AdapterResult:
        import torch

        self.load()
        model, processor = self._model, self._processor
        processor_kwargs = cosmos_frame_kwargs(frames)
        inputs = processor.apply_chat_template(
            self.messages(parts),
            tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            **({"processor_kwargs": processor_kwargs} if processor_kwargs else {}))

        frames_used: dict[str, int] = {}
        grid = inputs.get("video_grid_thw")
        if grid is not None:
            for index, row in enumerate(grid):
                frames_used[f"video{index}"] = int(row[0])

        device = getattr(model, "device", None) or next(model.parameters()).device
        inputs = inputs.to(device)

        eos = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **generation_kwargs(self.temperature, self.max_new_tokens, eos))
        new_ids = [output[len(input_ids):]
                   for input_ids, output in zip(inputs["input_ids"], output_ids)]
        output = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        return AdapterResult(text=output, frames_used=frames_used)


def build(model, protocol, runtime=None) -> Cosmos3EdgeAdapter:
    return Cosmos3EdgeAdapter(model, protocol, runtime)
