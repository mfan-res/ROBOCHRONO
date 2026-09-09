#!/usr/bin/env python3
# coding: utf-8
"""Adapter for models served through the Qwen3-VL chat template.

That is more than the Qwen family: any checkpoint whose processor follows the
same template and whose vision inputs go through ``qwen_vl_utils`` runs here
(Cosmos-Reason2 among them; RynnBrain subclasses this adapter unchanged).

The frame-sampling spec must be passed to ``qwen_vl_utils`` **explicitly** —
left alone, the library applies its own default (fps=2), and the protocol's
sampling declaration silently stops describing what actually happens. The
number of frames the library then really sampled is read back off the video
tensors and recorded, because it cannot be reconstructed afterwards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (Adapter, AdapterResult, generation_kwargs, max_memory_map,
                   weights_path)


def qwen_video_extra(frames: dict[str, Any]) -> dict[str, Any]:
    """Translate the protocol's frame spec into qwen_vl_utils video kwargs.

    The library reads ``fps`` or ``nframes`` off the video element; the two
    are mutually exclusive, and ``nframes`` must be a multiple of the
    library's FRAME_FACTOR (2)."""
    mode = frames.get("mode")
    if mode == "fps" and frames.get("value"):
        return {"fps": float(frames["value"])}
    if mode == "uniform" and frames.get("value"):
        n = max(2, int(frames["value"]))
        return {"nframes": n - (n % 2)}
    return {}


def single_frame_fallback(path: str) -> str | None:
    """A one-frame video is an image; send it as one.

    qwen_vl_utils merges frames pairwise (FRAME_FACTOR=2) and rejects a
    single-frame video outright. The extracted frame is cached beside the
    video and produced once."""
    try:
        from decord import VideoReader, cpu

        video = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        if len(video) >= 2:
            return None
        cache = Path(str(path)).with_suffix(".singleframe.jpg")
        if not cache.exists():
            from PIL import Image

            Image.fromarray(video[0].asnumpy()).save(cache, quality=95)
        return str(cache)
    except Exception:  # noqa: BLE001 — fall through to sending the video as-is
        return None


def qwen_content(parts: list[dict[str, Any]],
                 frames: dict[str, Any]) -> list[dict[str, Any]]:
    video_extra = qwen_video_extra(frames)
    content: list[dict[str, Any]] = []
    for part in parts:
        kind = part.get("type")
        if kind == "text":
            content.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind == "image":
            content.append({"type": "image", "image": str(part["path"])})
        elif kind == "video":
            path = str(part["path"])
            frame = single_frame_fallback(path)
            if frame is not None:
                content.append({"type": "image", "image": frame})
            else:
                content.append({"type": "video", "video": path, **video_extra})
        else:
            raise ValueError(f"unsupported content part type: {kind}")
    return content


class Qwen3VLAdapter(Adapter):
    def __init__(self, model, protocol, runtime=None) -> None:
        super().__init__(model, protocol, runtime)
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch

        if not hasattr(torch, "float8_e8m0fnu"):   # older torch, newer checkpoints
            torch.float8_e8m0fnu = torch.uint8
        from transformers import AutoModelForImageTextToText, AutoProcessor

        weights = weights_path(self.model.weights)
        print(f"Loading {self.model.name} from {weights}", flush=True)
        self._processor = AutoProcessor.from_pretrained(weights, trust_remote_code=True)
        extra: dict[str, Any] = {}
        memory_spec = self.model.resources.get("max_memory")
        if memory_spec and torch.cuda.is_available():
            extra["max_memory"] = max_memory_map(memory_spec, torch.cuda.device_count())
        self._model = AutoModelForImageTextToText.from_pretrained(
            weights,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            **extra,
        )
        self._model.eval()

    def messages(self, parts: list[dict[str, Any]],
                 frames: dict[str, Any]) -> list[dict[str, Any]]:
        """The chat messages as sent. Separate from ``call`` so tests can pin it."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": qwen_content(parts, frames)},
        ]

    def call(self, parts: list[dict[str, Any]], *, frames: dict[str, Any],
             key: str = "") -> AdapterResult:
        import torch
        from qwen_vl_utils import process_vision_info

        self.load()
        model, processor = self._model, self._processor
        messages = self.messages(parts, frames)
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True, return_video_metadata=True)
        except TypeError:
            image_inputs, video_inputs = process_vision_info(messages)
            video_kwargs = {}

        frames_used: dict[str, int] = {}
        for index, video in enumerate(video_inputs or []):
            tensor = video[0] if isinstance(video, (tuple, list)) else video
            shape = getattr(tensor, "shape", None)
            if shape is not None and len(shape) >= 1:
                frames_used[f"video{index}"] = int(shape[0])

        if video_inputs and isinstance(video_inputs[0], tuple):
            videos, metadata = [], []
            for video, meta in video_inputs:
                videos.append(video)
                metadata.append(meta)
            video_inputs = videos
            video_kwargs["video_metadata"] = metadata

        device = next(model.parameters()).device
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt", **video_kwargs).to(device)

        eos = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **generation_kwargs(self.temperature, self.max_new_tokens, eos))
        new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        output = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        return AdapterResult(text=output, frames_used=frames_used)


def build(model, protocol, runtime=None) -> Qwen3VLAdapter:
    return Qwen3VLAdapter(model, protocol, runtime)
