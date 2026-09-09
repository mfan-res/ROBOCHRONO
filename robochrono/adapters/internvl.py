#!/usr/bin/env python3
# coding: utf-8
"""Adapter for InternVL-architecture checkpoints (SenseNova-SI among them).

InternVL has no processor pipeline; the adapter owns the preprocessing the
model card prescribes — tiling each image into 448×448 crops chosen by aspect
ratio, plus a thumbnail — and talks to the model through its ``chat()``
method with ``<image>`` placeholders.

Tiling budgets come from the model's ``media`` configuration, because the
right budget depends on the footage: tiling a 640×480 frame into twelve crops
upsamples it 2.8× before slicing, spending visual tokens without adding
information.

The LLM attention implementation is forced to the configured one after
loading (``sdpa`` in the shipped configuration). The model code hard-wires a
choice between flash-attention and ``eager``, and eager materialises the full
fp32 attention matrix — which is an OOM on episode-length inputs that sdpa,
shipped with PyTorch, avoids; measured outputs are byte-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Adapter, AdapterResult, weights_path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> Any:
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: set[tuple[int, int]],
                              width: int, height: int, image_size: int) -> tuple[int, int]:
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target)
        if diff < best_diff:
            best_diff = diff
            best = ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def dynamic_preprocess(image: Any, *, min_num: int = 1, max_num: int = 12,
                       image_size: int = 448, use_thumbnail: bool = True) -> list[Any]:
    width, height = image.size
    aspect_ratio = width / height
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1)
                     for j in range(1, n + 1)
                     if min_num <= i * j <= max_num}
    ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size)
    target_width, target_height = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((target_width, target_height))
    tiles = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_image(path: Path, input_size: int, max_num: int) -> Any:
    import torch
    from PIL import Image

    image = Image.open(path).convert("RGB")
    tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num)
    transform = build_transform(input_size)
    return torch.stack([transform(tile) for tile in tiles])


def midpoint_indices(frame_count: int, max_frame: int) -> list[int]:
    """One index per segment, at the segment midpoint."""
    if frame_count <= 1:
        return [max_frame // 2]
    seg = max_frame / frame_count
    return [min(max_frame, int((seg / 2) + round(seg * idx)))
            for idx in range(frame_count)]


class InternVLAdapter(Adapter):
    def __init__(self, model, protocol, runtime=None) -> None:
        super().__init__(model, protocol, runtime)
        media = model.media
        for required in ("max_image_tiles", "max_video_tiles"):
            if required not in media:
                raise ValueError(f"model {model.slug!r}: media.{required} is required "
                                 f"for the internvl adapter — the right tiling budget "
                                 f"depends on the footage and must be a decision")
        self.max_image_tiles = int(media["max_image_tiles"])
        self.max_video_tiles = int(media["max_video_tiles"])
        self.use_flash_attn = bool(media.get("use_flash_attn", False))
        self.attn_implementation = str(media.get("attn_implementation", ""))
        self._model = None
        self._tokenizer = None

    # -- loading -----------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        weights = weights_path(self.model.weights)
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"{self.model.name}: weights not found at {weights}")
        print(f"Loading {self.model.name} from {weights}", flush=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = self.runtime.get("device_map")
        if device_map is None:
            device_map = {"": 0} if torch.cuda.is_available() else None

        # transformers renamed torch_dtype to dtype in 4.56; the two RynnBrain
        # generations alone span both sides of that line, so try both, and
        # with/without use_flash_attn, which older signatures reject.
        last_error: Exception | None = None
        for dtype_kwarg in ({"dtype": dtype}, {"torch_dtype": dtype}):
            for extra in ({"use_flash_attn": self.use_flash_attn}, {}):
                try:
                    self._model = AutoModel.from_pretrained(
                        weights, low_cpu_mem_usage=True, trust_remote_code=True,
                        device_map=device_map, **dtype_kwarg, **extra).eval()
                    last_error = None
                    break
                except TypeError as exc:
                    last_error = exc
            if last_error is None:
                break
        if last_error is not None:
            raise last_error

        # The model code offers only flash-attention or eager; sdpa has to be
        # forced in after construction, module by module.
        if self.attn_implementation:
            language_model = getattr(self._model, "language_model", None)
            if language_model is not None:
                language_model.config._attn_implementation = self.attn_implementation
                for module in language_model.modules():
                    cfg = getattr(module, "config", None)
                    if cfg is not None and hasattr(cfg, "_attn_implementation"):
                        cfg._attn_implementation = self.attn_implementation
                print(f"  LLM attn_implementation -> {self.attn_implementation}", flush=True)

        self._tokenizer = AutoTokenizer.from_pretrained(
            weights, trust_remote_code=True, use_fast=False, fix_mistral_regex=True)

    # -- frame counts ------------------------------------------------------

    def video_frame_count(self, path: Path, frames: dict[str, Any],
                          total_frames: int, fps: float) -> int:
        """The protocol's frame spec, converted through the shared rounding.

        ``frames_from_fps`` is the same function every adapter uses; a private
        conversion here would let the same declared fps sample a different
        number of frames on this model than on the others."""
        mode = frames.get("mode")
        if mode == "fps" and frames.get("value"):
            duration = total_frames / fps if fps > 0 else 0.0
            return min(total_frames,
                       self.protocol.frames_from_fps(duration, float(frames["value"])))
        if mode == "uniform" and frames.get("value"):
            return int(frames["value"])
        raise ValueError(f"{self.model.slug}: no usable frame spec {frames!r} — "
                         f"declare one in protocol.frames.by_dimension")

    def load_video(self, path: Path, frames: dict[str, Any]) -> tuple[Any, list[int]]:
        import torch
        from decord import VideoReader, cpu
        from PIL import Image

        video = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        max_frame = len(video) - 1
        if max_frame < 0:
            raise ValueError(f"empty video file: {path}")
        count = self.video_frame_count(path, frames, len(video),
                                       float(video.get_avg_fps()))
        transform = build_transform(self._input_size())
        pixel_values_list, patches = [], []
        for index in midpoint_indices(count, max_frame):
            image = Image.fromarray(video[index].asnumpy()).convert("RGB")
            tiles = dynamic_preprocess(image, image_size=self._input_size(),
                                       max_num=self.max_video_tiles)
            pixel_values = torch.stack([transform(tile) for tile in tiles])
            pixel_values_list.append(pixel_values)
            patches.append(pixel_values.shape[0])
        return torch.cat(pixel_values_list), patches

    def _input_size(self) -> int:
        return int(getattr(getattr(self._model, "config", None),
                           "force_image_size", 448) or 448)

    # -- call --------------------------------------------------------------

    def call(self, parts: list[dict[str, Any]], *, frames: dict[str, Any],
             key: str = "") -> AdapterResult:
        import torch

        self.load()
        model, tokenizer = self._model, self._tokenizer
        input_size = self._input_size()
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        pixel_values_list: list[Any] = []
        num_patches_list: list[int] = []
        prompt_chunks: list[str] = []
        text_chunks: list[str] = []
        frames_used: dict[str, int] = {}
        media_index = 1

        for part in parts:
            kind = part.get("type")
            if kind == "text":
                text_chunks.append(str(part.get("text", "")))
            elif kind == "image":
                pixel_values = load_image(Path(str(part["path"])), input_size,
                                          self.max_image_tiles)
                pixel_values_list.append(pixel_values)
                num_patches_list.append(pixel_values.shape[0])
                prompt_chunks.append(f"Image{media_index}: <image>\n")
                media_index += 1
            elif kind == "video":
                path = Path(str(part["path"]))
                video_pixels, patches = self.load_video(path, frames)
                frames_used[path.name] = len(patches)
                pixel_values_list.append(video_pixels)
                num_patches_list.extend(patches)
                prompt_chunks.extend(f"Video{media_index} Frame{i + 1}: <image>\n"
                                     for i in range(len(patches)))
                media_index += 1
            else:
                raise ValueError(f"unsupported content part type: {kind}")

        if not pixel_values_list:
            raise ValueError("internvl requires at least one image or video part")

        device = next(model.parameters()).device
        pixel_values = torch.cat(pixel_values_list).to(dtype).to(device)
        user_prompt = "\n".join(chunk for chunk in text_chunks if chunk)
        question = "".join(prompt_chunks) + user_prompt
        if self.system_prompt:
            question = f"{self.system_prompt}\n\n{question}"

        do_sample = self.temperature > 0
        generation_config: dict[str, Any] = {"max_new_tokens": self.max_new_tokens,
                                             "do_sample": do_sample}
        if do_sample:
            generation_config["temperature"] = self.temperature
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            generation_config["pad_token_id"] = eos

        with torch.no_grad():
            output = model.chat(tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list,
                                history=None, return_history=False)
        if isinstance(output, tuple):
            output = output[0]
        return AdapterResult(text=str(output).strip(), frames_used=frames_used)


def build(model, protocol, runtime=None) -> InternVLAdapter:
    return InternVLAdapter(model, protocol, runtime)
