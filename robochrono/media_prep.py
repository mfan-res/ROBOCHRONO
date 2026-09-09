#!/usr/bin/env python3
# coding: utf-8
"""Fitting media into an API request budget. Local models never need this.

Remote providers inline media as base64, and servers cap the request body —
the measured limit on one production endpoint sits near 10 MiB, which full
episodes exceed. Without preparation, the episode-based dimension simply
cannot run on API models. Some endpoints also refuse videos shorter than a
minimum duration, which silently drops questions API-side that local models
answer — and a cross-model comparison with different question sets under each
model is not a comparison.

Three rules:

1. **Off by default.** Only a model configuration that declares
   ``max_request_bytes`` or ``min_video_seconds`` activates this; otherwise
   media is sent as-is and an oversized request fails loudly and is recorded.
2. **Spatial resolution first, never the time axis.** Timing is what the
   episode dimension measures; frame sampling belongs to the serving side.
   Videos that must grow to a minimum duration are padded with a **frozen
   last frame** — looping would show actions happening twice and speed change
   would alter motion, either of which can change the answer.
3. **Every transformation is recorded** — sizes, scale, CRF — and lands in
   the result row, so what the model actually saw stays auditable.

Products are cached by content, so one video is re-encoded once.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

# base64 turns 3 bytes into 4 characters
BASE64_RATIO = 4 / 3
# prompt text and JSON structure ride along with the media
OVERHEAD_BYTES = 64 * 1024

# Tried in order: resolution first, quality only when resolution is exhausted.
SCALE_LADDER = (0.75, 0.5, 0.375, 0.25, 0.1875)
CRF_LADDER = (23, 28, 32)


def encoded_size(path: Path) -> int:
    return int(path.stat().st_size * BASE64_RATIO)


def _cache_path(source: Path, scale: float, crf: int, cache_dir: Path) -> Path:
    stat = source.stat()
    digest = hashlib.md5(
        f"{source.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{scale}|{crf}".encode()
    ).hexdigest()[:16]
    return cache_dir / f"{source.stem}__{digest}.mp4"


def _reencode(source: Path, dest: Path, scale: float, crf: int) -> bool:
    """Re-encode to ``dest``, via a temp file and an atomic rename.

    Concurrent API threads can be shrinking the same video; writing ``dest``
    directly would let one thread ship another's half-written file as a cache
    hit.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.stem}.{os.getpid()}.{threading.get_ident()}.tmp.mp4")
    # trunc(...*scale/2)*2 keeps dimensions even, which libx264 requires
    vf = f"scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(source), "-vf", vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-an", str(tmp)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            return False
        os.replace(tmp, dest)          # same directory: atomic
        return True
    finally:
        tmp.unlink(missing_ok=True)    # failure path; success was moved away


def _record(source: Path, dest: Path, scale: float, crf: int) -> dict[str, Any]:
    return {"source": str(source), "prepared": str(dest),
            "scale": scale, "crf": crf,
            "source_encoded_bytes": encoded_size(source),
            "prepared_encoded_bytes": encoded_size(dest)}


def shrink_video(source: Path, budget_bytes: int,
                 cache_dir: Path) -> tuple[Path, dict[str, Any] | None]:
    """Shrink until the base64 size fits ``budget_bytes``.

    Returns (usable path, transformation record). When nothing fits, returns
    the original path with ``{"failed": True}`` — the caller decides whether
    to give up or send anyway.
    """
    if encoded_size(source) <= budget_bytes:
        return source, None

    for crf in CRF_LADDER:
        for scale in SCALE_LADDER:
            dest = _cache_path(source, scale, crf, cache_dir)
            if dest.exists() and dest.stat().st_size > 0:
                if encoded_size(dest) <= budget_bytes:
                    return dest, _record(source, dest, scale, crf)
                continue
            if not _reencode(source, dest, scale, crf):
                continue
            if encoded_size(dest) <= budget_bytes:
                return dest, _record(source, dest, scale, crf)
            # Still over budget: keep it anyway. Deleting could remove a file
            # another thread just validated, and the next run would have to
            # re-encode just to relearn that this rung is too big.

    return source, {"source": str(source), "failed": True,
                    "reason": "cannot shrink into the budget",
                    "source_encoded_bytes": encoded_size(source),
                    "budget_bytes": budget_bytes}


def video_duration(path: Path) -> float | None:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def pad_video(source: Path, min_seconds: float,
              cache_dir: Path) -> tuple[Path, dict[str, Any] | None]:
    """Extend a too-short video to ``min_seconds`` with a frozen last frame."""
    duration = video_duration(source)
    if duration is None or duration >= min_seconds:
        return source, None

    # 0.1s over, so server-side frame rounding cannot land just short again
    pad = min_seconds - duration + 0.1
    digest = hashlib.md5(f"{source.resolve()}|pad|{min_seconds}".encode()).hexdigest()[:16]
    dest = cache_dir / f"{source.stem}__pad{digest}.mp4"

    if not (dest.exists() and dest.stat().st_size > 0):
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.stem}.{os.getpid()}.{threading.get_ident()}.tmp.mp4")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
               "-vf", f"tpad=stop_mode=clone:stop_duration={pad:.3f}",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an", str(tmp)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
                return source, {"source": str(source), "failed": True,
                                "reason": f"padding failed: {result.stderr.strip()[:120]}"}
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)

    return dest, {"source": str(source), "prepared": str(dest),
                  "op": "pad_to_min_duration",
                  "source_seconds": round(duration, 3),
                  "target_seconds": min_seconds,
                  "padded_seconds": round(pad, 3)}


def prepare_parts(
    parts: list[dict[str, Any]],
    max_request_bytes: int,
    cache_dir: Path,
    min_video_seconds: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare parts against a whole-request budget.

    Two passes whose order matters: padding first (a server-side hard
    constraint), then shrinking — padding grows files slightly, so the budget
    must be computed after it. The budget covers the **whole request**, not
    each file: one call can carry a clip and half a dozen option images.
    Images are left alone — they are small, and they are the answer.
    """
    media = [p for p in parts if p.get("type") in {"image", "video"}]
    if not media:
        return parts, []

    transforms: list[dict[str, Any]] = []
    replacement: dict[str, str] = {}
    videos = [p for p in media if p["type"] == "video"]

    if min_video_seconds > 0:
        for part in videos:
            padded, record = pad_video(Path(part["path"]), min_video_seconds, cache_dir)
            if record is not None:
                transforms.append(record)
            if str(padded) != part["path"]:
                replacement[part["path"]] = str(padded)

    def current(part: dict[str, Any]) -> Path:
        return Path(replacement.get(part["path"], part["path"]))

    budget = max_request_bytes - OVERHEAD_BYTES
    image_bytes = sum(encoded_size(Path(p["path"])) for p in media if p["type"] == "image")
    total = image_bytes + sum(encoded_size(current(p)) for p in videos)

    if videos and total > budget:
        per_video = max(1, (budget - image_bytes) // len(videos))
        for part in videos:
            source = current(part)
            prepared, record = shrink_video(source, per_video, cache_dir)
            if record is not None:
                transforms.append(record)
            if prepared != source:
                replacement[part["path"]] = str(prepared)

    if not replacement:
        return parts, transforms
    new_parts = [{**p, "path": replacement[p["path"]]}
                 if p.get("path") in replacement else p for p in parts]
    return new_parts, transforms
