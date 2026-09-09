#!/usr/bin/env python3
# coding: utf-8
"""What each adapter sends, pinned by structure and by fingerprint.

Construction must work without an inference stack — that is the load-bearing
property the environment switcher rests on — so everything here runs on a
machine with neither torch nor a GPU. Request structures are pinned by sha so
that an accidentally deleted branch cannot pass unnoticed just because the
code still runs.

DECLARED DIFFERENCE vs the pre-refactor implementation: **API adapters now
send the system prompt** (a ``system`` message on chat/completions,
``systemInstruction`` on Gemini). The provider configuration always declared
one, but the API request builders never sent it — the same
declared-but-not-sent defect family as ``max_tokens`` and the thinking
toggle, both fixed before. Every local adapter sends it, so omitting it
API-side made cross-model comparison unequal on exactly the axis the shared
protocol exists to fix.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochrono import adapters  # noqa: E402
from robochrono.adapters import cosmos3_edge, gemini, openai_compat, qwen3_vl  # noqa: E402
from robochrono.config.models import load_model, load_models  # noqa: E402
from robochrono.config.protocol import load_protocol  # noqa: E402

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:56} {detail}")
    if not passed:
        failures.append(name)


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


PROTOCOL = load_protocol(ROOT / "configs/protocol.json")
MODELS = load_models(ROOT / "configs/models")

print("1. every configured model builds its adapter, without torch")
EXPECTED_CLASS = {
    "qwen3_vl": "Qwen3VLAdapter", "rynnbrain": "RynnBrainAdapter",
    "internvl": "InternVLAdapter", "cosmos3_edge": "Cosmos3EdgeAdapter",
    "openai_compat": "OpenAICompatAdapter", "gemini": "GeminiAdapter",
    "replay": "ReplayAdapter",
}
import os
for env in ("DASHSCOPE_API_KEY", "GEMINI_API_KEY", "ARK_API_KEY"):
    os.environ.setdefault(env, "test-key")
for name, model in sorted(MODELS.items()):
    adapter = adapters.build(model, PROTOCOL)
    check(f"{model.slug} -> {type(adapter).__name__}",
          type(adapter).__name__ == EXPECTED_CLASS[model.adapter])
check("no inference stack was imported", "torch" not in sys.modules)

print("2. per-model generation settings resolve as decided")
EXPECTED_GENERATION = {
    "qwen3-vl-2b-instruct":          (0.0, 4096, "disabled"),
    "qwen3-vl-8b-instruct":          (0.0, 4096, "disabled"),
    "qwen3-vl-8b-thinking":          (0.0, 40960, "always_on"),
    "cosmos-reason2-2b":             (0.0, 4096, "disabled"),
    "cosmos3-edge-2b":               (0.0, 4096, "disabled"),
    "rynnbrain-2b":                  (0.0, 4096, "disabled"),
    "rynnbrain1-1-2b":               (0.0, 4096, "disabled"),
    "sensenova-si-1-1-internvl3-2b": (0.0, 4096, "disabled"),
    "qwen3-vl-30b-a3b-instruct":     (0.0, 4096, "disabled"),
    "qwen3-vl-32b-instruct":         (0.0, 4096, "disabled"),
    "rynnbrain1-1-9b":               (0.0, 4096, "disabled"),
    "internvl3-2b":                  (0.0, 4096, "disabled"),
    "qwen3-vl-235b-a22b-instruct":   (0.0, 4096, "disabled"),
    "rynnbrain1-1-122b-a10b":        (0.0, 4096, "disabled"),
    # Reasons under a budget it never approaches: the longest output
    # measured across 38k questions stays under a third of the baseline,
    # so unlike the other always-on model it needs no raised ceiling.
    "mimo-vl-7b-rl":                 (0.0, 4096, "always_on"),
    "qwen3-8-max":                   (0.0, 4096, "disabled"),
    "qwen3-8-max-thinking":          (0.0, 4096, "enabled"),
    "gemini-3-6-flash":              (0.0, 4096, "disabled"),
    "doubao-seed-2-0-lite":          (0.0, 4096, "disabled"),
    "qwen3-vl-235b-a22b-api":        (0.0, 4096, "disabled"),
    # Reports thoughtsTokenCount on every call and offers no switch, so the
    # thinking state is what it does rather than what we asked for.
    "gemini-robotics-er-2":          (0.0, 4096, "always_on"),
}
for name, model in sorted(MODELS.items()):
    if model.adapter == "replay":
        # A replay configuration is a debugging stand-in, not a measured
        # model; there are no shipped settings to pin it against.
        continue
    a = adapters.build(model, PROTOCOL)
    got = (a.temperature, a.max_new_tokens, a.thinking)
    want = EXPECTED_GENERATION[model.slug]
    check(f"{model.slug} generation", got == want, f"{got} vs {want}")

print("3. frame specs translate per framework")
check("qwen fps", qwen3_vl.qwen_video_extra({"mode": "fps", "value": 2.0}) == {"fps": 2.0})
check("qwen uniform rounds to the frame factor",
      qwen3_vl.qwen_video_extra({"mode": "uniform", "value": 15}) == {"nframes": 14})
check("qwen empty spec adds nothing", qwen3_vl.qwen_video_extra({}) == {})
check("cosmos fps", cosmos3_edge.cosmos_frame_kwargs({"mode": "fps", "value": 1.0}) == {"fps": 1.0})
check("cosmos uniform must null the built-in fps",
      cosmos3_edge.cosmos_frame_kwargs({"mode": "uniform", "value": 16})
      == {"num_frames": 16, "fps": None})

print("4. message structures, pinned")
PARTS = [{"type": "video", "path": "/abs/clip.mp4"}, {"type": "text", "text": "Q?"}]
qwen_messages = adapters.build(MODELS["Qwen3-VL-2B-Instruct"], PROTOCOL).messages(
    [{"type": "text", "text": "Q?"}], {"mode": "fps", "value": 2.0})
check("qwen sends the system prompt first",
      qwen_messages[0] == {"role": "system", "content": PROTOCOL.system_prompt})
check("qwen messages sha", sha(qwen_messages) == "9998ecc996ce", sha(qwen_messages))
cosmos_messages = adapters.build(MODELS["Cosmos3-Edge-2B"], PROTOCOL).messages(PARTS)
check("cosmos video becomes a url part",
      cosmos_messages[1]["content"][0] == {"type": "video", "url": "/abs/clip.mp4"})
check("cosmos messages sha", sha(cosmos_messages) == "a0aa6615091c", sha(cosmos_messages))

print("5. API payloads, pinned")


def api_model(adapter_name: str, **api):
    base = load_model(ROOT / "configs/models/local/qwen3-vl-2b-instruct.json")
    return dataclasses.replace(
        base, slug=f"test-{adapter_name}", adapter=adapter_name,
        api={"url": "https://api.example/v1/chat", "model": "served-model", **api})


TEXT_PARTS = [{"type": "text", "text": "Q?"}]
oc = openai_compat.build(api_model("openai_compat"), PROTOCOL)
payload = oc.payload(TEXT_PARTS)
check("openai system message first",
      payload["messages"][0] == {"role": "system", "content": PROTOCOL.system_prompt})
check("openai max_tokens actually sent", payload.get("max_tokens") == 4096)
check("openai payload sha", sha(payload) == "08f9a9e8020c", sha(payload))

oc = openai_compat.build(api_model("openai_compat", thinking_param="enable_thinking"), PROTOCOL)
check("thinking_param dialect sends the boolean",
      oc.payload(TEXT_PARTS).get("enable_thinking") is False)
oc = openai_compat.build(api_model("openai_compat", send_thinking=True), PROTOCOL)
check("send_thinking dialect sends the object",
      oc.payload(TEXT_PARTS).get("thinking") == {"type": "disabled"})
oc = openai_compat.build(
    api_model("openai_compat", extra_payload={"temperature": 0.7}), PROTOCOL)
check("extra_payload may override anything",
      oc.payload(TEXT_PARTS)["temperature"] == 0.7)

gm = gemini.build(api_model("gemini", url="https://api.example/{model}:generate"), PROTOCOL)
check("gemini url formats the model", gm.url == "https://api.example/served-model:generate")
gpayload = gm.payload(TEXT_PARTS)
check("gemini sends systemInstruction",
      gpayload["systemInstruction"] == {"parts": [{"text": PROTOCOL.system_prompt}]})
check("gemini maxOutputTokens actually sent",
      gpayload["generationConfig"]["maxOutputTokens"] == 4096)
check("gemini payload sha", sha(gpayload) == "e6dbe3891318", sha(gpayload))
gm = gemini.build(api_model(
    "gemini", url="u", extra_payload={"generationConfig": {"thinkingConfig": {"x": 1}}}), PROTOCOL)
check("gemini generationConfig merges key-by-key",
      gm.payload(TEXT_PARTS)["generationConfig"].get("temperature") == 0.0
      and gm.payload(TEXT_PARTS)["generationConfig"]["thinkingConfig"] == {"x": 1})

print("5b. per-card memory budgets translate for accelerate")
from robochrono.adapters.base import max_memory_map  # noqa: E402
check("caps with a default",
      max_memory_map({"0": "55GiB", "default": "68GiB"}, 4)
      == {0: "55GiB", 1: "68GiB", 2: "68GiB", 3: "68GiB"})
check("no default caps only named cards",
      max_memory_map({"0": "55GiB"}, 3) == {0: "55GiB"})
check("the 235B declaration parses",
      max_memory_map(MODELS["Qwen3-VL-235B-A22B-Instruct"].resources["max_memory"], 8)[0]
      == MODELS["Qwen3-VL-235B-A22B-Instruct"].resources["max_memory"]["0"])

print("6. required declarations are enforced")
try:
    openai_compat.build(dataclasses.replace(api_model("openai_compat"), api={}), PROTOCOL)
    check("openai without api.url refuses", False, "no exception")
except ValueError:
    check("openai without api.url refuses", True)
try:
    internvl_model = dataclasses.replace(
        MODELS["SenseNova-SI-1.1-InternVL3-2B"], media={})
    adapters.build(internvl_model, PROTOCOL)
    check("internvl without tiling budgets refuses", False, "no exception")
except ValueError as e:
    check("internvl without tiling budgets refuses", "max_image_tiles" in str(e))

check("still no inference stack imported", "torch" not in sys.modules)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
