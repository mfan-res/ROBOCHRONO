#!/usr/bin/env python3
# coding: utf-8
"""Per-model settings, compared field by field against the pre-refactor code.

The refactor discipline: a pure restructuring must resolve the same values as
the code it replaces, and every intended change is declared here — with its
reason — so an undeclared difference reads as "we broke it", not as noise.

The reference implementation lives in the internal working repository and is
imported in a subprocess (the two codebases share a package name). When it is
not present, this test prints why and skips: the pinned payload fingerprints
in test_adapter_payloads.py still guard the request structure.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochrono import adapters  # noqa: E402
from robochrono.config.models import load_models  # noqa: E402
from robochrono.config.protocol import load_protocol  # noqa: E402

# The reference tree used to live in `bench/`, which has been retired; a copy
# of it is kept with this repository's own notes. Either location works, and
# ROBOCHRONO_BENCH_EVAL overrides both.
BENCH_EVAL = Path(os.environ.get(
    "ROBOCHRONO_BENCH_EVAL",
    Path(__file__).resolve().parents[1] / "dev/archive/bench-lineage/eval-code"))
SKIP = 0

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:56} {detail}")
    if not passed:
        failures.append(name)


# slug here -> provider name there
PROVIDER = {
    "qwen3-vl-2b-instruct": "local_qwen3_vl_2b",
    "qwen3-vl-8b-instruct": "local_qwen",
    "qwen3-vl-8b-thinking": "local_qwen3_vl_8b_thinking",
    "cosmos-reason2-2b": "local_cosmos_reason2_2b",
    "cosmos3-edge-2b": "local_cosmos3_edge_2b",
    "rynnbrain-2b": "local_rynnbrain_2b",
    "rynnbrain1-1-2b": "local_rynnbrain1_1_2b",
    "sensenova-si-1-1-internvl3-2b": "local_sensenova_si_1_1_internvl3_2b",
    "qwen3-vl-30b-a3b-instruct": "local_qwen3_vl_30b_a3b",
    "qwen3-vl-32b-instruct": "local_qwen3_vl_32b",
    "rynnbrain1-1-9b": "local_rynnbrain1_1_9b",
    "internvl3-2b": "local_internvl3_2b",
    "qwen3-vl-235b-a22b-instruct": "local_qwen3_vl_235b",
    "rynnbrain1-1-122b-a10b": "local_rynnbrain1_1_122b",
}

# ── Declared differences ──────────────────────────────────────────────────
# (slug, field) -> why. Everything not listed must match exactly.
DECLARED = {
    ("qwen3-vl-8b-thinking", "thinking"):
        'was "disabled", now "always_on". The model cannot disable thinking; '
        'recording "disabled" wrote down a uniformity that was never delivered, '
        'and a recorded-but-false uniformity invites comparisons that are not '
        'valid. The model now declares its real state.',
    ("cosmos3-edge-2b", "timeout"):
        "was 1800, now the protocol's 300. The 1800 served a subprocess-based "
        "inference path that was retired; the transformers path never reads a "
        "timeout, so the field is inert for this model either way.",
}
# frames: declared wholesale. The reference sampled a fixed count (uniform 16,
# and uniform 8 on the InternVL provider alone); the protocol now fixes
# temporal density (fps 2.0, action_time 1.0) for every model, because with a
# fixed count video length confounds every comparison, and a per-model count
# measured different models on different evidence.
FRAMES_DECLARED = True

if not BENCH_EVAL.exists():
    print(f"skip: reference implementation not found at {BENCH_EVAL}")
    print("      set ROBOCHRONO_BENCH_EVAL to compare against it.")
    sys.exit(SKIP)

dump = subprocess.run(
    [sys.executable, "-", str(BENCH_EVAL)], text=True, capture_output=True,
    input="""
import json, sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from robochrono import vlm_api
providers = json.loads(sys.stdin.read()) if False else %s
out = {}
for slug, name in providers.items():
    r = vlm_api.runtime_config(Path(sys.argv[1]) / "configs/providers.json", name,
                               default_model="x")
    out[slug] = {k: r.get(k) for k in
                 ("temperature", "max_new_tokens", "thinking", "system_prompt",
                  "timeout", "max_retries", "frames")}
print(json.dumps(out))
""" % json.dumps(PROVIDER))
if dump.returncode != 0:
    print(f"skip: reference dump failed:\n{dump.stderr[-800:]}")
    sys.exit(SKIP)
reference = json.loads(dump.stdout)

protocol = load_protocol(ROOT / "configs/protocol.json")
models = {m.slug: m for m in load_models(ROOT / "configs/models").values()}

print("1. generation fields match, or the difference is declared")
for slug, provider in PROVIDER.items():
    old = reference[slug]
    a = adapters.build(models[slug], protocol)
    new = {"temperature": a.temperature, "max_new_tokens": a.max_new_tokens,
           "thinking": a.thinking, "system_prompt": a.system_prompt,
           "timeout": protocol.timeout, "max_retries": protocol.max_retries}
    for field in ("temperature", "max_new_tokens", "thinking",
                  "system_prompt", "timeout", "max_retries"):
        if old[field] == new[field]:
            check(f"{slug}.{field}", True)
        elif (slug, field) in DECLARED:
            check(f"{slug}.{field} (declared)", True,
                  f"{old[field]!r} -> {new[field]!r}")
        else:
            check(f"{slug}.{field}", False,
                  f"UNDECLARED: {old[field]!r} -> {new[field]!r}")

print("2. frame sampling: declared change from fixed count to fixed density")
old_frames = {slug: reference[slug]["frames"] for slug in PROVIDER}
uniform = {json.dumps({k: v for k, v in f.items() if k in ("mode", "value")},
                      sort_keys=True) for f in old_frames.values()}
check("reference sampled fixed counts",
      all(f["mode"] == "uniform" for f in old_frames.values()),
      f"{sorted(uniform)}")
new_specs = {d: protocol.frames_for(d) for d in
             ("current_action", "action_time")}
check("protocol now fixes density (declared)",
      FRAMES_DECLARED and all(s["mode"] == "fps" for s in new_specs.values()),
      json.dumps(new_specs))

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
