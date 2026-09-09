#!/usr/bin/env python3
# coding: utf-8
"""Adapter for the RynnBrain checkpoints.

RynnBrain ships a Qwen3-VL-compatible chat template and takes its vision
inputs through ``qwen_vl_utils``, so the whole call path is the Qwen3-VL
adapter's. What differs is packaging, not behaviour: the two generations pin
mutually exclusive transformers versions (4.x vs 5.x), which is an
environment mapping in the model configuration, not a code path here. The
class exists so a future divergence has a place to land without touching the
Qwen adapter.
"""

from __future__ import annotations

from .qwen3_vl import Qwen3VLAdapter


class RynnBrainAdapter(Qwen3VLAdapter):
    pass


def build(model, protocol, runtime=None) -> RynnBrainAdapter:
    return RynnBrainAdapter(model, protocol, runtime)
