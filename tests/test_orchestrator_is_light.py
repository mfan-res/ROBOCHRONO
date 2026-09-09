#!/usr/bin/env python3
# coding: utf-8
"""The orchestrator must not import an inference stack.

Environment switching works by starting a subprocess under the interpreter a
model needs. That only holds while the orchestrating process itself can run
anywhere — which means it must not import torch, transformers, or any of the
media libraries.

Nothing about that is self-enforcing. A single top-level import added to a
module the orchestrator loads would break it, and the failure would not appear
until someone ran from an environment that lacked the library. This test makes
the constraint explicit by blocking those imports and failing if one is
attempted.
"""
from __future__ import annotations

import importlib
import sys

FORBIDDEN = {"torch", "transformers", "decord", "cv2", "qwen_vl_utils", "PIL"}

# Modules the orchestrator loads; these must import from any environment.
LIGHT_MODULES = [
    "robochrono.config.protocol",
    "robochrono.config.models",
    "robochrono.config.suites",
    "robochrono.config.environments",
    "robochrono.dataset.manifest",
    "robochrono.dataset.loader",
    "robochrono.parsing",
    "robochrono.dimensions",
    "robochrono.results.store",
    "robochrono.results.runid",
    "robochrono.results.report",
    "robochrono.engine",
    "robochrono.media_prep",
    # Adapter modules must *import* everywhere — only load()/call() may touch
    # an inference stack, and only lazily. A top-level torch import in any of
    # these silently breaks dispatch-from-anywhere; this list is the lock.
    "robochrono.adapters",
    "robochrono.adapters.base",
    "robochrono.adapters.replay",
    "robochrono.adapters.qwen3_vl",
    "robochrono.adapters.rynnbrain",
    "robochrono.adapters.internvl",
    "robochrono.adapters.cosmos3_edge",
    "robochrono.adapters.openai_compat",
    "robochrono.adapters.gemini",
    "robochrono.config.runtime",
    "robochrono.orchestrate.matrix",
    "robochrono.orchestrate.execute",
    "robochrono.orchestrate.pool",
    "robochrono.orchestrate.dispatch",
    "robochrono.preflight",
    "robochrono.dataset.validate",
    "robochrono.cli",
]


class _Blocker:
    """Blocks inference-stack imports. Installed at the front of meta_path."""

    def find_module(self, fullname, path=None):
        return self.find_spec(fullname, path)

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in FORBIDDEN:
            raise AssertionError(
                f"the orchestrator imported {fullname!r}. This breaks dispatching "
                f"models to different interpreters: inference libraries belong in "
                f"robochrono/adapters/ and load only inside a worker subprocess.")
        return None


def main() -> int:
    for m in list(sys.modules):
        if m.split(".")[0] in FORBIDDEN or m.startswith("robochrono"):
            del sys.modules[m]

    sys.meta_path.insert(0, _Blocker())
    failed = []
    for name in LIGHT_MODULES:
        try:
            importlib.import_module(name)
            print(f"  ✅ {name}")
        except AssertionError as exc:
            print(f"  ❌ {name}\n     {exc}")
            failed.append(name)
    sys.meta_path.pop(0)

    print()
    if failed:
        print(f"FAILED: {len(failed)} module(s) pull in an inference stack")
        return 1
    print("orchestrator is import-light; it can start from any environment")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
