#!/usr/bin/env python3
# coding: utf-8
"""Pre-run checks: what would fail at minute 180 should fail at minute 0.

A full matrix runs for days. Anything discoverable up front — a missing key,
absent weights, an interpreter in the wrong environment, data that is not the
dataset the suite pins — is checked here before a single model call.

Four verdict levels:

    FAIL  the run cannot proceed; fix it
    WARN  it can run, but results may be wrong or incomplete; know about it
    OK    checked and fine
    SKIP  not needed for this selection — an API-only run does not need the
          local inference stack, and saying FAIL there would demand twenty
          gigabytes of torch from someone who only wants to call an endpoint

Only FAIL affects the exit status.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import adapters
from .config.environments import Environment, satisfies
from .config.models import ModelConfig
from .config.protocol import Protocol
from .config.suites import Suite
from .dataset.loader import load_questions, media_paths, resolve_media
from .dataset.manifest import load_scenario_manifest
from .dataset.render import load_question_bank
from .orchestrate.matrix import RunSpec

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"

API_PACKAGES = ("requests",)
LOCAL_PACKAGES = ("torch", "transformers", "decord", "PIL", "qwen_vl_utils",
                  "torchvision")


@dataclass
class Check:
    level: str
    name: str
    detail: str = ""


def has_failures(checks: list[Check]) -> bool:
    return any(c.level == FAIL for c in checks)


def format_checks(checks: list[Check]) -> str:
    mark = {OK: "✅", WARN: "⚠️ ", FAIL: "❌", SKIP: "⏭️ "}
    lines = [f"  {mark[c.level]} [{c.level:4}] {c.name:44} {c.detail}".rstrip()
             for c in checks]
    fails = sum(1 for c in checks if c.level == FAIL)
    warns = sum(1 for c in checks if c.level == WARN)
    lines.append("")
    lines.append(f"{len(checks)} checks: {fails} FAIL, {warns} WARN"
                 if fails or warns else f"{len(checks)} checks, all clear")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Individual check groups
# --------------------------------------------------------------------------

def check_configuration(models: list[ModelConfig], suite: Suite,
                        protocol: Protocol,
                        environments: dict[str, Environment]) -> list[Check]:
    out: list[Check] = []
    for model in models:
        try:
            adapters.build(model, protocol)
            out.append(Check(OK, f"{model.slug} configuration"))
        except Exception as exc:  # noqa: BLE001 — any config defect is the finding
            out.append(Check(FAIL, f"{model.slug} configuration", str(exc)))
        if model.environment not in environments:
            out.append(Check(FAIL, f"{model.slug} environment",
                             f"{model.environment!r} is not defined in "
                             f"configs/environments.json"))
    missing_frames = [d for d in suite.dimensions
                      if d not in protocol.frames_by_dimension]
    out.append(Check(FAIL if missing_frames else OK, "frame sampling declared",
                     f"missing: {missing_frames}" if missing_frames
                     else f"{len(suite.dimensions)} dimensions"))
    return out


def _current_environment(environments: dict[str, Environment],
                         repo_root: Any) -> Environment | None:
    """Which declared environment is this interpreter, if any?

    ``sys.prefix`` identifies the venv itself; comparing ``sys.executable``
    would follow the bin/python symlink to the base interpreter and compare
    the wrong thing.
    """
    here = Path(sys.prefix).resolve()
    for env in environments.values():
        raw = str(env.python or "").strip()
        if not raw:
            continue
        prefix = Path(raw)
        if not prefix.is_absolute():
            prefix = Path(repo_root) / prefix
        if prefix.parent.parent.resolve() == here:
            return env
    return None


def check_environment(models: list[ModelConfig],
                      environments: dict[str, Environment],
                      repo_root: Any) -> list[Check]:
    """Is this interpreter one of the declared environments, with the stack
    the selected models need?"""
    out: list[Check] = []
    needs_local = any(m.kind == "local" for m in models)

    current = _current_environment(environments, repo_root)
    out.append(Check(OK if current else WARN, "interpreter environment",
                     f"{current.name}" if current else
                     f"{sys.prefix} is none of the declared environments — "
                     f"fine for dispatching, not for running local models"))

    packages = (LOCAL_PACKAGES if needs_local else ()) + API_PACKAGES
    if not needs_local:
        for name in LOCAL_PACKAGES:
            out.append(Check(SKIP, f"package {name}",
                             "API-only selection needs no local stack"))
    for name in packages:
        try:
            module = importlib.import_module(name)
        except ImportError:
            out.append(Check(FAIL, f"package {name}", "not importable"))
            continue
        version = getattr(module, "__version__", "")
        if name == "transformers" and current:
            ok = satisfies(f"=={current.transformers}", version)
            out.append(Check(OK if ok else FAIL, "package transformers",
                             f"{version} vs {current.transformers} declared for "
                             f"{current.name}"))
        else:
            out.append(Check(OK, f"package {name}", version))
    return out


# Runs under the probed environment's interpreter, so it must not assume the
# repository is importable there: stdlib only, package names via argv, one
# JSON object out. A string value is the imported version; an object records
# why the import failed.
_PROBE = """\
import importlib, json, sys
report = {}
for name in sys.argv[1:]:
    try:
        module = importlib.import_module(name)
        report[name] = getattr(module, "__version__", "")
    except Exception as exc:
        report[name] = {"error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(report))
"""


def _sync_command(env: Environment) -> str:
    prefix = Path(str(env.python)).parent.parent
    return (f"UV_PROJECT_ENVIRONMENT={prefix} uv sync "
            f"--extra {env.extra} --python 3.11")


def check_foreign_environments(models: list[ModelConfig],
                               environments: dict[str, Environment],
                               repo_root: Any) -> list[Check]:
    """The declared environments this selection needs beyond this interpreter.

    ``eval`` dispatches each model to the interpreter its environment names,
    so a broken second environment would otherwise surface only when its
    child starts — hours into a long matrix. Probe each such interpreter now:
    a subprocess imports the packages the models mapped there will need and
    reports versions, which are compared against configs/environments.json.
    Environments the selection does not touch are not probed — an import of
    torch costs seconds, and paying it for no model would be pure overhead.
    """
    current = _current_environment(environments, repo_root)
    needed = sorted({m.environment for m in models
                     if m.environment in environments
                     and (current is None or m.environment != current.name)})
    if not needed:
        return [Check(SKIP, "other environments",
                      "selection runs entirely in this interpreter's "
                      "environment")]

    out: list[Check] = []
    for name in needed:
        env = environments[name]
        local_here = any(m.kind == "local" and m.environment == name
                         for m in models)
        packages = (LOCAL_PACKAGES if local_here else ()) + API_PACKAGES

        python = Path(str(env.python))
        if not python.is_absolute():
            python = Path(repo_root) / python
        if not python.exists():
            out.append(Check(FAIL, f"{name} interpreter",
                             f"{python} not found — build it: "
                             f"{_sync_command(env)}"))
            continue
        try:
            proc = subprocess.run([str(python), "-c", _PROBE, *packages],
                                  capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            out.append(Check(FAIL, f"{name} interpreter",
                             f"probe timed out after 120s under {python}"))
            continue
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()
            out.append(Check(FAIL, f"{name} interpreter",
                             f"probe failed under {python}: "
                             f"{tail[-1] if tail else 'no output'}"))
            continue

        out.append(Check(OK, f"{name} interpreter", str(python)))
        report = json.loads(proc.stdout)
        for pkg in packages:
            got = report.get(pkg)
            if not isinstance(got, str):
                why = got.get("error", "not importable") if got else "not probed"
                out.append(Check(FAIL, f"{name} package {pkg}",
                                 f"{why} — rebuild it: {_sync_command(env)}"))
            elif pkg == "transformers":
                ok = bool(got) and satisfies(f"=={env.transformers}", got)
                out.append(Check(
                    OK if ok else FAIL, f"{name} package transformers",
                    f"{got} vs {env.transformers} declared for {name}"
                    + ("" if ok else f" — rebuild it: {_sync_command(env)}")))
            else:
                out.append(Check(OK, f"{name} package {pkg}", got))
    return out


def check_gpu(models: list[ModelConfig]) -> list[Check]:
    if not any(m.kind == "local" for m in models):
        return [Check(SKIP, "GPU", "API-only selection")]
    try:
        import torch
    except ImportError:
        return [Check(FAIL, "GPU", "torch is not importable")]
    if not torch.cuda.is_available():
        return [Check(FAIL, "GPU", "no CUDA device visible")]
    return [Check(OK, "GPU", f"{torch.cuda.device_count()} device(s)")]


def check_weights(models: list[ModelConfig], repo_root: Any) -> list[Check]:
    out: list[Check] = []
    for model in models:
        if model.kind != "local":
            out.append(Check(SKIP, f"{model.slug} weights", "API model"))
            continue
        path = Path(model.weights)
        if not path.is_absolute():
            path = Path(repo_root) / path
        out.append(Check(OK, f"{model.slug} weights", str(path)) if path.exists()
                   else Check(FAIL, f"{model.slug} weights",
                              f"{path} not found — download it first; "
                              f"docs/TUTORIAL.md §1.4 maps every model "
                              f"to its repository"))
    return out


def check_api_keys(models: list[ModelConfig]) -> list[Check]:
    out: list[Check] = []
    for model in models:
        if model.kind != "api":
            continue
        key_env = model.api.get("key_env")
        if not key_env:
            out.append(Check(WARN, f"{model.slug} key", "no key_env declared"))
        elif os.environ.get(str(key_env)):
            out.append(Check(OK, f"{model.slug} key", f"{key_env} is set"))
        else:
            out.append(Check(FAIL, f"{model.slug} key", f"{key_env} is not set"))
    needs_ffmpeg = any(m.kind == "api" for m in models)
    if needs_ffmpeg:
        found = shutil.which("ffmpeg") and shutil.which("ffprobe")
        out.append(Check(OK if found else WARN, "ffmpeg",
                         "" if found else "not on PATH — media over the request "
                         "budget cannot be shrunk and will fail loudly"))
    return out



HASH_TOOL = Path("tools/compute_scenario_hash.py")


def _content_hash(data_root: Any, scenario: str) -> str | None:
    """The scenario's content hash, recomputed by the authoritative tool.

    Bare-hash mode reads only qa/ — no media inventory — so recomputing a
    whole suite costs seconds. None means the tool refused the directory or
    is not where a repository checkout keeps it.
    """
    import subprocess
    if not HASH_TOOL.exists():
        return None
    proc = subprocess.run([sys.executable, str(HASH_TOOL),
                           str(Path(data_root) / scenario)],
                          capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


@dataclass
class PinReport:
    """What the suite pins versus what the data root holds."""
    absent: list[str]
    mismatched: list[str]
    stale: list[str]
    unreadable: list[str]
    questions: int

    @property
    def ok(self) -> bool:
        return not (self.absent or self.mismatched or self.stale
                    or self.unreadable)


def verify_suite_pins(suite: Suite, data_root: Any) -> PinReport:
    """Recompute every pinned scenario's content hash against the disk.

    The one implementation of the question "is the data this suite pins
    actually here and unchanged" — preflight reports it, and eval refuses to
    start on it, so a half-downloaded pool can never produce results that
    look complete.
    """
    absent, mismatched, stale, unreadable = [], [], [], []
    total = 0
    for scenario, pin in sorted(suite.pins.items()):
        try:
            manifest = load_scenario_manifest(data_root, scenario)
        except FileNotFoundError:
            absent.append(scenario)
            continue
        except ValueError as exc:
            unreadable.append(f"{scenario} ({exc})")
            continue
        total += manifest.questions
        # Recompute from the qa on disk rather than trusting the stored
        # manifest: a tampered or half-updated scenario keeps its old
        # manifest, and the pin exists precisely to catch that.
        actual = _content_hash(data_root, scenario)
        if actual is None:
            unreadable.append(f"{scenario} (hash could not be recomputed)")
        elif actual != pin:
            mismatched.append(scenario)
        elif manifest.hash != actual:
            stale.append(scenario)
    return PinReport(absent=absent, mismatched=mismatched, stale=stale,
                     unreadable=unreadable, questions=total)


def check_data(specs: list[RunSpec], suite: Suite, data_root: Any,
               sample: int = 3) -> list[Check]:
    """Every scenario the suite pins is present, unchanged, and its media resolve.

    The whole suite is checked, not just the selection: a missing scenario
    would otherwise vanish from the expansion as "question file missing" and
    the run would quietly cover less than the suite promises. Each verdict
    names its scenarios — "something is wrong with the data" is not a finding
    anyone can act on.

    Loading goes through the same functions the evaluation uses; a check that
    loads data its own way validates something other than what runs. Media
    existence is sampled — the full sweep is ``validate-data``'s job.
    """
    out: list[Check] = []
    pins = verify_suite_pins(suite, data_root)
    absent, mismatched = pins.absent, pins.mismatched
    stale, unreadable, total = pins.stale, pins.unreadable, pins.questions
    if absent:
        out.append(Check(FAIL, "suite scenarios present",
                         f"missing under {data_root}: {absent} — fetch them: "
                         f"tools/download_data.py {' '.join(absent[:3])}"
                         f"{' …' if len(absent) > 3 else ''}"))
    if unreadable:
        out.append(Check(FAIL, "scenario manifests readable", str(unreadable)))
    if mismatched:
        out.append(Check(FAIL, "scenario pins",
                         f"content differs from the suite's pin — revised or "
                         f"corrupted: {mismatched}"))
    if stale:
        out.append(Check(FAIL, "scenario manifests current",
                         f"manifest.json disagrees with the qa on disk — "
                         f"regenerate with tools/compute_scenario_hash.py "
                         f"--write: {stale}"))
    if not (absent or unreadable or mismatched or stale):
        out.append(Check(OK, "scenario pins",
                         f"{len(suite.pins)} scenario(s) recomputed and "
                         f"verified, {total} questions"))
    if absent or unreadable:
        return out    # nothing below can load what is not there

    try:
        bank = load_question_bank(data_root, list(suite.scenarios))
    except Exception as exc:  # noqa: BLE001
        return out + [Check(FAIL, "question bank", str(exc))]

    checked = missing = 0
    combos = {(s.scenario, s.dimension) for s in specs}
    for scenario, dimension in sorted(combos):
        try:
            questions = load_questions(data_root, scenario, dimension, bank=bank)
        except Exception as exc:  # noqa: BLE001
            out.append(Check(FAIL, f"{scenario}/{dimension}", f"load: {exc}"))
            continue
        for question in questions[:sample]:
            for path in media_paths(question):
                checked += 1
                if not resolve_media(data_root, scenario, path).exists():
                    missing += 1
    out.append(Check(FAIL if missing else OK, "sampled media",
                     f"{missing} of {checked} missing" if missing
                     else f"{checked} paths from {len(combos)} file(s)"))
    return out


def run_preflight(
    specs: list[RunSpec],
    models: list[ModelConfig],
    *,
    suite: Suite,
    protocol: Protocol,
    environments: dict[str, Environment],
    data_root: Any,
    repo_root: Any = ".",
) -> list[Check]:
    """All checks for one selection, this interpreter and the foreign
    environments the selection dispatches to alike."""
    checks = check_configuration(models, suite, protocol, environments)
    checks += check_environment(models, environments, repo_root)
    checks += check_foreign_environments(models, environments, repo_root)
    checks += check_gpu(models)
    checks += check_weights(models, repo_root)
    checks += check_api_keys(models)
    checks += check_data(specs, suite, data_root)
    return checks
