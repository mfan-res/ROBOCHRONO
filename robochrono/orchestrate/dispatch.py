#!/usr/bin/env python3
# coding: utf-8
"""Starting each model under the interpreter its environment declares.

No single environment runs every model — the transformers requirements are
mutually exclusive — so the matrix cannot finish inside one interpreter. The
split is **per model**: the matrix is model-major anyway, so cutting it by
model disturbs nothing, and a broken environment or a crashed process takes
down only its own group.

The parent establishes the run directory before dispatching and passes it
down; a child that recomputed identity from its own subset of models would
derive a different fingerprint and file its results elsewhere.

Interpreter paths are joined lexically, **never resolved**: a venv's
``bin/python`` is a symlink to the base interpreter, and resolving it points
``sys.prefix`` at the base environment — the venv's packages silently stop
applying. (``multiprocessing.set_executable`` is avoided for the same family
of reasons: cross-interpreter spawn couples both sides' pickle protocols;
a plain subprocess is boring and reliable.)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..config.environments import Environment
from ..config.models import ModelConfig


def resolve_interpreter(env: Environment, repo_root: Any) -> str:
    """The interpreter for one environment. Empty means the current one."""
    raw = str(env.python or "").strip()
    if not raw:
        return sys.executable
    path = Path(raw)
    if not path.is_absolute():
        # lexical join only — resolving would follow the venv symlink and
        # silently swap in the base environment's packages
        path = Path(os.path.abspath(Path(repo_root) / path))
    if not path.exists():
        raise FileNotFoundError(
            f"interpreter for environment {env.name!r} not found: {path}\n"
            f"create it first: UV_PROJECT_ENVIRONMENT={path.parent.parent} "
            f"uv sync --extra {env.extra or env.name} — see "
            f"configs/environments.json and the Environments section of "
            f"docs/TUTORIAL.md")
    return str(path)


def group_by_environment(models: list[ModelConfig]) -> dict[str, list[ModelConfig]]:
    """Models per environment, in the order given. The model configuration
    itself names the environment; there is no separate mapping to drift."""
    groups: dict[str, list[ModelConfig]] = {}
    for model in models:
        groups.setdefault(model.environment, []).append(model)
    return groups


def build_commands(
    models: list[ModelConfig],
    *,
    environments: dict[str, Environment],
    repo_root: Any,
    run_dir: Any,
    passthrough: list[str],
) -> list[tuple[str, list[str]]]:
    """One (environment, command) per group. Raises on an unknown environment;
    a missing interpreter is reported per group at execution time instead,
    so one broken venv does not block the groups that work."""
    commands = []
    for env_name, group in group_by_environment(models).items():
        if env_name not in environments:
            raise ValueError(f"models {[m.slug for m in group]} name environment "
                             f"{env_name!r}, which configs/environments.json does "
                             f"not define; known: {sorted(environments)}")
        cmd = ["{interpreter}", "-m", "robochrono", "eval", *passthrough,
               "--run-dir", str(run_dir), "--models", *[m.slug for m in group],
               "--no-dispatch"]
        commands.append((env_name, cmd))
    return commands


def _tee(cmd: list[str], cwd: Any, log_path: Path) -> int:
    """Run a child, streaming its output to both console and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def dispatch(
    models: list[ModelConfig],
    *,
    environments: dict[str, Environment],
    repo_root: Any,
    run_dir: Any,
    passthrough: list[str],
    dry_run: bool = False,
) -> int:
    """Run each environment group under its interpreter. Returns failures."""
    commands = build_commands(models, environments=environments,
                              repo_root=repo_root, run_dir=run_dir,
                              passthrough=passthrough)
    groups = group_by_environment(models)
    print(f"{len(models)} model(s) across {len(commands)} environment(s):")
    for env_name, _ in commands:
        print(f"  {env_name:<16} {', '.join(m.slug for m in groups[env_name])}")
    print()

    failures = 0
    for env_name, cmd in commands:
        try:
            interpreter = resolve_interpreter(environments[env_name], repo_root)
        except FileNotFoundError as exc:
            if dry_run:
                # A dry run answers "what would run"; whether it *can* run is
                # preflight's question, so a missing venv is noted, not fatal.
                print(f"[{env_name}] interpreter not built yet: {exc}")
                continue
            print(f"[{env_name}] skipped: {exc}")
            failures += 1
            continue
        cmd = [interpreter, *cmd[1:]]
        print(f"{'=' * 70}\n[{env_name}] {interpreter}\n  {' '.join(cmd[1:])}\n{'=' * 70}",
              flush=True)
        if dry_run:
            continue
        code = _tee(cmd, repo_root, Path(run_dir) / "log" / f"{env_name}.log")
        if code != 0:
            print(f"[{env_name}] exit code {code}", flush=True)
            failures += 1
    return failures
