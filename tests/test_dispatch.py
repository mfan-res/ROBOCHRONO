#!/usr/bin/env python3
# coding: utf-8
"""Dispatch: each model group under its own interpreter, one run directory."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochrono.config.environments import Environment, load_environments  # noqa: E402
from robochrono.config.models import load_models  # noqa: E402
from robochrono.orchestrate.dispatch import (  # noqa: E402
    build_commands, dispatch, group_by_environment, resolve_interpreter)

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:52} {detail}")
    if not passed:
        failures.append(name)


MODELS = sorted(load_models(ROOT / "configs/models").values(), key=lambda m: m.slug)
ENVS = load_environments(ROOT / "configs/environments.json")

print("1. models group by the environment their configuration names")
groups = group_by_environment(MODELS)
check("two environments", sorted(groups) == ["transformers-4x", "transformers-5x"])
check("tf5 gets the four 5.x models",
      sorted(m.slug for m in groups["transformers-5x"])
      == ["cosmos3-edge-2b", "rynnbrain1-1-122b-a10b",
          "rynnbrain1-1-2b", "rynnbrain1-1-9b"])
check("tf4 gets the rest", len(groups["transformers-4x"]) == len(MODELS) - 4)

print("2. interpreter resolution is lexical and honest")
check("empty python means this interpreter",
      resolve_interpreter(Environment("x", "", "", ""), ROOT) == sys.executable)

base = Path(tempfile.mkdtemp())
real = base / "real_env" / "bin"
real.mkdir(parents=True)
(real / "python").write_text("")
link = base / "venv"
link.symlink_to(base / "real_env")
env = Environment("x", "venv/bin/python", "", "")
resolved = resolve_interpreter(env, base)
check("relative paths join against the repo root",
      resolved == os.path.abspath(base / "venv/bin/python"), resolved)
check("the symlink is not resolved away", "real_env" not in resolved)
try:
    resolve_interpreter(Environment("tf9", ".venvs/tf9/bin/python", "tf9", ""), base)
    check("a missing interpreter raises with a hint", False, "no exception")
except FileNotFoundError as e:
    check("a missing interpreter raises with a hint", "tf9" in str(e))

print("3. commands carry the run directory and never re-dispatch")
commands = build_commands(MODELS, environments=ENVS, repo_root=ROOT,
                          run_dir="/results/2026-08-27_abc",
                          passthrough=["--suite", "smoke"])
check("one command per environment", len(commands) == 2)
for env_name, cmd in commands:
    check(f"{env_name} passes the parent's run dir",
          "--run-dir" in cmd and "/results/2026-08-27_abc" in cmd)
    check(f"{env_name} child will not dispatch again", "--no-dispatch" in cmd)
    check(f"{env_name} names only its own models",
          set(cmd[cmd.index("--models") + 1:-1])
          == {m.slug for m in groups[env_name]})

print("4. an unknown environment is a configuration error, loudly")
try:
    import dataclasses
    odd = dataclasses.replace(MODELS[0], environment="transformers-9x")
    build_commands([odd], environments=ENVS, repo_root=ROOT,
                   run_dir="x", passthrough=[])
    check("unknown environment raises", False, "no exception")
except ValueError as e:
    check("unknown environment raises", "transformers-9x" in str(e))

print("5. dry run prints the plan and touches nothing")
run_dir = Path(tempfile.mkdtemp())
fails = dispatch(MODELS, environments=ENVS, repo_root=ROOT,
                 run_dir=run_dir, passthrough=["--suite", "smoke"], dry_run=True)
check("dry run reports no failures", fails == 0)
check("dry run writes no logs", not (run_dir / "log").exists())

print("6. a real child's output is teed into the run's log directory")
echo_env = {"echo": Environment("echo", sys.executable, "", "")}
import dataclasses
model = dataclasses.replace(MODELS[0], environment="echo")
# the child here is python -m robochrono eval ... which does not exist yet;
# what is under test is the tee and the exit-code accounting
fails = dispatch([model], environments=echo_env, repo_root=ROOT,
                 run_dir=run_dir, passthrough=[])
log = run_dir / "log" / "echo.log"
check("a failing child counts as a failure", fails == 1)
check("its output landed in log/<env>.log",
      log.exists() and log.read_text().strip() != "")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
