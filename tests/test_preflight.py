#!/usr/bin/env python3
# coding: utf-8
"""Preflight verdicts: what fails, what skips, and what passes on real data."""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochrono import preflight  # noqa: E402
from robochrono.config.environments import Environment, load_environments  # noqa: E402
from robochrono.config.models import load_models  # noqa: E402
from robochrono.config.protocol import load_protocol  # noqa: E402
from robochrono.config.suites import load_suite  # noqa: E402
from robochrono.orchestrate.matrix import expand  # noqa: E402

failures: list[str] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    print(f"  {'✅' if passed else '❌'} {name:52} {detail}")
    if not passed:
        failures.append(name)


def levels(checks, name_part):
    return [c.level for c in checks if name_part in c.name]


PROTOCOL = load_protocol(ROOT / "configs/protocol.json")
SUITE = load_suite("official-v1", ROOT / "configs/suites")
ENVS = load_environments(ROOT / "configs/environments.json")
MODELS = sorted(load_models(ROOT / "configs/models").values(), key=lambda m: m.slug)
DATA = ROOT / "scenarios"

LOCAL = MODELS[0]
API = dataclasses.replace(
    MODELS[0], slug="test-api", kind="api", adapter="openai_compat",
    api={"url": "https://x", "model": "m", "key_env": "ROBOCHRONO_TEST_KEY"})

print("1. configuration checks")
checks = preflight.check_configuration(MODELS, SUITE, PROTOCOL, ENVS)
check("all real models configure", not preflight.has_failures(checks))
broken = dataclasses.replace(MODELS[0], slug="broken", adapter="nope")
checks = preflight.check_configuration([broken], SUITE, PROTOCOL, ENVS)
check("unknown adapter is a FAIL", preflight.FAIL in levels(checks, "broken"))
odd_env = dataclasses.replace(MODELS[0], slug="odd", environment="tf-9")
checks = preflight.check_configuration([odd_env], SUITE, PROTOCOL, ENVS)
check("undefined environment is a FAIL",
      preflight.FAIL in levels(checks, "odd environment"))

print("2. weights and keys")
ghost = dataclasses.replace(LOCAL, slug="ghost", weights="models/does-not-exist")
checks = preflight.check_weights([ghost, API], ROOT)
check("absent local weights FAIL",
      levels(checks, "ghost weights") == [preflight.FAIL])
check("API models skip the weights check",
      levels(checks, "test-api weights") == [preflight.SKIP])

os.environ.pop("ROBOCHRONO_TEST_KEY", None)
checks = preflight.check_api_keys([API])
check("unset key_env is a FAIL", levels(checks, "test-api key") == [preflight.FAIL])
os.environ["ROBOCHRONO_TEST_KEY"] = "sk-test"
checks = preflight.check_api_keys([API])
check("set key_env is OK", levels(checks, "test-api key") == [preflight.OK])

print("3. environment checks scale to the selection")
checks = preflight.check_environment([API], ENVS, ROOT)
check("API-only selection skips the local stack",
      all(l == preflight.SKIP
          for l in levels(checks, "package torch")))
check("requests still required",
      levels(checks, "package requests") == [preflight.OK])
checks = preflight.check_gpu([API])
check("API-only selection skips the GPU check",
      [c.level for c in checks] == [preflight.SKIP])

print("4. foreign environments are probed only when the selection needs them")
# These tests run under one of the declared environments (tf4); a model mapped
# to that same environment must not trigger a subprocess probe.
tf4_model = dataclasses.replace(LOCAL, slug="tf4-model",
                                environment="transformers-4x")
checks = preflight.check_foreign_environments([tf4_model], ENVS, ROOT)
check("same-environment selection skips the probe",
      [c.level for c in checks] == [preflight.SKIP]
      and "other environments" in checks[0].name)

ghost_env = Environment(name="ghost-env", python=".venvs/ghost/bin/python",
                        extra="tf9", transformers="9.9.9")
ghost_model = dataclasses.replace(LOCAL, slug="ghost-model",
                                  environment="ghost-env")
checks = preflight.check_foreign_environments(
    [ghost_model], {**ENVS, "ghost-env": ghost_env}, ROOT)
check("missing interpreter is a FAIL that names it",
      levels(checks, "ghost-env interpreter") == [preflight.FAIL]
      and ".venvs/ghost/bin/python" in checks[0].detail)
check("and gives the build command",
      "uv sync --extra tf9" in checks[0].detail)

# A real probe with a deliberately wrong declaration: point a fake environment
# at the tf4 interpreter (present wherever the tests run) but declare a
# transformers version it cannot have. The dict keeps the real environments
# first so the current-interpreter match still resolves to transformers-4x
# and the fake one counts as foreign.
drift_env = Environment(name="drift-env", python=ENVS["transformers-4x"].python,
                        extra="tf4", transformers="9.9.9")
drift_model = dataclasses.replace(LOCAL, slug="drift-model",
                                  environment="drift-env")
checks = preflight.check_foreign_environments(
    [drift_model], {**ENVS, "drift-env": drift_env}, ROOT)
check("probed interpreter reports OK",
      levels(checks, "drift-env interpreter") == [preflight.OK])
drift = [c for c in checks if c.name == "drift-env package transformers"]
check("version drift is a FAIL naming both versions",
      [c.level for c in drift] == [preflight.FAIL]
      and "vs 9.9.9 declared" in drift[0].detail
      and "uv sync" in drift[0].detail)
check("other probed packages import fine",
      [c.level for c in checks
       if c.name == "drift-env package torch"] == [preflight.OK])

tf5_python = ROOT / ENVS["transformers-5x"].python
if tf5_python.exists():
    tf5_model = dataclasses.replace(LOCAL, slug="tf5-model",
                                    environment="transformers-5x")
    checks = preflight.check_foreign_environments([tf5_model], ENVS, ROOT)
    check("healthy second environment passes end to end",
          not preflight.has_failures(checks))
    check("and its transformers version is checked against the declaration",
          any(c.name == "transformers-5x package transformers"
              and ENVS["transformers-5x"].transformers in c.detail
              for c in checks))
else:
    print("  ⏭️  transformers-5x not built here — healthy-probe case skipped")

print("5. data checks run the evaluation's own loading path")
specs, _ = expand([LOCAL], SUITE, DATA,
                  only_scenarios=["make_tea_tianji"],
                  only_dimensions=["current_action", "view_match"])
checks = preflight.check_data(specs, SUITE, DATA)
check("real dataset passes", not preflight.has_failures(checks),
      str([c for c in checks if c.level == preflight.FAIL][:1]))
check("media was actually sampled",
      any("sampled media" in c.name and c.level == preflight.OK for c in checks))

checks = preflight.check_data(specs, SUITE, "/nonexistent")
check("missing dataset is a FAIL", preflight.has_failures(checks))
check("and it names the missing scenarios",
      any("suite scenarios present" in c.name and "make_tea_tianji" in c.detail
          for c in checks if c.level == preflight.FAIL))
wrong = dataclasses.replace(
    SUITE, pins={**SUITE.pins, "make_tea_tianji": "0" * 64})
checks = preflight.check_data(specs, wrong, DATA)
check("a pin the content does not match is a FAIL",
      any("scenario pins" in c.name and c.level == preflight.FAIL
          and "make_tea_tianji" in c.detail for c in checks))

print("6. the full pass, formatted")
checks = preflight.run_preflight(specs, [API], suite=SUITE, protocol=PROTOCOL,
                                 environments=ENVS, data_root=DATA, repo_root=ROOT)
text = preflight.format_checks(checks)
check("summary line present", "checks" in text.splitlines()[-1])
check("exit condition computable", isinstance(preflight.has_failures(checks), bool))

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("all checks passed")
