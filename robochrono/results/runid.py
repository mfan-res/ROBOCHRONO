#!/usr/bin/env python3
# coding: utf-8
"""Run identity: which experiment a set of results belongs to.

A result file that does not say which experiment produced it is a trap that
springs later: rerun with changed parameters, and question-level resume will
happily extend the old file, leaving half the rows under one configuration and
half under another, with nothing recording the seam.

The identity is a directory name::

    results/2026-08-27_a1b2c3d4e5f6/
            └── created ──┘└─ config fingerprint

The two parts carry two separate jobs. The date is for humans sorting
directories; it is fixed when the directory is first created and never moves.
Equality is judged by the fingerprint alone — which is why resuming after an
interruption works: the same command produces the same fingerprint, the
existing directory is found by suffix, and work continues. A current-timestamp
name would send every rerun to a fresh empty directory.

The fingerprint covers **exactly the things that change results**: the
protocol, the suite, the dataset fingerprint, the participating models'
configuration files, and the code version. It deliberately excludes anything
that varies by machine — concurrency, proxies, cache paths, GPU counts —
because two runs differing only in those are the same experiment.

Configuration files are hashed as raw bytes. A stricter rule than "the parsed
values changed", and chosen on purpose: deciding which edits are harmless
would need a maintained list of which bytes count, and the cost of being
strict is only an extra directory.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FINGERPRINT_LENGTH = 12


# -- code version ----------------------------------------------------------

def code_version(repo_root: Any = ".") -> dict[str, Any]:
    """The code identity: commit, whether the tree is dirty, and if so a
    digest of the uncommitted diff.

    A dirty tree participates in the fingerprint through the diff digest, so
    editing one line moves subsequent results to a new directory — the same
    rule as committing the line. Without git metadata (an exported tree) the
    code component is inert; ``preflight`` reports that, this module does not
    guess.
    """
    def _git(*args: str) -> str | None:
        try:
            proc = subprocess.run(["git", "-C", str(repo_root), *args],
                                  capture_output=True, text=True, timeout=30)
        except OSError:
            return None
        return proc.stdout if proc.returncode == 0 else None

    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return {"git": None, "dirty": False}
    status = _git("status", "--porcelain") or ""
    dirty = bool(status.strip())
    out: dict[str, Any] = {"git": sha.strip()[:12], "dirty": dirty}
    if dirty:
        diff = (_git("diff", "HEAD") or "") + status
        out["diff_sha"] = hashlib.sha256(diff.encode("utf-8")).hexdigest()[:12]
    return out


# -- fingerprint -----------------------------------------------------------

def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_fingerprint(*, protocol_path: Any, suite_path: Any,
                       model_paths: list[Any], dataset_digest: str,
                       code: dict[str, Any]) -> str:
    """Digest of everything that determines the results, and nothing else.

    ``dataset_digest`` is the ordered digest of the suite's scenario pins
    (``config.suites.pins_digest``) — the data identity as the suite declares
    it, independent of where the pool happens to sit on disk.
    """
    composite = {
        "protocol": _file_sha(Path(protocol_path)),
        "suite": _file_sha(Path(suite_path)),
        "models": {Path(p).stem: _file_sha(Path(p)) for p in model_paths},
        "dataset": dataset_digest,
        "code": [code.get("git"), code.get("diff_sha")],
    }
    payload = json.dumps(composite, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:FINGERPRINT_LENGTH]


# -- directory resolution ---------------------------------------------------

@dataclass(frozen=True)
class RunDir:
    path: Path
    run_id: str
    fingerprint: str
    resumed: bool


_RUN_DIR = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<fp>[0-9a-f]+)(?:-(?P<ordinal>\d+))?$")


def _matches(results_root: Path, fingerprint: str) -> list[tuple[int, Path]]:
    out = []
    if not results_root.exists():
        return out
    for entry in results_root.iterdir():
        m = _RUN_DIR.match(entry.name)
        if entry.is_dir() and m and m.group("fp") == fingerprint:
            out.append((int(m.group("ordinal") or 1), entry))
    return sorted(out)


def resolve_run_dir(results_root: Any, fingerprint: str, *,
                    fresh: bool = False, today: str | None = None) -> RunDir:
    """Find the directory for this fingerprint, or create one.

    Lookup is by fingerprint suffix: an existing directory means this exact
    experiment has run before, so enter it and resume. ``fresh`` skips the
    lookup and creates a sibling with the next ordinal — an explicit "run the
    same experiment again from nothing", never the default.
    """
    root = Path(results_root)
    today = today or _dt.date.today().isoformat()
    existing = _matches(root, fingerprint)

    if existing and not fresh:
        path = existing[-1][1]
        return RunDir(path=path, run_id=path.name, fingerprint=fingerprint,
                      resumed=True)

    ordinal = existing[-1][0] + 1 if existing else 1
    name = f"{today}_{fingerprint}" if ordinal == 1 else f"{today}_{fingerprint}-{ordinal}"
    path = root / name
    path.mkdir(parents=True, exist_ok=False)
    return RunDir(path=path, run_id=name, fingerprint=fingerprint, resumed=False)


# -- run.json ---------------------------------------------------------------

def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def write_run_record(run_dir: RunDir, record: dict[str, Any]) -> dict[str, Any]:
    """Write ``run.json`` once; on resume, verify instead of overwrite.

    The record is the run's configuration snapshot, written exactly once, at
    creation. Resuming re-verifies that the directory's stored fingerprint
    matches the current one — by construction it must, so a mismatch means the
    directory was renamed or copied by hand, and extending it would rebuild
    the half-old half-new files this layer exists to prevent.
    """
    path = run_dir.path / "run.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("fingerprint") != run_dir.fingerprint:
            raise ValueError(
                f"{path} records fingerprint {stored.get('fingerprint')!r} but the "
                f"current configuration gives {run_dir.fingerprint!r}. This directory "
                f"was moved or edited by hand; refusing to extend it.")
        return stored
    full = {"run_id": run_dir.run_id, "fingerprint": run_dir.fingerprint,
            "started": _now(), "finished": None, **record}
    path.write_text(json.dumps(full, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return full


def mark_finished(run_dir: RunDir) -> None:
    path = run_dir.path / "run.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["finished"] = _now()
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def read_run_record(run_path: Any) -> dict[str, Any]:
    path = Path(run_path) / "run.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — this directory is not a run, or the run was "
            f"created by hand. Every run directory carries its configuration "
            f"snapshot; without one its results cannot be attributed.")
    return json.loads(path.read_text(encoding="utf-8"))
