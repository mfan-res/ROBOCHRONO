#!/usr/bin/env python3
# coding: utf-8
"""Download scenarios from the dataset repositories and leave them ready to run.

    python3 tools/download_data.py pack_airpods_gim make_tea_tianji
    python3 tools/download_data.py --all
    python3 tools/download_data.py --all --qa-only
    python3 tools/download_data.py cap_pen_gim --media-types clips,frames

Scenarios live across two public Hugging Face repositories; which scenario
sits where is discovered by listing them, not hardcoded, so a scenario that
moves or arrives later needs no change here. Media ships as one tar per kind
per scenario (a scenario that asks no frame questions has no frames.tar);
archives are unpacked into the layout the questions reference and deleted
unless ``--keep-archives`` is given. The download ends with the same sweep
``robochrono validate-data`` runs, so "it downloaded" and "it is usable"
are never confused.

Resume comes from huggingface_hub itself: partially downloaded files are kept
as ``*.incomplete`` and continue where they stopped — rerun the same command
after an interruption. Errors that a retry cannot fix (a missing scenario,
a revoked repository, an authentication demand) abort immediately rather
than spinning against the server's rate limits; transient network failures
are retried a bounded number of times with growing pauses.

Standard library plus ``huggingface_hub`` (already present in the tf4/tf5
environments; ``pip install huggingface_hub`` anywhere else).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_REPOS = ("gimai/RoboChrono-GIM", "gimai/RoboChrono-Tianji")
MEDIA_TYPES = ("clips", "episodes", "frames")
TRANSIENT_RETRIES = 3


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub is not installed in this interpreter. It ships "
                 "with the tf4/tf5 environments (.venvs/tf4/bin/python), or: "
                 "pip install huggingface_hub")
    return HfApi()


def discover(api) -> dict[str, str]:
    """scenario id -> repository, from the repositories' own listings."""
    where: dict[str, str] = {}
    for repo in DATASET_REPOS:
        for f in api.list_repo_files(repo_id=repo, repo_type="dataset"):
            top, _, rest = f.partition("/")
            if rest and top not in where:
                where[top] = repo
    return where


def _is_fatal(exc: Exception) -> bool:
    """Errors a retry cannot fix — retrying them only burns rate limit."""
    from huggingface_hub.errors import (EntryNotFoundError, GatedRepoError,
                                        RepositoryNotFoundError)
    if isinstance(exc, (RepositoryNotFoundError, GatedRepoError,
                        EntryNotFoundError)):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 403, 404)


def fetch(repo: str, scenario: str, patterns: list[str], out: Path) -> None:
    from huggingface_hub import snapshot_download
    for attempt in range(1, TRANSIENT_RETRIES + 1):
        try:
            snapshot_download(repo_id=repo, repo_type="dataset",
                              local_dir=str(out), allow_patterns=patterns)
            return
        except Exception as exc:  # noqa: BLE001 — classified right below
            if _is_fatal(exc) or attempt == TRANSIENT_RETRIES:
                raise
            wait = 15 * attempt
            print(f"  transient failure ({type(exc).__name__}), "
                  f"retry {attempt}/{TRANSIENT_RETRIES - 1} in {wait}s", flush=True)
            time.sleep(wait)


def _safe_members(tar: tarfile.TarFile, archive: Path):
    for member in tar.getmembers():
        p = Path(member.name)
        if p.is_absolute() or ".." in p.parts:
            sys.exit(f"{archive} contains an unsafe path {member.name!r}; "
                     f"refusing to extract")
        yield member


def unpack(scenario_dir: Path, keep: bool) -> int:
    """Extract whichever media archives exist; return how many."""
    unpacked = 0
    for kind in MEDIA_TYPES:
        archive = scenario_dir / "media" / f"{kind}.tar"
        if not archive.exists():
            continue
        with tarfile.open(archive) as tar:
            tar.extractall(scenario_dir, members=_safe_members(tar, archive))
        unpacked += 1
        if not keep:
            archive.unlink()
    return unpacked


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download scenarios from the dataset repositories.")
    ap.add_argument("scenarios", nargs="*", help="scenario ids to download")
    ap.add_argument("--all", action="store_true", help="every scenario there is")
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "scenarios",
                    help="pool directory to download into (default: scenarios/)")
    ap.add_argument("--qa-only", action="store_true",
                    help="manifest and qa only, no media")
    ap.add_argument("--media-types", default=",".join(MEDIA_TYPES),
                    help="comma-separated subset of clips,episodes,frames")
    ap.add_argument("--keep-archives", action="store_true",
                    help="keep the media tars after unpacking")
    ap.add_argument("--skip-validate", action="store_true",
                    help="do not run validate-data afterwards")
    args = ap.parse_args()

    if bool(args.scenarios) == args.all:
        ap.error("name scenarios or pass --all, one or the other")
    kinds = [] if args.qa_only else [k.strip() for k in args.media_types.split(",")]
    unknown_kind = sorted(set(kinds) - set(MEDIA_TYPES))
    if unknown_kind:
        ap.error(f"unknown media types {unknown_kind}; known: {list(MEDIA_TYPES)}")

    api = _api()
    where = discover(api)
    wanted = sorted(where) if args.all else args.scenarios
    missing = sorted(set(wanted) - set(where))
    if missing:
        sys.exit(f"not in any dataset repository: {missing}\n"
                 f"available: {sorted(where)}")

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    for i, scenario in enumerate(wanted, 1):
        patterns = [f"{scenario}/manifest.json", f"{scenario}/qa/*"]
        patterns += [f"{scenario}/media/{k}.tar" for k in kinds]
        print(f"[{i}/{len(wanted)}] {scenario}  <-  {where[scenario]}", flush=True)
        fetch(where[scenario], scenario, patterns, out)
        n = unpack(out / scenario, args.keep_archives)
        if kinds:
            print(f"  unpacked {n} media archive(s)", flush=True)

    # huggingface_hub leaves its bookkeeping next to the download; the pool
    # should hold scenarios and nothing else.
    cache = out / ".cache"
    if cache.exists():
        shutil.rmtree(cache)

    if args.skip_validate:
        return 0
    if args.qa_only or set(kinds) != set(MEDIA_TYPES):
        print("\nnote: media was skipped or filtered on request — expect "
              "validate-data to report the missing files.", flush=True)
    print("\nrunning validate-data over the pool:", flush=True)
    proc = subprocess.run([sys.executable, "-m", "robochrono", "validate-data",
                           "--data-root", str(out.resolve())], cwd=REPO_ROOT)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
