"""Deterministic provenance for the port source tree.

The reference container deliberately excludes ``.git``.  Build scripts therefore
pass the Git identity in environment variables, while this module independently
hashes the exact research files copied into the image.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any


_EXCLUDED_DIRECTORY_NAMES = {
    ".cache",
    ".deps",
    ".git",
    ".pytest_cache",
    ".uv_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "gpudrive_cache",
}


def _is_port_source_file(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    if any(
        part in _EXCLUDED_DIRECTORY_NAMES or part.endswith(".egg-info")
        for part in relative.parts
    ):
        return False
    return path.is_file() and path.suffix not in {".pyc", ".pyo"}


def port_source_tree_sha256(root: Path | str) -> str:
    """Hash stable relative paths and bytes for the research build context."""

    source_root = Path(root).resolve()
    digest = hashlib.sha256()
    files = sorted(
        path for path in source_root.rglob("*") if _is_port_source_file(source_root, path)
    )
    for path in files:
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout


def _git_patch_sha256(root: Path, status: bytes) -> str:
    """Hash tracked diffs plus the bytes of every untracked, non-ignored file."""

    digest = hashlib.sha256()
    digest.update(b"status\0")
    digest.update(status)
    digest.update(b"diff\0")
    digest.update(_git_bytes(root, "diff", "--binary", "HEAD", "--"))
    untracked = _git_bytes(
        root, "ls-files", "--others", "--exclude-standard", "-z"
    ).split(b"\0")
    for encoded_path in sorted(path for path in untracked if path):
        path = root / os.fsdecode(encoded_path)
        if not path.is_file() or not _is_port_source_file(root, path):
            continue
        digest.update(b"untracked\0")
        digest.update(encoded_path)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _parse_environment_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def port_identity(root: Path | str) -> dict[str, Any]:
    """Return commit, dirtiness, patch, and exact source-tree fingerprints."""

    source_root = Path(root).resolve()
    identity: dict[str, Any] = {
        "commit": os.environ.get("GPUDRIVE_PORT_GIT_COMMIT"),
        "dirty": _parse_environment_bool(os.environ.get("GPUDRIVE_PORT_DIRTY")),
        "diff_sha256": os.environ.get("GPUDRIVE_PORT_DIFF_SHA256"),
        "source_tree_sha256": port_source_tree_sha256(source_root),
        "declared_source_tree_sha256": os.environ.get(
            "GPUDRIVE_PORT_SOURCE_TREE_SHA256"
        ),
        "identity_source": "container_build_arguments",
    }
    try:
        commit = _git_bytes(source_root, "rev-parse", "HEAD").decode().strip()
        status = _git_bytes(
            source_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        identity.update(
            {
                "commit": commit,
                "dirty": bool(status),
                "diff_sha256": _git_patch_sha256(source_root, status),
                "identity_source": "git_worktree",
            }
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, UnicodeError):
        pass

    declared = identity["declared_source_tree_sha256"]
    identity["source_tree_matches_declared"] = (
        declared is None or declared == identity["source_tree_sha256"]
    )
    return identity
