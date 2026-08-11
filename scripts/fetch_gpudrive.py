#!/usr/bin/env python3
"""Fetch and verify the immutable GPUDrive source checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gpudrive_adversary.pins import (  # noqa: E402
    PinError,
    load_pins,
    required_checks_pass,
    verify_source_tree,
)


def run(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise PinError(
            f"command failed with exit code {result.returncode}: {' '.join(command)}"
        )


def fetch(destination: Path, pins: dict) -> None:
    gpudrive = pins["gpudrive"]
    if destination.exists():
        if not (destination / ".git").exists():
            raise PinError(
                f"destination exists but is not a Git checkout: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                gpudrive["repository"],
                str(destination),
            ]
        )
    run(["git", "config", "core.longpaths", "true"], cwd=destination)
    run(
        [
            "git",
            "fetch",
            "--depth=1",
            "origin",
            gpudrive["commit"],
        ],
        cwd=destination,
    )
    run(["git", "checkout", "--detach", gpudrive["commit"]], cwd=destination)
    run(
        ["git", "submodule", "update", "--init", "--recursive", "--depth=1"],
        cwd=destination,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=REPOSITORY_ROOT / ".deps/gpudrive",
    )
    parser.add_argument("--pins", type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not clone; only verify an existing checkout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        pins = load_pins(args.pins)
        destination = args.destination.resolve()
        if not args.verify_only:
            fetch(destination, pins)
        checks = verify_source_tree(destination, pins)
        report = {
            "source": str(destination),
            "ok": required_checks_pass(checks),
            "checks": checks,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1
    except PinError as exc:
        print(f"GPUDrive fetch/verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
