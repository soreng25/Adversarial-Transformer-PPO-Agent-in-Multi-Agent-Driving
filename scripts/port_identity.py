#!/usr/bin/env python3
"""Print one build-context provenance field without third-party dependencies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpudrive_adversary.provenance import port_identity  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "field",
        choices=("commit", "dirty", "diff_sha256", "source_tree_sha256"),
    )
    args = parser.parse_args()
    value = port_identity(ROOT)[args.field]
    if isinstance(value, bool):
        print(str(value).lower())
    elif value is None:
        return 1
    else:
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
