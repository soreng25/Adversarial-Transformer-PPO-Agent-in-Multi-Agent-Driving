#!/usr/bin/env python3
"""Download or verify the exact published PPO victim checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpudrive_adversary.victim.checkpoint import (  # noqa: E402
    VictimCheckpointError,
    default_checkpoint_directory,
    download_checkpoint,
    load_victim_pin,
    verify_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--without-model-card", action="store_true")
    parser.add_argument("--gpudrive-source", type=Path)
    args = parser.parse_args()
    try:
        pin = load_victim_pin(args.pin)
        destination = (
            args.destination.resolve()
            if args.destination is not None
            else default_checkpoint_directory(pin)
        )
        if args.verify_only:
            report = verify_checkpoint(
                destination, pin, gpudrive_source=args.gpudrive_source
            )
        else:
            download_checkpoint(
                destination,
                pin,
                include_model_card=not args.without_model_card,
            )
            report = verify_checkpoint(
                destination, pin, gpudrive_source=args.gpudrive_source
            )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1
    except (OSError, VictimCheckpointError) as exc:
        print(f"victim checkpoint operation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
