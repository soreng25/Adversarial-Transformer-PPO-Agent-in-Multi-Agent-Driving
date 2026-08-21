"""Command-line entry point for the staged GPUDrive research repository."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from .adversary.checkpoint import (
    AdversaryCheckpointError,
    validate_adversary_checkpoint,
)
from .adversary.config import AdversaryConfigError
from .adversary.training import AdversaryTrainingError, train_adversary_smoke
from .adversary.training_artifact import (
    AdversaryTrainingArtifactError,
    validate_adversary_training_artifact,
)
from .doctor import build_doctor_report
from .multiagent.artifact import (
    MultiAgentArtifactError,
    summarize_nonfocal_system_run,
    validate_multiagent_training_artifact,
)
from .multiagent.calibration import BoundCalibrationError, run_highway_bound_sweep
from .multiagent.environment import MultiAgentEnvironmentError
from .multiagent.replay import MultiAgentReplayError, render_highway_failure
from .multiagent.scene import MultiAgentSceneError
from .multiagent.training import MultiAgentTrainingError, train_highway_multiagent
from .pins import (
    PinError,
    load_pins,
    repository_root,
    required_checks_pass,
    verify_source_tree,
)
from .smoke import (
    SmokeError,
    compare_smoke_artifacts,
    run_fresh_process_smoke,
    run_scene_smoke,
    validate_smoke_artifact,
)
from .victim.checkpoint import (
    VictimCheckpointError,
    default_checkpoint_directory,
    load_victim_pin,
    verify_checkpoint,
)
from .victim.evaluation import (
    VictimEvaluationError,
    compare_victim_artifacts,
    run_fresh_victim_evaluation,
    run_victim_evaluation,
    validate_victim_artifact,
)


def _write_or_print(value: dict, output: Path | None) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True)
    if output is None:
        print(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(output.resolve())


def _add_pin_and_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pins", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        default=repository_root() / ".deps/gpudrive",
        help="Pinned GPUDrive checkout (default: .deps/gpudrive).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gda")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect pins, host, and runtime.")
    _add_pin_and_source(doctor)
    doctor.add_argument("--output", type=Path)
    doctor.add_argument("--skip-runtime", action="store_true")
    doctor.add_argument("--reference", action="store_true")
    doctor.add_argument("--allow-uninitialized-submodules", action="store_true")
    doctor.add_argument("--strict", action="store_true")

    verify = subparsers.add_parser("verify-source", help="Verify immutable GPUDrive source.")
    _add_pin_and_source(verify)
    verify.add_argument("--output", type=Path)
    verify.add_argument("--allow-uninitialized-submodules", action="store_true")

    smoke = subparsers.add_parser("scene-smoke", help="Run one scene twice in one process.")
    _add_pin_and_source(smoke)
    smoke.add_argument("--config", type=Path)
    smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    smoke.add_argument("--output", type=Path, required=True)

    fresh = subparsers.add_parser("fresh-smoke", help="Run and compare two fresh processes.")
    _add_pin_and_source(fresh)
    fresh.add_argument("--config", type=Path)
    fresh.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    fresh.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare-smoke", help="Compare two smoke artifacts.")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--pins", type=Path)
    compare.add_argument("--output", type=Path)

    validate = subparsers.add_parser("validate-smoke", help="Validate one smoke artifact.")
    validate.add_argument("artifact", type=Path)
    validate.add_argument("--pins", type=Path)
    validate.add_argument("--output", type=Path)

    victim_verify = subparsers.add_parser(
        "verify-victim-checkpoint",
        help="Verify the immutable published PPO checkpoint without loading Torch.",
    )
    victim_verify.add_argument("--pin", type=Path)
    victim_verify.add_argument("--checkpoint", type=Path)
    victim_verify.add_argument(
        "--source",
        type=Path,
        default=repository_root() / ".deps/gpudrive",
    )
    victim_verify.add_argument("--output", type=Path)

    victim_eval = subparsers.add_parser(
        "victim-eval",
        help="Run the pinned PPO twice after one reset and save its deterministic trace.",
    )
    _add_pin_and_source(victim_eval)
    victim_eval.add_argument("--victim-pin", type=Path)
    victim_eval.add_argument("--checkpoint", type=Path)
    victim_eval.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    victim_eval.add_argument("--output", type=Path, required=True)

    victim_fresh = subparsers.add_parser(
        "victim-fresh-eval",
        help="Run and compare the pinned PPO in two fresh processes.",
    )
    _add_pin_and_source(victim_fresh)
    victim_fresh.add_argument("--victim-pin", type=Path)
    victim_fresh.add_argument("--checkpoint", type=Path)
    victim_fresh.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    victim_fresh.add_argument("--output", type=Path, required=True)

    victim_compare = subparsers.add_parser(
        "compare-victim", help="Compare two deterministic victim artifacts."
    )
    victim_compare.add_argument("left", type=Path)
    victim_compare.add_argument("right", type=Path)
    victim_compare.add_argument("--victim-pin", type=Path)
    victim_compare.add_argument("--pins", type=Path)
    victim_compare.add_argument("--output", type=Path)

    victim_validate = subparsers.add_parser(
        "validate-victim", help="Validate one deterministic victim artifact."
    )
    victim_validate.add_argument("artifact", type=Path)
    victim_validate.add_argument("--victim-pin", type=Path)
    victim_validate.add_argument("--pins", type=Path)
    victim_validate.add_argument("--output", type=Path)

    adversary_train = subparsers.add_parser(
        "adversary-train-smoke",
        help="Run the Linux/CUDA-only one-scene Transformer-PPO training smoke.",
    )
    _add_pin_and_source(adversary_train)
    adversary_train.add_argument("--victim-pin", type=Path)
    adversary_train.add_argument(
        "--victim-checkpoint",
        type=Path,
        help="Pinned victim checkpoint (default: location from --victim-pin).",
    )
    adversary_train.add_argument("--config", type=Path)
    adversary_train.add_argument("--output", type=Path, required=True)

    adversary_validate = subparsers.add_parser(
        "validate-adversary-checkpoint",
        help="Validate a safe Transformer-PPO checkpoint and its fingerprints.",
    )
    adversary_validate.add_argument("artifact", type=Path)
    adversary_validate.add_argument("--output", type=Path)

    adversary_run_validate = subparsers.add_parser(
        "validate-adversary-run",
        help="Validate an indivisible Milestone C training-run directory.",
    )
    adversary_run_validate.add_argument("artifact", type=Path)
    adversary_run_validate.add_argument("--output", type=Path)

    highway_train = subparsers.add_parser(
        "highway-train",
        help="Train the focal-car adversary against ten frozen PPO vehicles.",
    )
    _add_pin_and_source(highway_train)
    highway_train.add_argument("--victim-pin", type=Path)
    highway_train.add_argument("--victim-checkpoint", type=Path)
    highway_train.add_argument("--adversary-config", type=Path)
    highway_train.add_argument("--experiment-config", type=Path)
    highway_train.add_argument(
        "--scene-source",
        type=Path,
        help="Pinned original GPUDrive-mini JSON (default: .deps/datasets/GPUDrive_mini/<configured path>).",
    )
    highway_train.add_argument("--output", type=Path, required=True)

    highway_bound_sweep = subparsers.add_parser(
        "highway-bound-sweep",
        help="Measure random-prior failure rates at several disturbance bounds without PPO updates.",
    )
    _add_pin_and_source(highway_bound_sweep)
    highway_bound_sweep.add_argument("--victim-pin", type=Path)
    highway_bound_sweep.add_argument("--victim-checkpoint", type=Path)
    highway_bound_sweep.add_argument("--adversary-config", type=Path)
    highway_bound_sweep.add_argument("--experiment-config", type=Path)
    highway_bound_sweep.add_argument("--sweep-config", type=Path)
    highway_bound_sweep.add_argument("--scene-source", type=Path)
    highway_bound_sweep.add_argument("--episodes-per-bound", type=int)
    highway_bound_sweep.add_argument("--output", type=Path, required=True)

    highway_validate = subparsers.add_parser(
        "validate-highway-run",
        help="Validate a ten-PPO-agent highway training artifact.",
    )
    highway_validate.add_argument("artifact", type=Path)
    highway_validate.add_argument("--output", type=Path)

    highway_summary = subparsers.add_parser(
        "summarize-highway-system-run",
        help="Summarize qualifying slots-1-through-9 failures and collision pairs.",
    )
    highway_summary.add_argument("artifact", type=Path)
    highway_summary.add_argument("--output", type=Path)

    highway_render = subparsers.add_parser(
        "render-highway-failure",
        help="Replay a deterministic ten-agent failure and export a GIF and diagnostic plots.",
    )
    _add_pin_and_source(highway_render)
    highway_render.add_argument("--run", type=Path, required=True)
    highway_render.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory name beneath RUN/checkpoints, for example iteration-0094.",
    )
    highway_render.add_argument("--victim-pin", type=Path)
    highway_render.add_argument("--victim-checkpoint", type=Path)
    highway_render.add_argument("--zoom-radius", type=int, default=70)
    highway_render.add_argument(
        "--fps",
        type=int,
        default=10,
        help="GIF playback frames per second (1-30; lower is slower).",
    )
    highway_render.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            report = build_doctor_report(
                source=args.source,
                pin_path=args.pins,
                probe_runtime=not args.skip_runtime,
                reference=args.reference,
                require_initialized_submodules=not args.allow_uninitialized_submodules,
            )
            _write_or_print(report, args.output)
            return 0 if report["ok"] or not args.strict else 1

        if args.command == "verify-source":
            checks = verify_source_tree(
                args.source,
                load_pins(args.pins),
                require_initialized_submodules=not args.allow_uninitialized_submodules,
            )
            report = {
                "schema": "gpudrive_source_verification",
                "schema_version": 1,
                "ok": required_checks_pass(checks),
                "source": str(args.source.resolve()),
                "checks": checks,
            }
            _write_or_print(report, args.output)
            return 0 if report["ok"] else 1

        if args.command == "scene-smoke":
            manifest = run_scene_smoke(
                source=args.source,
                output=args.output,
                device=args.device,
                pin_path=args.pins,
                config_path=args.config,
            )
            print(json.dumps({"ok": True, "artifact": str(args.output.resolve()), "artifact_id": manifest["artifact_id"]}, indent=2))
            return 0

        if args.command == "fresh-smoke":
            report = run_fresh_process_smoke(
                source=args.source,
                output=args.output,
                device=args.device,
                pin_path=args.pins,
                config_path=args.config,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["ok"] else 1

        if args.command == "compare-smoke":
            report = compare_smoke_artifacts(
                args.left, args.right, output=args.output, pin_path=args.pins
            )
            if args.output is None:
                print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["ok"] else 1

        if args.command == "validate-smoke":
            report = validate_smoke_artifact(args.artifact, pin_path=args.pins)
            _write_or_print(report, args.output)
            return 0 if report["ok"] else 1

        if args.command == "verify-victim-checkpoint":
            victim_pin = load_victim_pin(args.pin)
            checkpoint = (
                args.checkpoint
                if args.checkpoint is not None
                else default_checkpoint_directory(victim_pin)
            )
            report = verify_checkpoint(
                checkpoint,
                victim_pin,
                gpudrive_source=args.source,
            )
            _write_or_print(report, args.output)
            return 0 if report["ok"] else 1

        if args.command in {"victim-eval", "victim-fresh-eval"}:
            victim_pin = load_victim_pin(args.victim_pin)
            checkpoint = (
                args.checkpoint
                if args.checkpoint is not None
                else default_checkpoint_directory(victim_pin)
            )
            if args.command == "victim-eval":
                manifest = run_victim_evaluation(
                    source=args.source,
                    checkpoint_directory=checkpoint,
                    output=args.output,
                    device=args.device,
                    victim_pin_path=args.victim_pin,
                    gpudrive_pin_path=args.pins,
                )
                print(
                    json.dumps(
                        {
                            "ok": True,
                            "artifact": str(args.output.resolve()),
                            "artifact_id": manifest["artifact_id"],
                        },
                        indent=2,
                    )
                )
                return 0
            report = run_fresh_victim_evaluation(
                source=args.source,
                checkpoint_directory=checkpoint,
                output=args.output,
                device=args.device,
                victim_pin_path=args.victim_pin,
                gpudrive_pin_path=args.pins,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["ok"] else 1

        if args.command == "compare-victim":
            report = compare_victim_artifacts(
                args.left,
                args.right,
                output=args.output,
                victim_pin_path=args.victim_pin,
                gpudrive_pin_path=args.pins,
            )
            if args.output is None:
                print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["ok"] else 1

        if args.command == "validate-victim":
            report = validate_victim_artifact(
                args.artifact,
                victim_pin_path=args.victim_pin,
                gpudrive_pin_path=args.pins,
            )
            _write_or_print(report, args.output)
            return 0 if report["ok"] else 1

        if args.command == "adversary-train-smoke":
            manifest = train_adversary_smoke(
                source=args.source,
                output=args.output,
                checkpoint_directory=args.victim_checkpoint,
                adversary_config_path=args.config,
                victim_pin_path=args.victim_pin,
                gpudrive_pin_path=args.pins,
            )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "artifact": str(args.output.resolve()),
                        "artifact_id": manifest["artifact_id"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "validate-adversary-checkpoint":
            report = validate_adversary_checkpoint(args.artifact)
            _write_or_print(report, args.output)
            return 0 if report["ok"] else 1

        if args.command == "validate-adversary-run":
            report = validate_adversary_training_artifact(args.artifact)
            _write_or_print(report, args.output)
            return 0 if report["ok"] else 1

        if args.command == "highway-train":
            manifest = train_highway_multiagent(
                source=args.source,
                output=args.output,
                checkpoint_directory=args.victim_checkpoint,
                scene_source=args.scene_source,
                adversary_config_path=args.adversary_config,
                experiment_config_path=args.experiment_config,
                victim_pin_path=args.victim_pin,
                gpudrive_pin_path=args.pins,
            )
            print(json.dumps({"ok": True, "artifact": str(args.output.resolve()), "artifact_id": manifest["artifact_id"], "clean_eligibility": manifest["clean_eligibility"], "final_metrics": manifest["metrics"][-1]}, indent=2, sort_keys=True))
            return 0

        if args.command == "highway-bound-sweep":
            manifest = run_highway_bound_sweep(
                source=args.source,
                output=args.output,
                checkpoint_directory=args.victim_checkpoint,
                scene_source=args.scene_source,
                adversary_config_path=args.adversary_config,
                experiment_config_path=args.experiment_config,
                sweep_config_path=args.sweep_config,
                victim_pin_path=args.victim_pin,
                gpudrive_pin_path=args.pins,
                episodes_per_bound=args.episodes_per_bound,
            )
            print(json.dumps({"ok": True, "artifact": str(args.output.resolve()), "artifact_id": manifest["artifact_id"], "results": manifest["results"]}, indent=2, sort_keys=True))
            return 0

        if args.command == "validate-highway-run":
            report = validate_multiagent_training_artifact(args.artifact)
            _write_or_print(report, args.output)
            return 0 if report["ok"] else 1

        if args.command == "summarize-highway-system-run":
            report = summarize_nonfocal_system_run(args.artifact)
            _write_or_print(report, args.output)
            return 0

        if args.command == "render-highway-failure":
            manifest = render_highway_failure(
                source=args.source,
                run=args.run,
                checkpoint=args.checkpoint,
                output=args.output,
                checkpoint_directory=args.victim_checkpoint,
                victim_pin_path=args.victim_pin,
                gpudrive_pin_path=args.pins,
                zoom_radius=args.zoom_radius,
                fps=args.fps,
            )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "artifact": str(args.output.resolve()),
                        "artifact_id": manifest["artifact_id"],
                        "failure": manifest["failure"],
                        "files": sorted(manifest["files"]),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
    except (
        AdversaryCheckpointError,
        AdversaryConfigError,
        AdversaryTrainingArtifactError,
        AdversaryTrainingError,
        BoundCalibrationError,
        PinError,
        MultiAgentEnvironmentError,
        MultiAgentArtifactError,
        MultiAgentReplayError,
        MultiAgentSceneError,
        MultiAgentTrainingError,
        SmokeError,
        VictimCheckpointError,
        VictimEvaluationError,
    ) as exc:
        print(f"gda {args.command} failed: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
