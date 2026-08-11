import json
from pathlib import Path

import pytest

from gpudrive_adversary.adversary.training import AdversaryTrainingError
from gpudrive_adversary.cli import build_parser, main
from gpudrive_adversary.pins import repository_root


pytestmark = pytest.mark.unit


def test_adversary_train_parser_exposes_only_pinned_training_inputs() -> None:
    args = build_parser().parse_args(
        [
            "adversary-train-smoke",
            "--source",
            "source",
            "--pins",
            "pins.json",
            "--victim-pin",
            "victim.json",
            "--victim-checkpoint",
            "victim-checkpoint",
            "--config",
            "adversary.json",
            "--output",
            "run",
        ]
    )

    assert args.command == "adversary-train-smoke"
    assert args.source == Path("source")
    assert args.pins == Path("pins.json")
    assert args.victim_pin == Path("victim.json")
    assert args.victim_checkpoint == Path("victim-checkpoint")
    assert args.config == Path("adversary.json")
    assert args.output == Path("run")
    assert not hasattr(args, "device")


def test_adversary_train_dispatches_without_native_imports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, object] = {}

    def fake_train(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"artifact_id": "adversary-train-test"}

    monkeypatch.setattr("gpudrive_adversary.cli.train_adversary_smoke", fake_train)
    output = tmp_path / "run"
    status = main(
        [
            "adversary-train-smoke",
            "--source",
            str(tmp_path / "source"),
            "--pins",
            str(tmp_path / "pins.json"),
            "--victim-pin",
            str(tmp_path / "victim.json"),
            "--victim-checkpoint",
            str(tmp_path / "victim"),
            "--config",
            str(tmp_path / "adversary.json"),
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert observed == {
        "source": tmp_path / "source",
        "output": output,
        "checkpoint_directory": tmp_path / "victim",
        "adversary_config_path": tmp_path / "adversary.json",
        "victim_pin_path": tmp_path / "victim.json",
        "gpudrive_pin_path": tmp_path / "pins.json",
    }
    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "artifact": str(output.resolve()),
        "artifact_id": "adversary-train-test",
    }


def test_adversary_training_error_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(**_: object) -> dict[str, object]:
        raise AdversaryTrainingError("CUDA is required")

    monkeypatch.setattr("gpudrive_adversary.cli.train_adversary_smoke", fail)
    status = main(
        ["adversary-train-smoke", "--output", str(tmp_path / "run")]
    )

    assert status == 1
    assert "gda adversary-train-smoke failed: CUDA is required" in capsys.readouterr().err


@pytest.mark.parametrize("ok, expected_status", [(True, 0), (False, 1)])
def test_validate_adversary_checkpoint_status_and_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ok: bool,
    expected_status: int,
) -> None:
    artifact = tmp_path / "checkpoint"
    output = tmp_path / "validation.json"
    report = {"schema": "test", "ok": ok, "failed_checks": [] if ok else ["file"]}
    monkeypatch.setattr(
        "gpudrive_adversary.cli.validate_adversary_checkpoint",
        lambda path: report if path == artifact else pytest.fail("wrong artifact"),
    )

    status = main(
        [
            "validate-adversary-checkpoint",
            str(artifact),
            "--output",
            str(output),
        ]
    )

    assert status == expected_status
    assert json.loads(output.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize("ok, expected_status", [(True, 0), (False, 1)])
def test_validate_adversary_run_status_and_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ok: bool,
    expected_status: int,
) -> None:
    artifact = tmp_path / "run"
    output = tmp_path / "validation.json"
    report = {"schema": "test", "ok": ok, "failed_checks": [] if ok else ["run"]}
    monkeypatch.setattr(
        "gpudrive_adversary.cli.validate_adversary_training_artifact",
        lambda path: report if path == artifact else pytest.fail("wrong artifact"),
    )

    status = main(
        ["validate-adversary-run", str(artifact), "--output", str(output)]
    )

    assert status == expected_status
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_linux_cuda_adversary_runner_uses_the_pinned_smoke_contract() -> None:
    script = (repository_root() / "scripts/run_adversary_reference.sh").read_text(
        encoding="utf-8"
    )

    assert "adversary-train-smoke" in script
    assert "validate-adversary-run" in script
    assert "--gpus all" in script
    assert "smoke_transformer_ppo.json" in script
    assert "1532950cad84dafc6e9d976a2bcc524ee481a1a1" in script
    assert "cpu" not in script.lower()
