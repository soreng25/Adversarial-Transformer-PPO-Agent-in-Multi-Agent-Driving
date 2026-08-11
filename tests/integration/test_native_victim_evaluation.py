"""Opt-in tests for the pinned deterministic PPO in the reference runtime."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gpudrive_adversary.victim.evaluation import (
    run_fresh_victim_evaluation,
    run_victim_evaluation,
    validate_victim_artifact,
)


pytestmark = [pytest.mark.gpudrive, pytest.mark.smoke]


def _native_inputs() -> tuple[Path, Path]:
    if os.environ.get("GPUDRIVE_RUN_NATIVE_TESTS") != "1":
        pytest.skip("set GPUDRIVE_RUN_NATIVE_TESTS=1 in the reference container")
    source = Path(os.environ.get("GPUDRIVE_SOURCE", "/opt/gpudrive"))
    checkpoint = Path(
        os.environ.get(
            "GPUDRIVE_VICTIM_CHECKPOINT",
            "/opt/checkpoints/policy_S10_000_02_27/"
            "1532950cad84dafc6e9d976a2bcc524ee481a1a1",
        )
    )
    if not source.is_dir() or not checkpoint.is_dir():
        pytest.skip("pinned GPUDrive source or victim checkpoint is unavailable")
    return source, checkpoint


def test_frozen_slot0_victim_repeats_after_reset(tmp_path: Path) -> None:
    source, checkpoint = _native_inputs()
    artifact = tmp_path / "victim"
    run_victim_evaluation(
        source=source,
        checkpoint_directory=checkpoint,
        output=artifact,
        device=os.environ.get("GPUDRIVE_SMOKE_DEVICE", "cuda"),
    )
    assert validate_victim_artifact(artifact)["ok"]


@pytest.mark.gpu
@pytest.mark.slow
def test_frozen_slot0_victim_matches_across_processes(tmp_path: Path) -> None:
    source, checkpoint = _native_inputs()
    report = run_fresh_victim_evaluation(
        source=source,
        checkpoint_directory=checkpoint,
        output=tmp_path / "fresh-victim",
        device=os.environ.get("GPUDRIVE_SMOKE_DEVICE", "cuda"),
    )
    assert report["ok"]
