"""Opt-in tests for the pinned native GPUDrive reference runtime."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gpudrive_adversary.smoke import (
    run_fresh_process_smoke,
    run_scene_smoke,
    validate_smoke_artifact,
)


pytestmark = [pytest.mark.gpudrive, pytest.mark.smoke]


def _native_source() -> Path:
    if os.environ.get("GPUDRIVE_RUN_NATIVE_TESTS") != "1":
        pytest.skip("set GPUDRIVE_RUN_NATIVE_TESTS=1 in the reference container")
    source = Path(os.environ.get("GPUDRIVE_SOURCE", "/opt/gpudrive"))
    if not source.is_dir():
        pytest.skip(f"pinned GPUDrive source is unavailable: {source}")
    return source


def test_one_scene_reset_and_action_transport(tmp_path: Path) -> None:
    artifact = tmp_path / "one-process"
    run_scene_smoke(
        source=_native_source(),
        output=artifact,
        device=os.environ.get("GPUDRIVE_SMOKE_DEVICE", "cuda"),
    )
    assert validate_smoke_artifact(artifact)["ok"]


@pytest.mark.gpu
@pytest.mark.slow
def test_two_fresh_processes_match(tmp_path: Path) -> None:
    report = run_fresh_process_smoke(
        source=_native_source(),
        output=tmp_path / "fresh-processes",
        device=os.environ.get("GPUDRIVE_SMOKE_DEVICE", "cuda"),
    )
    assert report["ok"]
