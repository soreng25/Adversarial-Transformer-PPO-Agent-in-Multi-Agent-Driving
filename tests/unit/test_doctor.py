from pathlib import Path

import pytest

from gpudrive_adversary.doctor import build_doctor_report


pytestmark = pytest.mark.unit


def test_doctor_reports_missing_source_without_importing_gpudrive(tmp_path: Path) -> None:
    report = build_doctor_report(
        source=tmp_path / "missing",
        probe_runtime=False,
        reference=False,
    )
    assert not report["ok"]
    checks = {check["name"]: check for check in report["checks"]}
    assert not checks["source.exists"]["ok"]
    assert "runtime" in report


def test_reference_doctor_cannot_skip_runtime_probe(tmp_path: Path) -> None:
    report = build_doctor_report(
        source=tmp_path / "missing",
        probe_runtime=False,
        reference=True,
    )
    checks = {check["name"]: check for check in report["checks"]}
    assert not checks["reference.runtime_probe_enabled"]["ok"]
    assert checks["reference.runtime_probe_enabled"]["required"]
    assert not report["ok"]
