from pathlib import Path

import pytest

from gpudrive_adversary.provenance import port_source_tree_sha256


pytestmark = pytest.mark.unit


def test_port_tree_hash_ignores_generated_directories(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("value = 1\n", encoding="utf-8")
    before = port_source_tree_sha256(tmp_path)

    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "result.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "noise.txt").write_text("ignored", encoding="utf-8")
    assert port_source_tree_sha256(tmp_path) == before


def test_port_tree_hash_changes_with_research_source(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    before = port_source_tree_sha256(tmp_path)
    source.write_text("value = 2\n", encoding="utf-8")
    assert port_source_tree_sha256(tmp_path) != before
