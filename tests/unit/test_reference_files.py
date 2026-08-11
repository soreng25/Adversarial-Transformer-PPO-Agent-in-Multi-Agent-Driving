from pathlib import Path

import pytest

from gpudrive_adversary.pins import repository_root


pytestmark = pytest.mark.unit


def test_dockerfile_pins_base_source_and_uv() -> None:
    root = repository_root()
    dockerfile = (root / "containers/gpudrive/Dockerfile").read_text(
        encoding="utf-8"
    )
    assert (
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:"
        "0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4"
        in dockerfile
    )
    assert "aa48a431ed127a37610cc2176db30ec73d0c55df" in dockerfile
    assert "uv 0.12.3" in dockerfile
    assert "ninja-build" in dockerfile
    assert "PORT_SOURCE_TREE_SHA256" in dockerfile
    assert "scripts/fetch_victim_checkpoint.py" in dockerfile
    assert "1532950cad84dafc6e9d976a2bcc524ee481a1a1" in dockerfile
    assert (
        "MADRONA_MWGPU_KERNEL_CACHE=/var/cache/gpudrive/megakernel.bin"
        in dockerfile
    )
    assert "1482d1462b1aecd18ee33627363fe1c63d6a194f12d40d37efc446d9e0d800a1" in dockerfile


def test_reference_victim_runners_use_fresh_process_evaluation() -> None:
    root = repository_root()
    for relative in (
        "scripts/run_victim_reference.sh",
        "scripts/run_victim_reference.ps1",
    ):
        script = (root / relative).read_text(encoding="utf-8")
        assert "victim-fresh-eval" in script
        assert "1532950cad84dafc6e9d976a2bcc524ee481a1a1" in script


def test_no_gpu_import_at_package_import_time() -> None:
    import gpudrive_adversary

    assert gpudrive_adversary.__version__ == "0.1.0"
