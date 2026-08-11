#!/usr/bin/env bash
set -euo pipefail

image_name="${GPUDRIVE_IMAGE_NAME:-gpudrive-adversary:a-aa48a431}"
device="${1:-cuda}"

if [[ "${device}" != "cpu" && "${device}" != "cuda" ]]; then
    echo "usage: $0 [cpu|cuda]" >&2
    exit 2
fi

command -v docker >/dev/null || {
    echo "Docker is required. Install Docker Engine and, for CUDA, NVIDIA Container Toolkit." >&2
    exit 1
}

command -v python3 >/dev/null || {
    echo "Python 3 is required to fingerprint the exact build context." >&2
    exit 1
}

port_commit="$(python3 scripts/port_identity.py commit)"
port_dirty="$(python3 scripts/port_identity.py dirty)"
port_diff_sha256="$(python3 scripts/port_identity.py diff_sha256)"
port_source_tree_sha256="$(python3 scripts/port_identity.py source_tree_sha256)"

docker build \
    --platform linux/amd64 \
    --file containers/gpudrive/Dockerfile \
    --tag "${image_name}" \
    --build-arg "PORT_GIT_COMMIT=${port_commit}" \
    --build-arg "PORT_DIRTY=${port_dirty}" \
    --build-arg "PORT_DIFF_SHA256=${port_diff_sha256}" \
    --build-arg "PORT_SOURCE_TREE_SHA256=${port_source_tree_sha256}" \
    .

mkdir -p artifacts/milestone-a .cache/gpudrive

# GPUDrive's native module is CUDA-linked even for ExecMode.CPU. Both smoke
# modes therefore run in the same NVIDIA-enabled reference container.
gpu_args=(--gpus all)

docker run --rm "${gpu_args[@]}" \
    --shm-size=20g \
    --volume "${PWD}/artifacts:/workspace/port/artifacts" \
    --volume "${PWD}/.cache/gpudrive:/var/cache/gpudrive" \
    "${image_name}" \
    python -m gpudrive_adversary doctor \
        --source /opt/gpudrive \
        --reference \
        --strict \
        --output artifacts/milestone-a/doctor-${device}.json

docker run --rm "${gpu_args[@]}" \
    --shm-size=20g \
    --volume "${PWD}/artifacts:/workspace/port/artifacts" \
    --volume "${PWD}/.cache/gpudrive:/var/cache/gpudrive" \
    "${image_name}" \
    python -m gpudrive_adversary fresh-smoke \
        --source /opt/gpudrive \
        --device "${device}" \
        --output "artifacts/milestone-a/fresh-${device}"
