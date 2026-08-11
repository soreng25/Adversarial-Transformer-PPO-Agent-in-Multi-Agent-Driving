#!/usr/bin/env bash
set -euo pipefail

image_name="${GPUDRIVE_IMAGE_NAME:-gpudrive-adversary:a-aa48a431}"
device="${1:-cuda}"

if [[ "${device}" != "cpu" && "${device}" != "cuda" ]]; then
    echo "usage: $0 [cpu|cuda]" >&2
    exit 2
fi

command -v docker >/dev/null || {
    echo "Docker and the NVIDIA Container Toolkit are required." >&2
    exit 1
}

docker image inspect "${image_name}" >/dev/null 2>&1 || {
    echo "Reference image ${image_name} is missing; run scripts/bootstrap_linux.sh first." >&2
    exit 1
}

mkdir -p artifacts/milestone-b .cache/gpudrive

docker run --rm --gpus all \
    --shm-size=20g \
    --volume "${PWD}/artifacts:/workspace/port/artifacts" \
    --volume "${PWD}/.cache/gpudrive:/var/cache/gpudrive" \
    "${image_name}" \
    python -m gpudrive_adversary verify-victim-checkpoint \
        --source /opt/gpudrive \
        --checkpoint /opt/checkpoints/policy_S10_000_02_27/1532950cad84dafc6e9d976a2bcc524ee481a1a1 \
        --output "artifacts/milestone-b/checkpoint-${device}.json"

docker run --rm --gpus all \
    --shm-size=20g \
    --volume "${PWD}/artifacts:/workspace/port/artifacts" \
    --volume "${PWD}/.cache/gpudrive:/var/cache/gpudrive" \
    "${image_name}" \
    python -m gpudrive_adversary victim-fresh-eval \
        --source /opt/gpudrive \
        --checkpoint /opt/checkpoints/policy_S10_000_02_27/1532950cad84dafc6e9d976a2bcc524ee481a1a1 \
        --device "${device}" \
        --output "artifacts/milestone-b/fresh-${device}"
