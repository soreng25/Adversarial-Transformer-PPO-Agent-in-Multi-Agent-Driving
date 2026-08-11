#!/usr/bin/env bash
set -euo pipefail

image_name="${GPUDRIVE_IMAGE_NAME:-gpudrive-adversary:a-aa48a431}"
training_output="${GPUDRIVE_ADVERSARY_OUTPUT:-artifacts/milestone-c/tiny-train}"
victim_checkpoint="/opt/checkpoints/policy_S10_000_02_27/1532950cad84dafc6e9d976a2bcc524ee481a1a1"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This reference training runner requires a Linux host with CUDA." >&2
    exit 1
fi

command -v docker >/dev/null || {
    echo "Docker and the NVIDIA Container Toolkit are required." >&2
    exit 1
}

docker image inspect "${image_name}" >/dev/null 2>&1 || {
    echo "Reference image ${image_name} is missing; run scripts/bootstrap_linux.sh cuda first." >&2
    exit 1
}

mkdir -p artifacts/milestone-c .cache/gpudrive

docker run --rm --gpus all \
    --shm-size=20g \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --volume "${PWD}/artifacts:/workspace/port/artifacts" \
    --volume "${PWD}/.cache/gpudrive:/var/cache/gpudrive" \
    "${image_name}" \
    python -m gpudrive_adversary adversary-train-smoke \
        --source /opt/gpudrive \
        --victim-checkpoint "${victim_checkpoint}" \
        --config configs/adversary/smoke_transformer_ppo.json \
        --output "${training_output}"

docker run --rm --gpus all \
    --volume "${PWD}/artifacts:/workspace/port/artifacts" \
    "${image_name}" \
    python -m gpudrive_adversary validate-adversary-run \
        "${training_output}" \
        --output artifacts/milestone-c/run-validation.json
