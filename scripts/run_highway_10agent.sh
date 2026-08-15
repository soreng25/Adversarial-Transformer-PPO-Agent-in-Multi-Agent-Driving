#!/usr/bin/env bash
set -euo pipefail

python_bin="${GPUDRIVE_PYTHON:-.deps/gpudrive/.venv/bin/python}"
output="${GPUDRIVE_HIGHWAY_OUTPUT:-artifacts/highway-10agent/train-100}"
scene="${GPUDRIVE_HIGHWAY_SCENE:-.deps/datasets/GPUDrive_mini/validation/tfrecord-00107-of-00150_281.json}"

[[ "$(uname -s)" == "Linux" ]] || { echo "This experiment requires Linux/CUDA." >&2; exit 1; }
[[ -x "${python_bin}" ]] || { echo "Missing GPUDrive Python environment: ${python_bin}" >&2; exit 1; }
[[ ! -e "${output}" ]] || { echo "Output already exists: ${output}" >&2; exit 1; }

bash scripts/prepare_highway_scene.sh "${scene}"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MADRONA_MWGPU_KERNEL_CACHE="${MADRONA_MWGPU_KERNEL_CACHE:-$PWD/.cache/gpudrive/megakernel.bin}"
mkdir -p "$(dirname "${MADRONA_MWGPU_KERNEL_CACHE}")" "$(dirname "${output}")"

"${python_bin}" -m gpudrive_adversary highway-train \
  --source "$PWD/.deps/gpudrive" \
  --adversary-config configs/adversary/highway_10agent_transformer_ppo.json \
  --experiment-config configs/multiagent/highway_10agent.json \
  --scene-source "${scene}" \
  --output "${output}"

"${python_bin}" -m gpudrive_adversary validate-highway-run "${output}" \
  --output "${output}-validation.json"
