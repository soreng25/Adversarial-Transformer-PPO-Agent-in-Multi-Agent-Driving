#!/usr/bin/env bash
set -euo pipefail

python_bin="${GPUDRIVE_PYTHON:-.deps/gpudrive/.venv/bin/python}"
output="${GPUDRIVE_BOUND_SWEEP_OUTPUT:-artifacts/calibration/highway-bound-sweep-500}"
episodes="${GPUDRIVE_BOUND_SWEEP_EPISODES:-500}"
scene="${GPUDRIVE_HIGHWAY_SCENE:-.deps/datasets/GPUDrive_mini/validation/tfrecord-00107-of-00150_281.json}"

[[ "$(uname -s)" == "Linux" ]] || { echo "Bound calibration requires Linux/CUDA." >&2; exit 1; }
[[ -x "${python_bin}" ]] || { echo "Missing GPUDrive Python environment: ${python_bin}" >&2; exit 1; }
[[ ! -e "${output}" ]] || { echo "Output already exists: ${output}" >&2; exit 1; }

bash scripts/prepare_highway_scene.sh "${scene}"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MADRONA_MWGPU_KERNEL_CACHE="${MADRONA_MWGPU_KERNEL_CACHE:-$PWD/.cache/gpudrive/megakernel.bin}"
mkdir -p "$(dirname "${MADRONA_MWGPU_KERNEL_CACHE}")" "$(dirname "${output}")"

"${python_bin}" -m gpudrive_adversary highway-bound-sweep \
  --source "$PWD/.deps/gpudrive" \
  --adversary-config configs/adversary/highway_10agent_nonfocal_system_transformer_ppo.json \
  --experiment-config configs/multiagent/highway_10agent_nonfocal_system.json \
  --sweep-config configs/calibration/highway_nonfocal_bound_sweep.json \
  --scene-source "${scene}" \
  --episodes-per-bound "${episodes}" \
  --output "${output}"

printf 'Bound sweep complete: %s/summary.json\n' "${output}"
