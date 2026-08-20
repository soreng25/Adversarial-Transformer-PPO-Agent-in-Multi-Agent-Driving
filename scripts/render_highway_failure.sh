#!/usr/bin/env bash
set -euo pipefail

python_bin="${GPUDRIVE_PYTHON:-.deps/gpudrive/.venv/bin/python}"
run="${GPUDRIVE_HIGHWAY_RUN:-artifacts/highway-10agent/train-100}"
checkpoint="${1:-iteration-0094}"
zoom_radius="${2:-70}"
output="${GPUDRIVE_FAILURE_OUTPUT:-artifacts/highway-10agent/failure-${checkpoint#iteration-}}"

[[ "$(uname -s)" == "Linux" ]] || { echo "Failure replay/rendering requires Linux/CUDA." >&2; exit 1; }
[[ -x "${python_bin}" ]] || { echo "Missing GPUDrive Python environment: ${python_bin}" >&2; exit 1; }
[[ -d "${run}" ]] || { echo "Missing training run: ${run}" >&2; exit 1; }
[[ -d "${run}/checkpoints/${checkpoint}" ]] || { echo "Missing checkpoint: ${run}/checkpoints/${checkpoint}" >&2; exit 1; }
[[ ! -e "${output}" ]] || { echo "Output already exists: ${output}" >&2; exit 1; }

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MADRONA_MWGPU_KERNEL_CACHE="${MADRONA_MWGPU_KERNEL_CACHE:-$PWD/.cache/gpudrive/megakernel.bin}"
mkdir -p "$(dirname "${MADRONA_MWGPU_KERNEL_CACHE}")" "$(dirname "${output}")"

"${python_bin}" -m gpudrive_adversary render-highway-failure \
  --source "$PWD/.deps/gpudrive" \
  --run "${run}" \
  --checkpoint "${checkpoint}" \
  --zoom-radius "${zoom_radius}" \
  --output "${output}"

printf 'Failure visualization ready: %s\n' "${output}"
