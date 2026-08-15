#!/usr/bin/env bash
set -euo pipefail

revision="f0abb649610379e70d0b3105745550fa7acc90d0"
relative_path="validation/tfrecord-00107-of-00150_281.json"
expected_sha="1f2e221207cefe8c43a27eef93cb82a3fe3d6bba7553a70de8f0762ef17173ec"
target="${1:-.deps/datasets/GPUDrive_mini/${relative_path}}"
url="https://huggingface.co/datasets/EMERGE-lab/GPUDrive_mini/resolve/${revision}/${relative_path}?download=true"

mkdir -p "$(dirname "${target}")"
if [[ ! -f "${target}" ]]; then
  temporary="${target}.partial"
  trap 'rm -f "${temporary}"' EXIT
  curl --fail --location --retry 3 --output "${temporary}" "${url}"
  mv "${temporary}" "${target}"
  trap - EXIT
fi

observed_sha="$(sha256sum "${target}" | awk '{print $1}')"
[[ "${observed_sha}" == "${expected_sha}" ]] || {
  echo "Highway scene hash mismatch: expected ${expected_sha}, got ${observed_sha}" >&2
  exit 1
}
printf 'Pinned highway scene ready: %s\n' "${target}"
