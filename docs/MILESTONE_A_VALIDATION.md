# Milestone A validation record

Validation date: 2026-08-11

This record distinguishes repository/source validation from native GPUDrive
certification. A missing native result is not treated as a pass.

## Completed on this checkout

- Port baseline: `08eb402d47f3d9c39e6f8842885bbbf527dccaa0`, with the current dirty patch and exact source-tree SHA-256 recorded dynamically by `scripts/port_identity.py`.
- GPUDrive checkout: exact detached commit `aa48a431ed127a37610cc2176db30ec73d0c55df` and Git tree `33240941cc9e2504f2cbc9f61f7169b2a7d5ac25`.
- Top-level gitlinks: json `0457de21cffb298c22b629e538036bfeb96130b7`; Madrona `4bda33465340fabc2e61fb27f95aa04795a15466`.
- Recursive source audit: all 17 reported GPUDrive/Madrona submodule checkouts were initialized at their recorded gitlinks.
- Upstream `uv.lock`, package version, scene bytes, scenario ID, source SDC index, SDC object ID, type, and expert flag all matched their machine-readable pins.
- Python compilation of `src`, `scripts`, and `tests`: passed.
- Unit suite under Python 3.11.9: 15 passed; no GPUDrive import, simulator build, or training job required.
- CLI parser/help and source-only verification: passed.

The ignored reports produced locally are:

- `artifacts/milestone_a/source-verification.json`
- `artifacts/milestone_a/doctor-source-only.json`

They are local evidence, not versioned substitutes for rerunning validation.

## Not completed on this host

This machine has no Docker/WSL2 installation, CMake/Ninja compiler toolchain,
CUDA toolkit, NVIDIA runtime, `torch`, or built `madrona_gpudrive` extension.
Consequently, none of the following is claimed to pass here:

- building the digest-pinned Linux/amd64 reference image;
- importing the pinned native extension;
- strict reference doctor CUDA/driver checks;
- same-process native scene replay; or
- two-fresh-process native scene replay.

Run `scripts/bootstrap_linux.sh cuda` or
`scripts/bootstrap_windows.ps1 -Device cuda` on a supported NVIDIA host. Step 1
is certified only when that command produces a strict doctor report and a
passing fresh-process comparison under `artifacts/milestone-a/`.
