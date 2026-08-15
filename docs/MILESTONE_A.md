# Milestone A: pinned GPUDrive installation and one-scene smoke

Status: **repository tooling implemented; native certification requires a supported build host**

This milestone contains no victim, adversary, research failure predicate, or training code. It verifies only the simulator/runtime boundary needed by later work.

## Immutable inputs

The machine-readable source of truth is [`configs/runtime/gpudrive-pins.json`](../configs/runtime/gpudrive-pins.json).

| Component | Pin |
|---|---|
| GPUDrive | `aa48a431ed127a37610cc2176db30ec73d0c55df` |
| GPUDrive Git tree | `33240941cc9e2504f2cbc9f61f7169b2a7d5ac25` |
| Madrona submodule | `4bda33465340fabc2e61fb27f95aa04795a15466` |
| nlohmann/json submodule | `0457de21cffb298c22b629e538036bfeb96130b7` |
| Upstream `uv.lock` SHA-256 | `bd0af4c8fda0c7932f9296f22ab1df3c52cbe92320884b0ea1aa994ab6fc28a2` |
| Reference Python | `3.11.9` |
| Reference CUDA | `12.4` |
| Linux/amd64 CUDA image manifest | `sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4` |
| `uv` wheel | `0.12.3`, SHA-256 `1482d1462b1aecd18ee33627363fe1c63d6a194f12d40d37efc446d9e0d800a1` |

The CUDA digest was resolved for the amd64 image, not merely its multi-platform tag. The corresponding index digest is also stored in the pin file.

## Reference runtime

The normative runtime is the digest-pinned Linux/amd64 CUDA container in [`containers/gpudrive/Dockerfile`](../containers/gpudrive/Dockerfile). It:

1. verifies the pinned `uv` wheel before installation;
2. checks out GPUDrive at the exact detached commit;
3. initializes and verifies both recursive submodules;
4. verifies the upstream lock and smoke-scene bytes;
5. creates the exact Python 3.11.9 environment with `uv sync --frozen`;
6. builds the native simulator; and
7. exposes a persistent, fingerprinted Madrona kernel-cache path.

The bootstrap also fingerprints this port's Git commit, dirty state, complete
dirty patch (including untracked research files), and exact filtered source tree.
Those values are injected into the image and checked again against the copied
build context before a smoke artifact is accepted.

Native Windows is diagnostic only. Windows users should run the same Linux image through Docker Desktop with WSL2. GPUDrive's native module is CUDA-linked even when the simulator uses CPU execution mode, so both reference commands require NVIDIA Container Toolkit, `--gpus all`, and a compatible driver. `cpu` selects GPUDrive's CPU simulator backend; it does not make the container GPU-independent.

### Build and run on Linux

```bash
bash scripts/bootstrap_linux.sh cuda
```

For a CPU execution-mode diagnostic inside the same pinned image:

```bash
bash scripts/bootstrap_linux.sh cpu
```

### Build and run from Windows PowerShell

```powershell
./scripts/bootstrap_windows.ps1 -Device cuda
```

The scripts build the reference image, run strict runtime diagnostics, launch two isolated smoke processes, and compare their typed traces.

## Source-only setup

Python 3.11 is required for the port tooling. Fetching is non-destructive: an existing non-Git destination or wrong checkout fails instead of being replaced.

```powershell
py -3.11 -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[test]"
./.venv/Scripts/python.exe scripts/fetch_gpudrive.py
./.venv/Scripts/python.exe -m gpudrive_adversary verify-source --source .deps/gpudrive
```

The equivalent POSIX activation is optional; commands can call `.venv/bin/python` directly.

## Doctor

For a source-only host audit, skip native imports explicitly:

```bash
python -m gpudrive_adversary doctor \
    --source .deps/gpudrive \
    --skip-runtime \
    --strict \
    --output artifacts/milestone-a/doctor.json
```

The reference bootstrap omits `--skip-runtime` and adds `--reference`. The doctor checks command exit codes—not merely executable names—and verifies the source commit/tree, tracked cleanliness, unexpected untracked files, every recursive submodule, package version, upstream lock, scene bytes, Python, GCC/G++, CMake/Ninja, native imports, native-extension bytes, Torch CUDA version, live CUDA devices, `nvidia-smi` driver identity, digest-pinned base image, port commit/dirty/patch/tree identity, and Madrona cache identity. A reference report cannot disable runtime probes.

## One-scene smoke contract

Configuration: [`configs/smoke/scene.json`](../configs/smoke/scene.json).

| Field | Expected value |
|---|---|
| Scene | `tfrecord-00000-of-01000_325.json` |
| Scene SHA-256 | `69bd2b9ae49d43745651262abf3956309e9c0092ca24aff72e0f9abb32f9b948` |
| Scenario ID | `ef3a8f65142f41ac` |
| Controlled actor | the single SDC, simulator slot 0, stable ID 271 |
| Background actors | logged trajectory playback |
| Dynamics | classic |
| Collision response | `ignore`, for signal inspection only |
| Neutral discrete action | index 45, decoded `[0, 0, 0]` |
| Physical command order | `[acceleration, steering, head_angle]` |
| Raw info order | road contact, vehicle collision, non-vehicle collision, goal, actor type |

The smoke asserts the one-SDC binding, corrected stable ID, 91-action table, observation and info shapes, finite observations, and named action transport into the first three native action fields. It steps three fixed, non-research commands—neutral discrete index 45, mild acceleration `[1.25, 0, 0]`, and mild steering `[0, 0.125, 0]`—then resets and repeats them. Native action transport, event fields, and the raw-to-high-level info mapping must match exactly; state-like floating tensors use the configured tolerance.

Inside the reference image, run one process:

```bash
python -m gpudrive_adversary scene-smoke \
    --source .deps/gpudrive \
    --device cpu \
    --output artifacts/milestone-a/cpu-one-process
```

Run two fresh processes and compare them:

```bash
python -m gpudrive_adversary fresh-smoke \
    --source .deps/gpudrive \
    --device cuda \
    --output artifacts/milestone-a/cuda-fresh
```

Each successful run atomically publishes `manifest.json` and a non-pickled `trace.npz`; failures remove the private temporary directory rather than leaving an apparently usable artifact. Before publication, the validator recomputes the trace, config, source-verification, port, scene, native-extension, cache, runtime, driver, and reference-image contracts. The manifest also stores raw signal order, stable victim identity, and explicit `null` checkpoint/failure fields with reasons.

Validate an artifact independently:

```bash
python -m gpudrive_adversary validate-smoke artifacts/milestone-a/cpu-one-process
```

## Unit tests

The default tests need neither GPUDrive nor a training job:

```bash
python -m pytest -m unit
```

They cover immutable pins, scene identity, failure-definition absence, canonical and source-tree hashing, doctor failure modes, numeric versus exact-event trace comparison, tamper-rejecting artifact validation, and static container pins. Opt-in tests under `tests/integration/` exercise the same native one-process and fresh-process contracts without starting training.

The validation performed in this checkout is recorded in [MILESTONE_A_VALIDATION.md](MILESTONE_A_VALIDATION.md).

## What remains before Milestone A certification

Repository implementation alone is not a native-runtime certificate. A supported host must successfully produce:

- a strict reference doctor report;
- one CPU and/or CUDA one-scene artifact;
- a passing same-process reset comparison;
- a passing two-fresh-process comparison; and
- recorded native extension, driver/GPU, and kernel-cache fingerprints.

On a machine without Docker/WSL2 or the C++/CUDA toolchain, source verification and unit tests can pass while the native checks correctly remain unverified.
