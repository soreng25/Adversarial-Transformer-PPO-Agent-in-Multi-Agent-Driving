# Milestone B validation record

Status: **repository checks pass; native GPUDrive certificate pending**

The published PPO is pinned to Hugging Face revision
`1532950cad84dafc6e9d976a2bcc524ee481a1a1`. Local acquisition verified the
exact `config.json` and `model.safetensors` sizes and SHA-256 values, all 24
F32 tensor names/shapes, 51,228 parameters, the four pinned GPUDrive source
files, and the derived loaded-state hash
`75fc9f6f54636b32a560aaa9eb7f2464b586aa478f6ccab130b67a91d0e166bb`.

The default unit suite requires neither Torch, GPUDrive, CUDA, nor a training
job. It validates checkpoint parsing, action-table identity, lowest-index
argmax ties, slot-0 SDC binding, kernel-cache byte hashing, exact event/action
comparison, tolerant numeric comparison, and reference-script pins.
On 2026-08-11 the full local suite reported `33 passed, 4 skipped`; the four
skips are exactly the opt-in native Milestone A/B tests.

```powershell
./.venv/Scripts/python.exe -m pytest -m unit -q
./.venv/Scripts/python.exe -m gpudrive_adversary `
    verify-victim-checkpoint --source .deps/gpudrive
```

The deterministic evaluator itself is opt-in because this host has no Docker,
CUDA toolkit, NVIDIA runtime, or built `madrona_gpudrive` extension. On a
supported machine, build the reference image with the Milestone A bootstrap,
then run:

```powershell
./scripts/run_victim_reference.ps1 -Device cuda
```

The native certificate is valid only if both closed-loop processes independently
recompute identical action indices/events/termination, all numeric fields meet
the declared tolerance, the policy state equals the pinned state hash before
and after, slot 0 remains SDC object 271, and both artifacts pass
`validate-victim`.

This milestone intentionally has no failure predicate. Its artifact declares
`eligibility.status=not_assessed`, `failure_definition=null`, and
`failure_timestep=null`. The scene is a smoke fixture, not the R10 research
cohort.
