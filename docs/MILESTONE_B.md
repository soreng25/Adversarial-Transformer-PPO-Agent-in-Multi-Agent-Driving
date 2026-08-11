# Milestone B: frozen deterministic victim evaluation

Status: **one-scene repository implementation complete; native evaluation requires the reference runtime**

The user approved the two victim choices on 2026-08-11:

- control the dataset-designated self-driving car in simulator slot 0; and
- use the existing published GPUDrive PPO rather than train a new victim.

This milestone does not implement disturbances, an adversary, a research
failure predicate, or nominal-success eligibility.

## Immutable policy

The machine-readable source of truth is
[`configs/victim/pretrained_ppo.json`](../configs/victim/pretrained_ppo.json).

| Field | Pinned value |
|---|---|
| Model repository | `daphne-cornelisse/policy_S10_000_02_27` |
| Revision | `1532950cad84dafc6e9d976a2bcc524ee481a1a1` |
| Weights | `model.safetensors`, 206,856 bytes |
| Weights SHA-256 | `f3f26475def35f375c6c72d8f8f20b2b091f77175010345dc3fa968a860521b7` |
| Configuration SHA-256 | `e3f4cb2599f8b36aa8da4eb5c8fcb520a8e9b799ced35bfa6806fb80e0e98829` |
| Architecture | feed-forward late fusion, 51,228 parameters |
| Input/output | 2,984 observation values, 91 discrete actions |

The downloader uses revision-qualified URLs, verifies bytes before publishing
them to the ignored `.deps` cache, and refuses to overwrite an unexpected
existing file. Loading uses safetensors plus strict state-dictionary matching;
pickle-based `torch.load` and an unpinned Hub lookup are not used.

```powershell
./.venv/Scripts/python.exe scripts/fetch_victim_checkpoint.py `
    --gpudrive-source .deps/gpudrive

./.venv/Scripts/python.exe -m gpudrive_adversary `
    verify-victim-checkpoint `
    --source .deps/gpudrive
```

The pinned model card does not declare a license. The repository therefore
stores only hashes and acquisition instructions; it does not vendor the model
bytes.

## Slot 0 versus model layout

These two values intentionally differ:

- `max_cont_agents=1` tells GPUDrive that only simulator slot 0 receives PPO
  actions; and
- `max_controlled_agents=64` reconstructs the pretrained network's observation
  layout, including 63 partner slots.

Setting the network layout to 1 would corrupt its observation reshape. Every
reset must also assert that the sole controlled slot is slot 0, is marked SDC,
and has the pinned scene's stable object ID 271. All other dynamic actors use
GPUDrive's logged trajectory playback. That differs from multi-agent self-play
training and is recorded as an evaluation-distribution change.

## Checkpoint-compatible environment

Milestone A's simulator-only configuration is not reused. The victim requires
the configuration published with the checkpoint:

- classic normalized ego, partner, and road observations;
- non-vehicles removed;
- collision response `ignore`;
- observation radius 50 m and polyline reduction threshold 0.1;
- weighted reward coefficients `[-0.75, 1.0, -0.75]` for collision, goal, and
  road contact; and
- 7 acceleration values by 13 steering values, in native
  `[acceleration, steering, head_angle]` order.

The complete float32 action table is hashed. Index 45 must decode to
`[0, 0, 0]`.

## Deterministic contract

Evaluation places the network in evaluation mode, disables gradients on every
parameter, and runs inference mode. At every transition it computes logits and
selects the lowest index attaining their maximum. No sampling RNG is involved.
The complete module-state hash is checked before and after rollout.
That state hash must also equal the value derived from the pinned safetensors
bytes; merely remaining unchanged is insufficient.

A native evaluation trace records state-aligned observations, simulator state,
raw event channels, done flags, and rewards, plus transition-aligned logits,
values, log probabilities, action indices, decoded commands, and commands
observed in the native tensor. The same closed-loop policy is rerun after reset;
actions are recomputed rather than replayed forcibly. Fresh-process comparison
additionally requires identical checkpoint, scene, configuration, action-table,
source, build, and runtime identities.

## Research limits still in force

The pinned scene is only an installation/evaluation fixture. Because the
research scene cohort and nominal-success eligibility rule have not been
approved, artifacts state `eligibility.status=not_assessed`. Raw collision,
road-contact, goal, and done signals are retained, but `failure_definition` and
`failure_timestep` remain explicitly null.

This host can verify the checkpoint and run pure tests, but it lacks the native
GPUDrive/CUDA runtime. A passing native victim rollout and fresh-process replay
must therefore be produced by the digest-pinned reference container before
Milestone B receives a runtime certificate.

After building the reference image with the Milestone A bootstrap, run:

```powershell
./scripts/run_victim_reference.ps1 -Device cuda
```

On Linux, use `./scripts/run_victim_reference.sh cuda`. Both commands verify
the checkpoint again and run two fresh closed-loop policy processes. See
[`MILESTONE_B_VALIDATION.md`](MILESTONE_B_VALIDATION.md) for the current
evidence and exact limitation.
