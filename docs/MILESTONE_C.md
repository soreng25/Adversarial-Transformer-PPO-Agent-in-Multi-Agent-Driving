# Milestone C: bounded Transformer-PPO adversary

Milestone C is implemented as a **tiny, one-scene training smoke**. It is not
a research-scale experiment and it has not been executed on this host because
the pinned native GPUDrive/CUDA runtime is unavailable here.

## Settled smoke contract

- Victim: the pinned published PPO, frozen in evaluation mode, controlling the
  WOMD SDC in simulator slot 0. Other dynamic actors follow logged trajectories.
- Intervention: after deterministic victim action decoding, add a continuous
  physical residual `[delta_acceleration, delta_steering]`; preserve the
  nominal head-angle command and do not requantize.
- Residual bounds: `+/-0.667` acceleration and `+/-0.262` steering.
- Final command envelope: acceleration `[-4, 4]`, steering
  `[-3.142, 3.142]`.
- Prior: independent zero-mean tanh-squashed Normals with latent standard
  deviations `[0.5, 0.5]`. Both the PPO density and the prior density include
  the exact change-of-variables term and have no clipping atoms.
- Failure: the first victim road-object contact, vehicle collision, or
  nonvehicle collision. Goal and horizon are nonfailures, and safety wins a
  simultaneous goal tie.
- Eligibility: the zero-disturbance victim must first reach its goal without a
  safety event on the exact pinned scene. Training fails closed otherwise.
- Adversary input at transition `t`: current 2,984-D victim observation,
  previous nominal 3-D command, and previous effective 2-D disturbance. The
  current victim action is unavailable to the adversary.
- History/model: a right-aligned 50-token causal history, one Transformer
  layer/head, 64-D model state, and explicit 64-D `pre_actor_features` captured
  before the disturbance head.
- Smoke reward: `1.0 * failure - 0.01 * NLL_excess_from_zero`, with no horizon
  reward. These two numbers are plumbing values, not approved research
  calibration.

The PyTorch model is a methodology port, not an RLlib line-by-line copy. Its
single attention head has width 64, whereas the inspected CartPole wrapper used
a 32-D attention setting. The stable 64-D pre-actor latent and causal 50-step
semantics are explicit and tested. A research-scale architecture comparison
must be declared before interpreting results.

The pinned victim configuration removes nonvehicles. The failure classifier
still records the correct raw nonvehicle channel, but that subtype is likely
unreachable in this exact smoke environment. Its absence must not be reported
as evidence of safety.

## Implemented components

- `configs/adversary/smoke_transformer_ppo.json`: immutable smoke choices and
  CartPole/GPUDrive/victim provenance inputs.
- `adversary/distribution.py`: exact bounded prior density and NLL.
- `adversary/intervention.py`: named residual composition and saturation
  accounting.
- `adversary/failure.py`: post-step victim failure and nominal eligibility.
- `adversary/environment.py`: causal transition scheduling and validated T+1/T
  in-memory rollouts.
- `adversary/model.py` and `ppo.py`: causal Transformer actor-critic, GAE, and
  clipped PPO update.
- `adversary/checkpoint.py`: safetensors weights, typed non-pickle optimizer
  state, hashes, and validator.
- `adversary/training.py`: Linux/CUDA one-scene eligibility preflight,
  collection, PPO updates, deterministic post-update evaluation, frozen-victim
  checks, and atomic artifact publication.
- CLI commands `adversary-train-smoke`, `validate-adversary-checkpoint`, and
  `validate-adversary-run`.

Every run/checkpoint carries the methodology commit, GPUDrive/submodule pins,
scene hash/scenario ID, stable victim ID, victim checkpoint hash, adversary
configuration hash, native-extension hash, port identity, and runtime details.

## Reference execution

On a Linux x86-64 machine with Docker, an NVIDIA GPU, and NVIDIA Container
Toolkit:

```bash
scripts/bootstrap_linux.sh cuda
scripts/run_adversary_reference.sh
```

The runner sets deterministic CUDA/CUBLAS controls and uses the digest-pinned
reference image. It writes the atomic run directory under
`artifacts/milestone-c/tiny-train`, then validates the indivisible run and all
child checkpoints.

The eligibility preflight is deliberately real. This repository has not yet
proved that the bundled scene is a clean nominal goal success. If it is not,
the command stops before training; selecting and pinning a different eligible
scene is a research-cohort decision, not an error to bypass.

## Validation boundary

Platform-neutral unit tests exercise bounds, exact densities, failure
precedence, causal token/history construction, PPO arithmetic, schemas, CLI
dispatch, and validators without a training job. Torch execution tests are
skipped when Torch is absent. The final native acceptance item is a successful
reference run plus a repeated deterministic evaluation on Linux/CUDA; that
certificate is still pending.

Full-scale training remains blocked on the research scene cohort, reward-scale
calibration, number of seeds, and total transitions. Milestone D will promote
rollouts into the complete replay-certified trajectory schema.
