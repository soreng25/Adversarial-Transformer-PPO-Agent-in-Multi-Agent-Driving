# Adversarial Transformer-PPO for GPUDrive

A source-audited methodology port for replayable adversarial failure discovery in GPUDrive.

Current implementation status: **Milestones A-C repository paths are implemented; native certificates are pending**. Milestone B pins, verifies, freezes, and deterministically evaluates the published slot-0 SDC PPO. Milestone C implements the approved bounded 2-D intervention, victim-only failure predicate, causal Transformer-PPO environment/trainer, safe checkpoints, and a Linux/CUDA tiny-training runner. The current host lacks the reference native CUDA runtime, so no native training result is claimed. Baselines, replay-certified trajectory artifacts, MCMC, and latent analysis remain future milestones.

Pinned revisions:

- CartPole methodology: `315b14a90b252ba416eb329e8003d5926806ba67`
- GPUDrive: `aa48a431ed127a37610cc2176db30ec73d0c55df`

Start with [Milestone A](docs/MILESTONE_A.md), [Milestone B](docs/MILESTONE_B.md), and [Milestone C](docs/MILESTONE_C.md). The unresolved research-scale choices remain in the [methodology map](docs/CARTPOLE_TO_GPUDRIVE_MAP.md) and [research gate](docs/PROJECT_SPEC.md#8-research-decisions-and-gates).
