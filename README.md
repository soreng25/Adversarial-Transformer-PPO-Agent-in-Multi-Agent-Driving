# Adversarial Transformer-PPO for GPUDrive

A source-audited methodology port for replayable adversarial failure discovery in GPUDrive.

Current implementation status: **Milestones A-C repository paths are implemented; native certificates are pending**. Milestone B pins, verifies, freezes, and deterministically evaluates the published PPO. Milestone C contains both the original one-victim plumbing smoke and the current ten-PPO-agent highway pilot: only the focal slot-0 car receives the bounded 2-D disturbance, while failure covers any controlled car. The current host lacks the reference native CUDA runtime, so no native ten-agent result is claimed. Baselines, replay-certified trajectory artifacts, MCMC, and latent analysis remain future milestones.

Pinned revisions:

- CartPole methodology: `315b14a90b252ba416eb329e8003d5926806ba67`
- GPUDrive: `aa48a431ed127a37610cc2176db30ec73d0c55df`

Start with [Milestone A](docs/MILESTONE_A.md), [Milestone B](docs/MILESTONE_B.md), [Milestone C](docs/MILESTONE_C.md), and the current [ten-agent highway pilot](docs/HIGHWAY_10AGENT_EXPERIMENT.md). The unresolved research-scale choices remain in the [methodology map](docs/CARTPOLE_TO_GPUDRIVE_MAP.md) and [research gate](docs/PROJECT_SPEC.md#8-research-decisions-and-gates).
