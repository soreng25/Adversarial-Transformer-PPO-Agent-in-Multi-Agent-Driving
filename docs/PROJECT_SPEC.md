# Project specification: adversarial Transformer-PPO for GPUDrive

Status: **source-audited research draft; implementation is intentionally gated**

Prepared: 2026-08-10

Target repository: `soreng25/Adversarial-Transformer-PPO-Agent-in-Multi-Agent-Driving`

## 1. Objective

This repository will transfer the failure-discovery methodology from a pinned CartPole implementation to GPUDrive. It is a methodology port, not a line-by-line rewrite. The port preserves the experimental roles, temporal ordering, failure-conditioned sampling invariant, and representation-analysis pipeline while replacing CartPole-specific physics with explicit driving definitions.

The required stages are:

1. train or load a victim PPO policy;
2. freeze it and evaluate it deterministically;
3. train a causal Transformer-PPO adversary that emits bounded sequential disturbances and pays a declared negative-log-likelihood penalty;
4. save complete replayable trajectories;
5. evaluate constant and simple stochastic baselines;
6. run failure-conditioned Metropolis MCMC from a replay-verified adversarial failure;
7. replay distinct MCMC failure states and capture the final pre-failure Transformer latent; and
8. cluster those latents and visualize them with their traces.

This planning phase does not authorize the simulator wrapper, intervention, failure objective, training, or experiments. Those remain behind the research decisions in Section 8.

## 2. Immutable provenance

### 2.1 Methodology source pin

The methodology source is now resolved and audited:

| Field | Immutable value |
|---|---|
| Repository | [`soreng25/Adversarial-Transformer-PPO-agent-in-Cartpole`](https://github.com/soreng25/Adversarial-Transformer-PPO-agent-in-Cartpole) |
| Revision | [`315b14a90b252ba416eb329e8003d5926806ba67`](https://github.com/soreng25/Adversarial-Transformer-PPO-agent-in-Cartpole/tree/315b14a90b252ba416eb329e8003d5926806ba67) |
| Commit time | `2026-08-03T17:14:31-07:00` |
| Commit subject | `other mcmc` |
| Tag at revision | none; the full commit SHA is normative |

The commit, not `main`, is the source of truth. When prose and executable code differ, this specification records the code behavior and calls out the discrepancy.

The source audit established these core semantics:

- The victim is an RLlib PPO policy. Evaluation uses `explore=False`.
- The adversary controls one scalar wind value. It is clipped to `[-max_wind, max_wind]` and added to the victim's physical CartPole force.
- The adversary observes six values: CartPole's four state values, the previous victim action sign, and the previous applied wind.
- The adversary uses RLlib's attention model with one Transformer unit, a 64-dimensional attention representation, one attention head, a 50-step memory, and a 64-unit feed-forward layer.
- Its per-step quantity called an NLL penalty is the unnormalized Gaussian energy `0.5 * (wind / sigma)^2`.
- Failure is the first strict CartPole position or pole-angle threshold violation after a transition. Horizon truncation is not failure.
- The source MCMC samples a direct, fixed-length wind plan with a bounded Gaussian-energy target conditioned on replay failure. Every rejected proposal is stored as a repeated current chain row.
- Latent extraction replays row zero and every accepted-successor MCMC row, calls the adversary only to update causal context, captures the 64-D model `_features` immediately before each step, and selects the feature preceding the failure-causing transition. The stored MCMC wind, not the adversary's newly proposed action, is applied.
- Primary analysis treats row zero and every accepted successor once, standardizes the final latents, selects K-means `k` from 2 through 10 by silhouette score, and creates PCA, t-SNE, and UMAP visualizations linked to wind traces. Rejected dwell rows are retained in the raw chain but are not separate rows in this primary analysis.

The source also contains reproducibility gaps that the port must fix rather than reproduce:

- ordinary evaluation traces save winds and a few scalars, not complete state/action/seed/checkpoint provenance;
- training and random-baseline clipping induce boundary mass, while MCMC uses a continuous bounded-density target, so the probability model is not literally identical across stages;
- committed checkpoints declare Python 3.8-era RLlib `2.10.0` cloudpickle format, while `requirements.txt` pins Ray `2.40.0`; arbitrary-code pickle loading is also unsuitable for portable artifacts;
- the committed file named `500000iteration_episode6_mcmc.npz` records `iterations=100000`; manifests, not filenames, must be authoritative; and
- source `failure_step` is a one-based transition count, whereas this port's canonical `failure_timestep` is a zero-based action index.

These are documented adaptations, not claims that the source already provided stronger guarantees.

### 2.2 GPUDrive pin selected before implementation

| Component | Immutable revision |
|---|---|
| GPUDrive repository | [`aa48a431ed127a37610cc2176db30ec73d0c55df`](https://github.com/Emerge-Lab/gpudrive/tree/aa48a431ed127a37610cc2176db30ec73d0c55df) |
| Commit time | `2025-12-01T15:15:03Z` |
| Commit subject | `Repaired a bug causing wrong agent ids in abs_self_obs (#530)` |
| Python package version at the commit | `0.4.0` |
| `external/madrona` gitlink | `4bda33465340fabc2e61fb27f95aa04795a15466` |
| `external/json` gitlink | `0457de21cffb298c22b629e538036bfeb96130b7` |
| Upstream `uv.lock` SHA-256 | `d4f6fe3df752ae9f5cf4fd3e8c510870273fefc4e3c0f3df401f16a62ef7a04d` |

This exact GPUDrive commit and both recursive submodule revisions are normative. The reference runtime will be Linux x86-64, Python 3.11, CUDA 12.4, and a digest-pinned NVIDIA container. The native extension and Madrona kernel cache are build artifacts and receive their own hashes.

### 2.3 Non-scientific one-scene fixture

Milestone A will use the scene already present in the pinned GPUDrive tree:

| Field | Value |
|---|---|
| Relative path | `data/processed/examples/tfrecord-00000-of-01000_325.json` |
| `scenario_id` | `ef3a8f65142f41ac` |
| Stable SDC object ID | `271` |
| SHA-256 | `69bd2b9ae49d43745651262abf3956309e9c0092ca24aff72e0f9abb32f9b948` |

This is only an installation/replay fixture. It is not the experiment corpus and does not settle R10.

## 3. Scope

### In scope

- Reproducible installation of the pinned simulator and recursive dependencies.
- A victim interface that supports a pinned pretrained PPO or repository-owned PPO training.
- Frozen deterministic victim inference and nominal evaluation.
- One approved bounded sequential intervention in physical units.
- A causal Transformer actor/critic trained with PPO and an explicit prior penalty.
- Schema-versioned trajectory, checkpoint, baseline, MCMC, latent, and analysis artifacts.
- Source-matched constant, stochastic, and failure-conditioned comparisons adapted to driving.
- Replay certification, latent capture, clustering, and trace-linked visualization.
- Unit tests that do not require full training and a tiny documented end-to-end smoke path.

### Out of scope for the first port

- Updating the victim during adversary training.
- Mixing observation attacks, actuation attacks, and adversarial traffic actors in one result.
- Editing scene files or maps as an attack.
- Treating GPUDrive `done` as synonymous with safety failure.
- Claims of real-world safety or physical realizability beyond the selected simulator model.
- Large training or hyperparameter sweeps before replay and MCMC invariants pass.

## 4. Normative experimental contracts

These contracts become executable only after their scientific parameters are approved.

### 4.1 Frozen deterministic victim

- One immutable victim checkpoint, preprocessing definition, action table, and environment configuration are used throughout an experiment family.
- The victim is in evaluation mode, its parameters are frozen, inference uses `torch.inference_mode()`, and adversary optimization cannot update it.
- Deterministic discrete inference is argmax with a documented stable tie rule.
- Missing or incompatible checkpoints fail closed. The CartPole source's random-victim fallback will not be ported.
- A zero-disturbance nominal rollout is saved before a scene enters adversarial evaluation.

### 4.2 Sequential intervention and probability accounting

- The adversary emits one bounded disturbance vector at each approved decision step.
- Bounds are enforced by construction and asserted immediately before simulator input.
- The trace separately records the nominal victim command, raw policy output, requested bounded disturbance, effective disturbance after actuator saturation, final command, and saturation mask.
- PPO policy log probability and nominal-prior log probability are distinct stored quantities.
- The prior support, measure at boundaries, scale, temporal factorization, exact NLL/energy formula, aggregation, and coefficient are versioned research fields.
- The source's clipped-Gaussian training/baseline behavior and continuous bounded MCMC density are not silently described as one coherent distribution. R6 must choose exact source compatibility or a coherent adaptation.

### 4.3 Failure and indexing

The port uses this event order:

```text
                         -> victim nominal action --\
s_t and permitted past --                            -> applied command
                         -> adversary latent/d_t ----/   -> simulator step
                                                          -> s_(t+1) evidence
```

The source-matched adversary does not observe the current victim action; both decisions depend on the same pre-action state and permitted past. Their implementation may be scheduled in either order, but that must not change the adversary information set.

If failure first becomes true in `s_(t+1)`, then:

- `failure_timestep = t` is the zero-based failure-causing action index;
- `failure_step_count = t + 1` is stored for compatibility with the CartPole artifacts;
- the final pre-failure latent is the representation captured before applying action `t`; and
- states have length `T+1` while transition fields have length `T`.

Raw simultaneous event flags are retained. Goal, horizon, invalid actor, simulator error, and approved safety failure have distinct termination reasons.

### 4.4 Replay certification

A trajectory is replay-verified only if an isolated one-world reconstruction from pinned inputs reproduces:

- exact scene bytes, scenario ID, victim stable ID, and configuration;
- the same first-failure flag set and termination reason;
- the same zero-based failure-causing action index; and
- action/state values within declared tolerances on the certified runtime.

"Fails eventually" is insufficient. Same-process reset and fresh-process replay are separate checks. Cross-device bitwise equality is not assumed.

### 4.5 Failure-conditioned MCMC

- Chain row zero is a replay-verified failure under the exact chain fingerprints.
- The source-matched state is a direct fixed-horizon disturbance plan.
- Out-of-support, invalid-replay, and nonfailure proposals have zero acceptance probability.
- The current state is asserted to be a failure before every row is serialized.
- Every iteration appends a current-state row. A rejection appends the unchanged prior current state, preserving dwell time.
- Proposal diagnostics and current-state evidence are stored separately.
- The target and vector-valued proposal adaptation remain subject to R6 and R12.

### 4.6 Transformer latent

The source-backed default is a **teacher-forced context latent**:

1. build the causal adversary input and memory for `s_t`;
2. run the adversary model;
3. capture an explicit 64-D `pre_actor_features` tensor before action application;
4. ignore the action proposed by that call during direct-plan MCMC replay;
5. apply the stored MCMC disturbance; and
6. if that transition first fails, select the feature captured at step `t`.

The port will expose a stable named feature instead of reading RLlib's private `_features` attribute. It must not label this representation as the action-producing latent for an externally overridden disturbance.

## 5. Artifact and fingerprint contract

Every checkpoint, trajectory, baseline result, replay certificate, MCMC chain, latent table, cluster assignment, plot-data bundle, and figure has a validated manifest. A plot without a manifest and source table is non-normative.

### 5.1 Required common fields

- artifact schema/name/version, artifact ID, parent IDs, creation time, and producing command;
- methodology repository and exact commit;
- this repository commit, dirty flag, and patch/tree hash when dirty;
- GPUDrive commit, recursive submodule SHAs, native-extension hash, and kernel-cache identity;
- complete resolved configuration and canonical SHA-256;
- Python lock, container digest, OS, Python, PyTorch, CUDA, driver, GPU, and determinism settings;
- dataset repository/revision or local manifest hash;
- scene path, split, `scenario_id`, byte hash, ordered loader inputs, and reset protocol;
- seed ledger for scene selection, Python, NumPy, Torch CPU/CUDA, victim, adversary, baselines, and MCMC as applicable;
- victim selector, simulator slot, stable object ID, and role assertions;
- victim/adversary checkpoint byte hashes plus architecture, preprocessing, normalization, action-space, and configuration hashes;
- intervention, actuator, prior, failure, trace-schema, and replay-tolerance specification hashes.

Non-applicable fields are explicit `null` values with a reason.

### 5.2 Hashing and storage

- Canonical manifests use sorted-key UTF-8 JSON, a fixed finite-number representation, and SHA-256.
- Files and scenes are hashed over raw bytes. Directory checkpoints use a sorted relative-path/content hash manifest.
- Array payloads use typed non-pickled storage; loading an artifact never executes arbitrary code.
- A dataset fingerprint is the hash of an ordered manifest containing each relative path, size, SHA-256, scenario ID, split, and preprocessing provenance.
- Schema upgrades write new artifacts through explicit converters; no artifact is mutated in place.

## 6. Required validation

| User requirement | Acceptance test |
|---|---|
| Saved failure replays identically | Fresh process matches scene/victim, first failure flags, reason, and failure action index; numeric fields satisfy declared tolerances. |
| Every artifact stores fingerprints | Common-schema validator rejects a missing source, simulator, scene/config, or checkpoint fingerprint. |
| MCMC never enters nonfailure | Initializer and every serialized current row carry replay-verified failure evidence; corrupted rows fail validation. |
| Rejections repeat chain states | Forced nonfailure and Metropolis rejections produce a new row with the unchanged current state/hash. |
| Latent is immediately pre-failure | Instrumented toy ordering is `encode -> capture -> action/override -> step -> first failure`; off-by-one fixtures fail. |
| Unit tests need no full training | Default suite uses toy simulator/victim data and one tiny in-memory PPO update. |
| Tiny end-to-end smoke is documented | One-scene commands generate and verify all artifact types through a plot-data bundle. |

## 7. Planned repository boundaries

```text
configs/                 approved research specs and non-claim smoke configs
containers/              digest-pinned reference GPU build
docs/                    spec, mapping, plan, decisions, smoke guide
src/gpudrive_adversary/
  artifacts/             schemas, fingerprints, manifests, migrations
  envs/                  GPUDrive adapter, intervention, failure evaluator
  victims/               train/load/freeze/evaluate
  adversary/             causal Transformer and PPO
  baselines/             source-matched simple disturbances
  replay/                reconstruction and certification
  mcmc/                  conditional target, proposals, chain validation
  analysis/              distinct states, latent capture, clusters, plots
tests/                   pure unit, native integration, GPU, and smoke tests
third_party/gpudrive/    exact recursive source pin or equivalent vendoring
```

Any GPUDrive compatibility patch is explicit, fingerprinted, tested, and listed in manifests.

## 8. Research decisions and gates

`docs/CARTPOLE_TO_GPUDRIVE_MAP.md` gives the evidence and alternatives. Statuses below distinguish questions resolved by source inspection from GPUDrive definitions that still require approval.

| ID | Decision | Status |
|---|---|---|
| R0 | CartPole source URL and immutable revision | **Resolved:** commit `315b14a...`. |
| R1 | Canonical GPUDrive victim checkpoint: pinned pretrained or repository-trained PPO | **Blocking before B.** |
| R2 | Victim actor/cohort and non-victim behavior | **Blocking before B.** |
| R3 | Intervention locus: victim actuation, observation, or another actor | **Blocking before C.** Source analogue favors victim actuation. |
| R4 | Placement, dimensions, action order, and decision period | **Blocking before C.** Proposed 2-D decoded physical residual. |
| R5 | Numerical disturbance bounds and final actuator envelope | **Blocking before C.** |
| R6 | Nominal prior/measure, NLL or energy, temporal model, aggregation, and coefficient | **Blocking before C/F.** Source inconsistency requires an explicit choice. |
| R7 | Victim failure predicate and simultaneous-event policy | **Blocking before C.** |
| R8 | Nominal scene eligibility | **Blocking before B/C.** |
| R9 | Collision response (`ignore`, `stop`, or `remove`) | **Blocking before C.** |
| R10 | Dataset/split, scene selector, warm-up/start frame, and seeds | **Blocking before B.** |
| R11 | MCMC state and conditional target family | **Source-resolved state:** direct fixed-horizon plan. Its fixed-plan target density is conditioned on replay failure and declared separately; R6 decides whether it is identical to the PPO/baseline prior. |
| R12 | Vector-valued MCMC proposal, bounds, adaptation, and retained-chain protocol | **Blocking before F.** Source default perturbs every coordinate with symmetric Gaussian noise and rejects out-of-box proposals. |
| R13 | Distinct state and weighting semantics | **Source-resolved default:** row zero plus each accepted successor once; retain chain state IDs and dwell weights. |
| R14 | Latent and forced-replay semantics | **Source-resolved default:** 64-D pre-action teacher-forced context feature at the failure-causing step. |
| R15 | Primary clustering/visualization protocol | **Source-resolved default:** unweighted distinct states, StandardScaler, K-means/silhouette, PCA/t-SNE/UMAP, trace plots. |
| R16 | GPUDrive adversary information set, tokenization, memory, masks, and frequency | **Blocking before C.** Source has a fixed 6-D vector and 50-step memory; driving requires an explicit adaptation. |
| R17 | Failure reward/shaping and scale relative to the prior penalty | **Blocking before C.** Source uses `+1000` on failure and a terminal survival-margin penalty; those scales do not transfer automatically. |

After the user explicitly approves these planning documents, Milestone A may begin as a pinned installation and simulator-semantics audit without first settling the attack definitions. No research wrapper begins until the user approves R1-R10 and R16-R17 as applicable. Milestone F additionally requires R11-R12 confirmation; Milestone G follows the source-resolved R13-R15 protocol unless a documented scientific deviation is approved.

## 9. Definition of done

The repository is complete only when Milestones A-G pass their exit criteria; every artifact has immutable provenance; at least one learned-adversary failure passes fresh-process replay; all stored MCMC current states are certified failures with rejection repeats intact; every analyzed latent is proven to precede the failure-causing action; the source-matched analysis can be regenerated from artifacts; and the documented tiny pipeline runs without a full training job.

Any deliberate departure from the pinned CartPole semantics must be named in the decision record and mapping, with the scientific reason and compatibility consequences.
