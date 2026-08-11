# Implementation plan

Status: **source audit complete; implementation not started**

Methodology source: `soreng25/Adversarial-Transformer-PPO-agent-in-Cartpole@315b14a90b252ba416eb329e8003d5926806ba67`

GPUDrive: `Emerge-Lab/gpudrive@aa48a431ed127a37610cc2176db30ec73d0c55df`

## 1. Entry gate and dependency order

The requested CartPole source is now pinned and inspected, so R0 is closed. This plan still does not authorize implementation by itself. After the user approves these documents, they may authorize Milestone A alone before settling the intervention/failure choices, because A is restricted to installation and simulator-semantics evidence. The research wrapper and Milestones B-G remain gated as described below.

Milestone A is limited to reproducible installation and read-only simulator-semantics validation. It may not introduce the attack or failure objective. Before Milestone B, approve R1, R2, R8, and R10. Before Milestones C-E, also approve R3-R7, R9, R16, and R17. Before F, confirm R6, R11, and R12. G uses the source-backed R13-R15 defaults unless a deliberate deviation is recorded.

```text
explicit approval of planning documents
    -> A. pinned GPUDrive install + one-scene smoke
A + approved B decisions
    -> B. frozen deterministic victim
B + approved C-E decisions
    -> C. adversary environment + Transformer-PPO
    -> D. complete trajectory schema + replay certificate
    -> E. source-matched baselines
E + approved F decisions
    -> F. failure-conditioned Metropolis MCMC
F + approved G decisions
    -> G. latent extraction + clustering/visualization
```

Each milestone ends in reviewable artifacts and tests. A later milestone cannot compensate for a failed earlier exit criterion.

## 2. Source fidelity policy

The port preserves these inspected CartPole choices unless a GPUDrive-specific decision overrides them:

- one frozen deterministic PPO victim;
- an additive bounded physical-control residual selected from the same pre-action state as the victim;
- a causal Transformer adversary with a 50-step memory and explicit previous-action/disturbance context;
- a Gaussian-likelihood disturbance cost and failure-seeking reward;
- direct fixed-horizon disturbance-plan MCMC conditioned on deterministic replay failure;
- raw rejection rows retained as repeated chain states;
- row zero plus accepted successors as the source "distinct" jump-chain population;
- a 64-D teacher-forced context feature captured immediately before the failure transition; and
- standardized full-space K-means/silhouette analysis with PCA, t-SNE, UMAP, and trace plots.

The following source quirks are not copied silently:

- incomplete wind-only traces;
- missing seed/config/checkpoint/code fingerprints;
- cloudpickle checkpoints and Ray 2.10 versus 2.40 incompatibility;
- clipped-Gaussian versus continuous-bounded-prior inconsistency;
- length thresholds substituted for actual baseline failure;
- positive-only constant sweeps;
- CartPole-specific sine frequency;
- filename/metadata disagreement in a committed MCMC artifact;
- an initializer that need only "fail eventually" rather than match the recorded event time; and
- private `_features` access without a stable feature contract.

Every change from source behavior is marked either an approved scientific adaptation or an ordinary engineering strengthening.

## 3. Ordinary software decisions

These choices do not settle the research questions:

- Python package: `gpudrive_adversary` in a `src/` layout.
- CLI: a `gda` entry point with subcommands and resolved configuration output.
- Configuration: versioned YAML input parsed into typed objects, then canonical JSON for hashing.
- Artifacts: one directory per artifact, `manifest.json`, typed non-pickled arrays/tensors, and atomic publication.
- Fingerprints: SHA-256 over raw files and canonical sorted manifests.
- Tests: `pytest` markers `unit`, `gpudrive`, `gpu`, `smoke`, and `slow`.
- CI: default pure unit/schema jobs; separate native CPU/GPU integration jobs.
- Figures: always save input tables/arrays, resolved plotting parameters, assignments, and manifests with PNG/PDF outputs.
- Generated datasets, checkpoints, chains, and results are not committed by default; small license-compatible fixtures may be.

Planned layout:

```text
configs/
  research/
  smoke/
containers/
docs/
  PROJECT_SPEC.md
  CARTPOLE_TO_GPUDRIVE_MAP.md
  IMPLEMENTATION_PLAN.md
  RESEARCH_DECISIONS.md
  SMOKE_PIPELINE.md
src/gpudrive_adversary/
  artifacts/
  config/
  envs/
  victims/
  adversary/
  baselines/
  replay/
  mcmc/
  analysis/
tests/
  unit/
  integration/
  smoke/
third_party/gpudrive/
```

## 4. Cross-cutting implementation rules

### 4.1 One manifest system

All artifact-producing commands use one manifest builder and validator. It records source/port commits, dirty-tree patch identity, GPUDrive/submodules/native build, environment/config, dataset/scene, checkpoint, intervention, failure, prior, schema, runtime/hardware, and seed fingerprints. Moving revisions and missing mandatory fields fail before artifact publication.

Parent IDs form a provenance DAG. A verifier can validate one artifact or recursively verify a results tree.

### 4.2 One transition clock

A shared transition record owns:

```text
pre-action state s_t
victim output from s_t
adversary input/memory and pre-actor latent from s_t
requested/effective disturbance
applied command
post-step state/evidence s_(t+1)
failure_timestep t or no failure
```

The adversary input cannot consume the current victim action under the source-matched threat model, even if implementation scheduling computes the victim first. Training, baselines, replay, MCMC, and latent extraction share this clock.

### 4.3 One intervention adapter

Only one module decodes victim actions and composes disturbances with physical commands. It uses named acceleration/steering/head fields rather than GPUDrive's ambiguous continuous tuple helper. It reports nominal, requested, effective, applied, and saturation values. Learned attacks, baselines, replay, and MCMC all call it.

### 4.4 One failure evaluator

The approved failure specification is a versioned pure function over victim post-step evidence. It returns a lossless flag set and derived termination. GPUDrive `done` cannot silently substitute for it. Reward, trace stopping, baseline scoring, replay, and MCMC use the same evaluator.

### 4.5 One canonical disturbance dtype

Generation, simulator application, storage, hashing, and MCMC scoring use one declared dtype/canonicalization boundary. This prevents the CartPole source issue where live float64 chain scores are paired with float32 saved/applied values that do not exactly reproduce the logged score.

## 5. Milestone A - reproducible GPUDrive installation and one-scene smoke

### Goal

Build and identify the exact simulator, step one immutable scene, establish action/info/reset semantics, and measure replay determinism before adding RL code.

### Work

1. Add GPUDrive at `aa48a431ed127a37610cc2176db30ec73d0c55df` recursively with:
   - `external/madrona@4bda33465340fabc2e61fb27f95aa04795a15466`;
   - `external/json@0457de21cffb298c22b629e538036bfeb96130b7`.
2. Create a Linux x86-64 reference container using Python 3.11 and CUDA 12.4. Resolve `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` to an image digest and record compiler, CMake, driver compatibility, GPU, extension hash, and Madrona kernel-cache identity.
3. Install using the pinned upstream `uv.lock` and assert its expected hash.
4. Add `gda doctor` for source/submodule/lock/native/runtime checks.
5. Add a one-world smoke for scene `data/processed/examples/tfrecord-00000-of-01000_325.json`, scenario `ef3a8f65142f41ac`, stable SDC ID 271, scene SHA-256 `69bd2b9ae49d43745651262abf3956309e9c0092ca24aff72e0f9abb32f9b948`.
6. Step several safe named physical commands and read observations, state, reward, done, and raw info.
7. Prove raw action order `[acceleration, steering, head_angle]` with a focused dynamics test.
8. Audit vehicle, non-vehicle, road-contact, goal, and horizon signals, including immediate collision-flag lifetime under `collision_behavior=ignore`.
9. Compare fixed-action repeated resets and fresh processes on CPU and CUDA. Store event/timestep equality and numeric drift; do not claim cross-device bitwise identity.

### Deliverables and exit criteria

- Digest-pinned build instructions, runtime doctor, one-scene smoke artifact, and simulator semantics report.
- Runtime commit, submodules, lock, scene bytes, scenario ID, and SDC ID match expected pins.
- Import/reset/step succeeds on the reference GPU runtime.
- Action order and raw info semantics are test-backed.
- Same-process and fresh-process traces meet a measured replay tolerance.
- No victim/adversary training and no provisional attack/failure wrapper are introduced.

## 6. Milestone B - deterministic victim evaluation

### Goal

Create one immutable GPUDrive victim with reproducible preprocessing, actor selection, action decoding, and nominal rollouts.

### Work

1. Define a `VictimPolicy` protocol returning native deterministic output plus canonical physical command.
2. Implement the approved canonical path:
   - load the pinned published PPO snapshot; or
   - train a repository-owned PPO on a pinned dataset/cohort.
3. Preserve a separately versioned alternate train/load path without mixing checkpoint families.
4. Build a safe checkpoint envelope containing weights, architecture, preprocessing, normalization, action grid/order, environment config, dataset, and source/runtime provenance.
5. Enforce `eval()`, `requires_grad_(False)`, and inference mode. Fail if checkpoint/config or victim identity mismatches.
6. Apply R2's stable actor assertions after every reset.
7. Run zero-disturbance evaluation over the R10 cohort, record R8 eligibility outcomes/exclusions, and replay-certify nominal traces.

### Tests and exit criteria

- Deterministic argmax and tie rule are unit tested.
- Mismatched observation/action tables fail closed.
- Victim parameters remain byte-identical and gradient-free through a mock adversary update.
- Repeated nominal action indices, events, and termination timesteps meet the Milestone A contract.
- Every output contains the common fingerprint bundle.

## 7. Milestone C - adversary environment and Transformer-PPO

### Goal

Implement the approved driving threat model and train a bounded causal adversary without changing the victim.

### Source-informed starting point

The compatibility architecture has one causal Transformer unit, 64-D pre-actor representation, one attention head, head dimension 32, 50-step memory, and a 64-unit ReLU feed-forward layer. Source PPO uses gamma 0.99, learning rate `3e-4`, and batch 4000. These values are versioned starting points, not justification for copying CartPole reward magnitudes or token fields.

### Work

1. Encode R3-R7, R9, R16, and R17 in immutable typed specs.
2. Implement the shared named intervention adapter, including requested/effective disturbance and saturation.
3. Implement the approved causal driving token/input. Source continuity requires permitted current driving state plus previous victim action/command and previous applied disturbance, with no current nominal action exposed to the adversary. Actor/map masks and padding are explicit.
4. Implement a 50-step causal Transformer actor/critic with stable `pre_actor_features` output before policy/value heads. Any architecture deviation is recorded.
5. Implement the R6 bounded policy/prior accounting. Store PPO policy log probability separately from nominal-prior density/energy.
6. Implement R17 reward components separately: failure term, likelihood penalty, optional safety-risk/terminal shaping, and total. Do not copy source `failure_bonus=1000` or CartPole boundary margin without calibration/approval.
7. Implement PPO rollout/update with causal masks, termination masks, fixed victim inference, and metrics for failure subtype/time, likelihood, magnitude, and saturation.
8. Save resumable and inference-only adversary checkpoints with parent victim and full research provenance.
9. Provide a toy PPO update and a few-update GPUDrive plumbing smoke, both labeled non-claim.

### Tests and exit criteria

- Every disturbance/final command satisfies approved support/envelope at boundary cases.
- Current victim action is provably absent from the source-matched adversary input.
- Policy and prior log probabilities cannot be interchanged.
- Hand-computed density/energy cases and any normalization tests pass.
- Future tokens cannot affect an earlier action or latent.
- Victim bytes/gradients do not change.
- Tiny rollout/update completes without a full training run.

## 8. Milestone D - complete trajectory schema and replay certificate

### Goal

Make every saved outcome independently attributable and certify at least one bounded failure at the same event/timestep.

### Work

1. Implement `TrajectoryV1` with `T+1` states and `T` transitions exactly as mapped.
2. Store scene-wide identities/masks/poses/velocities/yaws/validity; exact victim input; native/nominal/disturbed/applied controls; prior/PPO terms; reward components; raw info/done; termination reason; zero-based failure action; and one-based compatibility count.
3. Implement atomic write/read, typed payload validation, hashes, and non-mutating schema migrations.
4. Reconstruct in a fresh one-world process from exact scene/config/checkpoint/disturbance/seed inputs.
5. Require exact scene/victim/failure flags/reason/timestep plus declared numeric comparison.
6. Save a failure from a trained adversary or explicitly non-claim smoke checkpoint and issue a replay certificate.
7. Negative controls alter scene hash, victim bytes, action order, one disturbance, failure spec, and config; all must fail closed with useful diagnostics.

### Tests and exit criteria

- A saved failure replays to the same failure flag set and causing action index.
- Missing/moving fingerprints are rejected.
- Off-by-one failure/latent fixtures fail.
- Unit schema/replay tests use toy traces and require no training.

## 9. Milestone E - source-matched baselines

### Goal

Compare learned failures against simple sequences under identical support, scenes, victim, event, horizon, and replay rules.

### Work

1. Implement zero disturbance.
2. Implement signed per-axis constants and an approved vector/corner sweep. Unlike the source's positive-only sweep, cover directions justified by the symmetric/vector budget.
3. Implement the R6 stochastic prior sequence. If source-compatible clipping is selected, name it `clipped_gaussian` and model/report boundary mass accurately.
4. Optionally implement episode-constant prior draws if approved as an additional simple baseline.
5. Add a periodic/spectral baseline only after an approved driving-domain frequency rule; never hardcode CartPole's `0.23` cycles/step.
6. Route all sequences through the shared adapter, trace writer, failure evaluator, and replay verifier.
7. Pair scene/victim/reset inputs across methods. Report true failure rate/subtype/timestep, likelihood/energy, requested/effective magnitude, and saturation, not an episode-length proxy.

### Tests and exit criteria

- Constant sequences stay constant over their realized prefix.
- Stochastic sequences reproduce from stored values and RNG identity.
- No baseline exceeds bounds or uses different failure logic.
- Selected failures replay-certify and all artifacts pass fingerprint validation.

## 10. Milestone F - failure-conditioned Metropolis MCMC

### Goal

Sample the approved fixed-scene failure-conditioned target while making a nonfailure current chain state structurally impossible.

### Source-matched statistical contract

The default state is a direct physical disturbance plan `d[0:H]`. The full fixed-dimensional target is the separately declared R6 fixed-plan density multiplied by an indicator that deterministic replay satisfies R7. R6 decides whether this density is identical to the PPO penalty and stochastic-baseline prior; the CartPole artifact used a different MCMC scale. The source scores even the unused post-failure tail, so R6/R12 must explicitly retain this choice or approve a statistically valid alternative.

The source default independently perturbs every plan coordinate with symmetric Gaussian noise and rejects out-of-box proposals. Single-coordinate and contiguous-block variants exist. GPUDrive's vector/high-dimensional setting makes kernel choice scientific rather than a hidden performance tweak.

### Work

1. Load a learned-adversary failure and require its fresh replay certificate, including exact failure flags/timestep, under chain fingerprints.
2. Canonicalize the plan dtype before hashing, scoring, storage, and application.
3. Implement target density and symmetric/asymmetric proposal terms independently with hand-checkable tests.
4. Replay every proposal. Out-of-support, invalid, or nonfailure proposals have acceptance zero.
5. For every iteration store proposal plan/hash/outcome, previous/current plan/hash, acceptance/uniform draw, target/proposal terms, failure evidence/timestep, and fingerprints.
6. On any rejection append the unchanged current plan as the new raw chain row.
7. Cache replay only under keys containing every replay-determining fingerprint.
8. Compute diagnostics without deleting repeats: acceptance, nonfailure/OOB proposal rates, autocorrelation/effective sample size where applicable, dwell weights, failure subtype/time, and proposal-scale history.

### Tests and exit criteria

- A toy finite-state target matches its enumerable stationary distribution.
- Initialization fails if event flags or failure timestep differ from the source certificate.
- Forced nonfailure, out-of-box, and Metropolis rejections append repeated current states.
- Every serialized current row validates as failure; injected nonfailure rows are rejected.
- Raw row count equals proposals plus one and no rejection is missing.

## 11. Milestone G - latent extraction and analysis

### Goal

Replay source-defined distinct jump-chain failures, capture the exact pre-failure context representation, and connect latent clusters to driving traces without losing raw-chain multiplicity.

### Work

1. Preserve the raw chain and derive row zero plus every accepted successor, exactly matching the source jump-chain rule.
2. Map every raw row to a selected-state ID and store dwell weight. Content-hash selected plans and record accepted revisits/canonical collisions rather than silently calling them globally unique.
3. Replay every selected failure under exact fingerprints and require exact event/timestep.
4. At each pre-action state:
   - build the approved causal context and 50-step memory;
   - compute the current token through the Transformer;
   - copy the stable 64-D `pre_actor_features` vector;
   - ignore the adversary's predicted disturbance for direct-plan replay;
   - apply the stored MCMC disturbance;
   - step and evaluate failure.
5. Select the feature at `failure_timestep` and label it `teacher_forced_context_latent`.
6. Build the primary unweighted selected-state matrix while retaining dwell weights for sensitivity analysis.
7. Apply the source-matched analysis:
   - StandardScaler on 64-D final latents;
   - K-means `k=2..10`, `n_init=20`, seed 42;
   - silhouette selection on at most 5,000 seeded rows, smaller tied `k` first;
   - label ordering by increasing mean failure step;
   - PCA, t-SNE, and UMAP visualization only.
8. Link clusters to realized disturbance/control/saturation traces, victim/all-agent state, failure evidence, top-down paths, and representative/medoid IDs.
9. Save all arrays, resolved versions/parameters, assignments, plot data, figures, and manifests.

### Tests and exit criteria

- Instrumented ordering proves `encode -> capture -> forced action -> step -> failure`.
- Previous-step and post-failure latent fixtures fail validation.
- Rejection/dwell derivation never modifies or drops raw chain rows.
- Teacher-forced context latents cannot be mislabeled as action-producing.
- Re-running analysis from artifacts reproduces assignments and plot data under the declared library/runtime contract.

## 12. Validation and test strategy without full training

The default suite is useful on a development machine with no GPUDrive build and no long optimization job.

### Pure unit tests

- canonical JSON, directory/file hashes, and required manifest fields;
- schema versions, migrations, corruption, and parent provenance;
- action order, decoding, bounds, clipping, and saturation;
- failure bit sets, simultaneous events, termination, and action indexing;
- prior density/energy and PPO-versus-prior likelihood separation;
- causal masks, 50-step memory, and latent hook ordering;
- PPO loss on a tiny in-memory batch;
- constant/stochastic sequence generation;
- MCMC target/proposal ratios, failure support, rejection repeats, and dtype canonicalization;
- jump-chain IDs, dwell weights, accepted revisits, trace prefix masking, and clustering parameters.

### Native integration tests

Marked tests use the pinned GPUDrive build and one tracked scene for reset, stable identity, action ordering, post-step info, trace writing, and replay. CPU is diagnostic; the reference CUDA runtime is authoritative for GPU artifacts.

### Tiny smoke, not a research result

One scene, a few environment steps, a very small batch, and one or two PPO updates test data flow/checkpointing. A checked-in or release-hosted replay-verified smoke failure may exercise MCMC/analysis without waiting for convergence. Every smoke artifact has `purpose=smoke_only` and result aggregators exclude it.

## 13. Documented tiny end-to-end pipeline

Milestone A will turn this outline into tested commands in `docs/SMOKE_PIPELINE.md`:

```bash
gda doctor --strict
gda scene-smoke --config configs/smoke/scene.yaml
gda victim evaluate --config configs/smoke/victim.yaml
gda adversary train --config configs/smoke/adversary.yaml
gda replay verify --artifact artifacts/smoke/failure
gda baselines run --config configs/smoke/baselines.yaml
gda mcmc run --config configs/smoke/mcmc.yaml --init artifacts/smoke/failure
gda mcmc validate --artifact artifacts/smoke/chain
gda latents extract --chain artifacts/smoke/chain
gda analyze cluster --latents artifacts/smoke/latents
gda artifacts verify-tree artifacts/smoke
```

The smoke ends with one complete runtime/scene/victim manifest, a frozen nominal trace, proof of a tiny adversary update without victim changes, one replay-verified failure, baseline traces, a short all-failure chain with at least one repeated rejection row, correctly indexed pre-failure context latents, and a manifested cluster/trace plot-data bundle. The smoke fixture must contain at least three selected jump-chain states and two nondegenerate clusters; it uses the explicit smoke-only override `min_k=2, max_k=2`. Reported research keeps the source-matched `k=2..10` selection and must have enough selected states for that range.

## 14. User decisions needed before research implementation

The minimum decision record is:

1. R1/R2: canonical victim checkpoint, one SDC or another actor selector, and logged versus reactive background.
2. R3/R4: approve victim actuation residual and its 2-D `[delta_accel, delta_steer]` placement/decision period, or choose another threat model.
3. R5: numeric residual bounds and final actuator envelope, or an approved calibration protocol.
4. R6: source-compatible mixed probability mechanics or one coherent bounded prior; dimension scales, penalty formula/coefficient, and post-failure-tail accounting.
5. R7/R9: exact victim failure bit set and collision behavior.
6. R8: whether attacks are conditioned on nominal safety-clean goal success.
7. R10: experiment dataset/split/cohort, preprocessing revision, and warm-up/start frame.
8. R16: exact adversary driving fields/tokens/masks, previous control/disturbance history, context length, and decision frequency.
9. R17: failure bonus/shaping and scale relative to the prior penalty.
10. R12: source-default all-coordinate MCMC or a block/single-coordinate adaptation, scale/adaptation, horizon, and retained-chain settings.

R11, R13, R14, and R15 now have source-backed defaults. They need no reinvention, but any requested deviation will be recorded as a scientific choice. Until the blocking answers above are approved, the correct state is planning complete and implementation not started.

## 15. Milestone review checklist

At every review:

1. show commands, artifact IDs, and exact pins;
2. record repository dirty state and patch/tree fingerprints;
3. run milestone unit tests and applicable native/GPU checks;
4. validate every produced artifact and provenance edge;
5. list scientific and engineering deviations from the pinned source;
6. update the decision record and map if evidence changes semantics; and
7. do not enter the next milestone until exit criteria pass.
