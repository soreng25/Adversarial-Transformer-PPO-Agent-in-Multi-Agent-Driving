# CartPole-to-GPUDrive methodology map

Status: **source-audited mapping; GPUDrive scientific definitions await approval**

Methodology source: `soreng25/Adversarial-Transformer-PPO-agent-in-Cartpole@315b14a90b252ba416eb329e8003d5926806ba67`

GPUDrive target: `Emerge-Lab/gpudrive@aa48a431ed127a37610cc2176db30ec73d0c55df`

## 1. Interpretation rule

The CartPole commit above was inspected directly. This document distinguishes:

- **source behavior**: what the pinned code actually implements;
- **preserved methodology**: the scientific role or invariant the port must retain;
- **GPUDrive adaptation**: an ordinary change forced by the simulator/domain; and
- **approval-required definition**: a scientifically meaningful choice that the code cannot settle for driving.

The destination repository's bootstrap commit `597350b07b88de0281bf3c807741955371a8fc94` contained only a README. It is not evidence for the CartPole method. Moving branch names and artifact filenames are not versions.

## 2. Stage-by-stage source audit and transfer

| Stage | Pinned CartPole behavior | Invariant to preserve | GPUDrive transfer | Gate |
|---|---|---|---|---|
| 1. Victim PPO | RLlib PPO; feed-forward ReLU MLP `[256,256,128]`; full 4-D or 100-D frame-stacked stateless input; deterministic evaluation with `explore=False`. | A victim is trained or loaded before attacks and remains a fixed experimental object. | Load a pinned GPUDrive PPO or train one from a pinned dataset; bind observation preprocessing and the action table to its checkpoint. | R1, R2, R10 |
| 2. Freeze/evaluate | The restored victim algorithm is never trained by the adversary, although parameters are not explicitly frozen. A missing victim can silently become random inside the environment. | No victim sampling or learning during attack/evaluation. | `eval()`, `requires_grad_(False)`, inference mode, deterministic argmax, gradient/byte checks, and fail-closed loading. | R1, R2, R8 |
| 3. Transformer-PPO adversary | One scalar wind residual is clipped and added to the same-step victim force. Six-value observation; one-unit 64-D RLlib GTrXL with 50-step memory; PPO reward includes Gaussian energy and failure incentives. | A causal sequential adversary applies bounded disturbances while paying a declared nominal-likelihood cost. | The approved smoke uses a 2-D decoded victim acceleration/steering residual, a 2,989-D causal token, exact tanh-Normal accounting, and explicitly non-claim reward scales. | R3-R7, R9, R16, R17 |
| 4. Trajectories | Normal adversary evaluation saves only NaN-padded winds, lengths, failure flags/steps, bound, sigma, and horizon. | Failure traces must be replayable and attributable. | Deliberately strengthen to complete scene-wide state/control/evidence traces plus immutable manifests. | Required schema below |
| 5. Baselines | Positive constant sweep; IID Gaussian samples then clipped; a phase-zero sine sweep at a CartPole-specific frequency. Some reports use episode-length thresholds instead of true failure. | Compare learned attacks with simple deterministic/stochastic sequences under one budget and one event definition. | Use the shared adapter and true approved failure predicate. Do not inherit the threshold or frequency without justification. | R5-R10 |
| 6. Failure-conditioned MCMC | Direct scalar plan of length 350; bounded Gaussian-energy target conditioned on deterministic replay failure; symmetric Gaussian random walk; every rejection repeats the current state. | Initialize inside failure support and never let the current chain leave it. Preserve dwell rows. | Use a fixed `[H,d]` physical-disturbance plan, exact replay, and a vector-valued source-matched or approved proposal. | R6, R7, R11, R12 |
| 7. Replay/latent | Select row zero and each accepted successor; replay its winds; call adversary only for context; copy 64-D `_features` before each forced step; select the feature before the recorded failure transition. | Capture the causal final pre-failure representation, not a post-failure or previous-step vector. | Expose a stable `pre_actor_features` interface and label direct-plan results as teacher-forced context latents. | R13, R14, R16 |
| 8. Cluster/visualize | Unweighted accepted-state/jump-chain rows; StandardScaler; K-means `k=2..10` selected by silhouette; PCA, t-SNE, UMAP, and pre-failure wind traces. | Link failure representations to the trajectories that produced them. | Preserve this as the primary source-matched analysis, while adding driving control/state/event/top-down traces and complete manifests. | R15 |

## 3. Victim-controlled agent

### CartPole definition

There is exactly one controlled cart/pole victim. It chooses `Discrete(2)`: action 0 contributes `-10 N` and action 1 contributes `+10 N`. The adversary never controls a second actor because none exists.

The normal victim receives `[x, x_dot, theta, theta_dot]`. The optional "stateless" victim receives a 50-frame stack of `[x, theta]`, so its feed-forward input is 100-D. The Transformer adversary, not the victim, is the sequential policy in the main attack pipeline.

### Approved GPUDrive definition (R1/R2)

The user selected the pinned published GPUDrive PPO and one valid Waymo self-driving-car (SDC) vehicle as victim. It is identified by `scenario_id` plus stable object ID, never only by tensor slot. On every reset the adapter must assert:

1. exactly one victim is selected;
2. its controlled mask is true;
3. its SDC metadata flag is true;
4. its stable object ID matches the manifest; and
5. it is a valid vehicle at the attack start.

Scenes that fail these assertions are excluded with a recorded reason. For the first port, the proposed non-victim behavior is GPUDrive's logged trajectory playback. This is deterministic but non-reactive and changes the causal interpretation of collisions.

Not-selected alternatives include a track-to-predict vehicle, a preregistered ordinary-vehicle selector, all eligible controlled vehicles as separate samples, or reactive policy-controlled background actors. Any later switch is a new scientific decision. The pinned GPUDrive `womd_tracks_to_predict` selector must not be trusted without a focused fix/test because its current predicate can make non-track agents eligible.

The canonical checkpoint is `daphne-cornelisse/policy_S10_000_02_27`, pinned at revision `1532950cad84dafc6e9d976a2bcc524ee481a1a1`; `model.safetensors` SHA-256: `f3f26475def35f375c6c72d8f8f20b2b091f77175010345dc3fa968a860521b7`. The current fixture binds its sole controlled actor to internal slot 0 and stable object ID 271; logged playback supplies every background actor. R8/R10 still govern which scenes may enter reported research.

## 4. Adversary-controlled variable and dimensions

### CartPole definition

At decision `t`, the adversary observes:

```text
o_adv_t = [x_t, xdot_t, theta_t, thetadot_t,
           previous_victim_action_sign, previous_applied_wind]
```

and emits one scalar `w_t`. It does not observe the current victim action before choosing `w_t`. The victim and adversary act from the same pre-action state, after which the environment combines them:

```text
victim_sign_t in {-1, +1}
force_t = 10 * victim_sign_t + clip(w_t, -max_wind, max_wind)
```

Thus the threat model is an additive physical actuation residual, not an action-index, logit, observation, or second-agent attack.

### Approved GPUDrive smoke definition (R3/R4/R16)

Scientifically different choices are:

1. a residual on the victim's physical control;
2. corruption of victim observations;
3. direct control of another traffic actor; or
4. manipulation of scene/initial conditions.

**Approved smoke source analogue:** a 2-D victim-control residual

```text
d_t = [delta_acceleration_t, delta_steering_t] in R^2
```

with the dependency order

```text
u_nominal_t = decode(argmax victim_logits(s_t))
d_t         = adversary(s_t, permitted history)  # no current u_nominal input
u_requested = [u_nominal.accel + d_t.accel,
               u_nominal.steer + d_t.steer,
               u_nominal.head]
u_applied   = actuator_clip(u_requested)
```

The computation may be scheduled in either order, but the adversary information set must exclude the current victim action if source fidelity is intended. The trace stores native action index, decoded nominal command, requested residual, effective residual, and applied command separately.

The canonical simulator order is `[acceleration, steering, head_angle]`. The pinned GPUDrive continuous-space helper exposes acceleration/steering in a conflicting order from the simulator consumer, so the port will use named fields and a raw-tensor action-order test. The adversary does not modify head angle under this proposal; it preserves the victim's decoded value. For the candidate 91-way classic action table, that value is separately expected and asserted to be zero.

The source acts every CartPole transition (`0.02 s`). GPUDrive's default step is `0.1 s`; the approved smoke adversary acts every simulator step. This is recorded as a driving-domain adaptation, not inferred by equating the physical time units.

## 5. Disturbance bounds, prior, and penalty

### Exact CartPole behavior

- Training CLI defaults: `max_wind=4.0`, `wind_sigma=1.0`.
- The committed adversary/evaluation artifact uses `max_wind=1.0`, `wind_sigma=1.0`.
- Action is converted to float32 and hard-clipped to the symmetric interval.
- Per-step penalty is

```text
energy(w_t) = 0.5 * (w_t / sigma)^2
```

  which is a zero-mean Gaussian negative log density only up to constants.
- The intended temporal prior is IID; temporal structure comes from the Transformer policy.
- The random baseline samples an unbounded Gaussian and clips it, creating point masses at both bounds.
- MCMC instead uses a continuous Gaussian density restricted to the box. With fixed dimension/bounds/sigma, its omitted normalization cancels in Metropolis ratios.

The source therefore does not implement one identical probability measure across training, the stochastic baseline, and MCMC. It also accumulates adversary penalties only until termination, while MCMC scores the full fixed horizon, including a tail never applied after failure.

### Approved C-smoke definition and later R6 gate (R5/R6)

For the proposed 2-D intervention, define

```text
B = [-b_accel, b_accel] x [-b_steer, b_steer]
```

The approved smoke uses `b_accel=0.667`, `b_steer=0.262`, then clips the final physical acceleration to `[-4,4]` and steering to `[-3.142,3.142]`, with head angle preserved. These are explicit smoke values rather than a claim that upstream grid endpoints define a general attack budget.

For C only, the declared prior is an IID zero-mean tanh-squashed Normal with latent standard deviations `[0.5,0.5]`, exact change-of-variables density, and a per-step `0.01 * NLL_excess_from_zero` penalty. The failure bonus is `1`. Those scales are labeled non-claim plumbing values.

Before baselines/MCMC or research-scale training, R6 must still settle cross-stage mechanics explicitly:

- **Source-compatible mechanics:** use the Gaussian energy on bounded actions for PPO, a clipped-Gaussian stochastic baseline, and the continuous bounded-Gaussian density for MCMC; document that these are different measures.
- **Coherent bounded-prior adaptation (recommended):** use the same properly normalized componentwise truncated Gaussian for PPO prior accounting, the stochastic baseline, and the fixed-plan MCMC target. This preserves the Gaussian-likelihood idea while removing boundary atoms and cross-stage ambiguity.
- **Another declared model:** for example a tanh-squashed Gaussian or temporally correlated prior, with exact density and proposal math.

The choice must also specify dimension-specific scales, whether the PPO penalty is exact NLL or excess energy, sum/mean aggregation, coefficient, realized-prefix versus fixed-plan accounting, and whether post-failure tail coordinates belong to the MCMC target. These choices affect which failures are called likely.

## 6. Failure predicate and timestep

### CartPole definition

After applying `w_t` and integrating the physics, failure is the first strict crossing of:

```text
abs(cart_position) > 2.4
OR abs(pole_angle) > 12 degrees
```

Exact equality is not failure. Failure is `terminated=True`. Reaching the horizon (default 500) without a crossing is `truncated=True` and not failure. If crossing occurs on the horizon transition, failure wins. Source `failure_step` and `episode_len` are one-based transition counts, so the causing wind is at index `failure_step - 1`.

### Approved GPUDrive smoke definition (R7/R8/R9)

GPUDrive exposes victim post-step channels corresponding to road-object contact, vehicle collision, non-vehicle collision, and goal reached. Its Python `off_road` name refers to the road-collision channel and is not evidence of a general geometric lane-departure predicate. `done` also includes goal/horizon and cannot define failure.

**Approved smoke source analogue:** victim-only safety failure on the first post-action transition with any of:

```text
collidedWithVehicle
OR collidedWithNonVehicle
OR collidedWithRoad
```

Goal and horizon remain non-failure termination reasons. Simultaneous flags are stored losslessly and safety failure wins only for the derived outcome. Under the leading pretrained victim configuration, keep `collision_behavior=ignore`, inspect transient collision flags immediately after every step, and have the research wrapper terminate at the first approved event.

The pinned victim-compatible environment also has `remove_non_vehicles=true`.
The classifier retains the raw nonvehicle channel for a stable definition, but
that subtype is likely unreachable in the bundled smoke scene; observing none
does not establish safety against nonvehicles.

Any later change such as excluding road contact, defining geometric lane departure, including timeout/non-goal, counting any-agent events, or using `stop`/`remove` is a new scientific definition and requires approval.

**Approved eligibility strengthening:** attack only scene/victim pairs whose zero-disturbance rollout reaches the goal without the approved safety event. The CartPole source does not impose this condition, so adopting it changes the estimand but avoids labeling an already-failing nominal rollout as adversarially induced. The bundled scene has not yet passed this native preflight.

Canonical port indexing is:

```text
s_t -> (victim action, adversary latent/disturbance) -> step -> s_(t+1)
```

If failure first appears in `s_(t+1)`, store zero-based `failure_timestep=t`, compatibility `failure_step_count=t+1`, transition count `T=t+1`, and select latent `h_t`.

## 7. Scene/reset determinism

### CartPole source

`reset(seed)` uses Gymnasium's RNG to sample four initial state values uniformly from `[-0.05, 0.05]`. Victim and adversary evaluation use `explore=False`; adversary memory is reset per episode. The source assumes deterministic inference/dynamics but neither enables deterministic Torch algorithms nor fingerprints the runtime. Several artifacts omit the reset seed entirely.

### GPUDrive contract

One integer seed does not identify a driving reset. Certified replay and MCMC will:

1. run exactly one world;
2. load an explicit scene byte hash and internal `scenario_id` with no resampling;
3. assert the stable victim object ID and slot after reset;
4. use exact environment, observation, dynamics, action-adapter, initialization, collision, and warm-up settings;
5. freeze victim/adversary checkpoints and deterministic rules;
6. use the approved logged or reactive background model;
7. record scene-loader, Python, NumPy, Torch, policy, baseline, and MCMC seeds separately;
8. record OS, Python, Torch, CUDA, driver, GPU, compiler, native-extension, and kernel-cache identities; and
9. verify repeated reset and a fresh-process reconstruction.

The pinned GPUDrive reset path is structurally deterministic for fixed scene/config/actions, but cross-hardware bitwise equality is not promised. Replay requires exact failure flag set and timestep; numeric state comparison uses a declared tolerance on the certified backend.

## 8. Trace representation

### Source artifact boundary

CartPole adversary evaluation saves only `winds`, `episode_lengths`, `victim_failed`, `failure_steps`, `max_wind`, `wind_sigma`, and `horizon`. It does not save observations, initial seeds, victim actions, rewards, termination subtype, checkpoint identity, or code/runtime provenance. That is insufficient for the requested GPUDrive repository.

### Required GPUDrive logical schema

For `T` applied transitions, `A` actor slots, observation width `D`, disturbance dimension `d`, and latent width `L`, the trace has `T+1` states and `T` transition records:

| Field | Shape/type | Meaning |
|---|---|---|
| Manifest | JSON object | All fingerprints, configs, scene/victim identity, seed ledger, termination, and parent artifacts. |
| `agent_id`, `actor_type` | `[A]` | Stable IDs and actor types; padding sentinel is explicit. |
| `victim_slot`, `victim_id` | scalars | Runtime slot and stable scene identity. |
| `controlled_mask`, `sdc_mask` | `[A] bool` | Reset-time roles. |
| `valid` | `[T+1,A] bool` | State validity/padding. |
| `position_xy`, `velocity_xy` | `[T+1,A,2]` | Scene-wide motion in a declared coordinate frame. |
| `yaw` | `[T+1,A]` | Scene-wide heading. |
| `victim_observation` | `[T+1,D]` | Exact checkpoint-preprocessed victim input. |
| `raw_info_flags`, `done` | state aligned | Collision subtypes, goal, type, and raw done evidence. |
| `victim_action_native` | `[T,...]` | Index/logits or native deterministic output needed to audit decoding. |
| `victim_nominal_command` | `[T,3]` | Physical `[accel, steer, head]`. |
| `adversary_raw` | `[T,d]` or distribution params | Policy output before its bounded transform. |
| `disturbance_requested` | `[T,d]` | Requested bounded intervention. |
| `disturbance_effective` | `[T,d]` | Actual residual after final saturation. |
| `applied_command` | `[T,3]` | Exact simulator command. |
| `saturation_mask` | `[T,d] bool` | Which residual dimensions were changed by the actuator envelope. |
| `adversary_log_prob` | `[T]` | PPO policy log probability. |
| `prior_log_prob`, `prior_penalty` | `[T]` | Declared nominal model, separate from PPO likelihood. |
| `reward_components` | `[T,*]` | Failure, prior, shaping, and total reward. |
| `transformer_latent` | `[T,L]` or selected row | Pre-action feature when requested. |
| `failure_flags` | bit set | First-failure evidence without lossy precedence. |
| `failure_timestep` | optional int | Zero-based causing action. |
| `failure_step_count` | optional int | One-based compatibility count. |
| `termination_reason` | enum | Safety subtype/union, goal, horizon, invalid victim, simulator error, or external limit. |

For MCMC, store the complete fixed plan `[H,d]` separately from its realized prefix and never infer suffix values from padding. Save proposal and current rows separately. "Complete" means sufficient state/control/evidence for attribution and reconstruction; it does not promise a native simulator memory snapshot.

## 9. Checkpoint and dataset versioning

### Source behavior and gap

The committed victim/adversary checkpoints use RLlib checkpoint version `1.1`, `cloudpickle`, Ray `2.10.0`, and Ray commit `09abba26...`; `requirements.txt` instead pins Ray `2.40.0`. Source paths and directory names are not immutable identity, and pickle loading can execute code. Later latent extraction hashes MCMC and checkpoint directories, but that does not recover the provenance missing when the chain was generated.

### Port contract

Every victim/adversary checkpoint includes safe typed weights and a manifest with:

- checkpoint/schema kind and version;
- byte hash of every weight/optimizer payload;
- architecture and constructor configuration;
- observation layout, mask, normalization, and tokenization hashes;
- action table/order, dynamics, bounds, and transform hashes;
- training step and optimizer/scheduler metadata for resumable checkpoints;
- parent checkpoint and victim/adversary relationship;
- training/evaluation dataset cohort hashes; and
- code, simulator, build, runtime, intervention, prior, and failure-spec hashes.

Every dataset is pinned by immutable upstream revision plus an ordered scene manifest containing relative path, byte length, SHA-256, scenario ID, split, and preprocessing provenance. The eligible cohort gets a separate ordered hash with every exclusion and reason.

## 10. Baseline mapping

### Exact CartPole baselines

- **Constant wind:** 100 paired seeds; default positive sweep `0, 0.5, ..., 4.0`. Its headline "success" is length at least 450, not true absence of boundary failure.
- **Random wind:** IID `Normal(0, sigma)` samples subsequently clipped by the environment; defaults bound 1 and sigma 1. Its histogram summary also uses an episode-length threshold.
- **Sine wind:** phase-zero `A*sin(2*pi*0.23*t)`, amplitude 0.1 through 1.0, bound 1; finds/replays on seed 1006 and evaluates other seeds. Frequency 0.23 came from CartPole trace analysis and is not a domain-independent baseline.

### GPUDrive transfer

All baselines use the same victim, scenes, intervention adapter, bounds, actuator envelope, decision period, failure predicate, horizon, trace writer, and replay verifier as the learned adversary.

The minimum approved set should include zero, signed per-axis constants, selected constant corners or magnitudes, and an IID draw from the approved nominal prior. A source-compatible clipped-Gaussian baseline is used only if R6 selects that mixed-measure behavior. Report true failure rate/subtype/timestep and likelihood/magnitude; never replace the event with an episode-length threshold.

A periodic baseline should be added only after a preregistered driving-domain spectral rule identifies a frequency, or as an explicitly exploratory analysis. The CartPole value `0.23` must not transfer numerically.

## 11. Failure-conditioned Metropolis MCMC

### Exact source target and state

The source selects a failed adversary rollout, zero-pads it to `H=350`, and uses a direct scalar plan `w in R^H`. The committed initializer comes from episode 6, fails after 289 transitions, and has 61 zero tail coordinates. The target is:

```text
pi(w) proportional to
    exp(-0.5 * sum_t (w_t / sigma_natural)^2)
    * I(all |w_t| <= max_wind)
    * I(deterministic replay fails by H)
```

Every coordinate, including an unapplied post-failure tail, contributes to the score. The committed chain uses `H=350`, `max_wind=1`, `sigma_natural=0.3333`, environment seed 1006, MCMC seed 123, and `proposal_mode=all`. Its target scale `0.3333` is distinct from the source adversary-evaluation `wind_sigma=1.0`; the source therefore does not support calling these one shared nominal prior without qualification.

The symmetric proposal can perturb all coordinates, one uniformly selected coordinate, or a uniformly located contiguous block. Out-of-box proposals are rejected rather than clipped. A failing in-box proposal receives the usual Metropolis ratio; a nonfailure proposal is rejected. Row zero is verified to fail, `N` proposals produce `N+1` chain rows, and every rejection writes the unchanged current plan/failure step.

The source initializer check only requires some replay failure, not the exact recorded failure step. The port strengthens this: initialization must reproduce the same approved failure flags and exact causing timestep before MCMC begins.

### GPUDrive transfer (R6/R11/R12)

The source-backed default state is a fixed direct plan `d in R^(H x d_control)`. Its target is a separately declared fixed-plan density conditioned on exact replay failure for one scene/victim/config. R6 decides whether that density is identical to the PPO penalty and stochastic-baseline prior. This settles direct plans versus adversary policy-noise innovations unless the user requests a deliberate methodological deviation.

Still requiring approval:

- `H`, whether all post-failure tail coordinates remain probabilistically active, and canonical dtype;
- the vector prior/bounds from R5/R6;
- all-coordinate, single-coordinate, or block proposal and scale;
- boundary handling and any burn-in-only adaptation; and
- retained-chain/burn-in settings.

The raw chain is never deduplicated. Proposal diagnostics cannot substitute for current rows, and an accepted state is not serialized until replay certification passes.

## 12. Distinct failures, latent extraction, and analysis

### Source distinctness

The source selects row zero plus `accepted_index + 1` for every accepted proposal. It validates that every rejected row and failure step exactly repeat the predecessor, maps all 100,001 chain rows to cumulative state IDs, and stores dwell counts. "Distinct" therefore means jump-chain transitions, not global content uniqueness. An accepted revisit would still receive another selected row.

The primary source analysis ignores dwell counts and treats each of the 9,627 selected states in the committed artifact once. This estimates modes of the accepted-state/jump-chain population, not prevalence under stationary chain dwell mass.

### Source latent

For every selected state and every step through its one-based failure count:

1. restore the deterministic victim and adversary;
2. reset the CartPole seed and zero-pad the 50-step attention memory;
3. call the adversary on the pre-action six-vector;
4. copy exactly one finite 64-D `policy.model._features` vector;
5. ignore the adversary's predicted wind and apply the stored MCMC wind;
6. reject any early/late/nonfailure replay mismatch; and
7. retain the last copied vector as `final_latent`.

The CartPole extractor itself establishes only that `_features` is populated by the policy call and contains one finite 64-D vector before the forced step. Inspection of the expected Ray 2.10 attention wrapper indicates that it is the contextualized current-token GTrXL feature used before the separate policy/value heads, but the port will not rely on that private-field inference: a model-level test must prove the explicit `pre_actor_features` layer/token contract. The source also saves full-prefix and final-50 mean latents, but clusters only final latents.

### GPUDrive latent contract (R13/R14/R16)

- Preserve the raw chain and create a derived jump-chain index identical to the source rule.
- Validate accepted canonical rows; record content hashes and any revisits rather than silently claiming global uniqueness.
- Preserve dwell weights even though the primary source-matched analysis is unweighted.
- Replay stored direct disturbances, teacher-force the permitted history, and capture a stable explicit 64-D `pre_actor_features` vector immediately before the failure-causing action.
- Label it `teacher_forced_context_latent`; it is not the action-producing representation for the overridden MCMC action.
- Store trace, checkpoint, feature-hook, failure-timestep, and replay-certificate identities with each vector.

### Source-matched primary analysis (R15)

- population: each jump-chain selected state once, not dwell weighted;
- input: final pre-failure 64-D context latents;
- preprocessing: `StandardScaler` fitted across all selected rows;
- clustering: K-means in standardized 64-D space, `k=2..10`, `n_init=20`, seed 42;
- selection: maximum silhouette score using at most 5,000 seeded sample rows, with smaller `k` winning an exact tie;
- label order: increasing unweighted mean failure step;
- PCA: two components;
- t-SNE: two components, perplexity 30, PCA init, 1,000 iterations, seed 42;
- UMAP: two components, 30 neighbors, minimum distance 0.1, Euclidean metric, seed 42, one job; and
- trace means/plots: realized prefix only, unweighted.

PCA/t-SNE/UMAP visualize the full-space K-means labels; they do not define the clusters. The port additionally visualizes nominal/applied driving control, saturation, victim/all-agent motion, safety flags, event geometry, and top-down paths. It saves resolved parameters, library versions, arrays/tables, assignments, and figure fingerprints.

## 13. CartPole assumptions that do not transfer

| Source assumption | Why it fails in GPUDrive | Required adaptation |
|---|---|---|
| One anonymous controlled system | GPUDrive has variable actors and padded slots. | Bind each sample to scene plus stable victim ID and role assertions. |
| Four homogeneous state scalars | Driving observations contain ego, partners, map features, masks, and IDs. | Approve/fingerprint the adversary information and token schema. |
| Scalar force with one unit and scale | Driving residuals span acceleration and steering with different units/nonlinear effects. | Dimension-specific bounds, scales, and actuator envelope. |
| Binary policy action directly maps to `+/-10 N` | GPUDrive victim actions may be 91 indices decoded to physical triples. | Store native index, decoded command, residual, and final command. |
| Current physical state fully specifies the small system | Driving depends on scene identity, logged actors, map, and validity. | Complete scene/config/checkpoint fingerprint and scene-wide trace. |
| Synthetic seeded reset identifies an episode | GPUDrive reset depends on scene bytes, preprocessing, actor selector, native build, and loader behavior. | Explicit scene manifests and a seed/runtime ledger. |
| One analytic threshold union is failure | GPUDrive collision, road contact, lane departure, goal, and horizon are distinct signals. | Approve a victim event bit set and separate termination reason. |
| Failure is absorbing termination | GPUDrive collisions can be transient under `ignore`. | Inspect post-step evidence immediately and terminate in the research wrapper. |
| No background agents | Non-victim GPUDrive actors usually replay logged, non-reactive paths. | Declare background model and its causal limitation. |
| IID scalar Gaussian has one natural scale | Vector controls need unit-specific scales and possibly temporal modeling. | Explicit joint prior and exact density. |
| Clipping is innocuous | It creates boundary point masses and disagrees with continuous MCMC density. | Choose source-compatible mixed mechanics or one coherent bounded prior. |
| Fixed 350-D plan is cheap | A 2-D GPUDrive plan doubles dimension and each replay is expensive/discontinuous. | Approve `H`, kernel, caching identity, and diagnostics without weakening invariants. |
| Never-applied tail is harmless | It changes target score and acceptance despite not changing the observed failure. | Explicitly retain this source choice or define a valid alternative length model. |
| One `failure_step` integer is enough | Driving has simultaneous failure types and nonfailure terminations. | Store bit set, zero-based causing action, one-based count, and reason. |
| Checkpoint path identifies policy | Source paths/cloudpickle lack portable provenance and can execute code. | Safe typed weights plus byte/config/source hashes. |
| Private RLlib `_features` is a stable latent API | Model implementations and layer semantics can change. | Named, tested pre-actor feature interface and hook-order evidence. |
| Accepted transition means globally distinct | Accepted chains can revisit states and float canonicalization can collide. | Preserve source jump-chain index and separately record content identity/revisits. |
| Unweighted jump-chain prevalence equals posterior prevalence | Dropping dwell weights changes the measure. | Label source analysis unweighted and retain weights for sensitivity analysis. |
| Sine frequency `0.23` is generic | It reflects CartPole dynamics and one trace analysis. | Derive any driving periodic baseline by an approved domain-specific rule. |

## 14. Decision register for user approval

Recommended answers are proposals, not decisions already made.

| ID | Scientific decision | Source evidence | Current GPUDrive answer | Status |
|---|---|---|---|---|
| R0 | Methodology source/revision | Exact code now audited. | Commit `315b14a...`. | Resolved |
| R1 | Canonical victim | Source trains/loads one PPO. | Pinned published PPO revision `1532950...`; strict safetensors load. | Resolved |
| R2 | Victim/cohort/background | One victim; no background. | Slot-0 SDC with stable identity; other actors logged playback. | Resolved |
| R3 | Intervention locus | Additive victim physical-force residual. | Victim physical-control residual. | Resolved for C smoke |
| R4 | Placement/dimensions/period | One same-step scalar, no current victim action in adversary input. | 2-D decoded `[delta_accel, delta_steer]`, every GPUDrive step, excluding current nominal action from adversary input. | Resolved for C smoke |
| R5 | Bounds/actuator envelope | Symmetric scalar bound; experiment uses 1 despite CLI default 4. | Residual `[+/-0.667, +/-0.262]`; final acceleration `[-4,4]`, steering `[-3.142,3.142]`, head preserved. | Resolved for C smoke |
| R6 | Prior/NLL/plan accounting | Gaussian energy, clipped random baseline, bounded continuous MCMC, full-H MCMC score. | C smoke: IID zero-mean tanh-Normal, latent sigma `[0.5,0.5]`, exact NLL excess, coefficient `0.01`. | C smoke resolved; research/F target and tail blocking |
| R7 | Failure | Strict post-step physical threshold; horizon nonfailure. | Victim vehicle/non-vehicle/road-contact union; goal/horizon nonfailure; safety wins ties. | Resolved for C smoke |
| R8 | Eligibility | Source attacks all evaluated seeds. | Require nominal safety-clean goal success before attack. | Resolved rule; native scene result pending |
| R9 | Collision response | Failure immediately terminates. | Keep simulator `ignore` for checkpoint compatibility, but wrapper stops immediately on approved event. | Resolved for C smoke |
| R10 | Dataset/start | Synthetic reset distribution. | Exact bundled scene/seed for smoke; pin a larger eligible cohort for reported research. | Smoke resolved; research blocking |
| R11 | MCMC state/target | Direct fixed-H disturbance plan under a separately parameterized bounded Gaussian density conditioned on failure. | Same state family in 2-D; R6 decides whether its target density equals the PPO/baseline prior. | Source-resolved state; confirm target in R6 |
| R12 | Proposal | Symmetric Gaussian `all` default; `single`/`block` supported; out-of-box rejection. | Source-matched `all` or approved block adaptation; freeze scale after burn-in. | Blocking before F |
| R13 | Distinct/weighting | Row zero plus accepted successors; dwell weights retained; analysis unweighted. | Preserve exactly and record content revisits. | Source-resolved |
| R14 | Latent | Finite 64-D private `_features` copied before the forced failure action; exact wrapper layer semantics require verification. | Stable, model-tested 64-D `pre_actor_features`, teacher-forced context label. | Source-resolved timing; engineering verification required |
| R15 | Analysis | StandardScaler, K-means/silhouette, PCA/t-SNE/UMAP, unweighted traces. | Preserve as primary; add manifested driving views. | Source-resolved |
| R16 | Information/token/history/frequency | Six-vector plus previous action sign/wind; 50-step memory. | Current 2,984-D victim observation plus previous 3-D nominal command and 2-D effective disturbance; causal 50-step history; every simulator step. | Resolved for C smoke |
| R17 | Failure reward | `-0.5(w/sigma)^2`, `+1000` failure, and subtraction of remaining normalized boundary margin only on nonfailure horizon, with gamma 0.99. | C smoke: failure `+1`, NLL coefficient `0.01`, no horizon shaping. | C smoke resolved; research calibration blocking |

The approved C implementation is a one-scene non-claim smoke. It cannot be promoted to reported research until the native eligibility result, R10 cohort, and R6/R17 research calibration are recorded. R11-R12 plus the fixed-plan R6 tail/target rule are additionally required for MCMC. Source-resolved R13-R15 will be used unless the user approves a named deviation.

## 15. Evidence anchors

### CartPole source

- [Victim PPO training and deterministic evaluation](https://github.com/soreng25/Adversarial-Transformer-PPO-agent-in-Cartpole/blob/315b14a90b252ba416eb329e8003d5926806ba67/train_victim.py)
- [Adversarial CartPole dynamics, reward, failure, and trace info](https://github.com/soreng25/Adversarial-Transformer-PPO-agent-in-Cartpole/blob/315b14a90b252ba416eb329e8003d5926806ba67/envs/adversarial_cartpole.py)
- [Transformer-PPO architecture and evaluation artifacts](https://github.com/soreng25/Adversarial-Transformer-PPO-agent-in-Cartpole/blob/315b14a90b252ba416eb329e8003d5926806ba67/train_adversary.py)
- [Failure-conditioned Metropolis chain](https://github.com/soreng25/Adversarial-Transformer-PPO-agent-in-Cartpole/blob/315b14a90b252ba416eb329e8003d5926806ba67/mcmc_failure_trace.py)
- [MCMC replay and latent extraction](https://github.com/soreng25/Adversarial-Transformer-PPO-agent-in-Cartpole/blob/315b14a90b252ba416eb329e8003d5926806ba67/extract_transformer_latents.py)
- [Latent clustering and trace visualization](https://github.com/soreng25/Adversarial-Transformer-PPO-agent-in-Cartpole/blob/315b14a90b252ba416eb329e8003d5926806ba67/analyze_transformer_latents.py)
- [Constant baseline](https://github.com/soreng25/Adversarial-Transformer-PPO-agent-in-Cartpole/blob/315b14a90b252ba416eb329e8003d5926806ba67/eval_constant_wind.py)
- [Random baseline](https://github.com/soreng25/Adversarial-Transformer-PPO-agent-in-Cartpole/blob/315b14a90b252ba416eb329e8003d5926806ba67/plot_episode_length_histogram.py)
- [Sine baseline](https://github.com/soreng25/Adversarial-Transformer-PPO-agent-in-Cartpole/blob/315b14a90b252ba416eb329e8003d5926806ba67/eval_sine_wind.py)

### GPUDrive target

- [Environment tensor/action APIs](https://github.com/Emerge-Lab/gpudrive/blob/aa48a431ed127a37610cc2176db30ec73d0c55df/gpudrive/env/env_torch.py)
- [Environment configuration and action grids](https://github.com/Emerge-Lab/gpudrive/blob/aa48a431ed127a37610cc2176db30ec73d0c55df/gpudrive/env/config.py)
- [Scene loading and selection](https://github.com/Emerge-Lab/gpudrive/blob/aa48a431ed127a37610cc2176db30ec73d0c55df/gpudrive/env/dataset.py)
- [Info layout](https://github.com/Emerge-Lab/gpudrive/blob/aa48a431ed127a37610cc2176db30ec73d0c55df/src/types.hpp)
- [Reset, collision, and done behavior](https://github.com/Emerge-Lab/gpudrive/blob/aa48a431ed127a37610cc2176db30ec73d0c55df/src/sim.cpp)
- [Classic/delta/bicycle dynamics](https://github.com/Emerge-Lab/gpudrive/blob/aa48a431ed127a37610cc2176db30ec73d0c55df/src/dynamics.hpp)
- [Deterministic victim action selection](https://github.com/Emerge-Lab/gpudrive/blob/aa48a431ed127a37610cc2176db30ec73d0c55df/gpudrive/networks/late_fusion.py)
- [Pinned published victim-policy revision](https://huggingface.co/daphne-cornelisse/policy_S10_000_02_27/commit/1532950cad84dafc6e9d976a2bcc524ee481a1a1)
- [Pinned `model.safetensors` file](https://huggingface.co/daphne-cornelisse/policy_S10_000_02_27/blob/1532950cad84dafc6e9d976a2bcc524ee481a1a1/model.safetensors)
