# Ten-agent highway pilot

This is the original one-scene any-agent pilot used to test mechanics and observe multi-agent failure behavior; it is not a population-level result.

> Historical objective: this document preserves the original any-agent pilot.
> Its checkpoint-94 slot-0 road contact revealed that the failure reward was too
> permissive. New training uses the separate
> [slots 1-9 system-failure objective](NONFOCAL_SYSTEM_EXPERIMENT.md); the old
> artifact remains valid only as a diagnostic and reproducibility record.

## Exact experiment

- Dataset: `EMERGE-lab/GPUDrive_mini`, validation split, revision `f0abb649610379e70d0b3105745550fa7acc90d0`.
- Source scene: `tfrecord-00107-of-00150_281.json`, scenario `a8ce823178390d2c`, SHA-256 `1f2e221207cefe8c43a27eef93cb82a3fe3d6bba7553a70de8f0762ef17173ec`.
- Highway evidence at reset: the source SDC travels at `30.6 m/s` (about `68 mph`); selected moving vehicles travel at roughly `25.5-32.2 m/s`; the nearby road graph has lanes, road lines, and road edges, with no nearby intersection, crosswalk, stop sign, or speed bump.
- Selection: source SDC first, then the nine nearest controllable vehicles that are valid at reset, have at least 10 valid source samples, initial speed at least `5 m/s`, and recorded displacement at least `10 m`. Ties use ascending object ID.
- Stable IDs by simulator slot: `[1460, 844, 846, 845, 843, 850, 858, 862, 857, 859]`.
- The derived scene contains exactly those ten vehicles. Every other dynamic object is removed; there are no logged-playback actors.
- One frozen pretrained PPO network is shared by all ten cars and evaluated as one deterministic batch. Each car has its own observation and goal.
- Only slot 0 / ID `1460` receives the adversarial residual. The other nine receive their PPO commands unchanged.

The source scene and selected IDs are immutable inputs. The derived scene is rebuilt deterministically and must have canonical SHA-256 `125ea611a5d7dd9aae52d08e0d5c22931e2db7dbdfc55fdf2c6c434df215f1e4`.

## Intervention, eligibility, failure, and reward

The focal residual bounds remain `[+/-0.667 acceleration, +/-0.262 steering]`. The residual is added after decoding the focal PPO's discrete action; the final acceleration and steering are clamped to the existing physical command envelopes. Head angle is unchanged.

Before training starts, a zero-disturbance rollout must show that all ten PPO cars reach their individual goals and that none has a safety failure. If this clean-eligibility test fails, the command stops instead of training on an invalid scene.

A failure is the first post-step road-object contact, vehicle collision, or nonvehicle collision reported for **any** of the ten controlled cars. Goal and horizon are not failures. Because the derived scene has no pedestrians or cyclists, the nonvehicle-agent collision subtype is structurally absent in this particular pilot; road geometry contacts remain possible.

Distance-to-failure is the minimum signed clearance between every active pair of oriented vehicle rectangles. Positive means separated, zero means touching, and negative means overlap. The reward follows the CartPole timing pattern:

- every step: subtract `0.01 * NLL_excess`, favoring likely/small residuals;
- failure: add `+1` on the failure-causing transition;
- nonfailure: on the final transition, subtract the episode's closest clearance divided by the positive initial clearance and clipped to `[0,1]`.

Thus a nonfailing rollout receives a better terminal score when some pair came closer, without adding an arbitrary dense reward at every timestep.

## Run on Ananke

From the repository root with the previously built pinned GPUDrive environment:

```bash
bash scripts/run_highway_10agent.sh 2>&1 | tee highway-10agent.log
```

The script downloads only the pinned source scene if missing, checks its SHA-256, builds the ten-object scene, certifies the clean rollout, then trains for 100 PPO iterations. Each iteration prints only sampled training results: episode count, failure count/rate, and mean minimum clearance. The separate deterministic evaluation remains stored in the checkpoint metrics for replay selection, but is intentionally omitted from the training-results line to avoid mixing two different populations.

To monitor a detached run, launch it using the site's approved job/session mechanism (for example `tmux` if available), then follow `highway-10agent.log`. Do not assume an SSH process survives logout unless the NASA host's scheduler or session manager guarantees it.

## Replay and visualize a failure

The training run stores every iteration checkpoint, including its separate
deterministic evaluation result. Choose a checkpoint whose
`metrics.deterministic_evaluation.failure_timestep` is an integer. In the first
100-iteration pilot, iteration 94 was one such checkpoint. Render it from the
repository root on the same Linux/CUDA machine:

```bash
bash scripts/render_highway_failure.sh iteration-0094
```

The default output is `artifacts/highway-10agent/failure-0094/`. The command:

1. validates the complete parent training artifact and selected checkpoint;
2. reloads the same frozen victim and trained adversary weights;
3. performs deterministic mean-action replay twice from the exact derived scene;
4. requires the training-time, first-replay, and repeated-replay failure action
   index to agree; and
5. exports:
   - `failure.gif`, a 10-frame-per-second top-down animation;
   - `failure-frame.png`, the immediate post-action failure state;
   - `trajectories.png`, paths for all ten agents;
   - `clearance.png`, signed minimum pairwise vehicle clearance over time;
   - `controls.png`, nominal/applied focal commands and effective disturbance;
   - `failure-trace.npz`, the numerical trajectory behind the figures; and
   - `manifest.json`, hashes, identities, event details, and the repeat-replay
     certificate.

The disturbed slot-0 vehicle is outlined and labeled in red. Any agent carrying
a post-step safety flag at the first failure is marked with a gold cross. A
finished vehicle that GPUDrive has moved to its out-of-bounds sentinel position
is excluded from automatic camera framing and from subsequent trajectory points.
Thus goal-completed agents cannot cause the animation to zoom away from the
remaining traffic. A
failure GIF is not published if the selected checkpoint no longer reproduces
the recorded failure or if the repeated replay differs. This command provides
a same-process repeat-replay check; the separate fresh-process replay gate
required for the complete Milestone D research artifact remains to be added.

To use another output directory, a wider minimum camera radius, or slower GIF
playback, pass radius as the second argument and frames per second as the third.
For example, `3` fps is about one-third real-time because the simulator advances
at 10 steps per simulated second:

```bash
GPUDRIVE_FAILURE_OUTPUT=artifacts/highway-10agent/failure-0094-wide \
  bash scripts/render_highway_failure.sh iteration-0094 90 3
```

After completion:

```bash
.deps/gpudrive/.venv/bin/python -m gpudrive_adversary validate-highway-run \
  artifacts/highway-10agent/train-100
```

The artifact contains the derived scene, nominal eligibility trace, final sampled and deterministic traces, per-iteration metrics, safe model/optimizer checkpoints, all ten action streams, focal disturbances, global failure evidence, goal clocks, clearance/pair traces, and source/config/checkpoint/runtime fingerprints.

## Current limitation

This code has passed pure unit and artifact-contract tests locally, but the selected scene has not yet passed the native Linux/CUDA clean-eligibility gate. The first Ananke run is the required certification. A result from one scene cannot establish a general adversarial failure rate or emergent-property distribution; that requires a preregistered multi-scene train/validation cohort after this pilot works.
