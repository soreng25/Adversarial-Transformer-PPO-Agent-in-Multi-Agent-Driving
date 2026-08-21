# Highway disturbance-bound calibration

This diagnostic runs no PPO updates. It applies IID zero-mean tanh-Normal
disturbances to slot 0 while the same ten frozen PPO vehicles replay the pinned
highway scene. Current, half, and quarter physical bounds receive corresponding
latent draws from the same episode seeds.

The default is 500 episodes per bound. The target range is a 1--5% random-prior
qualifying failure rate. This range is a calibration heuristic, not a reported
research result or an automatic scientific decision.

Run on the Linux/CUDA host:

```bash
nohup bash scripts/run_highway_bound_sweep.sh > highway-bound-sweep.log 2>&1 &
echo $! > highway-bound-sweep.pid
disown
```

Monitor with:

```bash
tail -f highway-bound-sweep.log
```

Results are written to
`artifacts/calibration/highway-bound-sweep-500/summary.json`. Each row reports
the aggregate failure rate, affected slots and event kinds, as well as whether
the focal slot was still active or had already reached its goal at failure.

For a short operational test, override the episode count and output path:

```bash
GPUDRIVE_BOUND_SWEEP_EPISODES=10 \
GPUDRIVE_BOUND_SWEEP_OUTPUT=artifacts/calibration/highway-bound-sweep-test \
bash scripts/run_highway_bound_sweep.sh
```
