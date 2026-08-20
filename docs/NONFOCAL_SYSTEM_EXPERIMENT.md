# Slots 1-9 system-failure experiment

This experiment supersedes the original any-agent failure objective for new
training. The original run and configuration remain unchanged so checkpoint 94
and its slot-0 road-contact visualization stay reproducible as a diagnostic of
why the earlier objective was too permissive.

## Qualifying failure

Slot 0 remains the only disturbed vehicle, but it is not an eligible failing
vehicle. A reward-bearing terminal failure is now exactly one of:

- a road-object contact reported for a vehicle in slots 1-9;
- a nonvehicle collision reported for a vehicle in slots 1-9; or
- a vehicle collision whose two attributed participants are both in slots 1-9.

Vehicle-pair attribution requires both non-focal vehicles to carry GPUDrive's
post-step vehicle-collision flag and their oriented footprints to touch or
overlap within `0.00001 m`. A slot-0 road contact, a slot-0-only event, or a
direct slot-0-to-nonfocal collision is stored as diagnostic evidence but gives
no failure bonus and does not terminate the episode.

The nonfailure clearance shaping is also restricted to pairwise oriented-box
clearance among slots 1-9. This prevents the adversary from receiving shaping
credit merely by moving slot 0 close to another car. Clean eligibility remains
strict: before training, all ten cars must reach their goals with no safety
event under zero disturbance.

## Training and reporting

The new adversary starts from fresh random weights because the old checkpoint
was optimized for the rejected slot-0 self-crash objective. The simulator,
scene, ten vehicles, victim checkpoint, disturbance bounds, NLL coefficient,
Transformer architecture, PPO settings, seed, and 100-iteration workload are
otherwise unchanged.

Run on Ananke from the repository root:

```bash
nohup bash scripts/run_highway_10agent_nonfocal_system.sh \
  > highway-10agent-system.log 2>&1 &
echo $! > highway-10agent-system.pid
disown
```

Monitor it with:

```bash
tail -f highway-10agent-system.log
```

Each iteration reports the qualifying episode failure rate and a compact
`failure_slots=` count. Slot-0 events are not included in that failure count.
After completion, produce the full totals and ranked failing slots/pairs:

```bash
.deps/gpudrive/.venv/bin/python -m gpudrive_adversary \
  summarize-highway-system-run artifacts/highway-10agent-system/train-100 \
  --output artifacts/highway-10agent-system/summary.json

cat artifacts/highway-10agent-system/summary.json
```

The summary includes total qualifying failure episodes/rate, counts for every
slot 1-9, counts by failure kind, ranked non-focal collision pairs, and the
number of episodes containing focal safety events that correctly received no
automatic failure credit. Slot counts are episode incidences: a collision
between slots 2 and 3 increments both slot counters, so slot-count totals may
exceed the number of failed episodes.
