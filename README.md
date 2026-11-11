# DAROM-TPR: E76→D3 single-agent driving

This repository contains the single-agent DAROM SAC/GRU environment for an Ego
vehicle that starts on E76 and reaches D3 through E77. The task is fixed:

- Ego origin: E76; destination: D3; departure lane: random legal E76 lane
- scenarios: `easy`, `medium`, `hard`
- SUMO step: 0.1 s; episode limit: 400 s / 4,000 steps
- road limit: 120 km/h (`33.3333333333 m/s`) in `sumo_new/Antwerp.net.xml`
- one Ego agent, `laneChangeMode=0`, custom safety interception disabled
- pending lane-change requests are observational only and never block a new command
- original DAROM delayed observation, action semantics, SAC and GRU encoder are retained

## Layout

```text
train_sac_delay.py          fresh SAC training entry
evaluate_sac_delay.py      checkpoint evaluation entry
src/merging.py             single-agent SUMO environment
src/wrapper.py             delay/action wrappers
src/delay_encoder.py       DAROM delayed-state encoders
src/exit_e76_d3_config.py  E76→D3 experiment profiles
src/geometry.py            traffic geometry helpers
src/utils.py               shared utilities
src/road_boundary.py       road-boundary diagnostics
src/rewards/events.py      terminal-event rewards
src/rewards/config.py      Hydra/OmegaConf reward configuration
src/rewards/terms.py       RewardTerm and RewardTracker accounting
sumo_new/Antwerp.net.xml   Antwerp network with the fixed corridor limit
sumo_new/Antwerp_*_exit_e76_d3.rou.xml
                            Easy/Medium/Hard E76→D3 routes
sumo_new/Antwerp.sumocfg   Easy E76→D3 GUI configuration
```

Old entry, legacy O3→D1, dual-agent routes, detector definitions, root
`config.yaml`, old model archives, and continuation/replay-resume paths are not
part of the current task.

## Environment setup

Install the dependencies in `requirements.txt`, install SUMO, and export
`SUMO_HOME` so that its Python tools are available:

```bash
export SUMO_HOME=/path/to/sumo
python3 -m pip install -r requirements.txt
```

The training and evaluation code starts SUMO directly with the selected route
file. It does not load detector `additional` files and does not modify lane
speed limits through TraCI.

## Training

Every run starts from fresh random weights and an empty replay buffer. Models
and logs should use a new run name and remain in ignored output directories.
The E76→D3 training contract always keeps custom Safety interception off;
passing `--safety` is rejected rather than creating a `SafetyWrapper`.

```bash
python3 train_sac_delay.py \
  --scenario easy \
  --seed 42 \
  --total_timesteps 200000 \
  --checkpoint_freq 50000 \
  --gamma 0.999 \
  --reward-config configs/rewards/events_only_v1.json \
  --wandb_mode disabled
```

The command-line values are final; no historical Phase or reward file can
override seed, scenario, budget, checkpoint interval, or gamma. The default
resolved reward configuration is `conf/config.yaml` → `conf/reward/e76_d3.yaml`.
The B0 reward is zero on non-terminal steps and settles exactly one event:
success `+50`, collision/missed-exit/teleport/timeout `-100`.

The B0–B3 reward ablations use Hydra reward groups; these are candidate
calibration settings, not claims of optimal weights.  B1 enables clipped SECRM
efficiency, B2 enables clipped topology route guidance, and B3 enables both.
All preserve the terminal-event values and environment contract.

```bash
# B1 / B2 / B3 respectively
python3 train_sac_delay.py --hydra-override reward=e76_d3_b1_efficiency
python3 train_sac_delay.py --hydra-override reward=e76_d3_b2_route
python3 train_sac_delay.py --hydra-override reward=e76_d3_b3_combined
```

Both dense components have candidate `weight=0.005` and clip their raw value
to `[-1, 0]`.  Efficiency uses the existing SECRM safe-speed formula with a
2 m/s denominator floor.  In B1/B3, only the efficiency payment weight is
linearly decayed by the remaining route distance to D3: it is `0.005` at the
E76 reference state, `0.0025` at 50% remaining distance, and zero at the D3
endpoint.  The raw and clipped SECRM values remain unchanged and are logged
separately from the effective weight.  Route guidance computes the minimum legal lane
changes and applies `-min(1, n*200/max(d,200))`, where `d` is the remaining
distance through the final on-route edge that can still host a required lane
change; it is not distance to the D3 endpoint.  Internal connectors are marked
unavailable and pay no route term, rather than being treated as `n=0`; the
efficiency distance factor, in contrast, follows the full route through those
connectors and D3.

Reward terms can be disabled or reweighted with Hydra-style overrides. These
change only payment, not termination or SUMO control:

```bash
python3 train_sac_delay.py \
  --hydra-override reward.terms.collision.enabled=false \
  --hydra-override reward.terms.timeout.weight=0.5 \
  --total_timesteps 200000 --checkpoint_freq 50000 \
  --wandb_mode disabled
```

Each run records the resolved reward YAML, configuration hashes and Git SHA in
the episode-log manifest. It also records `git_dirty` and the porcelain status
so an uncommitted source snapshot is never presented as a clean commit.
Episode-level raw, weighted, discounted, active,
tail-quantile and clipping statistics use the training gamma and are emitted
under `reward/episode/*` and `reward/total/*`. W&B training, reward and
diagnostic series all use monotonic `env_steps`; TensorBoard remains a separate
local log and is not auto-synchronized into W&B.

## Evaluation

```bash
python3 evaluate_sac_delay.py \
  models_experiments/e76-d3/<run>.zip \
  --scenario easy \
  --episodes 50 --seed 200001 \
  --no-safety \
  --output tmp/ai/e76-d3/eval.json
```

Evaluation requires the checkpoint's complete E76→D3 reward configuration,
profile and matching configuration hash; missing or inconsistent provenance is
an error and never falls back to B0. Reports distinguish
`policy_reward_return` (the checkpoint reward configuration) from
`b0_reference_return` (common terminal-event scoring: success `+50`, failure
`-100`, non-terminal `0`) and retain per-term totals. Safety is off for the
matched task contract unless explicitly requested for a separate diagnostic.

The optional `--observation-mode ego_lateral_v1` trains a separate GRU policy
with `paper_raw` inputs. The default `legacy` mode preserves old checkpoints.
The new 136-value layout is Ego `[distance, lane_index, speed, lateral_position,
lateral_speed]`, 30 unchanged neighbor triples, 20 action pairs, and delay.
TraCI reports lateral position relative to the current lane center in meters
and lateral speed in meters per second; positive is left in the direction of
travel, negative is right. Position can jump when the current lane changes.
The two values are read from the current Ego at each observation, while the
neighbor packet and action-history masking retain the existing delay rules.
Checkpoints record the mode and must match the environment and encoder shape;
134-value checkpoints cannot be resumed with the 136-value mode.

For read-only trajectory export, pass a new output directory. The recorder
writes one step row and one transition row per `env.step`, plus an immutable
manifest and column dictionary. Reward columns are derived from the
checkpoint-authenticated reward configuration; for B0 they are `success`,
`collision`, `missed_exit`, `teleport`, and `timeout`. Terminal transitions
whose Ego has already left SUMO retain the environment-returned observation
but mark it invalid rather than presenting it as a physical next state.

```bash
python3 evaluate_sac_delay.py \
  models_experiments/e76-d3/<run>.zip \
  --scenario easy --episodes 1 --seeds 200051 200052 \
  --no-safety --trajectory-dir tmp/ai/e76-d3/trajectory-export \
  --model-label B0 --run-label <experiment-id> \
  --output tmp/ai/e76-d3/trajectory-eval.json

python3 -m scripts.verify_trajectory_csv \
  --trajectory-dir tmp/ai/e76-d3/trajectory-export \
  --report-on tmp/ai/e76-d3/trajectory-eval.json \
  --report-off tmp/ai/e76-d3/baseline-eval.json
```

The optional baseline report must use the same checkpoint, arguments, and
seeds without `--trajectory-dir`; the verifier then checks that recording did
not change the evaluation report. Exported `next_obs_valid=0` rows must not be
used as physical successor states in downstream transition learning.

Each current export also contains `episode_summary.csv`, with one row per
trajectory. It aggregates the existing route diagnostics, E76/E77 request and
command stages, physical lane changes, braking/TTC safety diagnostics, and
available efficiency/route reward terms; a missing reward term remains empty.
It does not create labels or change any environment behavior. Compare several
recording directories read-only with:

```bash
python3 -m scripts.analyze_trajectory_collection \
  --group B0=/path/to/B0/trajectory \
  --group B1=/path/to/B1/trajectory \
  --group B2=/path/to/B2/trajectory \
  --group B3=/path/to/B3/trajectory \
  --output-dir tmp/ai/e76-d3/trajectory-comparison
```

The comparison reports evaluation-episode summaries, not independent
training-seed statistics. `exit_reachable_fraction` is computed over only the
steps where the route diagnostic is defined; request, command, and physical
lane-change counts remain separate quantities.

Existing preference exports can be audited without changing the rule or the
source CSV files. The audit replays the exact comparator branch for every pair,
checks A/B reversal and strict three-way cycles, and writes a cross-tab, 20
review samples, and a Markdown report to a new output directory:

```bash
python3 -m scripts.audit_trajectory_preferences \
  --group B0=/path/to/B0 --group B1=/path/to/B1 \
  --group B2=/path/to/B2 --group B3=/path/to/B3 \
  --preferences-dir /path/to/preferences \
  --output-dir tmp/ai/e76-d3/preference-audit
```

The P3-only v2 candidate is a separate implementation and never overwrites the
v1 CSV. It audits whether stable `min_lane_changes_required=0` has false
negatives among successful trajectories before enabling completion-dependent
branches, then writes only a candidate CSV and short comparison report:

```bash
python3 -m scripts.compare_trajectory_preferences_v2_candidate \
  --group B0=/path/to/B0 --group B1=/path/to/B1 \
  --group B2=/path/to/B2 --group B3=/path/to/B3 \
  --v1-preferences /path/to/trajectory_preferences.csv \
  --output-dir tmp/ai/e76-d3/preference-v2-candidate
```

Training defaults are Easy, training seed 42, gamma 0.999, 200,000 environment
steps and 50,000-step checkpoints. Evaluation defaults are Easy, 50 episodes
and evaluation seed range beginning at 200001. Training seed and evaluation
seeds are separate provenance fields. Explicit CLI values remain final.

SUMO `Emergency Stopping` diagnostics and the kinematic hard-braking threshold
(`acceleration <= -4.5 m/s²`) are logged separately. Both report step counts,
episode counts and episode rates; neither changes termination or B0 reward.

The network lane speed is 120 km/h and all three formal route files pin the Ego
trip to `speedFactor=1.0`. Background vehicle types retain their original
speed-factor behavior.

## GUI inspection

`sumo_new/Antwerp.sumocfg` opens the Easy E76→D3 route with a 600-second GUI
end time and 0.1-second step. The 400-second/4,000-step task timeout is owned
by the environment and is independent of this GUI end time. For Medium or Hard inspection, replace the route file
in the GUI configuration with the corresponding
`Antwerp_<scenario>_exit_e76_d3.rou.xml` file.

## E0 reachability diagnostic

`src.e0_reachability` is a privileged, non-learning diagnostic controller. It
uses the real SUMO state and the parsed lane-connection graph only to choose
ordinary `[acceleration, lateral_action]` environment actions. It does not
change vehicle position, route, speed limits, or the SAC observation.

Run the five fixed E76 departure lanes against the same ten traffic seeds in a
fresh, ignored output directory:

```bash
SUMO_HOME=/path/to/sumo python3 -m src.e0_reachability \
  --output-dir tmp/ai/e0-reachability/<run-id> \
  --lanes 0 1 2 3 4 \
  --seeds 100001 100002 100003 100004 100005 100006 100007 100008 100009 100010 \
  --run-label <experiment-id>
```

The command writes an immutable manifest, route/network/configuration hashes,
`episodes.csv`, per-lane outcome counts, and JSONL traces per episode. It
validates B0 reward settlement on every transition and rejects any terminal
truncation. These privileged-controller results are environment reachability
evidence, not SAC-policy performance.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

The real SUMO integration suite is opt-in:

```bash
python3 -m tests.test_sumo_integration
```

It checks all three scenarios, the fixed route contract, 0.1-second stepping,
4,000-step episode limits, `laneChangeMode=0`, and the 120 km/h network speed.
