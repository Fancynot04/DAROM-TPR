# DAROM: Delay-Aware Reinforcement Learning for Highway On-Ramp Merging

**DAROM** (Delay-Aware Reinforcement learning for On-ramp Merging) is a framework for autonomous highway on-ramp merging under stochastic V2I communication latency. The agent is modeled as a Random Delay Markov Decision Process (RDMDP) and trained with a Delay-Aware Encoder that jointly processes delayed observations, masked action histories, and delay magnitudes to recover the true latent traffic state.

## Overview

In RSU-assisted autonomous driving, the roadside unit (RSU) perceives surrounding vehicles and transmits state estimates to the ego vehicle over V2I links. Processing and transmission introduce stochastic delays, violating the Markov assumption and degrading standard RL performance. DAROM addresses this by:

- **RDMDP formulation**: Models the problem as a Random Delay MDP via state augmentation with the delayed observation, action buffer, and delay magnitude.
- **Delay-Aware Encoder**: A GRU-based encoder that implicitly infers the current latent state from the augmented input.
- **Physics-Based Safety Controller**: A supervisory layer that enforces kinematic stopping distance constraints, overriding unsafe actions before execution.
- **Unified SAC Agent**: A single soft actor-critic agent for joint longitudinal and lateral control.

Upstream experiments in SUMO using real-world NGSIM traffic data report DAROM-GRU achieving **>99% success** in high-density traffic with random delays up to 2.0 seconds and generalizing across five stochastic delay distributions without retraining. These are historical upstream results, not results of the Antwerp migration. Current local validation and pilot outcomes are recorded under **Version 2 staged validation** below.

![DAROM Framework](assets/network.png)

## Repository Structure

```
DAROM-TPR/
├── src/
│   ├── merging.py           # SUMO-based merging environment (gymnasium)
│   ├── wrapper.py           # DelayWrapper — RDMDP state augmentation
│   ├── delay_encoder.py     # Delay-Aware Encoder (MLP / GRU / Transformer)
│   ├── safety_controller.py # Physics-based safety controller
│   ├── geometry.py          # Captured traffic geometry and frame/time alignment
│   ├── diagnostics.py       # Full-episode rule diagnostics and per-step traces
│   └── utils.py             # Shared utilities
├── sumo_files/              # SUMO network and route files (Easy/Medium/Hard)
├── sumo_new/                # Antwerp network and three demand scenarios
├── tests/                   # Persistent regression and opt-in SUMO checks
├── models/
│   └── GRU-uniform-delay/   # Pretrained DAROM-GRU checkpoint
├── train_sac_delay.py       # Training script
├── evaluate_sac_delay.py    # Evaluation script
├── config.yaml              # Environment configuration
└── requirements.txt
```

## Git Submission Policy

The root `.gitignore` uses an allowlist. Only `src/`, `assets/`,
`sumo_files/`, `sumo_new/`, `models/`, `tests/`, and the explicitly listed root files
are eligible for new Git tracking. New root files or directories must be
added to the allowlist intentionally. Under `models/`, only the two existing
pretrained checkpoints in `GRU-uniform-delay/` are allowed.

AI workspace directories (`.codex/`, `.claude/`, `.cursor/`, `.agents/`),
local agent instructions (`AGENTS.md`, `CLAUDE.md`), temporary directories
(`tmp/`, `temp/`, `scratch/`, `artifacts/`), caches, credentials, and generated
training or simulation output are ignored even within allowed directories.
Store AI scratch work in these ignored directories: ordinary files placed
directly in an allowed source or asset directory can still be tracked.

Before committing, inspect `git status --short` and `git diff --cached`.
Ignore rules do not affect files already tracked by Git, and `git add -f`
can bypass them.

## Installation

**Requirements**: Python 3.10+, [SUMO](https://sumo.dlr.de/) ≥ 1.18

```bash
# Clone the repository
git clone https://github.com/<your-username>/DAROM.git
cd DAROM

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set SUMO_HOME (add to ~/.bashrc for persistence)
export SUMO_HOME=/usr/share/sumo
```

## Antwerp traffic routes

`sumo_new/Antwerp_easy.rou.xml`, `Antwerp_medium.rou.xml`, and
`Antwerp_hard.rou.xml` define 1-hour background demand of 1,800, 3,456,
and 7,108 vehicles, respectively. To run one scenario, pass its route file
to SUMO, for example:

```bash
sumo -c sumo_new/Antwerp.sumocfg --route-files sumo_new/Antwerp_hard.rou.xml
```

The origin mix, vehicle lengths, and OD seed come from the Antwerp detector
data on 2022-09-05 from 11:30 to 12:30. Matched trajectories are balanced
to observed origin counts and to the proportions of the three mapped exit
counts. Detector `3849` (`L4e`) has no verified D1/D2/D3 mapping in the
three-exit network; its observations are retained in origin totals, and
their destinations are projected onto the three mapped exits. The four
vehicle types use the paper's disjoint length ranges; cars and vans are
combined because the supplied length data cannot separate them. Each
scenario includes one `ego` trip at SUMO time 120 s from O3 to D1 using the
same `car_van` type as the traffic flow. One O3-to-D1 `car_van` was removed
from each flow, so the 1-hour vehicle and OD totals are unchanged.
The route file defines ego's trip and vehicle type for both pure SUMO and
training. Pure SUMO leaves its lane and speed decisions to SUMO; the training
environment instead applies `laneChangeMode=0` at departure and sends explicit
acceleration and lane commands through TraCI.
`src.merging.Merging(seed=579071, scenario="easy")` loads one of the
`easy`, `medium`, or `hard` Antwerp routes (`medium` by default). Each reset
advances SUMO to just before the scheduled departure, then waits for `ego`
to actually enter O3. The episode clock starts at takeover and has a uniform
limit of 6,000 control steps (600 simulated seconds at 0.1 s per step);
arrival at D1 can end it earlier. `reset()` returns the actual departure
time in `info`. The environment starts SUMO from the network and selected
route files with the timing and lateral settings of `Antwerp.sumocfg`,
without detector outputs. The observation uses ego route distance, lane
index, and speed, with nearby vehicle positions expressed in the ego's
forward/left frame. Reward gaps and current-state safety checks use these
local positions. Neighbor relative speed is the neighbor's world velocity
projected onto ego's forward axis minus ego speed. Delayed safety uses captured
world positions/velocity vectors and timestamps to predict neighbors at constant
velocity, then transforms them into the current ego frame before lane filtering.
Only current ego pose/state is used; current neighbor data is not read by that
prediction. Lane changes are
bounded by each Antwerp edge's actual lane count; merging is
recognized when ego leaves `E75_0` for a mainline lane or reaches `E76`.
The existing pretrained policy must be retrained for this observation layout.
The task is for ego to merge safely and reach D1 before the episode limit;
collision, emergency stopping, timeout, and any ego teleport exclude task success.
Raw `Arrived` remains available as a separate historical comparison metric.
Background queues
are recorded as scenario conditions, not optimized away by changing traffic
demand or network throughput. The existing network speed limit is 13.89 m/s
and remains unchanged for this stage.

## Training

Train DAROM-GRU from scratch:

```bash
python train_sac_delay.py \
    --scenario hard \
    --delay_mode uniform \
    --encoder_mode GRU \
    --total_timesteps 7000000 \
    --run_name GRU-hard-uniform-delay \
    --save_dir models_experiments/antwerp-hard \
    --wandb_mode disabled
```

Use the same scenario when evaluating a checkpoint in a separate process.
The safety controller is disabled by default; pass `--safety` to enable it.
DAROM-GRU is the default encoder for new training runs.
The pretrained checkpoint under `models/` is kept unchanged.
The Antwerp GRU scales route progress and local neighbor coordinates before
encoding, reads masked actions from oldest to newest, and outputs 256 features.
Models trained with the earlier GRU input convention should be retrained.

For a controlled Table IV input ablation, keep `--encoder_mode GRU` and choose
`--ablation` from `full`, `delayed_state_only`, `delayed_state_delay`,
`delayed_state_action`, or `linear_predictor`. All five preserve the 134-value
GRU input. Missing action-history or delay fields are zeroed; the linear
predictor advances each neighbor's relative longitudinal position by its
relative velocity times the received delay and exposes no history or delay to
the policy. This keeps the architecture fixed while changing available
information. Safety receives the same unmasked captured-neighbor metadata for
every ablation; its packet also retains full world velocity vectors, unlike
the policy's three values per vehicle. This extra safety input must be stated
when reporting experiments with `--safety`. Policy input dimensions and the
linear predictor's longitudinal-only extrapolation remain unchanged.
Choose one training-safety policy and one evaluation-safety policy across all
five configurations; those two policies need not be equal. Pass the matching
`--ablation` value to evaluation. Both entry points default to `full` with
safety off; use explicit evaluation `--safety` or `--no-safety` in experiments.
Checkpoints store `training_safety`, `ablation` and `antwerp_env_version=2`.
Evaluation rejects incompatible environment/ablation settings, but independently
selects evaluation safety and records both flags without modifying the model.
Training resume still requires matching training safety to avoid silently
changing the training intervention midway through a run.
Checkpoints predating the velocity/reward corrections must be retrained;
training resume and evaluation reject a missing or different environment version.

## Table IV reproduction protocol

The target is the original task and evaluation, not a shorter success condition
chosen to improve Antwerp scores. Pin the paper to
[arXiv v5, Sections V-A/V-D/V-E and Table IV](https://arxiv.org/html/2403.11852v5#S5).
The table contains **five configurations total**, mapped here as follows:

| Paper configuration | Project argument |
|---|---|
| Delayed State Only | `delayed_state_only` |
| Delayed State + Delay Magnitude | `delayed_state_delay` |
| Delayed State + Action Buffer | `delayed_state_action` |
| Linear State Predictor | `linear_predictor` |
| Full Augmented State | `full` |

The paper specifies Hard traffic, 7,000,000 training steps per agent, learning
rate 3e-5, batch size 512, and a 0.1 s step with delay draws up to 20 steps.
Evaluation uses 500 episodes repeated across three seeds. The full-model
reference is 99.80% success, 0% collision, 0.20% no-merge and 179.68 mean return.
These are reference results, not achieved Antwerp scores.

**Confirmed source differences and current choices** (upstream checked
2026-10-12):

- The [upstream training entry](https://github.com/amin-tabrizian/DAROM/blob/6dadb355a8a5dd3fc4ee092454a7b425d90afee4/train_sac_delay.py)
  does not wrap training in the safety controller and defaults to 10M steps.
  This project's default budget now follows the paper's 7M; explicit small
  `--total_timesteps` values remain available for smoke tests. No 7M run has
  been started by this protocol correction.
- The [upstream evaluation entry](https://github.com/amin-tabrizian/DAROM/blob/6dadb355a8a5dd3fc4ee092454a7b425d90afee4/evaluate_sac_delay.py)
  defaults to safety on and evaluates one checkpoint in three runs of 501
  episodes. Its reported error is `1.96 * sample_std / sqrt(n_runs)`, not
  standard deviation. Here formal commands explicitly use 500 episodes; reports
  preserve sample standard deviation and that normal-approximation interval
  separately. One-run reports use null uncertainty, not a zero-width interval.
- The paper does not specify each Table IV row's training/evaluation safety
  flags. Train-off/evaluate-on is supported by the public entry points, but
  does not establish every row's exact experiment configuration. Until the
  experiment configuration is verified, label that combination as a public-code
  reference convention; keep the same convention across the five configurations.
- Repeated evaluation of one checkpoint estimates evaluation variability only.
  It is not evidence of variation across independently trained models. Keep
  training seeds, evaluation seeds and checkpoint selection separate; final
  test seeds must not be used to choose the best checkpoint.

**Still not an exact reproduction:** Antwerp changes the network, OD demand,
vehicle profiles, ego speed mode, episode duration, coordinates and safety
geometry. Its GRU scaling/action ordering also differs from the pinned upstream
copy; the fixed-shape masking implementation of the five configurations is a
documented project choice. Raw returns across different routes/rewards are not
directly comparable. The original-scene smoke calibration below passes, but
per-row experiment configurations and checkpoint selection remain unverified.
Existing success definitions, traffic and environment version are unchanged.
The earlier 20k pilot is 1/350 of the paper budget and cannot establish
convergence or failure to learn.

### Original-scene calibration (2026-10-12)

The public-code reference is pinned to
[`amin-tabrizian/DAROM@6dadb355`](https://github.com/amin-tabrizian/DAROM/tree/6dadb355a8a5dd3fc4ee092454a7b425d90afee4).
SHA-256 comparisons confirm that this project's original network, SUMO config,
default route and all three difficulty routes match that revision byte for byte.
The default `mergingP.rou.xml` is identical to `mergingP_hard.rou.xml`.
Having these assets does not make the current training entry an original-scene
benchmark: `src.merging.Merging` still loads Antwerp exclusively.

| Setting | Pinned public implementation / packaged checkpoint | Current Antwerp implementation |
|---|---|---|
| Ego task | `E4 → E3 → E0`; arrive after merging | `O3 → D1`; arrive without any ego teleport |
| Clock | Depart at 10 s; timeout when absolute SUMO time exceeds 200 s | Depart at 120 s; 6,000 steps after actual takeover |
| Road / ego speed | Road limit 60 m/s; ego type and command cap 32 m/s | Road limit 13.89 m/s; `car_van` type max 45 m/s, SUMO constraints active |
| Control modes | Requests speed mode 96 and lane-change mode 0 | Actual speed mode 31, lane-change mode 0 |
| Hard demand | Five lane flows, probabilities 0.41/0.38/0.38/0.38/0.38 | 7,108 scheduled vehicles including ego, distributed over multiple OD pairs |
| Ego features | World x/y and speed | Route distance, lane index and speed |
| Neighbor features | Relative world x/y and scalar speed difference | Ego-frame positions and projected relative velocity |
| GRU input split | 91 observation + 42 action + 1 delay values; 21 history slots, 128 output features | 93 observation + 40 action + 1 delay values; 20 history slots, 256 output features |
| Safety input | First 5 entities, including ego | Up to 31 entities, with captured geometry aligned to current ego frame |

These are implementation facts, not a declaration that the public code exactly
implements every paper setting. In particular:

- The [public GRU training configuration](https://github.com/amin-tabrizian/DAROM/blob/6dadb355a8a5dd3fc4ee092454a7b425d90afee4/train_sac_delay.py)
  requests `hist_len=max_delay+1`, while its wrapper emits only `max_delay`
  actions. On a 134-value packet this silently consumes two observation values
  as action history. Both packaged checkpoints preserve this 91/42/1 split;
  the best checkpoint records 9.1M steps and the final checkpoint 10M, seed 42.
  Neither records training safety. They are legacy public-code references,
  not compatible Antwerp models or proven 7M Table IV checkpoints. Do not fix
  their encoder split and then reuse the same weights.
- The [public safety controller](https://github.com/amin-tabrizian/DAROM/blob/6dadb355a8a5dd3fc4ee092454a7b425d90afee4/src/safety_controller.py)
  decodes only ego plus four neighbors. A synthetic probe moves the same
  dangerous leader from row 1 to row 5: the former triggers braking and the
  latter is ignored. This defect is retained only in the isolated reference
  probe; it is not reintroduced into Antwerp.
- The [public environment](https://github.com/amin-tabrizian/DAROM/blob/6dadb355a8a5dd3fc4ee092454a7b425d90afee4/src/merging.py)
  applies aggressive/cooperative overrides only to `f_1`/`f_2`; the other three
  flows keep their type defaults. Cooperative headway uses `N(0.7, 0.1)`
  rather than the paper's uniform interval. Original reward progress uses
  world-x displacement; Antwerp uses route distance. Event rewards retain the
  same nominal scale, but timeout precedence, gap handling and teleport
  accounting differ. Absolute returns cannot be used as a cross-scene ranking.
- Original flow probabilities imply nominal expected rates of 1,800 / 3,456 /
  6,948 vph. The paper's Hard lane counts sum to 7,108. Expected probabilistic
  demand, observed arrivals and Antwerp's scheduled total are different
  quantities; matching a total alone does not match traffic difficulty.
- Local SUMO is **1.18.0**. The original request `speedMode=96` reads back as
  `32`: the [SUMO 1.18 implementation](https://github.com/eclipse-sumo/sumo/blob/v1_18_0/src/microsim/MSVehicle.cpp)
  implements bits 0–5 but not the speed-limit override bit 6 described in the
  [current documentation](https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html#speed_mode_0xb3).
  The paper's exact SUMO version is not established. No simulator upgrade or
  production control-mode change was made during calibration.

The paper's Table IV full row matches Table III's Hard DAROM-GRU row with SC,
which supports **evaluation safety on for the full reference**. It does not
resolve safety flags for every ablation row or training safety. The public
evaluation defaults to `bursty`; calibration explicitly selected `uniform`.
No per-row execution manifest was established from the inspected sources.

**Runtime smoke result:** the packaged `GRU-uniform-delay_best.zip` was loaded
with the pinned original encoder/environment/wrapper/controller, not Antwerp's
modules. Only the hard-coded launch path, output location and random seeding
were adapted: original assets were selected explicitly, SUMO `--random` was
replaced with each stated seed, and NumPy/Python/Torch were seeded. Traffic,
rewards, network and checkpoint bytes were unchanged. CPU inference used one
Torch thread. This deterministic calibration convention differs from the
public entry's unseeded SUMO runs and is not a formal reproduction claim.

| Hard seed | Evaluation SC | Outcome | Control time | Return | SC interventions |
|---|---|---|---:|---:|---:|
| 579091 | Off | Arrived, no teleport | 20.0 s | 180.589 | 0 |
| 579091 | On | Arrived, no teleport | 20.0 s | 180.589 | 0 |
| 579092 | Off | Arrived, no teleport | 20.0 s | 180.150 | 0 |
| 579092 | On | Arrived, no teleport | 20.0 s | 177.969 | 23 |

All four episodes merged. An additional repeat of seed 579091 with SC on
matched all 200 trace records and the return exactly. These are two paired
development seeds, not 500×3 independent test episodes or evidence of 99.8%
success. The archived checkpoint SHA-256 is
`f3d8521eee9a10584a036ba8f2856381e41f86247df9cca2fc7399850242e13b`.
Sources, hashes, probe scripts, JSON traces, logs and SUMO tripinfo remain under
ignored `tmp/ai/original-calibration/`; production code does not depend on them.
The runtime-metadata additions passed all 25 regression tests and three real
SUMO integration tests. A complete Antwerp Hard replay (20k checkpoint, seed
579081, SC off) also preserved the previous 631-step collision and return
exactly while exporting the new scene hashes and runtime settings.

**Calibration decision:** retain Antwerp as a transfer experiment and establish
a separately identified original-scene benchmark before claiming Table IV
reproduction. The current smoke pass validates the legacy runtime chain only.
The next implementation stage should provide a permanent original-scene entry,
make the public-code versus corrected/paper-aligned conventions explicit, and
verify a fresh short training run before committing to the 7M budget. Exact
per-row safety, training seeds, checkpoint selection and SUMO version still
need provenance; do not select those conventions by whichever gives the best
final-test score. Formal training and ablations have not started.

## Antwerp migration checklist

This checklist tracks the working baseline. Mark an item complete only after
its code and validation are both in place; keep unresolved assumptions visible
when comparing results with the [DAROM paper](https://arxiv.org/html/2403.11852)
and [upstream code](https://github.com/amin-tabrizian/DAROM).

**Completed**

- [x] Add Antwerp network and easy/medium/hard one-hour OD routes with four
  vehicle types in `sumo_new/` (1,800 / 3,456 / 7,108 vehicles).
- [x] Place one `car_van` ego on O3 to D1 at SUMO time 120 s in each route,
  replacing one background O3-to-D1 car without changing total demand.
- [x] Start the episode when ego actually enters O3; cap it at 6,000 control
  steps after takeover, with arrival, collision, and emergency-stop outcomes.
- [x] Express positions of up to 30 neighbors in ego-forward/left local
  coordinates; use route distance for ego progress. Adapt lane-change bounds,
  merge recognition, reward gaps, and safety checks to the Antwerp observation
  layout. Relative speed now projects the neighbor velocity onto ego's forward
  axis before subtracting ego speed; regression cases cover parallel, crossing,
  and opposing traffic.
- [x] Select the nearest 30 vehicles within the observation radius before
  applying the existing left/forward spatial order. Hard traffic reached 37
  candidates in a full probe; a separate 2,501-step run verified the selected
  states at all 113 over-capacity steps without changing the 31×3 shape.
- [x] Connect the training entry to all three routes. Training disables the
  safety controller by default; `--safety` enables it. New checkpoints go to
  `models_experiments/`.
- [x] Separate evaluation from training: training saves numbered checkpoints
  every 100,000 steps without starting a second SUMO environment in the same
  process. Run `evaluate_sac_delay.py` separately for Antwerp outcome metrics;
  checkpoints are no longer labeled "best" before evaluation.
- [x] Carry the ego's merge status in each environment step. Independent
  evaluation retains the total `Max Steps` count and splits it into `No Merge
  Timeout` and `Merged Timeout`; training logs separate episode indicators for
  arrival, collision, emergency stopping, and both timeout types. A TraCI
  probe checked no-merge timeouts in all three densities and an Easy arrival
  after merging.
- [x] Retain the public evaluation script's outcome proxy: success is `Arrived`, collision
  is `Collision`, and reported no-merge rate counts **all** `Max Steps` episodes.
  The merged/unmerged timeout split remains diagnostic because the Antwerp route
  continues beyond the merge. Emergency stopping stays separate and is not
  silently folded into the paper's three rates; if it occurs, those rates need
  not sum to 100%. This raw proxy is reported separately from uninterrupted
  O3-to-D1 task success. `Arrived` includes arrivals after teleportation;
  `success_rate` excludes them and `raw_arrival_rate` preserves the raw proxy.
- [x] Adapt the safety controller's target-lane filter to Antwerp geometry.
  It now projects the selected lane centerline into ego's local frame for both
  lane changes and lane keeping, accounting for lateral movement and heading.
  In three-density TraCI probes this
  eliminated the observed misses of `E75_1` vehicles during the O3 merge;
  fixed-action comparisons reached a mainline lane with `--safety` in all
  three scenarios before ego lane-mode restoration and again under lane
  mode 0. The corrected delayed safety path predicts all captured neighbors
  before lane filtering and aligns them with the current ego frame. Regression
  cases cover lateral movement, rotation, changing neighbor order, packet age,
  padding, and out-of-order packets. These checks do not establish a trained
  policy's safety or constant-velocity prediction accuracy in real maneuvers.
- [x] Track ego teleport start/end events for the entire episode. Teleport
  buffers skip ego control and position queries. Arrival/merge bonuses are
  disabled after a teleport; progress reward is suppressed during teleport
  transitions and off-road steps. Raw arrivals, uninterrupted arrivals, and
  episodes with any ego teleport are logged separately.
- [x] Connect the five Table IV input ablations to Hard-scenario training and
  independent evaluation. All retain the GRU input shape and separate safety
  input; five 3-step training and evaluation smoke runs loaded their matching
  temporary checkpoints, and a masked-input safety-wrapper smoke run completed.
  A separate 102-step linear-predictor run exercised training beyond SAC's
  default initial exploration period. These runs do not measure convergence or
  task success.
- [x] Align the GRU action-history length with the wrapper's 20-action buffer.
  The upstream training script used `max_delay + 1`, which mismatches that
  buffer's length.
- [x] Give delay sampling an independent RNG keyed by the experiment seed and
  episode index. An explicit `reset(seed=S)` restarts episode 0; later resets
  without a seed advance the index. All five delay modes now replay for the
  same seed without consuming SUMO's or global NumPy's random stream.
- [x] Check the legacy 32 m/s ego guard against Antwerp speeds. With seed
  579071 and a continuous +5 acceleration command, ego reached D1 in all
  three scenarios under the previous automatic lane mode; its maximum speed
  was 14.36 m/s and the guard never fired. Recheck route completion under the
  current lane mode before using these runs as reachability evidence.
- [x] Check ego acceleration against its `car_van` type. All three routes set
  `accel=2.6` and `decel=4.5`; an Easy-scenario TraCI probe measured +2.6
  m/s² from a +5 command and −4.5 m/s² from a −5 command, without an
  emergency-stop outcome.
- [x] Restore the upstream ego-only lane-control setting. The original DAROM
  disables SUMO's autonomous ego lane changes with `laneChangeMode=0`; Antwerp
  now does the same while retaining the `car_van` type and speed mode 31. A
  same-seed TraCI probe showed no spontaneous lane changes under mode 0 and
  confirmed that an explicit policy lane request still works in the E75 merge
  zone. A new 3-step Hard training smoke run saved a checkpoint successfully.

**Retained choices and recorded differences**

- Detector `3849 / L4e` has an unverified exit mapping: its counts
  remain in the demand totals and its destinations are projected to D1/D2/D3.
  Revisit only if a verified detector-to-exit mapping becomes available.
- Keep the current 120 s ego departure, 6,000-step episode limit,
  13.89 m/s network speed limit, and Hard-scenario background queues until
  measurements justify a change. The objective is ego reaching D1, not
  removing congestion from the network.
- Keep the 32 m/s guard as an untriggered legacy condition in the tested runs.
  The ego's `car_van` type has a 45 m/s maximum and SUMO speed mode 31; its sampled
  speed factor of 1.034 makes the effective speed limit about 14.36 m/s on
  13.89 m/s lanes. One internal connector is limited to 12.51 m/s. Recheck
  the guard if road limits or speed mode change; this probe does not establish
  the speed behavior of a trained policy.
- Keep the four Antwerp vehicle types and their SUMO parameters as the traffic
  baseline. The original `f_1`/`f_2` overrides targeted particular lane flows;
  Antwerp flow IDs represent scenario/OD/type and do not match those checks.
  Do not assign the old overrides to Antwerp types or invent a driver-profile
  ratio. The current `SetTrafficBehavior` checks therefore have no effect on
  Antwerp background vehicles; a sampled `car_van` retained `tau=1.0`, type
  maximum speed 45 m/s, and speed mode 31. Define any driver-profile groups
  separately if a future experiment needs them.
- Upstream also sets ego `speedMode=96`, but Antwerp keeps `speedMode=31` and
  the `car_van` acceleration/deceleration values. Changing lane mode does not
  change these physical parameters; the upstream speed-mode difference remains
  deliberate and should be evaluated separately. SUMO lane mode 0 also skips
  gap-safety checks for explicit TraCI lane requests, so policy collisions and
  the optional safety controller must be assessed under this setting.
- The policy action range remains −5 to +5 m/s². SUMO limits realized ego
  acceleration to the `car_van` type values in the measured probe, while the
  existing jerk reward uses the commanded action. Keep this reward convention
  for upstream comparison; review it before changing model behavior.
- The current Antwerp GRU is a **documented deviation**, not an exact copy
  of upstream: the upstream encoder reads its
  newest-first action buffer without reversal and uses 128 output features;
  the current encoder reverses that buffer, scales its inputs, and outputs 256
  features. The separate MLP encoder still receives unscaled inputs. These
  differences are recorded without changing the current code. Follow upstream
  model behavior as the reference for further migration; discuss any new
  model-behavior change separately before implementing it.
- The distributed checkpoint under `models/` is not an Antwerp result; the
  observation meaning and GRU interface have changed.
- Unlike upstream's global NumPy delay draws, Antwerp uses a separate seeded
  delay stream for reproducibility; the five distribution rules are unchanged.

**Pending work**

- [ ] Measure the remaining constant-velocity prediction error during neighbor
  acceleration/turning and crossing-junction conflicts in complete episodes.
  Coordinate alignment does not provide a collision guarantee. The policy's
  `linear_predictor` ablation is still longitudinal-only in the captured frame.
- [ ] Evaluate the trained Antwerp policy with `--safety` before claiming a
  collision reduction. The fixed-action probes cover merge behavior, not
  complete policy episodes or all delayed-observation cases.
- [ ] Select the best Antwerp checkpoint using independent evaluation across
  the required scenarios and delay profiles; no checkpoint has been ranked yet.
- [ ] Build and train the five Hard-scenario Table IV variants under a stated
  common safety-controller policy. Report success, collision, no-merge, and
  return using the outcome proxy above; keep emergency stops and the timeout
  split visible beside the table rates. Generate the corresponding Fig. 5
  training curves. No full Antwerp model training or Table IV evaluation has
  been completed yet.
- [ ] Train and evaluate the `full` baseline and four comparison variants at a sufficient
  budget under ego lane mode 0; the pre-restoration pilot checkpoints used
  SUMO's automatic lane mode and cannot measure the revised policy task.
  A historical 10,000-step pilot and a fresh version 2 20,000-step `full` pilot
  have completed under lane mode 0. Neither establishes convergence or a
  Table IV result; see the staged validation for held-out version 2 outcomes.
- [ ] Assess route-aware lane guidance after the policy-controlled baseline.
  `E77_0` connects only to D3; `E82_0`–`E82_2` connect to
  D2, while `E82_3`–`E82_5` connect to D1. One pre-restoration training episode
  ended with an ego emergency stop on `E77_0`; its per-step lane actions were not
  recorded, so the source of that lane choice is unknown. Keep emergency stops
  separate from collision and timeout counts.
- [ ] Reassess the 6,000-step limit using complete non-teleported trajectories.
  Do not count merely entering D1 as an `Arrived` episode.
- [x] Run an initial scripted end-to-end controller under the same TraCI ego
  takeover settings as training. Easy and Medium completed without ego
  teleportation in both current- and delayed-observation probes; none of four
  Hard seeds did. Pure SUMO validates native driving behavior, while these
  probes validate the policy action interface and expose Hard-scenario limits.
- [x] Compare native SUMO and scripted ego trajectories on Hard seeds 579071
  and 579074, then extend only the observation limit for the queued 579074
  rule run. The fixed rule over-constrains the D1 approach to `E78_2`, while
  native SUMO can use other D1-connected lanes. Extending to 9,000 steps did
  not clear its `E78_2` queue. These diagnostics do not justify changing Hard
  demand or the training episode limit.
- [x] Add connectivity-derived D1 lane choices and record requested versus
  completed lane changes; rerun the fixed Hard seeds (version 2 stage 2).
  These cases did not activate a beneficial alternative lane choice.
- [ ] Address continuous lateral maneuver safety and inspect J107 conflicts
  separately. The rule has an information advantage and unresolved collisions;
  it is not a Table IV comparator.

**Validation so far:** unit tests cover ego selection, geometry, wrappers,
and encoder input handling. All three Antwerp scenarios reset at 120.1 s
with a 134-value augmented observation (93 vehicle values + 40 action-history
values + one delay value). A controlled delay-wrapper probe confirmed current
ego values, delayed neighbors, newest-first masked actions, and matching
observation-space dtype/shape. A five-step GRU/SAC prediction in Antwerp Easy
completed after reset. A SUMO probe reproduced the same 12 delay draws after
resetting twice with one seed and a different sequence after changing it.
These are interface checks, not evidence of training
convergence or task success.

The pilot and trajectory results below predate environment version 2. They
remain historical diagnostics; the corrected observation, safety input, and
reward contract requires new training before reporting model performance.

A Hard/uniform/no-safety pilot with seed 579071 trained each Table IV input
variant for 256 steps (156 SAC updates). One complete independent episode per
variant at seed 579072 ended in `Merged Timeout`; this is a pipeline check, not
a variant ranking. A separate `full` baseline trained from scratch for 10,000
steps (9,900 updates). Its three independent complete episodes at seeds
579072–579074 yielded two D1 arrivals and one merged timeout, with no ego
collision or emergency stop. Three episodes are insufficient for a Table IV
performance claim; pilot models and logs remain under `tmp/ai/`.
All of these checkpoints were trained before ego lane mode 0 was restored, so
their arrival rates are historical and must not be treated as current results.

Deterministic traces of the 10,000-step `full` checkpoint at seeds 579073 and
579074 show a parsed lane-change action of zero on every step. SUMO still
changed the ego's lane: the successful episode moved from `E82_1` to `E82_3`
and reached D1, while the timeout moved from `E78_1` to `E78_0` and then
entered `:J107_0_0`. There it advanced only 72 m over nearly 3,000 steps;
between steps 3,500 and 4,500 its stopped leader was a D1-bound car with a
near-zero gap. The immediate timeout mechanism was a downstream queue, and
the current pilot does not demonstrate learned lane-changing behavior.
With the same old checkpoint and seed 579073 under lane mode 0, the policy
again issued zero lane-change actions, SUMO made no within-edge lane changes,
and ego timed out on `E78_0`. This is a control-attribution check, not a fair
performance comparison across training conditions.

Under lane mode 0, fixed `+1` lane commands merged into the mainline in Easy,
Medium, and Hard with the safety controller both disabled and enabled (seed
579071, 800-step probe). With safety enabled, fixed lane-keeping did not merge
within 500 steps in any density. These probes check the merge-control path,
not D1 arrival or policy safety. A new Hard/uniform/no-safety `full` pilot
trained from scratch for 10,000 steps. Three independent complete episodes
at seeds 579072–579074 produced zero D1 arrivals, one ego collision, and two
merged timeouts, with no ego emergency stop. In the collision episode, the
model issued two consecutive `+1` lane commands on `E82` (steps 2322–2323);
SUMO reported an ego collision during lane change at step 2339. This small
pilot demonstrates a policy-control failure, not a stable success rate or a
comparison with the earlier automatically changing-lane checkpoint. The
checkpoint, probe logs, and trace remain under `tmp/ai/lane-control-audit/`.

Full traces of the two merged timeouts separate their causes. At seed 579073,
ego remained on `E78_0` from step 2627 to the 6,000-step limit; it advanced
about 64 m in the final 1,000 steps and spent 82% of those steps below 1 m/s.
The model issued no lane-change commands on E78, so it had neither reached
E82 nor moved toward an E82 lane that connects to D1. At seed 579074, ego
spent about 3,100 steps on `:J107_0_0`, with speed below 1 m/s for 97% of
that interval. SUMO teleported it at simulation time 699.3 s from the jammed
connector to `E82_3`; this was not a policy-controlled lane change. It then
entered D1, but timed out 106 m before the D1 endpoint at simulation time
720 s. The model issued no lane-change commands on E78 or E82 in this run.
The observed delays include a downstream queue; simply extending max steps
would not establish uninterrupted ego completion or learned D1 lane choice.

A separate pure-SUMO reference used the same network, scenario routes,
0.1 s step, lateral resolution, and mapped SUMO seeds without TraCI or a model.
Ego departed at 120 s; each run was observed until simulation time 900 s.

| Scenario | Environment seed | Ego result | Ego teleported? |
|---|---:|---|---|
| Easy | 579071 | Arrived at 303.0 s | No |
| Medium | 579071 | Arrived at 320.9 s | No |
| Hard | 579071 | Arrived at 364.8 s | No |
| Hard | 579072 | Still unfinished at 900 s | No |
| Hard | 579073 | Arrived at 653.6 s | Yes, at 622.9 s |
| Hard | 579074 | Arrived at 318.9 s | No |

Thus the O3-to-D1 network is traversable with SUMO's autonomous driving in
multiple demand realizations, but Hard can also queue long enough to prevent
completion or require a teleport. These runs do not establish convergence of
the RL policy or feasibility under TraCI's ego-only lane mode 0. Outputs are
kept under `tmp/ai/pure-sumo-baseline/`.

A temporary scripted controller then reused `evaluate_sac_delay.make_env` with
the same scenario, seed, 120.1 s ego takeover, 0.1 s action step, 6,000-step
limit, `car_van` type, ego lane mode 0, speed mode 31, and default safety
controller disabled. It requested one lane at a time through the existing
action interface, checked neighbor gaps, and kept SUMO's collision and
teleport behavior unchanged. Both observation variants used the same static
O3-to-D1 target-lane map. The current-observation variant read the environment's
31-vehicle state; the delayed variant read the uniform-delay policy observation
from `DelayWrapper` (up to 20 steps). Ego state remained current in both.

| Scenario | Seed | Current observation | Delayed observation |
|---|---:|---|---|
| Easy | 579071 | Arrived, 199.3 s, no teleport | Arrived, 215.0 s, no teleport |
| Medium | 579071 | Arrived, 217.0 s, no teleport | Arrived, 217.0 s, no teleport |
| Hard | 579071 | Timeout at J107 connector | Timeout on D1 after ego teleport |
| Hard | 579072 | Timeout on E78_2 | Timeout on E82_0 |
| Hard | 579073 | Timeout on E78_2 | Timeout on E77_3 |
| Hard | 579074 | Timeout on E78_2 | Ego collision at J107 connector |

Times are simulated seconds after takeover. `Arrived` counts as success only
without an ego teleport. The rule also reads current edge/lane position and
an explicit lane-connectivity map, information not explicitly supplied to the
RL observation; these are engineering diagnostics, not a fair Table IV
comparison or evidence that delay compensation helps. The first exact-TraCI-
neighbor version of the rule also completed Easy and Medium but no Hard seed
without teleportation. The temporary script and per-seed logs, including
lane requests, transitions, final route distance, and stop counts, remain
under `tmp/ai/rule-baseline/`.

Hard trajectory follow-up used the same SUMO seeds and O3-to-D1 ego route.
Passive TraCI reproduced uninterrupted native arrivals (579071 at 364.9 s,
579074 at 319.0 s; one 0.1 s reporting step later than the batch SUMO run).
The current-observation rule was rerun with per-step lane, position, speed,
lane-change request, and queue traces:

| Hard seed | Key lane difference | Rule outcome and queue evidence |
|---|---|---|
| 579071 | Native ego changed `E75_0` to `E75_1` at 148.7 s. The rule stayed on `E75_0`; its three requests at 151.2–151.4 s did not produce a lane change. | Rule stopped 85.2 s on `E75_0`, then 187.4 s on `E78_1`; timeout on the J107 connector. Native ego had no stopped steps on either edge and reached D1. |
| 579074 | Native ego used `E76_4` → `E77_5` → `E78_4`; the rule selected `E76_2` → `E77_3` → `E78_2`. | Rule stopped 396.0 s on `E78_2` and timed out there; native ego had no stopped steps on `E78_4` and reached D1. |

The network permits `E78_2` → `E82_3` → D1, `E78_3` → `E82_4` → D1,
and `E78_4` → `E82_5` → D1. Thus the rule's single target lane is an
unnecessary restriction, not a demonstrated network or route error. The
579071 E75 requests were issued but not executed; these traces do not
identify SUMO's exact rejection reason. A separate delayed-observation 579074
run collided with `hard_O1_D2_long_vehicle.15` at J107 at 431.2 s. Its ego
was on `:J107_0_3` at 7.18 m/s and requested no lane change immediately
before the collision; the collision warning named the conflicting
`:J107_0_2` connector. That single event does not isolate an observation-
delay effect.

For a one-factor queue check, only the 579074 current-observation rule's
post-takeover limit was raised from 6,000 to 9,000 steps (600 to 900 simulated
seconds). The original Hard route demand, seed, rule, and SUMO settings were
unchanged. Ego still timed out on `E78_2` at the same 1,982.445 m route
distance, with no ego teleport; it remained stopped at the same position
through the additional 300 simulated seconds. This does not support extending
the production episode limit for that failure. All trace scripts, CSVs, and
logs remain under `tmp/ai/rule-baseline/` and are not project inputs.

The other two Hard rule seeds, 579072 and 579073, also timed out on `E78_2`,
but their native runs respectively did not finish by 900 s and arrived only
after an ego teleport. They are useful stress cases, not clean examples of a
controller-only failure. A short 579071 TraCI inspection showed the first
`E75_0` lane-change request was blocked by a left leader; extending only the
request duration from 0 to 1 s still left ego on `E75_0` at 156 s. In a
separate one-factor diagnostic, reducing the temporary rule's observed-gap
headway from 1.8 to 1.0 s made the E75 change occur, but ego later reached
`E82_1` and SUMO teleported it for a wrong lane at 660.3 s before `Arrived`.
This is not uninterrupted success and does not establish 1.0 s as a safe
setting. No production gap, lane-change, demand, or episode-limit parameter
was changed. Calibrate them only after the rule checks all D1-reachable lanes
and distinguishes failed lane-change commands from unavailable gaps.

**Key arguments:**

| Argument | Options | Description |
|---|---|---|
| `--delay_mode` | `uniform`, `bimodal`, `bursty`, `triangular`, `exponential` | Delay distribution during training |
| `--scenario` | `easy`, `medium`, `hard` | Antwerp traffic scenario; defaults to `medium` |
| `--safety` | flag | Enable the safety controller (disabled by default) |
| `--encoder_mode` | `GRU`, `MLP`, `Transformer` | Delay-Aware Encoder architecture |
| `--ablation` | `full`, `delayed_state_only`, `delayed_state_delay`, `delayed_state_action`, `linear_predictor` | Table IV input variant; non-full variants require GRU |
| `--run_name` | string | WandB run name |
| `--save_dir` | path | Directory to save model checkpoints |
| `--total_timesteps` | integer | Training budget; defaults to the paper's 7,000,000 steps |
| `--checkpoint_freq` | positive integer | Numbered checkpoint interval; defaults to 100,000 steps |
| `--torch_threads` | positive integer | Optional PyTorch CPU thread limit for this process |
| `--device` | `auto`, `cpu`, `cuda`, etc. | SB3 model device |
| `--log_dir` | path | TensorBoard/WandB output directory; defaults to `runs` |
| `--episode_log` | path | Optional new JSONL file for episode metrics and progress every 1,000 steps |

## Evaluation

Evaluate a newly trained Antwerp checkpoint in a separate process:

```bash
python evaluate_sac_delay.py \
    models_experiments/antwerp-hard/GRU-hard-uniform-delay/GRU-hard-uniform-delay_100000_steps.zip \
    --scenario hard --delay-mode uniform --episodes 51 --no-safety
```

Select `--safety` or `--no-safety` independently of training. Omission retains
the existing safety-off default. Pass the same `--ablation` used to train the
checkpoint. Logs, optional JSON reports, and model diagnostic summaries record
`training_safety` and `evaluation_safety` separately; missing historical training
metadata is reported as null/unknown rather than assumed false.
The script reports raw D1 arrival, uninterrupted D1 arrival, arrival after an
ego teleport, any-teleport episodes, collision, timeout, emergency stopping,
and mean return. `success_rate` means uninterrupted arrival; the clearly labeled
historical upstream arrival proxy uses `raw_arrival_rate`. Training retains `train/arrival`
for raw arrivals and adds `train/success`, `train/ego_teleported`, and
`train/teleported_arrival`. Each episode gets a distinct seed for SUMO and delay sampling, so the
same checkpoint and evaluation seed reproduce the episode delay sequences.

For a paired safety check, run the following command twice with the **same
checkpoint and seeds**, changing `--no-safety` to `--safety` and using a different
output filename. This is an Antwerp evaluation with Table IV-style metrics;
it does not restore the original scene or resolve the protocol gaps above:

```bash
python evaluate_sac_delay.py models_experiments/antwerp-step3/full-v2-hard-20k.zip \
    --scenario hard --delay-mode uniform --ablation full --no-safety \
    --episodes 500 --seeds 579100 580100 581100 \
    --output tmp/ai/reproduction-protocol/full-safety-off-formal.json
```

`--episodes` is per evaluation run. Each run uses consecutive episode seeds;
overlapping ranges are rejected. The example schedules 1,500 episodes; it is a
command template, not an experiment already performed. `--seed` remains the
single-run option and cannot be combined with `--seeds`. Evaluation defaults to
CPU with one PyTorch thread (`--device` / `--torch-threads` can override this).
`--output` creates a new report and refuses to overwrite an existing file.
Reports contain checkpoint SHA-256, training seed/step count, both safety flags,
each episode's seed/outcome/return, per-run metrics and cross-run summaries.
Each episode's `reset_info` additionally records the task/scenario, actual SUMO
version and seed, launch command, network/route SHA-256, ego route and actual
speed/lane-change modes, step length and takeover-based episode limit. Rule and
model diagnostic summaries retain the same reset metadata. These fields are
additive; the observation/reward contract remains version 2. In particular,
evaluation `--no-safety` disables the project wrapper, not SUMO's built-in
speed-mode-31 constraints. It is not equivalent to the original ego setup.
The reported `std` uses `ddof=1`; `ci95_half_width` uses the upstream normal
approximation and is not a guarantee of coverage with only a few runs. Neither
quantity should be confused with independently trained-model variability.
The legacy `no_merge_rate` still counts all `Max Steps`; its merged/unmerged
counts and separate emergency-stop and teleport results remain visible.

Protocol-correction validation used the existing 20k checkpoint (training safety
off) and the same two Hard seeds, without retraining:

| Seed | Evaluation safety off | Evaluation safety on |
|---:|---|---|
| 579081 | Collision, 63.1 s | Merged timeout, 600 s |
| 579084 | Collision, 359.3 s | Teleported arrival, 580.0 s; success false |

Both settings had zero uninterrupted successes. Safety-off steps and returns
matched the earlier diagnostic runs. All checkpoint hashes matched before and
after evaluation. These four episodes validate the independent safety switch
and accounting, not convergence, a safety guarantee, or a Table IV result.
All 25 regression tests passed. An additional single-run CLI replay verified
JSON export and null single-run uncertainty. Artifacts remain in
`tmp/ai/reproduction-protocol/`. The paired jobs started before the final
uncertainty correction; `paired-current.json` recomputes only their aggregates
using the final sample-std/CI implementation and links the unmodified source
reports. All recorded episode data are preserved.

Numbered checkpoints are saved every 100,000 training steps by default (set
`--checkpoint_freq` for a short pilot) under
`models_experiments/<run_name>/`. The latest model is also saved as
`models_experiments/<run_name>.zip` at training completion. Neither is marked
"best" until independent evaluation compares them.

## Version 2 staged validation

Step 1 (2026-10-12) completed a 12-episode matrix with the corrected environment:
three densities, seeds 579071/579074, and safety off/on. All runs used the
production 6,000-step limit, traffic demand, speed mode 31 and ego lane mode 0.
The diagnostic rule reads current neighbor observations and an explicit fixed
target-lane map (`E75:1`, `E76:2`, `E77:3`, `E78:2`, `E82:3`). Its headway is
1.8 s; the optional safety wrapper receives uniform-delay packets as usual.
This is a new, reproducible engineering rule, not the earlier temporary script
or a fair RL comparator. It is not a trained model.

| Scenario | Seed | Safety off | Safety on |
|---|---:|---|---|
| Easy | 579071 | Arrived, 188.8 s | Arrived, 188.9 s |
| Easy | 579074 | Arrived, 189.4 s | Arrived, 189.4 s |
| Medium | 579071 | Arrived, 222.0 s | Arrived, 222.0 s |
| Medium | 579074 | Arrived, 205.2 s | Arrived, 204.6 s |
| Hard | 579071 | Collision, 63.8 s | Collision, 119.1 s |
| Hard | 579074 | Collision, 54.3 s | Collision, 111.2 s |

Times are after ego takeover. None of these 12 episodes teleported. All ended
through the environment's terminal API, with finite observations/rewards and
consistent task-success flags. Easy/Medium had eight uninterrupted arrivals;
Hard had four collisions, including both safety-enabled runs. Completing this
validation establishes episode accounting, not collision-free control or a
success-rate estimate. The 11 regression and 3 SUMO integration tests also passed.

To reproduce (choose a new output directory; existing case folders are refused):

```bash
python -m src.diagnostics --scenarios easy medium hard \
    --seeds 579071 579074 --safety-modes off on \
    --output-dir tmp/ai/antwerp-step1-repeat
```

Each case writes `trace.csv` and `summary.json`, including desired and applied
lane commands, actual within-edge lane transitions, leader gaps/speeds,
per-lane stopped time, teleport events and outcomes. Repeated lane commands
are not counted as successful lane changes. Outputs from this run remain in
`tmp/ai/antwerp-step1/` and are excluded from Git. Formal input ablations are
outside this staged validation.

Step 2 completed 12 additional Hard diagnostics on SUMO 1.18.0. Four passive
native-SUMO references retain the mapped scenario seed, speed mode and 600 s
budget, but restore native ego lane mode 1621 and issue no policy actions:

| Seed | Native outcome | Final lane | Ego teleport |
|---:|---|---|---|
| 579071 | Timeout, 600 s | `:J107_0_2` | No |
| 579072 | Timeout, 600 s | `E78_1` | No |
| 579073 | Timeout, 600 s | `:J107_0_1` | No |
| 579074 | Arrived, 294.6 s | `D1_1` | No |

These local runs do not reproduce every historical native result above; the
historical simulator build and raw artifacts are unavailable in this clone.
Treat these fresh results separately. The three timeout cases spent 93.7%,
97.3%, and 97.7% of their last 1,000 steps below 1 m/s, respectively.

Two controlled rule variants each ran seeds 579071/579074 with safety off/on.
First, only longitudinal control changed to constant +2.6 commands, leaving
SUMO speed mode 31 to enforce longitudinal limits. Second, that variant also
used a connectivity-derived set of D1 lanes instead of one fixed target.
Lane paths without further within-edge changes include `E78_2/3/4` and
`E82_3/4/5`; an adjacent reachable lane can be chosen when a stopped queue has
less observed forward room. Gap/headway, traffic and episode settings were
unchanged. Both variants produced the same physical trajectories and outcomes
on these four cases, so they do not demonstrate a benefit from route guidance:

| Seed | Safety | Outcome for both variants |
|---:|---|---|
| 579071 | Off | Collision at E75, 52.4 s |
| 579071 | On | Collision at E76, 119.0 s |
| 579074 | Off | Timeout on `E78_1`, 600 s |
| 579074 | On | Raw arrival at 523.6 s **after teleport**, task success false |

The evidence separates three mechanisms:

- The original four rule collisions were SUMO `stage=laneChange` events.
  They occurred 18–40 steps after the last nonzero applied lane request.
  New traces show ego continuing lateral motion while subsequent commands are
  zero. A zero policy direction sends no TraCI lane command and therefore does
  not actively cancel the ongoing maneuver. Single-step gap checks and the
  optional safety filter have not established safe completion of that maneuver.
- Native 579074 changed `E75_0` to `E75_1` at 146.2 s simulation time; the
  constant-acceleration rule changed at 276.7 s after 107.5 stopped seconds on
  E75. The latter reached E78 much later, then queued behind a D2-bound vehicle.
  During its final 1,000 steps it moved only 31.10 m and was below 1 m/s for
  89.5% of steps. There were 229 applied lane requests in that interval but no
  successful move out of `E78_1`. This is not simply a missing request.
- With safety enabled, seed 579074 stopped 300.2 s on wrong-lane `E82_1`, then
  teleported to D1 at simulation time 625.0 s. The teleport-step reward was
  -0.0883 and final arrival reward -0.03; task success remained false. This is
  a naturally occurring full-episode check of the version 2 accounting fix.

No production lane-change, reward, road, demand or time-limit parameters were
changed in this diagnostic stage. Route-aware multi-lane selection alone did
not resolve the observed failures, and relaxing a gap threshold is not justified
by these results. Continuous lateral maneuver safety and queue-aware route
choices remain open. Fresh `full` training is an exploratory policy baseline,
not a claim that these control problems have been solved.

```bash
python -m src.diagnostics --controller native --scenarios hard \
    --seeds 579071 579072 579073 579074 \
    --output-dir tmp/ai/antwerp-step2/native-repeat
python -m src.diagnostics --scenarios hard --longitudinal sumo \
    --lane-strategy reachable --output-dir tmp/ai/antwerp-step2/reachable-repeat
```

Omit `--lane-strategy reachable` for the longitudinal-only comparison. Outputs
remain under `tmp/ai/antwerp-step2/`. All 12 regression tests, including the
D1 lane-connectivity check, passed after this stage.

Step 3 completed a fresh **20,000-step `full` GRU/SAC pilot** in Hard with
uniform delays and safety disabled, training seed 579071. The architecture,
learning rate (3e-5), batch size (512), rewards, traffic and episode limits
were unchanged from version 2. Training used CUDA on an RTX 4090 and one
PyTorch CPU thread, with WandB disabled. It took 1,094 s and performed 19,900
gradient updates. Five training episodes completed (two collisions and three
merged timeouts); none succeeded or teleported. The final 89 steps belong to
an unfinished sixth episode and are not counted as an outcome.

The 10,000-step numbered checkpoint and final 20,000-step model were evaluated
in separate processes, deterministically on CPU, using held-out seeds
579081–579084. The policy received only its 134-value augmented observation,
without the diagnostic rule's live road map or current-neighbor information.
All eight Hard evaluations completed without simulator/runtime errors:

| Hard seed | 10,000-step checkpoint | Final 20,000-step model |
|---:|---|---|
| 579081 | Collision, 60.0 s, E75 | Collision, 63.1 s, E75 |
| 579082 | Collision, 62.4 s, E75 | Merged timeout, 600 s, E78 |
| 579083 | Merged timeout, 600 s, E78 | Collision, 544.2 s, E82 |
| 579084 | Collision, 61.9 s, E75 | Collision, 359.3 s, E82 |

Each checkpoint had **0/4 uninterrupted arrivals, 3 collisions and 1 merged
timeout**, with no ego teleport or emergency-stop termination. Mean returns
were -324.258 and -550.869, respectively. These four paired cases are pilot
diagnostics, not stable rate estimates or evidence of improvement. No "best"
checkpoint has been selected. The final model made 0/2/4/2 within-edge lane
changes for those four seeds, so some maneuvers executed, but none completed
the task. Seed 579082 made no route progress during its last 1,000 steps and
remained on E78. Additional final-model checks at seed 579081 collided in
Easy (345.3 s, E82, four lane changes) and Medium (434.0 s, E82, three lane
changes), also without teleports.

Both checkpoints loaded with matching version/ablation and safety off. At that
time the evaluator also required training and evaluation safety to match; that
restriction was later removed as described in the reproduction protocol.
All final policy parameters and recorded actor/critic/entropy losses were
finite, and all 13 regression tests passed. This verifies training and
evaluation execution; it does not establish policy convergence or safety.
Continuous lane-change safety, route completion, larger-budget baseline
training and broader evaluation remain open. No formal five-variant ablation
or stage 4 experiment was run.

The final model is retained locally at
`models_experiments/antwerp-step3/full-v2-hard-20k.zip`; numbered 10k/20k
checkpoints are in the sibling `full-v2-hard-20k/` directory. Models are ignored
by Git. Logs, TensorBoard events and all per-episode evaluation CSV/JSON files
are retained in `tmp/ai/antwerp-step3/`. Reproduction commands (use new run,
episode-log and output paths to preserve existing artifacts):

```bash
python train_sac_delay.py --scenario hard --seed 579071 \
    --encoder_mode GRU --ablation full --delay_mode uniform \
    --total_timesteps 20000 --checkpoint_freq 10000 \
    --torch_threads 1 --device cuda --wandb_mode disabled \
    --run_name full-v2-hard-20k-repeat --save_dir models_experiments/antwerp-step3 \
    --log_dir tmp/ai/antwerp-step3/tensorboard-repeat \
    --episode_log tmp/ai/antwerp-step3/training-repeat.jsonl
python -m src.diagnostics --controller model \
    --checkpoint models_experiments/antwerp-step3/full-v2-hard-20k.zip \
    --scenarios hard --seeds 579081 579082 579083 579084 --safety-modes off \
    --output-dir tmp/ai/antwerp-step3/eval-repeat
```

## Hard reachability and learning audit

The follow-up diagnostic adds `--controller managed`. This reference uses the
same `Merging.step` acceleration/lane-direction actions, lane-change mode 0,
speed mode 31, traffic, network and 6,000-step limit as training. The modes are
checked at reset and every on-road step. It does not move vehicles, enable
automatic lane changing, modify routes, or shorten teleport thresholds.

It is a **privileged feasibility reference**, not a learned policy or a fair
delay-ablation comparator. It reads live SUMO neighbors and car-following
parameters, the D1 lane-connectivity graph and current lateral motion.
`getNeighbors` reports bumper gaps with minimum-gap offsets (including negative
overlaps); `getSecureGap` supplies the car-following gap requirement. See the
[SUMO vehicle retrieval documentation](https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html).
The controller adds a three-second closing-speed allowance, requests one
adjacent lane at a time, and holds that target until the vehicle is entirely
inside it with lateral speed below 0.1 m/s. During an active maneuver it limits
acceleration using the target leader's follow speed. These checks are heuristics,
not a proof of collision avoidance.

An exploratory implementation incorrectly required exact lane-center settling.
SUMO's sublane model settled ego around +/-0.6995 m from center while its whole
1.8 m body was inside a 3.2 m lane, preventing further requests. The corrected
test uses body containment, not exact centering; a regression test covers it.
The four exploratory episodes are retained in `committed-dev/` and
`reactive-dev/` for transparency (one timeout and three teleport-assisted raw
arrivals, no successes). They do not evaluate the final controller.

With that correction, development seeds 579071 and 579074 both reached D1
without collisions or teleports in 380.4 s and 228.3 s. The first seed was rerun
in a separate process: all 3,804 trace rows matched exactly, including controls,
states, delays and rewards. This establishes a reproducible successful path
under the training actuator constraints, not general Hard solvability or
learnability from the policy's restricted observation.

The controller was then frozen (`committed`, `passive`, safety off) for four
seeds not used in this controller's development. These are the same seeds
used in the earlier model pilot, permitting outcome comparison without
retraining that model:

| Hard seed | Managed reference | Terminal area | Observed failure evidence |
|---:|---|---|---|
| 579081 | Timeout, 600 s | `:J107_0_1` | Last 1,000 steps: 90.2% below 1 m/s, 26.80 m progress |
| 579082 | Collision, 53.4 s | E75 | Started lane change with 9.27 m remaining; still moving laterally at impact |
| 579083 | Timeout, 600 s | `E78_2` | D1-connected lane; last 1,000 steps entirely below 1 m/s, 7.94 m progress |
| 579084 | Collision, 233.2 s | E82 | Started lane change with 11.56 m remaining; still moving laterally at impact |

There were **0/4 independent successes, two collisions, two timeouts, and no
ego teleports**. The development successes must not be pooled with this set
as an unbiased success-rate estimate. The late lane changes reveal an unhandled
interaction between maneuver duration, lane-end braking and rear traffic;
bumper gaps at initiation did not establish safety throughout the maneuver.
The E78 timeout shows that selecting a statically D1-connected lane does not
guarantee traffic can advance. The logs identify queue locations and stopped
leaders, but do not prove the root cause of the junction congestion or that
those seeds are impossible for every controller. This reference is not ready
to replace the production safety controller.

The observation/reward audit also found two reasons to avoid simply extending
the previous training budget:

- The 134-value policy input provides distance traveled, lane index, speed,
  delayed relative neighbor positions/velocities, action history and delay.
  It does not explicitly expose D1 lane connections, distance to a required
  lane transition, lateral position/speed, or vehicle lengths. The GRU encodes
  a window of action history; its hidden state is not persisted between policy
  calls as a full trajectory memory. Distance may help infer road position on
  this fixed route, but information sufficiency has not been demonstrated.
- The saved 20k pilot has `gamma=0.99` at 0.1 s per step. The discount multipliers
  for rewards 10, 60 and 200 seconds away are approximately 0.366, 0.002405 and
  1.86e-9. Thus the terminal arrival bonus contributes very little to early
  decisions; dense progress and merge rewards still provide learning signal.
  This arithmetic identifies a temporal-credit risk, not proof that discount
  is the sole cause of failure. No discount/reward/model-input changes or new
  training were made in this audit.

Diagnostic CSVs now include lateral speed, remaining lane distance, D1 lane
reachability, maneuver phase and gap-blocking reason. JSON summaries include
phase durations, control modes, final pre-terminal state, and stopped fraction
and progress over the last 1,000 steps. A reachable lane flag refers to static
connectivity without further within-edge changes, not an open traffic path.
The opt-in `--gap-control yield` variant reduces speed to create an insertion
opportunity; it retains the same gap guard. Both development seeds timed out
with this variant: 579071 on D1-connected `E78_2`, and 579074 on `E77_3` after
an earlier ego teleport from E75. It did not improve these cases and is not
the default. `--maneuver-mode reactive` disables
the latched target and is exploratory; it is not the validated reference.

```bash
python -m src.diagnostics --controller managed --scenarios hard \
    --seeds 579071 579074 579081 579082 579083 579084 --safety-modes off \
    --output-dir tmp/ai/hard-reachability/repeat
```

The audit completed 13 full episodes: four exploratory centering checks, two
corrected development runs, four independent runs, one exact replay, and two
yield comparisons. All 17 regression tests passed, including overlapping/fast
followers, lane-index crossing before maneuver completion, off-center settling,
edge/teleport target reset, and yielding without relaxing the gap guard.
All audit artifacts remain local under `tmp/ai/hard-reachability/`; existing
output directories are not overwritten. No new models or dependencies were
created, and the production observation/reward/checkpoint contract remains
version 2.

## Regression validation

Use an existing Python environment with `requirements.txt` installed and
`SUMO_HOME` configured. These tests use standard-library `unittest` and add
no dependencies:

```bash
python -m unittest discover -s tests -p 'test_antwerp_contract.py' -v
python -m tests.test_sumo_integration -v
```

The second command requires SUMO and local TraCI sockets. Ordinary unittest
discovery skips those integration tests. The version 2 correction passed all
11 regression tests and 3 integration tests on SUMO 1.18.0. Coverage includes
five policy input ablations sharing the same delayed safety data, lateral and
heading changes, stale/out-of-order packets, buffer reinsertion, separate
arrival metrics, legacy-checkpoint rejection, and 125 control steps with
safety enabled in each of Easy/Medium/Hard.

The integration arrival/teleport fixtures reposition ego and temporarily
shorten the teleport timeout only inside the tests. They reproduced an actual
wrong-lane teleport, verified loss of the arrival bonus and task-success flag,
and verified clean reset. They do not measure end-to-end driving success;
production routes, demand, teleport settings, and episode limits are unchanged.
The buffer handling follows SUMO's documented
[teleport lifecycle](https://sumo.dlr.de/docs/Simulation/Why_Vehicles_are_teleporting.html).
Training/evaluation under version 2 and full-episode delayed-safety performance
remain to be measured. Constant-velocity prediction cannot account for unseen
neighbor acceleration or turns.

## Pretrained Model

A pretrained DAROM-GRU checkpoint is provided in `models/GRU-uniform-delay/`. The model was trained exclusively on the uniform delay distribution and evaluated zero-shot on all five distributions.

These are historical upstream checkpoints, not Antwerp version 2 models.
The Antwerp evaluation entry point requires a freshly trained version 2 model.

## Results

Historical upstream results on Hard traffic (1,374–1,490 vph/lane), with delays
up to 2.0 s. These tables are not measurements from the Antwerp migration;
uncertainty values are reproduced as reported by upstream.

| Method | Success (%) | Collision (%) | Avg. Return |
|---|---|---|---|
| MPC | 83.10 ± 1.47 | 16.70 ± 1.47 | — |
| DRL-ORMOC | 90.09 ± 0.86 | 9.78 ± 0.81 | 138.90 ± 0.20 |
| No Encoder | 97.41 ± 0.68 | 0.13 ± 0.13 | 161.15 ± 2.08 |
| **DAROM-GRU** | **99.80 ± 0.23** | **0.00 ± 0.00** | **179.68 ± 0.51** |

Generalization across delay distributions (trained on uniform only):

| Delay Profile | Success (%) | Collision (%) | Avg. Return |
|---|---|---|---|
| Uniform (train) | 99.80 ± 0.23 | **0.00 ± 0.00** | 179.68 ± 0.51 |
| Bimodal | **100.00 ± 0.00** | **0.00 ± 0.00** | 180.03 ± 0.06 |
| Bursty | **100.00 ± 0.00** | **0.00 ± 0.00** | 180.01 ± 0.03 |
| Triangular | **100.00 ± 0.00** | **0.00 ± 0.00** | 180.01 ± 0.07 |
| Exponential | **100.00 ± 0.00** | **0.00 ± 0.00** | **180.04 ± 0.04** |

## Citation

If you use this code, please cite:

```bibtex
@misc{tabrizian2024darom,
  title={Delay-Aware Reinforcement Learning for Highway On-Ramp Merging under Stochastic Communication Latency},
  author={Tabrizian, Amin and Huang, Zhitong and Aziz, Arsyi and Wei, Peng},
  year={2024},
  eprint={2403.11852},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  doi={10.48550/arXiv.2403.11852},
  url={https://arxiv.org/abs/2403.11852}
}
```

## Acknowledgements

This work is supported by the National Science Foundation Award #2229885. Traffic demand and vehicle speed data are derived from the [NGSIM US Highway 101 dataset](https://www.fhwa.dot.gov/publications/research/operations/07030/).
