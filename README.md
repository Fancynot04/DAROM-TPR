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

**Speed and demand provenance audit (2026-10-13).** The 13.89 m/s (50 km/h)
road limit already exists in the first repository commit `9a59395`; no evidence
was found that this uniform limit was calibrated from observed speeds or is
the real road's legal limit. It must not be described as an empirical speed
profile. The pinned original DAROM implementation caps ego at 32 m/s, but that
is also a configuration limit, not a measured mean speed.

Recounting the raw archive's 2022-09-05 11:30–12:30 entrance observations gives:

| Mapped entrance | Detectors | Observed vehicles in the hour | Mean valid crossing speed (km/h) |
|---|---|---:|---:|
| O1 | 467, 468, 469 | 3,771 | 40.21 |
| O2 | 471, 472 | 2,482 | 46.75 |
| O3 | 464 | 816 | 66.07 |
| Total | Six entrance detectors | **7,069** | — |

These are **flows, not spatial densities in vehicles/km**. No measured
corridor-wide spatial density follows from the counts alone. The current
Hard demand (7,108/h) is about 0.55% above this observed entrance total;
Easy/Medium are scaled experimental demands (1,800/3,456 per hour), not three
separately observed traffic-density windows. Entrance and downstream counts
must not be added together because vehicles cross multiple detectors.
The current routes distribute demand over an hour rather than replaying all
original passage timestamps.

Speed means use 3,747 / 2,454 / 808 valid samples for O1/O2/O3; nonpositive,
nonfinite and provisional missing-code 254 speeds are excluded, while their
vehicles remain in demand totals. The local source-paper/geometry audit
supports km/h for this field, but does not establish a statutory limit. Raw
source: `/home/cyl/datasets/New_Data.zip`, member
`New_Data/20220905/20220905.txt`, SHA-256
`6a7c01cba487e75326112bbac1391b28a66598427557c8a4fe3d36157aeda7e0`.
Time filtering preserves the supplied Epoch clock without a timezone shift.
The local aggregate is retained in
`tmp/ai/safety-bumper-gap/speed-demand-audit.json`. Speed-distribution and
background-flow calibration remain necessary before changing limits or
claiming equivalent traffic difficulty; this audit does not change speeds.

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

### Current Table III motion evaluation (2026-10-13)

The background group `AT-T3-MOTION-20261013` supplements the running baseline
with the physical metrics implemented in isolated `table3-metrics` commit
`7d1710f`. It evaluates Easy/Medium **1,000,000-step** checkpoints and the
currently available Hard **750,000-step** checkpoint, fixed at launch. Each
checkpoint uses development seeds 680001–680020 with safety off/on: **120
episodes**, serial CPU evaluation with one PyTorch thread. This is one training
seed per density, not a multi-training-seed study or an equal-budget comparison.

Environment, scene and original safety-controller files were checked against
the live baseline before launch. This group does **not** apply the separate
bumper-clearance correction or modify the live training/evaluation sources.
Existing matching archived episodes are checked for identical checkpoint hash,
scene fingerprints, outcome, success, teleport flag, step count and return.
The batch completed all 120 episodes at 23:01 Asia/Shanghai on 2026-10-13.
All 80 Easy/Medium episodes matched their archived evaluations exactly; all
120 episodes have valid speed and jerk statistics.

In addition to outcomes and return, reports include mean physical speed (m/s),
signed jerk (m/s³), absolute jerk and RMS jerk. Motion statistics weight episodes
equally and include all outcomes; teleport gaps are excluded from derivatives.
Success requires arrival without ego teleport. Partial groups must retain their
actual sample count and must not be presented as completed 20-episode results.

The temporary supervisor, launch/checkpoint/source manifest, per-episode JSON
and logs, `progress.jsonl`, and live aggregate `status.json` are retained under
`tmp/ai/table3-metrics/latest-evaluation/`, already ignored by Git. The supervisor
stops and records an error if evaluation fails, source fingerprints change, or
an archived parity check fails. No new training is launched by this group.

On 2026-10-14, the follow-up group `AT-T3-H1M-MOTION-20261014` supplements
the now-completed Hard **1,000,000-step** model using the same 20 development
seeds, safety off/on, original controller and isolated evaluator: **40 episodes**
on a serial CPU worker. Every episode must match its archived 1M evaluation
before contributing to the aggregate. Its logs, manifest and live `status.json`
are retained separately in `tmp/ai/table3-metrics/hard1m-evaluation/`.
After all 40 episodes pass, the runner combines these two Hard rows with the
previous four Easy/Medium 1M rows into `table3-1m.json` and `table3-1m.csv` in
that directory. It verifies matching evaluator fingerprints and seeds before
combining results; the 750k Hard metrics are not substituted for the 1M model.

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

### Hard success/failure trajectory comparison (2026-10-12)

This follow-up compares the **corrected** development traces in
`tmp/ai/hard-reachability/contained-dev/` (579071/579074) with the frozen
controller's four failures in `heldout/` (579081–579084). The earlier
`committed-dev/` traces used the centering bug and are excluded. These six
selected episodes are diagnostic cases, not an unbiased success-rate sample.
All use lane-change mode 0, speed mode 31 and wrapper safety off; none has an
ego teleport. Times below are elapsed control time unless marked SUMO time.

| Seed | Result | Distinguishing trajectory evidence |
|---:|---|---|
| 579071 | Arrived, 380.4 s | Spent 148.8 s below 1 m/s overall, but eventually crossed J107; started E82 lane 1→2 with 118.82 m remaining, then 2→3 with 58.70 m remaining |
| 579074 | Arrived, 228.3 s | Never fell below 1 m/s; started the same E82 changes with 119.31 m and 57.51 m remaining |
| 579081 | Timeout, 600 s | Spent 353.1 s on `:J107_0_1`, including 331.0 s below 1 m/s; progressed only 62.38 m within that interval |
| 579082 | Collision, 53.4 s | Started E75 0→1 at 9.31 m/s with only 9.27 m remaining; collision occurred 2.1 s into the maneuver |
| 579083 | Timeout, 600 s | Reached a statically D1-connected lane on E75, then spent 447.7 s on `E78_2`, including 386.8 s below 1 m/s |
| 579084 | Collision, 233.2 s | Entered E82 lane 1 but waited until only 11.56 m remained to start 1→2 at 10.43 m/s; still needed lane 3 for D1 |

**Maneuver timing and execution.** The successful E82 changes took about
4.0–4.2 s each; both successful cases began the first change near the E82
entrance. The failing changes began with approximately one second of travel
remaining at their initial speeds. This is a screening calculation, not a
constant-speed feasibility proof: the actual vehicles decelerated sharply.
Both were still moving laterally at 1 m/s immediately before impact, with
longitudinal speed approaching zero at the lane end.

Read-only replay instrumentation confirmed the collision participants:
579082 was struck by `hard_O2_D2_car_van.15` on `E75_1` (reported speeds
2.76 and 0 m/s); in 579084, SUMO reports ego as the collider with
`hard_O1_D2_car_van.48` on `E82_2` (0 and 11.52 m/s). The second event was
reported during lane changing. Thus these are concrete overlap/collision
events, not false failures from teleport accounting.

The controller's initial gap check is insufficient over the full maneuver:
the traces show a rear blocker emerging while the latched maneuver continues.
The implementation adjusts longitudinal acceleration for the target leader
but does not reject a maneuver using lane-end distance or replan its lateral
execution on a rear blocker. A blanket "never change near a lane end" rule
is also unsupported: seed 579084 successfully started an earlier E75 maneuver
from rest with only 0.101 m remaining. Speed, lateral completion, downstream
connections and surrounding motion must be considered together.

**Queue location and route choice.** Static D1 reachability is an existence
test for a lane path, not a congestion forecast. Seed 579083 followed the
D1-connected chain `E75_3 → E76_2 → E77_3 → E78_2` and still timed out;
its 386.8 s below 1 m/s on E78 were spent behind
`hard_O3_D2_car_van.14`. Conversely, the two successes approached E82 on lane 1,
which requires two more changes to reach D1. Those different seeds do not
establish that this corridor is always preferable.

For seed 579081, ego continuously requested +2.6 m/s² and no lane change on
the internal J107 connector. Its low speed therefore cannot be explained by
the reference controller voluntarily commanding a stop. In 28 samples spaced
5 s apart, from SUMO time 515 through 650 s, every vehicle on all five E78
lanes was halted according to SUMO while all six E82 lanes were empty. Some
sampled queue-front vehicles were stopped on `:J107_0_1` with no leader
reported within 200 m. Seed 579083 showed the same halted-E78/empty-E82
condition in 29 consecutive samples from SUMO time 465 through 605 s.
These are sampled observations, not measurements of continuous standstill.
This localizes a flow interruption around the internal
connector; it does **not** prove the entire downstream network is saturated,
identify a specific junction defect, or show that another ego strategy could
not avoid the queue. Junction/route/lane-choice behavior remains to be tested.

**Next implementation priorities:** audit whether the policy can distinguish
route-required lanes, remaining maneuver distance and lateral completion;
test a reference-controller change that accounts for maneuver duration and
rear traffic before changing the RL observation/reward contract. Separately
inspect J107 queue-front route and lane-choice behavior before treating queue
timeouts as a training-budget problem. Do not copy the successful seeds'
action sequence or assume that increasing training steps resolves these
specific failure mechanisms. Existing distance/neighbor inputs may encode
some of this information indirectly; this comparison alone does not prove
that the policy observation is insufficient.

Comparison scripts, source hashes, replay traces, collision events and
5-second queue samples are retained under
`tmp/ai/hard-trajectory-comparison/`. All four additional full replays matched
the archived **14,866 CSV rows** exactly, including actions, states, delays
and rewards; terminal outcomes/returns and scene hashes were also verified.
The passive probe collected 296 queue snapshots and both ego collision
events. It makes no changes to vehicle controls or traffic settings. No new
unit tests were needed for this documentation-only production change.
No production controller, observation,
reward, network, traffic demand, episode limit or model was changed in this
stage. Seeds 579081–579084 are now diagnostic/development cases and must not
be reused as unseen final-test seeds after tuning against these findings.

### Background 200k baseline (completed 2026-10-13)

A fresh Antwerp Hard `full` GRU/SAC run was launched in a detached process
at 2026-10-12 22:33 Asia/Shanghai. The target is **200,000 total steps**,
not 200,000 additional steps loaded from the old pilot. The previous 20k
checkpoint did not include its replay buffer, so this run starts from scratch
with the same seed 579071, uniform delay, training safety off, environment
version 2, learning rate 3e-5, batch size 512 and gamma 0.99. The road, traffic,
observations, rewards, encoder and episode limit are unchanged. Training uses
CUDA with one Torch thread and W&B disabled; no upload is scheduled.

The detached supervisor PID at launch was `3721008`, with training PID
`3721044`. These identify the original processes only; inspect the status file
and log progress for the current stage rather than assuming an old PID is
still valid. Closing the interactive session does not send a hangup to the
detached process; host shutdown or process failure can still interrupt it.

| Artifact | Location |
|---|---|
| Console output and errors | `tmp/ai/antwerp-200k/train.log` |
| Episode metrics and progress every 1,000 steps | `tmp/ai/antwerp-200k/training.jsonl` |
| TensorBoard events, including SAC losses | `tmp/ai/antwerp-200k/tensorboard/full-v2-hard-200k/` |
| Task stage, active PID and exit status | `tmp/ai/antwerp-200k/status.json` |
| Supervisor errors | `tmp/ai/antwerp-200k/supervisor.log` |
| Launch command, source/scene hashes and seeds | `tmp/ai/antwerp-200k/manifest.json` |
| Checkpoints every 50,000 steps | `models_experiments/antwerp-200k/full-v2-hard-200k/` |
| Final model, once training succeeds | `models_experiments/antwerp-200k/full-v2-hard-200k.zip` |

After successful training, the supervisor evaluates the 50k, 100k and 200k
checkpoints sequentially, each with safety off and on, using development seeds
579081–579084. This schedules 24 evaluation episodes in separate processes;
these are previously inspected seeds, not an unseen final test. Evaluation
outputs use `tmp/ai/antwerp-200k/eval-<steps>-safety-<0|1>.json` and `.log`.
Failure of a training/evaluation command stops the remaining sequence and
records the error; success ends with status `completed`. Launch is not evidence
of completed training or improved performance. No further training budget or
formal ablations are automatically scheduled.

Monitor from the repository root:

```bash
tail -f tmp/ai/antwerp-200k/train.log
tail -n 5 tmp/ai/antwerp-200k/training.jsonl
cat tmp/ai/antwerp-200k/status.json
```

The temporary supervisor lives in `tmp/ai/antwerp-200k/background.py`; it
orchestrates the existing production entry points without modifying them.
All generated logs, models and supervisor files remain ignored by Git.

The run completed successfully: training ended at **2026-10-13 01:03:53**
Asia/Shanghai and all six evaluation jobs ended at **01:25:56**. Training
reached 200,000 steps and 199,899 updates in 9,017.94 seconds (about 2 h 30 m).
Its 87 completed exploration episodes contained 34 uninterrupted arrivals,
37 collisions, eight timeouts and eight emergency-stop terminations. No ego
teleports were recorded. The first training success ended at step 36,474.
Training success (34/87, 39.1%) is not deterministic evaluation performance;
the final partially completed episode is excluded from these counts.

| Checkpoint | Evaluation safety | Success | Collision | Timeout | Emergency stop | Mean return |
|---|---|---|---|---|---|---|
| 50k | Off | 2/4 | 2/4 | 0/4 | 0/4 | -45.142 |
| 50k | On | 2/4 | 2/4 | 0/4 | 0/4 | -109.294 |
| 100k | Off | 2/4 | 2/4 | 0/4 | 0/4 | -220.810 |
| 100k | On | 2/4 | 0/4 | 1/4 | 1/4 | -759.690 |
| 200k | Off | 1/4 | 3/4 | 0/4 | 0/4 | -23.668 |
| 200k | On | 2/4 | 2/4 | 0/4 | 0/4 | -3.789 |

All 24 evaluation episodes had no ego teleport. In particular, the 200k
policy reached D1 on seed 579084 in 210.0 seconds with safety off; with safety
on it succeeded on seeds 579083 and 579084. These are learned-policy
successes on Antwerp Hard, distinct from the earlier reference-controller
and original-road calibration successes. Four development episodes per
configuration cannot establish a reliable ranking, monotonic improvement,
or Table IV-level performance. A higher return does not imply a higher
success rate. Source/scene hashes still match the launch manifest, and each
evaluation's checkpoint hash was checked against the saved model on October
13. The supervisor reports `completed`, with all six evaluations recorded
and the final subprocess exit code zero.

### Expanded checkpoint validation (started 2026-10-13)

Following the completed 200k baseline, a detached evaluation-only job was
launched at **2026-10-13 09:16:57 Asia/Shanghai**, supervisor PID `3843755`.
It compares the existing 50k, 100k and 200k checkpoints on the same 20 seeds
**680001–680020**, with evaluation safety off and on: **120 episodes**.
Each episode runs in a separate process using the existing deterministic
Hard/uniform/full evaluation entry point, on CPU with one Torch thread.
Training safety remains off. These additional seeds are development
validation cases for checkpoint selection, not a reserved final test.
No further training budget or formal ablation is scheduled in this stage.

The temporary supervisor `tmp/ai/antwerp-200k-validation/background.py`
checks baseline source/scene hashes and checkpoint hashes before execution.
It preserves existing outputs, records a manifest, and stops on a subprocess
failure or fingerprint mismatch. Each seed/safety/checkpoint combination
has a separate `.json` report and `.log`; `progress.jsonl` records each
completed episode. `status.json` records the active process, completed count
out of 120, and running outcome/success/return aggregates for each of the
six configurations. Incomplete groups are not final comparative results.
`supervisor.log` records progress and supervisor errors. Models are read
from `models_experiments/antwerp-200k/` without modification.

```bash
cat tmp/ai/antwerp-200k-validation/status.json
tail -f tmp/ai/antwerp-200k-validation/supervisor.log
```

The job plan was checked for 120 unique outputs, 20 paired seeds and matching
baseline source/checkpoint fingerprints. Generated files remain ignored;
no dependencies or ignore rules changed. Once complete, compare success,
collision, timeout and emergency-stop counts before choosing a subsequent
training experiment. The old checkpoints have no saved replay buffer, so a
future weight-loaded run must not be described as an exact training resume.

### Three-density 1M baseline: AT-B0-1M-20261013

The user authorized three fresh, independent 1,000,000-step runs on October
13. The experiment group is **`AT-B0-1M-20261013`**. `AT` identifies Antwerp;
`B0` identifies the current version 2 observation/action/reward baseline;
`1M` is the per-model training budget. No proposed route-preparation changes
are included. Use the complete group and run IDs when comparing later runs.

| Run | Density | Run / checkpoint experiment ID | Training seed | Steps |
|---|---|---|---|---|
| 1 | Hard | `AT-B0-H1M-S579071` | 579071 | 1,000,000 |
| 2 | Medium | `AT-B0-M1M-S579071` | 579071 | 1,000,000 |
| 3 | Easy | `AT-B0-E1M-S579071` | 579071 | 1,000,000 |

Each run starts from scratch, including Hard; the old 200k model is not
loaded. All use full GRU/SAC, uniform delay up to 20 steps, training safety
off, learning rate 3e-5, gamma 0.99, batch size 512, CUDA and one Torch thread.
The 0.1-second step, 6,000-step episode limit, traffic, road and policy inputs
remain the same. W&B is disabled. This is three density-specific policies,
not three training seeds or a cross-density generalization experiment.

`train_sac_delay.py --save_replay_buffer` enables SB3 replay-buffer saving
alongside numbered checkpoints and the final model. The run name is also
stored as the model's `experiment_id`. The switch is off by default, so old
commands retain their previous storage behavior. This group saves at 250k,
500k, 750k and 1M, plus the final model and buffer. The SB3 periodic callback
runs before inserting the current transition; its replay buffer therefore
lags the checkpoint step by one transition. Final saving occurs after
`learn()` returns. These files support future training continuation, but do
not restore live SUMO/RNG state or provide bit-identical interrupted-run
resumption. Automatic resumption is not scheduled.

A detached supervisor at `tmp/ai/antwerp-density-1m/background.py` executes
**three CUDA training jobs in parallel**, as requested after the initial
serial launch. Hard's existing process and accumulated training were retained;
Medium and Easy were started alongside it. An independent queue runs at most
one CPU evaluation process at a time.
The already-running 120-episode 200k validation is allowed to finish before
this group's CPU queue starts. Training does not wait for that evaluation.

Evaluate each density's 250k, 500k and 1M checkpoints on the matching density,
using development seeds 680001–680020, safety off and on, one episode per
seed: **360 episodes**. The same seeds are used in the old 200k validation
and are not an unseen final test. Numbered checkpoint/replay pairs become
eligible only after a later training progress entry or successful training
exit confirms saving has finished. Each evaluation gets an exclusive JSON
report/log. Compare uninterrupted success, collision, timeout, emergency-stop
and teleport counts; successful travel time can be derived from episode steps.
Incomplete groups and one training seed per density do not establish
Table IV-level performance or training-seed stability.

| Artifact | Location |
|---|---|
| Supervisor, launch and source/scene/command manifest | `tmp/ai/antwerp-density-1m/` |
| Group progress, PIDs, queue states and evaluation aggregates | `tmp/ai/antwerp-density-1m/status.json` |
| Current supervisor progress and errors | `tmp/ai/antwerp-density-1m/parallel-supervisor.log` |
| Scheduling change and archived serial status | `tmp/ai/antwerp-density-1m/parallel-handoff.json`, `serial-status-at-handoff.json` in the same directory |
| Console, episode/progress JSONL and TensorBoard per run | `tmp/ai/antwerp-density-1m/<run-id>/` |
| Per-episode evaluation reports and logs | `tmp/ai/antwerp-density-1m/evaluation/` |
| Evaluation completion records | `tmp/ai/antwerp-density-1m/evaluation-progress.jsonl` |
| Completed checkpoint hashes and buffer sizes | `tmp/ai/antwerp-density-1m/checkpoints.jsonl` |
| Models and replay buffers | `models_experiments/antwerp-density-1m/<run-id>/` |

The supervisor refuses existing output paths, verifies source/scene hashes
before launching each subprocess, and records failures instead of silently
skipping experiments. On failure it stops its own active children and leaves
all files for inspection; the adopted Hard process is preserved rather than
terminated by the replacement supervisor. Its actual process exit code cannot
be collected by the new parent: completion requires process exit, 1M logged
steps, final model/buffer files and the training success log footer. Previously
started independent experiments are not stopped. Changing experiment code/scenes while this group runs will prevent
further queued subprocesses from starting; future variants should be isolated.
All generated artifacts remain ignored, with no dependency or ignore-rule
changes. The earlier 38-hour training estimate assumed serial execution and
does not apply to the parallel schedule. Measure sustained per-run throughput
under GPU/CPU contention to estimate the new finish time.

Validation before launch: 25 contract tests passed. A 16-step local fixture
exercised the production training entry point with actual SAC updates, saved
and reloaded both periodic checkpoint/buffer pairs and the final pair, and
verified experiment IDs, transition counts and stored observations/actions.
Only this fixture uses a small replay buffer; the actual experiments retain
SAC's existing buffer configuration. The temporary fixture and outputs are
kept under `tmp/ai/antwerp-density-1m/verification/`. The supervisor plan was
checked for three unique training runs and 360 unique paired evaluation jobs.
A full supervisor dry run with simulated child processes also completed all
three ordered training jobs, 12 checkpoint registrations and 360 evaluations
in 18 groups. Its outputs are fixtures, not traffic-performance results.

The detached group was launched at **2026-10-13 09:59:45 Asia/Shanghai**
with supervisor PID `3914167`. Hard started with PID `3914268`, CUDA confirmed;
Medium and Easy were initially queued. At **10:05:58 Asia/Shanghai**, the
user-requested parallel handoff replaced only the serial scheduler with PID
`3941888`; Hard retained PID `3914268` and 7k logged steps, while Medium PID
`3942022` and Easy PID `3942023` started. The original manifest is preserved;
`parallel-handoff.json` records the scheduling change and matching source
hashes. The old scheduler was identity-checked, suspended, then terminated
without signaling the Hard trainer, preventing duplicate queued launches.
A simulated full parallel run verified three trainers started before waiting
for completion, all 360 evaluations completed, and a separate handoff fixture
verified only the scheduler received signals. These PIDs describe the
launch/handoff only; `status.json` and per-run progress logs are the live record.
Launch does not imply completed training or measured improvement.

```bash
cat tmp/ai/antwerp-density-1m/status.json
tail -f tmp/ai/antwerp-density-1m/AT-B0-H1M-S579071/train.log
```

### Learned-checkpoint route diagnosis (2026-10-13)

For the current Table III migration target, six diagnostic episodes replayed
the existing Hard and Medium **500k-step** GRU checkpoints, pairing evaluation
safety off/on for three selected development cases. All six exactly matched
the archived outcome, success/teleport/merge flags, step count, and return;
checkpoint and scene hashes also matched. These cases were selected to explain
different failure mechanisms, **not to estimate success rates**.

| Density / evaluation seed | Safety off | Safety on |
|---|---|---|
| Hard / 680009 | Success, 303.1 s | Timeout, 600.0 s |
| Hard / 680013 | Timeout, 600.0 s | Success, 316.0 s |
| Medium / 680005 | Success, 263.6 s | Teleported arrival, 503.1 s; task failure |

Times start at ego takeover. The observer logged policy requests, actual
safety-controller decisions, road/lane position, and adjacent traffic without
changing actions or the policy's normal 134-dimensional input. Replays ran
sequentially on CPU with one Torch thread; no new training was launched.

- **Late route preparation is present even in successful cases.** All three
  successful episodes entered the final 119.4 m E82 segment in lane 0 and
  crossed to lane 3 there. Medium kept its lane throughout E77 and E78.
  Arrival alone therefore does not demonstrate reliable early preparation.
- **Medium combines late requests with subsequent safety suppression.** With
  safety on, ego spent 310.9 s on E82, including 300.2 s below 1 m/s. Of 1,471
  per-step requests toward the D1 lanes, safety suppressed 1,468; only three
  commands passed, and one actual lane crossing completed. Ego remained on
  E82_1 at its end and teleported at about 483 s. Earlier, 33 one-second
  samples passed the diagnostic adjacent-gap check without a corresponding
  route-directed request. These are instantaneous heuristic opportunities,
  not proof that a complete safe maneuver was possible. Suppression counts
  are 0.1 s commands, not independent attempted maneuvers.
- **Hard also exhibits queue lock-up.** Seed 680009 with safety on made no
  progress in its final 100 s despite positive policy and applied acceleration
  in 991 of 1,000 steps; safety made no longitudinal changes on E78. The
  four-vehicle leader chain was stopped. Seed 680013 with safety off was
  already on graph-compatible E78_2, yet advanced only 0.0027 m in its final
  100 s while requesting positive acceleration throughout. Graph reachability
  means a lane-connected path exists, not that congestion permits arrival.
  Early route preparation alone cannot account for every timeout. Safety
  changes trajectories earlier in the episode, so paired outcomes cannot be
  attributed solely to the final blocked request.

A separate pure-function counterexample confirmed a **distance-definition
issue in safety checking**. Observation geometry uses vehicle reference-point
separation, which the controller treats as clearance without subtracting
vehicle length. For a stationary ego and leader with the same physical bumper
gap of 2.501 m, a 4.2 m leader produces reference separation 6.701 m and a brake
decision, whereas a 16.6 m leader produces 19.101 m and permits acceleration.
SUMO defines vehicle position at the front bumper; its leader-distance API
has an additional minGap convention, so these values must not be compared as
interchangeable physical gaps. See the official
[position definition](https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html)
and [leader/neighbor API definitions](https://eclipse.dev/sumo/pydoc/traci/_vehicle.html).
This confirms a controller-contract issue, not that it caused every observed
failure. Correcting and testing this contract should precede interpreting
safety-on performance or adding reward terms.

The current observation has no explicit target exit, lane connectivity, or
distance to the required divergence. Cumulative distance on this fixed route
can nevertheless convey location implicitly; these replays do not establish
that the task is unobservable or that adding route features/rewards improves
learning. Such claims require a controlled comparison. No production state,
reward, controller, checkpoint, or scene was changed by this diagnosis.

Local diagnostic artifacts are retained, ignored by Git, under
`tmp/ai/checkpoint-route-diagnosis/`: `probe.py`, `probe.log`, `manifest.json`,
`results.json`, per-case `trace.jsonl`/`summary.json`, `analyze.py`,
`analysis.json`, `analysis.log`, `gap-contract-check.json`, and
`trajectories.png`. Production code does not depend on these files. The active
training group's source/scene hashes remain unchanged; `.gitignore` needs no
update. New training, including the deferred No Encoder queue, remains paused
pending a separate decision.

### Safety correction and passive background diagnosis (2026-10-13)

The separate `safety-bumper-gap` branch, commit `b60ffb0`, corrects safety's
reference-point/bumper-clearance mismatch. Neighbor dimensions and headings
travel with delayed snapshots; policy input remains 134 values. Both safety
entry points use the corrected geometry and reports identify controller
`bumper-clearance-v1`. The original 10 m margin and stopping rule are retained.
The branch passed 32 unit tests, four SUMO integration tests, and six complete
checkpoint replays. All three safety-off controls preserved every recorded
action/reward/physical state and sampled full observation, as well as exact
terminal outcomes/returns. Main's source/scene hashes remain frozen for the
ongoing baseline; the correction is **not merged into that experiment**.

| 500k checkpoint case | Previous safety on | Corrected safety on |
|---|---|---|
| Hard / 680009 | Timeout, 600.0 s | Timeout, 600.0 s |
| Hard / 680013 | Success, 316.0 s | Teleported arrival, 592.4 s; failure |
| Medium / 680005 | Teleported arrival, 503.1 s; failure | Teleported arrival, 503.1 s; failure |

These selected cases show no arrival improvement. Correcting clearance can
remove previously allowed maneuvers and change subsequent policy decisions;
the result does not validate the unchanged threshold/policy combination.
Details, version/source manifests, and traces remain under
`tmp/ai/safety-bumper-gap/`, with formal changes committed in its `worktree/`.

After that commit, four **passive SUMO** diagnostics used archived commands,
scenario seeds, routes, demand, limits, and vehicle types, but let SUMO drive
ego as well as all background vehicles. No learned policy or external safety
controller was used. Ego's observed modes were speed 31 and lane change 1621.
Each simulation continued to absolute time 720 s, even after ego arrived,
recording network/lane conditions every 10 s and collision/teleport events
every step. This is a traffic diagnostic, not a DAROM benchmark.

| Scenario / seed | Peak sampled E78 halted vehicles | Background/network collision events | Teleport starts | Ego by absolute 720 s |
|---|---:|---:|---:|---|
| Easy / 680009 | 0 | 0 | 0 | Arrived at 335.4 s |
| Medium / 680005 | 0 | 0 | 0 | Arrived at 314.6 s |
| Hard / 680009 | 178 | 1 | 3 | Not arrived |
| Hard / 680013 | 170 | 2 | 0 | Not arrived |

Both Hard cases repeatedly had at least 100 halted E78 vehicles while E82,
D1 and D2 contained **no vehicles** (16 and 12 sampled instants respectively).
In Hard 680009 at absolute 620 s, all 178 E78 vehicles were halted and the
downstream edges were empty. Several stopped J107 heads had no queried leader;
their lane-change states reported urgent strategic changes blocked by lateral
leaders/followers and overlap. D1-bound vehicles requested left changes while
D2-bound vehicles requested right changes. This localizes a route/lane-change
bottleneck at the 125.36 m J107 internal connector; it is not evidence of a
full downstream exit queue and is not caused solely by the learned ego policy.
The relative contributions of connector modeling, SUMO lane-change behavior,
OD assignment and high demand still require controlled isolation. No road,
flow, departure time, threshold or speed change was made to obtain these data.

All four diagnostics completed, with 288 scheduled samples and unchanged
scene/source hashes. Scripts, logs, events, lane/head traces, manifests and
aggregates are retained under `tmp/ai/background-traffic-audit/`; they and the
safety probes remain ignored without changing `.gitignore`. The next priority
is connector/background behavior and measured-speed calibration before a
route-preparation training comparison. No new training or No Encoder queue
was launched.

### Small speed-limit sensitivity probe (2026-10-14)

Group `AT-SPEED50-60-PILOT-20261014` tests two previously diagnosed Hard seeds,
680009 and 680013, using isolated scene copies. Only the 46 external-lane
speed limits change from 13.89 to 16.67 m/s (approximately 50 to 60 km/h).
All internal connector limits, geometry, demand, vehicle types, OD routes and
random seeds remain fixed. The 60 km/h value is a sensitivity setting, not a
claim about the real road's legal limit. Because `departSpeed="speedLimit"`
is retained, realized entry speeds can change as part of this intervention.

Four native-SUMO episodes use the same absolute 720 s observation horizon.
Both 50 km/h baselines exactly reproduce all 72 archived ten-second traffic
snapshots (lane counts, halted vehicles, lane speeds, active and pending counts).

| Native SUMO seed | Limit km/h | Ego uninterrupted arrival by 720 s | Peak E78 halted | Background collision events | Network arrivals / departures |
|---|---:|---|---:|---:|---:|
| 680009 | 50 | No | 178 | 1 | 301 / 1435 |
| 680009 | 60 | Yes, at absolute 364.1 s | 110 | 12 | 464 / 1435 |
| 680013 | 50 | No | 170 | 2 | 310 / 1435 |
| 680013 | 60 | No | 178 | 1 | 278 / 1435 |

Network arrivals can include teleported vehicles and are not a success metric.
The increased collisions in the first higher-speed case mean that lower queues
cannot be attributed solely to better traffic service: collision participants
are removed under the unchanged SUMO collision configuration. The second seed
does not improve. These two selected seeds cannot estimate general success rates.

The same pair is also evaluated with the frozen Hard 1M GRU model, additional
safety off, SUMO speed mode 31, and the existing 600 s takeover-based timeout.
The 50 km/h model controls must match archived outcomes, steps and returns.
The existing model was trained at 50 km/h, so higher-speed evaluation tests
sensitivity of that policy, not performance after retraining at 60 km/h.

All eight cases completed. Both 50 km/h model controls exactly matched the
archived steps, outcomes, success/teleport/merge flags and returns. Model elapsed
times below are measured from takeover; both 60 km/h ego episodes had no teleport.

| Hard 1M GRU, extra safety off | 50 km/h | 60 km/h | Mean ego speed, 50 → 60 (m/s) |
|---|---|---|---:|
| Seed 680009 | Success, 258.6 s | Collision, 167.4 s | 10.055 → 13.834 |
| Seed 680013 | Timeout, 600.0 s | Success, 548.7 s | 3.460 → 4.736 |

One model case improves and one regresses; each limit has one success in these
two selected cases. This demonstrates speed sensitivity, not a general benefit
from raising the limit or proof that 50 km/h is the principal failure cause.
Background teleports remain possible and are counted separately from ego
success. Do not merge these sensitivity episodes into the baseline Table III.
Source, production-scene and checkpoint hashes were verified unchanged after
the test, and traffic logging covered each model episode through its terminal step.

Temporary scripts, isolated networks/routes, source/checkpoint manifests,
per-case traffic traces and summaries remain in `tmp/ai/speed-limit-probe/`.
An initial terminal-step logging error was fixed in the temporary probe;
the failed attempt and logs are retained, and only model probes were resumed.
No production scene, reward, model or dependency is changed.

### Training-loss and convergence audit (2026-10-14)

Read-only analysis of the three AT-B0 1M TensorBoard files exports actor loss,
critic loss, entropy-coefficient loss, entropy coefficient alpha, and success
diagnostics under `tmp/ai/training-loss-analysis/`. PNG/PDF figures, raw
`scalars.csv`, source hashes and descriptive `statistics.json` are retained
there; production code and models are unchanged. The verification smoke-run
event file is excluded. Easy/Medium/Hard have 122/120/110 logged points, ending
at steps 995324/998750/991929, respectively, although training completed 1M.
SB3 dumps these logs every four episodes; they do not record every optimizer
update. Plots retain raw values and overlay a trailing 50k-step median, with
a shared logarithmic critic-loss axis and the last 200k steps shaded.

| Scenario | Critic loss median, logged points in 800k–1M | Training success, episodes ending in 800k–1M | Final deterministic eval success, SC off / on |
|---|---:|---:|---:|
| Easy | 0.838 | 84/103 = 81.6% | 20% / 15% |
| Medium | 1.996 | 86/101 = 85.1% | 0% / 0% |
| Hard | 6.208 | 42/93 = 45.2% | 45% / 35% |

All inspected scalar values are finite. Easy/Medium critic loss and alpha
are comparatively stable late in training, but actor loss still drifts and
independent task performance does not show sustained improvement. Hard has
intermittent large critic errors and a final recorded spike of **9233.912 at
991929 steps**. One spike, with no subsequent logged loss, does not establish
sustained divergence or identify its cause. A low sampled Bellman error or
near-zero entropy-coefficient loss is not evidence of successful task learning;
SAC actor loss also has no required zero limit.

The training/evaluation discrepancy is especially large for Easy/Medium.
Training uses sampled actions and changing policies/scenarios; evaluation uses
frozen checkpoints, deterministic actions, and 20 development seeds. These
are not matched estimators. The data do not isolate action determinism, seed
generalization, reward alignment or overfitting. A matched stochastic versus
deterministic evaluation, including lane-change command distributions, is a
more informative next diagnostic than inferring convergence from loss alone.
This audit launches no new training or evaluation.

The same directory also contains `figure4_training_curves.png` and `.pdf`,
following the row/column layout of
[paper Figure 4](https://arxiv.org/html/2403.11852v5#S5.SS5): Easy/Medium/Hard
columns and average training return, critic loss, actor loss rows. This figure
uses the **arithmetic mean** of available samples in the trailing 50k environment
steps for all bold curves, rather than the median used in `loss_curves`.
Return data include all 490/480/442 completed training episodes from
`training.jsonl`; losses use the TensorBoard records described above.
Raw points, including negative-return outliers and the Hard critic spike, remain
visible. Return axes have independent ranges; critic axes use a shared log scale
to retain the full spike. `figure4_data.csv` exports raw values, rolling means
and window sample counts; `statistics.json` includes episode-source hashes.
Only one GRU training seed per scenario is available here: this is not a
multi-method comparison, independent evaluation, or multi-seed confidence band.
The final plotted return windows average 119.46/150.23/-370.17 for Easy/Medium/Hard
over 25/25/23 episodes, respectively; these are training returns, not Table III
evaluation returns. No training, model, environment or dependency is changed.

### Existing DAROM-original training comparison (2026-10-14)

The local `../DAROM-original/` contains an actual 10M-step reproduction,
`GRU-uniform-delay-full`, seed 42, GRU/uniform, learning rate 3e-5 and batch
size 512. Read-only extraction recovered 5,113 TensorBoard loss records,
20,454 completed training episodes and 100 historical evaluation checkpoints.
The short `GRU-uniform-mytest` run and distributed pretrained models are not
substituted for this run. W&B records were read locally with fragment checksums
verified; no W&B initialization, synchronization or network upload was used.
The cumulative episode lengths match every TensorBoard logging step exactly
(one dump per four episodes); cumulative returns also match the logged means.

| Original reproduction window | Mean training episode return | Median actor loss | Median critic loss |
|---|---:|---:|---:|
| 800k–1M | -261.77 | 44.74 | 2.16 |
| 6M–7M | -247.89 | 39.39 | 2.10 |
| 9M–10M | 143.94 | -38.29 | 1.81 |

Original actor loss initially rises from about -1.8 to 69.6, then falls;
it remains positive at 1M and 7M. Historical deterministic evaluation reports
0/51 arrivals at 1M and 7M, and 49/51 (96.08%) at 10M, using the original
callback's arrival definition, not Antwerp's stricter success definition.
Critic loss has a raw spike of 3,756,378.5 at step 8,278,650, followed by
recovery. Thus this local reproduction does not duplicate the paper's smooth
training trajectory either. Early positive actor loss or a transient critic
spike alone does not establish transfer failure; this one run also cannot
guarantee that longer Antwerp training will recover.

`tmp/ai/training-loss-analysis/original_reproduction_training.{png,pdf}` shows
the first 1M and full 10M history. `original_vs_antwerp_1m.{png,pdf}` compares
equal training budgets with the same trailing 50k-step arithmetic mean.
The same original run is repeated against Antwerp Easy/Medium/Hard, not three
matched original-density runs. All raw values are retained; critic axes are
logarithmic and the full-history original actor axis is symlog beyond +/-80.
`compare_original.py`, allowlisted CSV exports, source hashes, statistics and
the parsing/plot log remain in the same temporary directory.

The archived run records commit `eabd572fc0aece579e5b553bdc7af27a8b81d03a`.
Present original scene files match that commit, and its default route matches
Hard, but the archived environment points to the historical absolute directory
`/home/cyl/project/DAROM/`, which no longer exists; runtime scene hashes were
not recorded. Treat traffic provenance as qualified, not a controlled paired
experiment. Seeds differ and the original SUMO configuration used `--random`.
Original logging starts at step 254 versus Antwerp's 9608/5529/13911; neither
first recorded loss represents an identical initialization stage. Reasons for
the remaining learning differences require matched diagnostics, not inference
from curve shape alone. Neither repository's training code or models changed,
and no new training or evaluation was launched.

### Six-episode reevaluation timing pilot (2026-10-14)

The user narrowed the next task to a timing/compatibility check only. Group
`AT-T3-1M-1500-20261014/pilot` evaluated the existing Easy/Medium/Hard 1M
checkpoints with SC off/on, one archived development seed (680001) per case.
Two CPU evaluator processes, each with one PyTorch thread, completed all six
cases in **255.2 seconds** and stopped. All outcomes, steps, returns and motion
metrics exactly matched the archived evaluations; checkpoint hashes remained
unchanged. This verifies the evaluation pipeline, not task success rates.

| Scenario | SC off wall time (s) | SC on wall time (s) |
|---|---:|---:|
| Easy | 22.0 | 54.1 |
| Medium | 38.5 | 76.5 |
| Hard | 193.1 | 17.0 |

Mean per-process episode wall time was 66.87 seconds. Assuming the same case
mix and sustained two-worker utilization, 600/3000/9000 episodes would take
approximately 5.6/27.9/83.6 hours. These are rough projections from only six
cases, sensitive to episode lengths and server contention; they are not runtime
guarantees. Directly scaling this small batch's elapsed time also includes its
idle-worker tail and gives a more conservative 106.3 hours for 9000 episodes.

The temporary runner, frozen model/source manifest, per-case logs/reports,
`pilot/status.json` and `pilot/timing.json` remain under
`tmp/ai/table3-1m-1500/`. The prepared 9000-episode formal manifest is dormant:
**no formal evaluation was launched or queued, and the pilot does not launch
it automatically**. Further large-scale evaluation requires a new instruction.
Training/model/environment code and ignore rules are unchanged; the local
`agents.md` task boundary records the six-episode limit.

### Table III reevaluation plan after the timing pilot (2026-10-14)

Following the completed pilot, the user authorized the paper-sized evaluation
and Table III generation for the current three **1M** Antwerp checkpoints.
Group `AT-T3-1M-1500-20261014` uses the validated metrics worktree at `7d1710f`
and freezes the existing baseline safety controller and model/scene hashes.
No additional training, original pretrained-model evaluation or other algorithm
is included. The evaluation sample size follows the paper; the models remain
1M checkpoints, not the paper's 7M training runs.

| Scenario | Checkpoint | Evaluation safety | Episodes per setting |
|---|---|---|---:|
| Easy | AT-B0-E1M-S579071, 1M | off / on | 500 × 3 = 1500 |
| Medium | AT-B0-M1M-S579071, 1M | off / on | 500 × 3 = 1500 |
| Hard | AT-B0-H1M-S579071, 1M | off / on | 500 × 3 = 1500 |

Total: **9000 episodes**, excluding the six pilot cases and previous development
evaluations. Three disjoint seed ranges are 710001–710500, 720001–720500 and
730001–730500. SC off/on are paired on the same seeds; all cases use deterministic
actions and uniform delays up to 20 steps. Four CPU worker processes with one
PyTorch thread each interleave scenarios, safety settings and batches. The
workers execute episodes from six configurations (three densities × safety
off/on); they do not represent four densities or four models. Each completed
episode is saved immediately; an evaluator error stops new scheduling instead
of dropping the case. Successful cases are preserved when explicitly resuming.

The original two-worker timing projection was approximately **84 hours**.
At the same per-episode speed, four workers project approximately **42 hours**;
allow roughly **2–3 days** for planning, subject to case mix and server contention.
This is a pilot-based estimate, not measured sustained four-worker throughput.
Progress and estimated finish time are updated during execution.
All batch sizes and settings are fixed before inspecting the new results.

Formal evaluation started at 19:12:18 Asia/Shanghai on October 14 with two
workers. After the user requested more parallelism if resources permitted,
CPU sampling showed 30–34% idle, 22 GiB available memory and no active swap I/O.
The two in-flight episodes finished before replacing only the scheduler at
19:18:05. All nine saved reports were retained; new supervisor PID `2888031`
started four workers. Post-launch sampling showed 25–28% CPU idle and about
20 GiB available memory, with no active swap I/O. These are launch-time
observations; consult `formal/status.json` for live progress. The handoff is
recorded in `formal/parallel-handoff.json`. Runtime `control.json` can set
1–4 workers; lowering the limit lets active episodes finish before reducing
concurrency. The control bounds and unchanged 9000-job schedule were checked.
Model, scene, safety and metric definitions remain frozen.

Outputs under `tmp/ai/table3-1m-1500/formal/` include `status.json`,
`progress.jsonl`, `supervisor.log`, per-episode logs/JSON, and automatically updated
`table3.md`, `table3.csv`, `table3.json`. The table contains the six GRU safety-on/off
rows with success, collision, timeout/legacy No Merge, return, speed and signed
jerk. Partial results are explicitly labeled; the final ± values are sample
standard deviations across the three evaluation batches, not confidence bounds
or training-seed variation. JSON also retains Wilson 95% success intervals,
actual unmerged/merged timeouts, teleport outcomes, safety intervention counts,
absolute/RMS jerk and missing-motion counts. Only one training seed per density
is represented. The paper's additional algorithms remain outside this table.

Preparation validation passed for all 9000 unique paired jobs, source/model
fingerprints, and empty/partial/complete table exports with synthetic batch
statistics. Synthetic outputs remain isolated under `verification/`; the
formal table begins empty. The original six-episode parity validation remains
applicable because this extension changes scheduling authorization and report
export only, not environment, policy inference or motion measurement.

### Original full-GRU best: historical Table III and Figure 4 export (2026-10-14)

The requested checkpoint is specifically
`../DAROM-original/models/GRU-uniform-delay-full/GRU-uniform-delay-full_best.zip`.
Its metadata records **9,700,000 steps**, seed 42, learning rate 3e-5 and batch
size 512; SHA256 is `eff41757c0aac5ad535fe187f737e9828105fc247c5163057bfe92a6e1c2d82a`.
It must not be confused with similarly named models in another directory.
Two complete historical uniform-delay, safety-on logs name this model path:

| Historical log | Episodes | Legacy arrival (%) | Collision (%) | Timeout (%) | Return | Speed (m/s) | Signed jerk (m/s³) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `eval_uniform_best_uniform.log` | 501 × 3 | 99.53 ± 0.13 | 0.00 ± 0.00 | 0.00 ± 0.00 | 177.95 ± 0.40 | 29.24 ± 0.01 | -0.26 ± 0.01 |
| `eval_uniform.log` | 501 × 3 | 99.40 ± 0.45 | 0.00 ± 0.00 | 0.00 ± 0.00 | 177.64 ± 1.25 | 29.24 ± 0.01 | -0.25 ± 0.01 |

These are separate existing evaluations, not additional densities or newly
executed trials. The table transcribes the complete log footers. Their ± values
are the old evaluator's **1.96 × standard error**, computed across three batches
for outcomes/return and across episodes for motion; they are not the batch
standard deviations in the ongoing Antwerp table. Historical success means
`Arrived` without the stricter no-teleport check. No Merge remains a timeout
proxy. Current default scene files match Hard, but the historical scene and
checkpoint hashes were not logged, so their exact identity is unverified.
No separate Easy/Medium or safety-off records were found for this checkpoint.
Some per-episode lines are disrupted by stdout/stderr interleaving, and motion
trajectories were not retained; the footers cannot be fully independently
reconstructed. The incomplete `eval_uniform_best.log` is excluded.

`tmp/ai/original-best-report/` contains `table3.md/.csv/.json`,
`table3_original_best.pdf/.png`, and `figure4_original_best.pdf/.png` with
underlying `figure4_data.csv`. The figure uses the previously extracted actual
10M training history: 20,454 episode returns and 5,113 points per loss metric,
with a trailing 50k-step arithmetic mean and a vertical marker at 9.7M.
Post-best training is shaded. Critic uses a log axis and actor a symmetric-log
axis outside ±80; outliers are retained. This is one run's training history,
not evaluation losses or three independent density-specific histories.
Source W&B/TensorBoard hashes were rechecked, rolling means independently
recomputed, and both rendered figures inspected. Six complete historical logs
(including four other delay modes in a supplementary table) and one incomplete
log are inventoried in `manifest.json`; `build_report.py`, `build.log` and
`validation.json` retain the extraction and verification details.

Resource sampling found about 469 GiB free disk, 21 GiB available memory and
26–28% idle CPU while the four Antwerp evaluators continued normally. This
export required no additional SUMO job, training, dependency or ignore-rule
change. The files remain local temporary analysis artifacts; no upload occurred.

### Antwerp actor-loss decomposition and Hard critic spike audit (2026-10-15)

The user requested a concrete explanation of the current 1M loss curves.
This audit reads saved checkpoints and replay buffers on CPU, with one Torch
thread, without gradient updates or SUMO. The original figures and ongoing
Table III evaluation are unchanged. Source/model/replay hashes and results
are retained under `tmp/ai/loss-spike-audit/`.

The installed SB3 logger overwrites `train/*` metrics on each training call
and dumps them every four completed episodes. These runs use one gradient
step per interaction and batch size 512. Thus a plotted loss is the latest
minibatch loss at the dump, **not the average loss across the preceding four
episodes**. For the 991,929-step dump, `n_updates=991828` corresponds to training
through interaction 991,928; the arrival transition at 991,929 has been stored
but has not yet been used in an update. Its preceding reward is +0.1044 and
its arrival reward is +49.9683. The episode arrived without teleport, and the
next two completed episodes also arrived. There is no evidence that an ego
collision at that timestamp caused the spike. The last recorded critic losses
are 14.306, 2.383, 3.968, 4.883 and 9233.912. In Figure 4 the final bold value
1544.277 averages only six logged points; it is not evidence of a continuously
elevated loss over 50k steps.

With `gradient_steps=1` and `target_entropy=-2`, existing alpha/alpha-loss
records allow algebraic reconstruction of the logged actor objective's two
terms, subject to float rounding. Actor loss is `alpha * mean(log_pi) - mean(Q)`.
The latest reconstructed entropy and negative-Q terms are respectively
0.0864 + 17.4485 for Easy, 0.1046 + 27.5999 for Medium, and 0.0588 + 21.6757
for Hard. At the first recorded points, mean Q was +15.0388, +9.5233 and
+1.8447 respectively. Their subsequent negative value estimates account for
most of the initial-to-final actor-loss increase; this is not an independently
measured increase in driving error. The states/policies differ across these
historical minibatches. A separate fixed set of 8192 first-250k replay states
also shows Easy actor objective rising 10.105→15.896 between its 250k and 1M
checkpoints, dominated by the Q term, while critic MSE falls 1.546→0.936.

All three saved periodic replay buffers contain 999,999 transitions (the
checkpoint callback runs before storing the millionth), with finite states,
actions and rewards and episode boundaries matching the complete episode logs.
Hard single-step rewards range from -100.8635 to +50.1654. A full frozen-1M
scan of Hard, sampling one next action per transition, found:

- Median per-transition squared TD error 0.2675 versus mean 211.1338.
- The largest 10 errors contribute 79.58% of total squared error; the largest
  100 contribute 91.77%. There are 1389 replay states with absolute critic Q
  above 1000. This is rare-state value instability, not uniformly large error.
- At replay interaction 69,085, reward is -0.91865, current Q values about
  5.92/6.47, but the bootstrapped target is 6420.72. Its squared error alone
  contributes about 80,363 to a 512-item batch average if sampled once.
- For that same next state, the target network's minimum-Q estimate averages
  -32.26, -27.53 and +6486.65 at the 250k, 750k and 1M checkpoints. At 1M,
  128 sampled actions give 6485.29–6487.31; a deterministic action gives
  6486.73. The large value is not just one unlucky action draw.

This establishes that the saved final Hard value networks can produce large
bootstrap errors on rare historical states, sufficient to cause isolated
minibatch spikes. It **does not identify the exact batch or weights responsible
for the historical 9233.912 value**: neither the 991,928-step weights nor its
sample indices/RNG state were saved. The origin of the value drift (such as
representation, replay distribution, timeout handling or optimization) and its
effect on final driving performance remain unisolated. Do not describe the
model as globally divergent, the spike as harmless, or a learning-rate/reward
change as a proven cure. The next diagnostic should record Q/target ranges,
TD-error quantiles, replay indices and gradients during a separately authorized
small controlled continuation, rather than overwrite the active baseline.

Artifacts include `audit.py`, `scan_hard.py`, `check_tail.py`, `summarize.py`,
their logs, per-density JSON, `actor_decomposition.csv`, `hard_transition_td.npz`,
`hard_full_scan.json`, `tail_checks.json`, `validation.json`, and the inspected
`loss_spike_diagnostics.pdf/.png`. Model hashes matched after analysis; no
production code, model, dependency or ignore rule changed.

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
