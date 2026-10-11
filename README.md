# DAROM: Delay-Aware Reinforcement Learning for Highway On-Ramp Merging

**DAROM** (Delay-Aware Reinforcement learning for On-ramp Merging) is a framework for autonomous highway on-ramp merging under stochastic V2I communication latency. The agent is modeled as a Random Delay Markov Decision Process (RDMDP) and trained with a Delay-Aware Encoder that jointly processes delayed observations, masked action histories, and delay magnitudes to recover the true latent traffic state.

## Overview

In RSU-assisted autonomous driving, the roadside unit (RSU) perceives surrounding vehicles and transmits state estimates to the ego vehicle over V2I links. Processing and transmission introduce stochastic delays, violating the Markov assumption and degrading standard RL performance. DAROM addresses this by:

- **RDMDP formulation**: Models the problem as a Random Delay MDP via state augmentation with the delayed observation, action buffer, and delay magnitude.
- **Delay-Aware Encoder**: A GRU-based encoder that implicitly infers the current latent state from the augmented input.
- **Physics-Based Safety Controller**: A supervisory layer that enforces kinematic stopping distance constraints, overriding unsafe actions before execution.
- **Unified SAC Agent**: A single soft actor-critic agent for joint longitudinal and lateral control.

Experiments in SUMO using real-world NGSIM traffic data show DAROM-GRU achieves **>99% success** in high-density traffic with random delays up to 2.0 seconds. The trained policy generalizes across five stochastic delay distributions (uniform, bimodal, bursty, triangular, exponential) without retraining.

![DAROM Framework](assets/network.png)

## Repository Structure

```
DAROM/
├── src/
│   ├── merging.py           # SUMO-based merging environment (gymnasium)
│   ├── wrapper.py           # DelayWrapper — RDMDP state augmentation
│   ├── delay_encoder.py     # Delay-Aware Encoder (MLP / GRU / Transformer)
│   ├── safety_controller.py # Physics-based safety controller
│   └── utils.py             # Shared utilities
├── sumo_files/              # SUMO network and route files (Easy/Medium/Hard)
├── models/
│   └── GRU-uniform-delay/   # Pretrained DAROM-GRU checkpoint
├── train_sac_delay.py       # Training script
├── evaluate_sac_delay.py    # Evaluation script
├── config.yaml              # Environment configuration
└── requirements.txt
```

## Git Submission Policy

The root `.gitignore` uses an allowlist. Only `src/`, `assets/`,
`sumo_files/`, `sumo_new/`, `models/`, and the explicitly listed root files
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
`src.merging.Merging(seed=579071, scenario="easy")` loads one of the
`easy`, `medium`, or `hard` Antwerp routes (`medium` by default). Each reset
advances SUMO to just before the scheduled departure, then waits for `ego`
to actually enter O3. The episode clock starts at takeover and has a uniform
limit of 6,000 control steps (600 simulated seconds at 0.1 s per step);
arrival at D1 can end it earlier. `reset()` returns the actual departure
time in `info`. The environment starts SUMO from the network and selected
route files with the timing and lateral settings of `Antwerp.sumocfg`,
without detector outputs. The observation uses ego route distance, lane
index, and speed, with nearby vehicles expressed in the ego's forward/left
coordinate frame. Reward and
safety checks use the same coordinate convention. Lane changes are
bounded by each Antwerp edge's actual lane count; merging is
recognized when ego leaves `E75_0` for a mainline lane or reaches `E76`.
The existing pretrained policy must be retrained for this observation layout.
The task is for ego to merge safely and reach D1 before the episode limit;
collision, emergency stopping, and timeout are failures. Background queues
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
- [x] Replace old-world-axis observation logic with ego-forward/left local
  coordinates for up to 30 neighbors; use route distance for ego progress.
  Adapt lane-change bounds, merge recognition, reward gaps, and safety checks
  to the Antwerp observation layout.
- [x] Connect the training entry to all three routes. Training disables the
  safety controller by default; `--safety` enables it. New checkpoints go to
  `models_experiments/`.
- [x] Separate evaluation from training: training saves numbered checkpoints
  every 100,000 steps without starting a second SUMO environment in the same
  process. Run `evaluate_sac_delay.py` separately for Antwerp outcome metrics;
  checkpoints are no longer labeled "best" before evaluation.
- [x] Align the GRU action-history length with the wrapper's 20-action buffer.
  The upstream training script used `max_delay + 1`, which mismatches that
  buffer's length.
- [x] Give delay sampling an independent RNG keyed by the experiment seed and
  episode index. An explicit `reset(seed=S)` restarts episode 0; later resets
  without a seed advance the index. All five delay modes now replay for the
  same seed without consuming SUMO's or global NumPy's random stream.

**Retained choices and recorded differences**

- Detector `3849 / L4e` has an unverified exit mapping: its counts
  remain in the demand totals and its destinations are projected to D1/D2/D3.
  Revisit only if a verified detector-to-exit mapping becomes available.
- Keep the current 120 s ego departure, 6,000-step episode limit,
  13.89 m/s network speed limit, and Hard-scenario background queues until
  measurements justify a change. The objective is ego reaching D1, not
  removing congestion from the network.
- Keep the four Antwerp vehicle types and their SUMO parameters as the traffic
  baseline. The original `f_1`/`f_2` overrides targeted particular lane flows;
  Antwerp flow IDs represent scenario/OD/type and do not match those checks.
  Do not assign the old overrides to Antwerp types or invent a driver-profile
  ratio. The current `SetTrafficBehavior` checks therefore have no effect on
  Antwerp background vehicles; a sampled `car_van` retained `tau=1.0`, type
  maximum speed 45 m/s, and speed mode 31. Define any driver-profile groups
  separately if a future experiment needs them.
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

- [ ] Verify observation ordering under Hard traffic: `src/merging.py`
  currently truncates the vehicle list before sorting, so it may omit a closer
  neighbor when more than 30 are within the observation radius.
- [ ] Review the ego's 32 m/s cap against the network's 13.89 m/s limit as a
  separate speed-consistency question; preserve current values meanwhile.
- [ ] Before any `--safety` comparison, validate the controller at the angled
  O3-to-E75 merge. Its fixed 3.2 m target-lane offset does not use target-lane
  geometry or lane connectivity.
- [ ] Select the best Antwerp checkpoint using independent evaluation across
  the required scenarios and delay profiles; no checkpoint has been ranked yet.
- [ ] Build and train the five Hard-scenario Table IV variants under a stated
  common input-preprocessing and safety-controller policy. Report success,
  collision, no-merge/timeout, and return; generate the corresponding Fig. 5
  training curves. No full Antwerp model training or Table IV evaluation has
  been completed yet.

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

**Key arguments:**

| Argument | Options | Description |
|---|---|---|
| `--delay_mode` | `uniform`, `bimodal`, `bursty`, `triangular`, `exponential` | Delay distribution during training |
| `--scenario` | `easy`, `medium`, `hard` | Antwerp traffic scenario; defaults to `medium` |
| `--safety` | flag | Enable the safety controller (disabled by default) |
| `--encoder_mode` | `GRU`, `MLP`, `Transformer` | Delay-Aware Encoder architecture |
| `--run_name` | string | WandB run name |
| `--save_dir` | path | Directory to save model checkpoints |

## Evaluation

Evaluate a newly trained Antwerp checkpoint in a separate process:

```bash
python evaluate_sac_delay.py \
    models_experiments/antwerp-hard/GRU-hard-uniform-delay/GRU-hard-uniform-delay_100000_steps.zip \
    --scenario hard --delay-mode uniform --episodes 51
```

Pass `--safety` only when evaluating a run trained with the safety controller.
The script reports D1 arrival, collision, timeout, emergency stopping, and mean
return. Each episode gets a distinct seed for SUMO and delay sampling, so the
same checkpoint and evaluation seed reproduce the episode delay sequences.

Numbered checkpoints are saved every 100,000 training steps under
`models_experiments/<run_name>/`. The latest model is also saved as
`models_experiments/<run_name>.zip` at training completion. Neither is marked
"best" until independent evaluation compares them.

## Pretrained Model

A pretrained DAROM-GRU checkpoint is provided in `models/GRU-uniform-delay/`. The model was trained exclusively on the uniform delay distribution and evaluated zero-shot on all five distributions.

Use `GRU-uniform-delay_best` for evaluation.

## Results

Performance on Hard traffic mode (1,374–1,490 vph/lane) with random delays up to 2.0s:

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
