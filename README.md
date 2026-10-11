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
combined because the supplied length data cannot separate them. These
files contain background traffic only, not a controlled `ego` vehicle.
`src.merging.Merging(seed=579071, scenario="easy")` loads one of the
`easy`, `medium`, or `hard` Antwerp routes (`medium` by default). Each reset
warms up for at least 120 simulated seconds, draws a takeover time between
120 and 600 seconds, and selects the next `car_van` that actually enters O3
with destination D1. The selected flow vehicle keeps its original SUMO type;
`reset()` returns its real ID and departure time in `info`. The environment
starts SUMO from the network and selected route files with the timing and
lateral settings of `Antwerp.sumocfg`, without detector outputs. The old
lane/coordinate logic and 200-second absolute episode limit still need
adaptation before complete training or evaluation runs.
The existing network speed limit remains 13.89 m/s, so its simulated speed
and congestion should be calibrated before using these scenarios to assess
policy performance.

## Training

Train DAROM-GRU from scratch:

```bash
python train_sac_delay.py \
    --delay_mode uniform \
    --encoder_mode GRU \
    --run_name GRU-uniform-delay \
    --save_dir models/GRU-uniform-delay
```

**Key arguments:**

| Argument | Options | Description |
|---|---|---|
| `--delay_mode` | `uniform`, `bimodal`, `bursty`, `triangular`, `exponential` | Delay distribution during training |
| `--encoder_mode` | `GRU`, `MLP`, `Transformer` | Delay-Aware Encoder architecture |
| `--run_name` | string | WandB run name |
| `--save_dir` | path | Directory to save model checkpoints |

## Evaluation

Evaluate the pretrained model:

```bash
MODEL_PATH=models/GRU-uniform-delay/GRU-uniform-delay_best \
DELAY_MODE=uniform \
python evaluate_sac_delay.py $MODEL_PATH
```

Set `DELAY_MODE` to any of `uniform`, `bimodal`, `bursty`, `triangular`, or `exponential` to test generalization across delay profiles.

**Output metrics**: Success rate, Collision rate, No-merge (timeout) rate, Average episode reward, Ego velocity, Acceleration, Jerk — reported as mean ± std across 3 seeds × 500 episodes.

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
