# DAROM: Delay-Aware Reinforcement Learning for Highway On-Ramp Merging

## 本地复现实验

当前已选择唯一长训练方案 **D / seed 42 / 累计 7M**，配置见 [d_long.json](experiments/d_long.json)。新入口 `.venv/bin/python -B -m experiments.long_run check` 可检查配置，`train` 开始训练；每 2M 保存完整断点，并在 7M 保存最终状态，支持 `--stop-at 2000000` 保存暂停和 `--resume <完整断点目录>` 续训。完整命令、恢复语义及验证范围见 [实验计划末节](docs/reproduction-experiment-plan.md#唯一长训练方案d--seed-42--7m2026-10-17)。2026-10-17 10:22（北京时间）已后台启动正式 7M，实时曲线见 [D 长训练](https://forge.coreweave.com/wandb/arlikache-anhui-university/DAROM-reproduction/runs/118fna3s)；原首轮模型不具备新入口所需的完整环境状态。

2026-10-17 调整评估调度时，D 在 657,971 步保存完整状态并暂停，15:59 从完整断点后台续训。训练进程使用本地 JSONL/TensorBoard 记录；随后按用户要求，**独立同步进程读取本地日志，先补传历史，再约每 5 分钟更新原 Forge run `118fna3s`**，本次接入没有中断训练。网络失败仅延迟云端同步，不阻塞训练；实际可见延迟还受上传与页面刷新影响。后续每 1M 评估、每 2M 保存完整状态，loss/熵系数仍每 100 步记录，奖励按回合记录。同步器当前跟随本次原始及恢复运行段，未来切换运行目录时需重新核对数据来源。运行状态与恢复校验记录见 [实验计划](docs/reproduction-experiment-plan.md#forge-曲线补传与调度变更2026-10-17)。

四组 A/B/C/D 的参数、分阶段预算、论文一致性复核和 Forge 使用方式见 [复现实验计划](docs/reproduction-experiment-plan.md)，可执行配置见 [first_round.json](experiments/first_round.json)。在本项目使用 `.venv/bin/python -B -m experiments.run list` 查看 8 个任务，`.venv/bin/python -B -m experiments.run check` 检查目标依赖。目标依赖和 CPU/GPU 短验证已通过；当前为受控对照配置，并非所有组都与论文完全一致。2026-10-16 已启动正式首轮：四组各两个种子、每次 100k，最多两任务同时运行，余下任务自动排队。曲线在线同步至 [Forge 项目](https://forge.coreweave.com/wandb/arlikache-anhui-university/DAROM-reproduction)，同时保存本地 TensorBoard/JSONL；队列目录与首批曲线链接见实验计划末节。此前的 [512 步联通验证](https://forge.coreweave.com/wandb/arlikache-anhui-university/DAROM-reproduction/runs/t3mc5mjp) 只用于检查接入，不参与效果对照。四组实验使用新入口 `experiments.run`；下方保留上游说明及旧入口作为参考。

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
