# DAROM-TPR · 双 Ego CTDE-MASAC

本分支 `exp/o3-d3-homogeneous-masac-v1` 在 Antwerp SUMO 路网上研究双车驶离任务。两辆 Ego 分别从 O3 出发，使用各自的延迟感知 Actor 驶向 D3。训练时由集中式 Critic 估计联合状态下的动作价值，评估时由冻结的 Actor 各自决策。联合到达指两车在同一回合都到达目标。

## 环境与快速验证

从仓库根目录运行。需要 Python、PyTorch、Gymnasium、TraCI 和 SUMO。安装 SUMO 后，确认 `sumolib.checkBinary("sumo")` 能找到预期版本。图形评估另需 `sumo-gui`。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install pytest
sumo --version
```

下面是一组小规模训练和评估命令。使用新的输出目录。正式训练和评估在远程运行。

```bash
.venv/bin/python -B train_masac.py \
  --scenario easy --device cpu --seed 42 --total-steps 64 \
  --learning-starts 8 --batch-size 8 --buffer-capacity 128 \
  --no-tensorboard --no-resume --output-dir runs/readme_smoke

.venv/bin/python -B evaluate_masac.py \
  --checkpoint runs/readme_smoke/checkpoints/masac_seed42_step_64.pt \
  --episodes 1 --seed-start 200001 \
  --output runs/readme_smoke/eval/step_64/summary.json
```

测试入口为 `.venv/bin/python -B -m pytest tests/ -q`，包含真实 SUMO 集成测试。只检查 Actor、Critic 和 Replay 时，可运行以下子集。

```bash
.venv/bin/python -B -m pytest -q \
  tests/test_masac_actor.py tests/test_centralized_critic.py \
  tests/test_joint_replay_buffer.py tests/test_masac_trainer.py \
  tests/test_transition_collector.py
```

## 场景与控制

| 场景 | Ego 配置出发速度 | 背景车入网速度配置 |
| --- | ---: | --- |
| [`easy`](sumo_new/Antwerp_easy_o3_d3_homogeneous_dual.rou.xml)（默认） | 10 m/s | 小客车和摩托车按入口限速请求入网 |
| [`easy_speed18`](sumo_new/Antwerp_easy_speed18_o3_d3_homogeneous_dual.rou.xml) | 18 m/s | 小客车和摩托车按车型期望速度请求入网，倍率为 `normc(0.75,0.08,0.55,0.9)` |

两种场景的 `ego_1` 和 `ego_2` 均从 `O3_0` 出发，计划出发时刻为 120 s 和 123 s，实际插入由 SUMO 决定。货车按车型最高速度请求入网。双车共享一个 SUMO 时钟，步长为 0.1 s，横向分辨率为 0.8 m。每辆车插入后独立计时，新训练默认最多 2,500 个活跃控制步。两辆车都结束后才重置回合。

[控制器](src/envs/control.py)接收两个范围为 `[-5, 5]` 的连续动作。第一维是请求加速度，控制器按当前速度限幅，速度上限为 120 km/h。第二维小于 `−5/3` 时请求向较小车道编号换道，大于等于 `5/3` 时请求向较大编号换道，其余保持。允许提交换道请求的路段为 E75、E76、E77、E78 和 E82，车辆使用 `laneChangeMode=0`。

训练与评估都支持两种 TraCI 请求有效期。默认 `--lane-change-request-mode legacy` 在 E76、E77 请求 4 s，其余允许换道的路段请求 0 s。`uniform` 默认各路段 0.1 s；可用 `--lane-change-request-duration` 设为 0、0.1、0.5、1 或 4 s。单独传有效期会选用 `uniform`。请求有效期由 TraCI 使用，车身横移由 SUMO 推进。

单模型评估可通过 `--lane-change-gate ego_1|ego_2|both` 启用入口门控，默认 `off`。提交新换道请求前，门控用实时目标车道前后净间隙和闭合速度，按恒速估算未来 4 s 最小净间隙；任一侧低于 2 m 时拦截。轨迹分别记录 Actor 请求、TraCI 提交和实测运动。已执行中的同向横移可继续。

## Actor、Critic 与 Replay

- **Actor**。每辆车有独立的 [DAROM-GRU Gaussian Actor](src/agents/masac.py)。默认每车输入 136 维，依次为 3 维实时自车、30 个周车槽位各 3 维、2 维目标出口、20 个二维动作的掩码历史及 1 维观测年龄。周车从 50 m 范围内按距离选取，特征为 Ego 坐标系下的纵向距离、横向距离和纵向相对速度，空槽补零。每辆车独立为周车数据包均匀采样 0–20 步的传输延迟，自车前三维每步用实时数据刷新。
- **路线上下文**。默认环境版本 v5 的自车三维为到任务终点的沿路线剩余距离、带符号的最少主动换道次数、实时速度。第一、第二维分别按 1,000 m 和 5 缩放。换道次数的正负表示最优走法首次主动换道的方向，左正右负，当前拓扑不可达时使用原始值 `+6`。出口编码为 D1 `[0,0]`、D2 `[1,0]`、D3 `[0,1]`。
- **可选前车特征**。`--actor-leader-features` 把 Actor 输入扩为 139 维，新增 120 m 内的前车净间距、带符号的闭合速度和 TTC，随周车一起延迟。TTC 封顶 10 s，无前车时使用 `[120, 0, 10]`。评估从 checkpoint 恢复该设置。
- **集中式 Critic**。[个人和团队 Twin-Q](src/agents/centralized_critic.py)共享观测口径。新训练中，每辆车的自车三维取自同一步 Actor 输入，加上实时周车与目标出口后，每车观测 95 维。个人 Critic 拼接两车观测、连续联合请求动作和活跃掩码，输入 196 维，分别估计两车价值。团队 Critic 额外接收两车的到达标志，输入 198 维，估计一个团队价值。训练用均匀回放和平方 TD 误差，各车独立学习熵温度 α。
- **联合 Replay**。默认每个共享 SUMO 控制步生成联合经验，记录每车的奖励、终止及动作有效性。采集器暂存上一步经验，以便把后续的步前 Missed Exit 终止归到上一次有效动作。训练使用 `--decision-interval-steps 2` 时，两车按共享决策边界更新动作，新车激活可提前开启下一区间。Replay 改为按区间存储，奖励和 bootstrap 按实际区间长度折扣；训练预算、保存间隔和更新频率仍以 SUMO 步计数。

## 奖励与终止

新训练的奖励默认值来自 [`MASACRewardConfig` 和 `MASACRewardProtocol`](src/rewards/masac.py)。布尔模块可通过 `--reward-<模块>` 和 `--no-reward-<模块>` 切换。

| 模块 | 默认 | 结算 |
| --- | --- | --- |
| `task` | 开 | 到达 +50，碰撞 −500，错过出口、急停、停滞或超时 −400，每车结算一次终止事件。 |
| `merge` | 开 | O3 Ego 首次完全汇入 E75 主路车道或进入 E76 时 +5。 |
| `following`、`rear-following` | 开 | 分别用实时净间隙和 TTC 风险扣分，单侧每步最多 −0.1。参考时距 1.5 s、TTC 阈值 3 s、前瞻 120 m。 |
| `lane-change` | 开 | 物理横移期间，目标车道前后风险峰值每增加一单位扣 0.5。 |
| `efficiency` | 开 | 每活跃步按步前速度奖励 `0.01 × min(1, v / 20)`，速度单位 m/s。 |
| `comfort` | 开 | 有实测加速度对时，扣 `0.0025 × min(1, abs(a_after − a_before))`。 |
| `team` | 开 | 双车均到达后一次性 +100，独立于个人回报。Actor 目标中的团队 Q 权重默认 1.0。 |
| `route-preparation` | 开 | 默认 `linear_progress_v1`，按合法换道需求、距出口距离和有效前进量结算。 |
| `route-guidance` | 关 | 启用后，每辆车首次通过其任务规则指定的连接时默认 +5。 |
| `exit-retention` | 关 | 启用后，步首在 `E77_0`、步末在 E77 较大编号车道，本步扣 0.5；传送排除。 |

[`linear_progress_v1`](src/envs/route_preparation.py) 用公式 `0.02 × G(n) × clip(d/1000, 0, 1) × clip(Δs/1, 0, 1)`。所需换道次数 `n` 为 0 时 `G=1`，为 1 时 `G=0.5`，其余为 0。`d` 是步首到 E77 末端的路线距离（m），不计内部连接长度；`Δs` 是核实过的实际前进距离（m），跨车道连接时计入经过的内部车道。规划路线的上游路段也可计奖。旧检查点按其存档模式恢复路线准备。

错过出口由[路线车道图](src/envs/route_lane_plan.py)和[出口规则](src/envs/exit_rules.py)判定。进了不通往任务目标的连接，或在必经路段末端接近不可恢复的错误车道，都可触发终止。新训练默认启用停滞终止；可用 `--no-stalled-terminate --stalled-shadow` 只收集停滞诊断。v5 停滞检测核对实际车道的合法连接，外部阻挡适用 3 s 宽限。SUMO 原生 Emergency Stop 也可触发个体终止，碰撞优先结算；评估另行统计按实测减速度定义的急刹。SUMO 配置禁用 teleport，碰撞车辆移除。

`--terminal-penalty-scale` 可缩放负任务终止奖惩，`--collision-penalty` 可直接指定碰撞惩罚的幅值。SUMO 原生急停的扣分固定为 −400。`--reward-comfort-weight` 和 `--reward-lane-change-weight` 可调整对应权重，评估按 checkpoint 恢复。

## 训练与续训

以 `train_masac.py --help` 为完整参数表。以下为从零训练时的代码默认值。

| 选项 | 默认值 |
| --- | --- |
| 全局预算 / 训练 seed | 1,000,000 步 / 42 |
| Replay 容量 / batch | 1,000,000 / 512 |
| 初始随机探索 / 更新间隔 | 100 个全局步 / 1 个全局步，更新还需足够 Replay 样本 |
| 学习率 / γ / Polyak τ / 初始 α | `3e-5` / `0.99` / `0.005` / `0.01` |
| 设备与算子线程 | `auto`，可用时选 `cuda:0`（2 线程），否则 CPU（最多 8 线程） |
| 输出与保存 | `runs/masac`，每 500,000 步和最终步保存，JSONL 与 TensorBoard 默认开启 |

例如，远程 1M 训练可显式设定 20 万步检查点并保留最终 Replay。

```bash
.venv/bin/python -B train_masac.py \
  --scenario easy --device cuda:0 --seed 42 --total-steps 1000000 \
  --checkpoint-every 200000 --replay-final-only --no-resume \
  --output-dir runs/easy_s42_1m
```

每次启动写出 `resolved_config.json`，包含实际参数、网络、优化器、环境及 SUMO 命令与输入哈希。训练输出 `train/metrics.jsonl` 记录探索期回合、运行窗口与梯度更新；TensorBoard 写入 `train/tb/`。训练滚动到达率与冻结模型评估属于不同协议。

- **继续训练**。续训默认开启。沿用同一输出目录、移除 `--no-resume`，并将 `--total-steps` 设为新的累计目标。需要同一保存点的模型 `.pt`、配对 `.replay.npz` 和有效索引。训练核对关键配置和哈希，从下一回合开始。恢复时舍弃中断前尚未提交的那条经验，因此回合轨迹可能与连续训练不同。
- **保存空间**。模型和 Replay 原子发布，`checkpoints/index.json` 记录哈希。默认周期节点含 Replay；使用 `--replay-final-only` 时，只有最终保存点可完整续训，中间模型用于评估。
- **历史口径**。`critic_uses_actor_ego` 写入新检查点，续训继承存档值。缺少该字段的检查点沿用累计行驶距离、当前车道编号和速度，以匹配旧 Replay。v2–v4 的环境差异由版本和存档配置恢复，旧检查点缺少换道有效期配置时使用 `legacy`。
- **只初始化 Actor**。用 `--init-actors-from CHECKPOINT --init-actors-sha256 SHA256` 从兼容模型初始化两只 Actor。使用新输出目录，训练入口另行初始化 Critic、温度、优化器和 Replay。

## 评估与结果

### 单模型

`evaluate_masac.py` 在 CPU 上加载冻结 Actor。默认确定性动作、5 回合、种子从 `200001` 开始、每回合最多共享推进 4,500 步。共享步上限与 checkpoint 中的逐车活跃步超时分别生效。用 `--episodes 100 --seed-start 200001` 开展固定集评估，用 `--stochastic` 单独检查随机策略。默认继承 checkpoint 的场景、决策间隔、换道有效期和奖励配置。单次评估可覆盖场景、决策间隔、有效期或停滞终止，也可加 `--gui --gui-focus ego_1` 查看车辆。

评估专用的决策降频会在中间步保持上次请求的纵向加速度，按实时速度限幅，横向请求只在刷新动作的步提交。对每步决策训练的旧模型临时降频时按各车激活后的步数计时；降频训练的模型继续使用带新车激活边界的共享时钟。报告记录实际决策协议。

### 批量检查点

```bash
.venv/bin/python -B scripts/evaluate_masac_checkpoints.py runs/easy_s42_1m
```

脚本按保存步排序，每个检查点运行同一组 `200001–200100` 种子的确定性评估。场景、决策间隔和换道有效期从各检查点恢复。它校验模型与检查点索引以及既有评估的协议和来源。有不兼容的旧结果时会报错；明确接受重算该目录结果时才用 `--rerun`。训练入口本身不自动启动评估。

### 结果口径和诊断

- 单模型评估写 `summary.json`、逐种子 `episodes.csv` 及默认启用的 `eval_trajectory.jsonl.gz`。批量输出位于 `eval/step_<步数>/`，并写 `eval/index.json`。用 `--no-trajectory` 关闭单模型轨迹；`--sumo-lanechange-output` 另按种子输出原生换道 XML。
- 联合到达按同回合两车均到达计数；联合碰撞等事件率按任一车发生计数。普通到达与无车道末端受阻到达分别统计，后者依赖 SUMO 的车道末端无连接急停警告。缺少警告源时保留无法分类的到达数。比例统计保留分子、分母和 Wilson 95% 区间；跨训练种子的离散性用独立训练重复衡量。
- 速度和 Jerk 按有效样本计算回合均值，再按有效回合汇总。实测急刹（加速度 ≤ −4.5 m/s²）与 SUMO Emergency Stop 分开记账。行为统计区分 Actor 请求、TraCI 提交、同一道路上相邻有效车道观测的变化，以及轨迹中的横移完成事件。
- 评估核心统计标识为 `homogeneous-dual-ego-v1`，输出协议为 v4。报告同时保存请求的 `speed_mode=96` 和实际读回值。复现对照时核对 Git 提交、SUMO 与 Python/PyTorch 版本、路网及路线哈希、训练种子、评估种子、动作采样模式和实际生效的终止与控制协议。

逐步诊断及并排比较见[轨迹分析工具](scripts/trajectory_dashboard/README.md)。离线统计图使用 [`scripts/plot_masac_experiments.py`](scripts/plot_masac_experiments.py)，每个 `--experiment` 传入算法名、训练种子和运行目录。

```bash
.venv/bin/python -B scripts/plot_masac_experiments.py \
  --experiment MASAC 42 runs/easy_s42_1m \
  --output runs/easy_s42_1m/figures
```

## 代码与历史配置

| 路径 | 用途 |
| --- | --- |
| [`train_masac.py`](train_masac.py)、[`evaluate_masac.py`](evaluate_masac.py) | 训练、续训和独立评估入口 |
| [`src/agents/`](src/agents/) | Actor、集中 Critic、更新、联合经验和回放 |
| [`src/envs/`](src/envs/)、[`src/rewards/`](src/rewards/) | SUMO、双车生命周期、观测延迟、路线、控制与奖励 |
| [`sumo_new/`](sumo_new/) | 路网及路线。当前双 Ego CLI 提供 Easy 两个变体；Medium/Hard 路线为单车参考。 |
| [`conf/legacy/`](conf/legacy/README.md) | 历史 YAML；当前训练直接解析 CLI 和 checkpoint。 |
| [`tests/`](tests/) | 单元测试及真实 SUMO 训练、回放、评估回归 |
