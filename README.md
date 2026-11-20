# DAROM-TPR — Dual-Ego CTDE-MASAC for Highway D3 Exit

> **状态：研究原型 / v0.1 训练管线基线（尚未证明策略收敛）。** 适用分支：`exp/o3-d3-homogeneous-masac-v1`。本分支的主要任务是让两辆独立决策的 Ego 在同一个 SUMO 仿真中驶离 D3，而不是人为制造车辆冲突或同步出发时差。

## 1. 研究任务

使用 DAROM 的延迟感知 GRU 观测编码器，在 Antwerp 路网构建双 Ego 驶离场景，采用集中训练、分散执行（CTDE）的多智能体 SAC（MASAC）：

| 项目      | 当前实现                                                     |
| --------- | ------------------------------------------------------------ |
| 场景      | Antwerp **Easy**（双 Ego；Medium/Hard 双 Ego 尚未接入）      |
| `ego_1`   | O3 → D3；计划 120 s 出发；`O3_0` 起步                        |
| `ego_2`   | O3 → D3；计划 123 s 出发；`O3_0` 起步                      |
| 仿真      | 一个 SUMO 进程、一个 0.1 s 全局时钟；各车实际插入时刻可不同  |
| 速度      | 道路及 Ego 限速 120 km/h（33.333 m/s）；Ego 出发速度 10 m/s  |
| Episode   | 每辆 Ego 默认最多 **2,500 个活跃控制步（250 s 仿真时间）**；分别终止，双方结束后重置 |
| Ego 控制  | 纵向加速度 + 横向换道请求；`laneChangeMode=0`、无自定义安全拦截 |
| 奖励      | 模块化奖励，具体配置见下文 |

**训练目标**：分别提高两辆 Ego 的安全抵达率，最终提高**同一回合两车同时抵达 D3 的联合成功率**。两车均由各自的 Actor 决策，计划出发间隔为 3 s，实际插入时刻由 SUMO 决定。

## 2. CTDE-MASAC 结构

- **分散 Actor**：每辆 Ego 使用独立的 `GaussianActor`（DAROM `DelayAwareGRUEncoder` + tanh-Gaussian 策略）；仅使用自己的 136 维延迟策略输入。
- **Actor 输入（136）**：当前 Ego 状态及最多 30 辆邻车的 31×3 基础观测（93）＋目标出口编码（2）＋20×2 自身动作历史（40）＋观测年龄（1）。邻车状态带通信延迟；Ego 自身状态实时更新。新动作位于历史开头，按实际观测年龄决定保留数量。
- **集中 Critic**：`CentralizedTwinQ` 接收两辆 Ego 的当前真实交通观测与目标出口（每车 95 维）、联合请求动作及 active mask，拼接为 196 维输入；两套 Q 网络分别对两辆车输出 Q 值。**Critic 不使用 Actor 的 256 维 GRU 特征。**
- **联合 Replay**：每个 SUMO 全局步对应一条双 Ego 联合经验，分别保存奖励、done、active/acted/next-active mask。`JointTransitionCollector` 延迟一个全局步提交，以正确归因 pre-step Missed Exit。
- **优化**：SAC Twin-Q 目标、Polyak 目标网络、分 Ego Actor 优化器和独立自动熵系数 α；Critic 使用真实联合状态，执行时只需各自 Actor。

Actor 输出两个连续值，均限制在 `[-5,5]`。纵向值对应加速度（m/s²）；横向值经阈值离散化：小于 `−5/3` 请求朝车道 0 换道，在 `[-5/3, 5/3)` 保持，大于等于 `5/3` 请求反方向换道。当前训练中的 Critic 看到**连续请求动作**，而 SUMO 执行的是离散换道命令；这是后续需要检验的建模限制。

### 模块化奖励

新训练默认开启下列七个模块。参数采用 `--reward-<模块>` 和 `--no-reward-<模块>`，可独立组合。直接构造环境时传入 `MultiEgoEnv(reward_config=MASACRewardConfig(...))`。

| 模块 | 结算规则 |
| --- | --- |
| `task` | 到达指定出口 +50、碰撞 −500、错过出口 −400、超时 −400。每车只结算一次终止事件。 |
| `merge` | O3 车辆完全进入 E75 的主路车道，或经 E75_0 进入 E76 时，首次 +25。后续任务失败保留该奖励。 |
| `following` | 每个活跃控制步按步前实际前车净间隙结算 −0.1 × C_f，安全时距 1.5 s。 |
| `lane-change` | SUMO 出现实际换道运动时，按起点步的目标车道前后净间隙结算一次 −0.1 × C_lc。反应时暂用 1.5 s，最小缓冲 2 m，前后车共用参考减速度 4.5 m/s²。该参数仅用于计算风险间隙。 |
| `efficiency` | 每个活跃控制步按步前速度结算 +0.05 × min(1, v/20)，速度单位 m/s。 |
| `comfort` | 有实测步前和步后加速度时，结算 −0.01 × clip(加速度差的绝对值, 0, 1)。0.01 为待测权重。 |
| `team` | 双车均到达指定出口时，回合一次性记录 +100。环境通过 `env.team_reward` 提供本步值，日志和评估 JSON 用 `team_reward` 记录回合值。独立团队 Twin-Q 学习该回报，每辆 Actor 最大化个人 Q 加团队 Q。Actor 团队权重用 `--team-q-weight` 设置，默认 1.0。 |

路线引导奖励为 0。Actor 与 Critic 均收到各车自己的目标出口编码（D1 为 [0,0]，D2 为 [1,0]，D3 为 [0,1]）。SUMO 启动时固定将等待、高速路错车道、断路和双向车道 teleport 阈值设为 `-1`，车辆碰撞使用 `remove` 策略。所有奖励模式、Shadow 和正式 Stalled 开关均使用此配置。残余 Ego teleport 在 `teleport_events` 记录时间和起止标志。若同一步检测到碰撞则记录该证据，其余原因标记为 `unknown`。个人回报包括已启用的个人模块及显式开启的旧 USP，可从 `reward_components` 和 USP 统计追溯；团队回报单独记账。

团队 Q 按共享 SUMO 步折扣。一辆 Ego 先到达时，团队价值继续沿后续状态传递，直至全员到达或任意 Ego 失败。Replay 区分已到达与尚未出发；个人奖励和团队奖励分别写入日志。

例如，追加 `--no-reward-comfort --no-reward-team` 可关闭舒适度和团队奖励。开关保存在 checkpoint，评估自动复用。未含新奖励配置的旧 checkpoint 沿用 134 维 Actor 和 B0 终止奖励。

## 3. 目录结构

```text
train_masac.py                     双 Ego CTDE-MASAC 训练（当前为基础训练入口）
evaluate_masac.py                  双 Ego 独立评估（deterministic / --stochastic）
src/agents/
  masac.py                         两个独立 DAROM-GRU Gaussian Actor
  centralized_critic.py            集中式 Twin-Q，当前输入 196 维
  replay_buffer.py                 联合经验回放
  transition_collector.py          pre-step 终止奖励归因
  masac_trainer.py                 Critic / Actor / alpha 更新
src/envs/
  multi_ego_env.py                 双 Ego 独立状态与统一 SUMO 时钟
  sumo_world.py                    SUMO 启动、推进、关闭
  ego.py                            Ego 生命周期
  observation.py                    每车 31×3 交通观测
  delay_buffer.py                   每车独立通信延迟队列
  action_history.py                 每车动作历史掩码及 134 维编码
  control.py                        连续动作映射及 TraCI 控制
  exit_rules.py / lane_change.py    D3 终止判定与换道状态跟踪
src/delay_encoder.py                DAROM 延迟感知 GRU
src/rewards/events.py               终止事件映射及旧版 B0 奖励
src/rewards/masac.py                模块化奖励配置及回合记账
sumo_new/
  Antwerp.net.xml                   Antwerp 路网
  Antwerp_easy_o3_d3_homogeneous_dual.rou.xml  当前双 Ego Easy 路线
  Antwerp_easy_o3_d3.rou.xml        单 Ego O3 参考路线
  Antwerp_medium_o3_d3.rou.xml      单 Ego Medium 路线
  Antwerp_hard_o3_d3.rou.xml        单 Ego Hard 路线
tests/                               环境、模型、真实 SUMO、完整回合回归测试
train_sac_delay.py / evaluate_sac_delay.py
                                     历史单 Ego DAROM-SAC 入口（非当前 MASAC 入口）
```

`conf/` 中的旧 E76 配置及尚未接入代码的空 YAML **不是当前 MASAC 训练参数来源**；当前参数来自 `train_masac.py` 的 CLI 默认值和对应模块的构造参数。

## 4. 运行环境

需要 Python、PyTorch、Gymnasium、TraCI/SUMO。当前本地集成验证使用 **SUMO 1.20.0**；不同 SUMO 版本可能改变碰撞、换道与交通动力学，正式比较应统一版本。

```bash
# 先确认 SUMO 可执行文件在 PATH 中
sumo --version
python3 -m pip install -r requirements.txt
python3 -m pip install pytest
```

如安装方式需要 SUMO Python tools，再配置 `SUMO_HOME`。代码通过 `sumolib.checkBinary('sumo')` 启动仿真。运行命令应从仓库根目录执行。

## 5. 训练

当前入口提供**从零训练**、联合采样、梯度更新、结构化日志及周期 checkpoint，不提供精确断点续训。训练步数是 SUMO 全局控制步，不是两车活跃步数之和。

```bash
# 已经过本地 2,500 全局步的训练冒烟验证
python3 train_masac.py \
  --total-steps 2500 \
  --learning-starts 32 \
  --batch-size 32 \
  --buffer-capacity 5000 \
  --train-every 10 \
  --log-every 500 \
  --gamma 0.999 \
  --learning-rate 3e-5 \
  --initial-alpha 0.01 \
  --seed 42 \
  --output-dir runs/masac_smoke_2500
```

正式训练默认参数为 1,000,000 个共享 SUMO 全局步、Replay 容量 1,000,000、batch 512、Learning Starts 100、学习率 3e-5、γ=0.99、`--tau=0.005`、seed 42，每 500,000 步保存 checkpoint，默认输出 `runs/masac`。两辆 Ego 的 α 独立学习。上述 2,500 步命令显式使用小参数，仅为历史冒烟配置。训练开始的前 `learning-starts` 个全局步使用均匀随机动作，后续使用随机 Actor 策略。

训练结束保存 `runs/masac_smoke_2500/checkpoints/masac_seed42_step_2500.pt`，其中包含 Actor、Critic、Target Critic、α 和优化器状态；**不包含 Replay Buffer、SUMO 物理状态或随机数全状态，不能精确续训**。`runs/` 由 `.gitignore` 排除，模型文件需另行归档。

## 6. 独立评估

```bash
# 确定性策略（正式汇报优先使用）
python3 evaluate_masac.py \
  --checkpoint runs/masac_smoke_2500/checkpoints/masac_seed42_step_2500.pt \
  --episodes 10 \
  --seed-start 200001 \
  --output runs/masac_eval_2500/deterministic_10/summary.json

# 随机采样策略（探索行为诊断，不与确定性评估混为一谈）
python3 evaluate_masac.py \
  --checkpoint runs/masac_smoke_2500/checkpoints/masac_seed42_step_2500.pt \
  --episodes 10 \
  --seed-start 200001 \
  --stochastic \
  --output runs/masac_eval_2500/stochastic_10/summary.json
```

评估器加载已冻结的 Actor，不更新参数。默认生成 `runs/masac_eval/summary.json` 和同目录 `episodes.csv`；自定义 `--output` 保留 JSON 文件名，CSV 固定为同目录 `episodes.csv`。CSV 每个 seed 一行，以 `ego_1_`、`ego_2_`、`joint_` 区分。JSON 保留旧字段，并增加 `per_agent`、`joint`、统计版本和参考提交。旧 `success_rates` / `joint_success_rate` 使用 [0,1]；新增分组中的率使用百分数。

指标标准 `dual-ego-v1` 的统计参考固定为 `913a73fef22be64ca1ac0ade5426874374b566cd`，通过 `git show` 核实实现，不能用工作树后续版本替代。定义如下：

| 指标 | 采集与汇总 | 与旧版关系 |
| --- | --- | --- |
| 速度 m/s | 有效在路步前 TraCI 实测值；回合内均值，再对有效回合等权 | 与旧评估器采样、权重一致；双车各自从插入后开始 |
| Jerk m/s³ | 实测步后/步前加速度差除以实际时间差，取绝对值；排除每车首个控制转换和终止转换 | 与旧评估器一致；旧 recorder 使用固定 dt，只有 dt=0.1s 时数值一致 |
| 急刹 | 有效步后实际加速度 ≤ −4.5 m/s²，包含仍在路上的终止步；报告步数、发生回合数、回合率 | 阈值和终止步规则与旧 merging 一致 |
| SUMO emergency stopping | 原始 SUMO 事件，报告步数和发生回合率 | 与旧事件统计一致；不等同于急刹 |
| Teleport | 累计 starting 或 ending 事件，即使碰撞为最终原因也保留 | 与旧累计规则一致 |
| 成功 / raw arrival | 成功为 Arrived 且从未 Teleport；raw arrival 只判断 Arrived | 成功公式与旧评估器一致；旧 recorder 的 success 仅判断 Arrived，非严格一致 |

上述一致指同一有效输入上的统计规则；双 Ego 的插入时刻、路径和现有终止实现不同，不代表跨场景的成功事件或性能可直接等同。环境终止规则不因指标适配改变。

速度、Jerk、急刹分别记录 `speed_samples`、`jerk_samples`、`hard_braking_samples`；事件记录 `event_samples`，不得共用分母。连续量无有效样本时 JSON 为 `null`、CSV 为空；汇总另记各自有效回合数。急刹等回合率以完整回合数为分母，不以速度/Jerk 样本数为分母。

联合成功要求同回合两车均成功，联合 raw arrival 要求两车均 Arrived；联合碰撞、Timeout、Missed Exit、Teleport、急刹及 emergency stopping 回合率表示任一车发生，步数和样本数相加。联合速度/Jerk 先对同回合有效车辆等权，再跨有效回合等权；CSV 保留各回合有效车辆数。这些联合指标是新增口径，无旧单车严格对应项。**联合成功率不是两个单车成功率的乘积。**

`global_steps` 是评估循环中 SUMO 共享推进次数（不含 reset 的插入预热）；`agent_steps_sum` 是两车实际控制步数之和（与环境 `active_steps` 一致，不含步前 Missed Exit 已终止的车）。兼容字段 `steps` 等于 `global_steps`，不能将控制步数之和当作仿真推进次数。

随机模式每回合固定 Python、NumPy、PyTorch、SUMO 和 delay buffer 种子。Actor 保持 ego_1、ego_2 的固定采样顺序，包括 inactive Actor 采样后掩码；诊断只读且不消耗随机数。确定性和随机模式分别汇报，不混合。复现保证限于相同模型、软件与硬件环境。

缺失/空终止原因、Unknown/UnknownRemoval、环境异常和步数上限导致的未完成评估直接报错，不计完整回合，也不输出本次最终文件。`Other` 仅用于环境正常返回的其他非空合法终止事件，原始原因同时保留。

验证：`.venv/bin/python -m pytest tests/test_evaluation_metrics.py -q` 包括固定提交函数对照、独立掩码/分母、真实 SUMO 诊断开关前后确定性行为和随机状态对照、随机重复性、完整输出重算与中断拒绝。该验证属于小规模本地功能验证，不是正式远程性能评估。

### 6.1 训练记录与周期 checkpoint

默认 `--device auto`，CUDA 可用时选择 `cuda:0` 并用 2 个 PyTorch CPU 算子线程，否则在 CPU 训练，线程数为 PyTorch 当前默认值与 8 的较小者。可用 `--device cpu`、`--device cuda:N` 或 `--torch-threads N` 显式覆盖。SUMO 与 Replay 存储在 CPU。请求设备、实际设备和线程数写入配置。评估器可在 CPU 读取 CUDA 训练生成的 checkpoint。

训练参数 `--checkpoint-every 500000`；默认每 500,000 个共享 SUMO 控制步保存一次，结束时保存最终状态，恰逢节点只保存一次。`--tensorboard` 默认开启，可用 `--no-tensorboard` 关闭；JSONL 始终开启。动作窗口复用 `--log-every`（默认 40 步），最后不足窗口仍写实际长度。实验目录不可重复写入已有 `train/metrics.jsonl`，避免混合两次训练。

```text
runs/<experiment_id>/
  train/metrics.jsonl
  train/tb/
  checkpoints/masac_seed42_step_500000.pt
  checkpoints/index.json
  eval/step_500000/summary.json
  eval/step_500000/episodes.csv
  eval/index.json
  figures/
```

JSONL 以 `event` 区分：

- `config`：完整 CLI、Git 提交及工作树是否有修改；`global_step=0`。
- `episode`：仅两车均完整终止后写入，包含 `episode_index`、`episode_global_steps`、每车终止原因/奖励/严格成功/有效控制步数、联合成功；最近 100 回合的 ego_1 / ego_2 成功、碰撞、Missed Exit、Timeout 百分率和平均奖励，以及联合成功率和两车奖励和的均值。`rolling_episodes` 是实际分母，`policy=exploration_training`，不能冒充正式评估。
- `stalled_shadow`：用 `--stalled-shadow` 开启 Stalled-v1。每个共享步输出两辆 Ego 的步前诊断，环境在 `infos[ego]["stalled_shadow"]` 中提供同样的数据。每车独立累计连续有效的 10 秒滑窗，用 SUMO 里程表计算前进距离，小于 20 m 时输出 `would_stall=true`。事件包含首次触发仿真时刻、道路、车道、窗内前进量、累计里程、50 m 内前车及判定原因。前车拥堵、计划停车、邻近红灯或路口冲突、无可通行连接及里程采样中断都会重置窗口。前车通行空间按净间隙扣除最小间距，加上前车按当前速度预计 10 秒的位移判断；路口只检测当前内部连接或距车道末端 20 m 内的冲突。影子输出仅供分析。
- `--stalled-terminate`：默认关闭，可独立于 `--stalled-shadow` 开启。复用 Stalled-v1 的步前检测，在 SUMO 步完成后，按 Collision、Ego Teleported、Missed Exit、Arrived、Stalled、Timeout 的顺序结算。Stalled 仅结束触发的 Ego，单次任务分为 −400；正常步进奖励按原规则结算。团队目标在首次失败时结束。正式 Stalled 与 USP 互斥。评估默认从 checkpoint 继承，也可用 `--stalled-terminate` 或 `--no-stalled-terminate` 覆盖。训练滚动指标与评估报告均含 `stalled_rate`，自动绘图在失败指标图中展示。
- `control_trace`：使用 `--trace-controls` 时，每个实际控制步记录前车 ID 和实际净间隙、Ego 道路和车道位置、步前和步后速度、策略请求加速度、TraCI 提交加速度、SUMO 步后实测加速度和原始到达标志。前车由 TraCI 沿当前路线搜索，搜索最短前瞻 100 m。查无前车或车辆已在步后移除时，相应字段为 `null`。
- `update`：第一轮、每隔 `--update-log-every` 轮（默认 40）以及最后一个可执行更新步记录一次。设置为 1 可记录每轮。每条记录保留该轮的原始 `critic_loss`、双 Q 均值 `mean_q`、`mean_target`、`valid_agents`、`update_index`；各车 `actor_loss`、`alpha`、`alpha_loss`、`mean_log_prob`、`valid_samples`。log-prob 是当前 actor 更新中已有采样的实际动作 log-prob，未再额外前向；无有效样本的损失为 null、样本数为 0。
- `runtime`：窗口实际步数、速度（SUMO 全局训练步/墙钟秒）、Replay 大小、累计更新数、各车动作样本数/有效动作占比、加速度均值与总体标准差、横向动作均值、换道请求比例。只统计真正执行动作的样本；换道比例直接调用当前控制器 `parse_action()`，不另定义阈值。

TensorBoard 的标量标签统一带 `exploration_training/` 前缀，以 `global_step` 为横轴；例如 `tensorboard --logdir runs/<experiment_id>/train/tb`。更新曲线展示实际记录轮次的 minibatch 值，横轴使用真实全局步。运行窗口仍记录累计更新总数。

checkpoint 保留原有模型/优化器字段，并增加 Git 和源码 SHA256；索引记录训练 seed、global_step、绝对路径、文件 SHA256、提交、源码哈希和配置。文件先写 `.partial`，flush/fsync 后原子替换，索引同样原子写入；残留 `.partial` 不参与发现。双 Ego 联合 Replay 的 float32 数组容量 1,000,000 时约 3.49 GiB/组（不含模型、Python 和采样临时张量）；同样的容量数值不代表与单 Ego 相同的内存或样本结构。checkpoint 仍缺 Replay、SUMO 状态、全部 RNG 状态，**不提供精确续训**。

### 6.2 固定种子评估与只读行为诊断

```bash
# 以下是正式评估入口示例；本次开发验证没有执行 100 回合评估
.venv/bin/python scripts/evaluate_masac_checkpoints.py runs/<experiment_id>
# 配置不一致或损坏时默认报错；明确重算时加 --rerun
```

入口按 checkpoint 内部 `global_step` 排序；每个模型固定 Easy、两车、确定性 Actor、100 回合、种子 `200001–200100`，不提供修改这些设置的选项。`--max-steps` 默认 4500；只限制评估器等待回合结束的共享步数。环境按 checkpoint 的 `max_active_steps` 分别判定每辆 Ego 超时。新训练可用 `--max-active-steps` 设置，默认 2,500。旧 checkpoint 缺少该项时按原 4,000 步评估。批量评估不在训练过程中自动运行。

复用结果前核对 JSON/CSV 所有回合字段、分组汇总重算、checkpoint SHA256、种子序列、协议版本、Git 提交、源码哈希、路网/路线和 SUMO 二进制哈希、环境配置及实际 speed mode 读回记录。任何失败记录到 `eval/index.json` 后报错，后续项保留 pending，不能当作已评估；索引缺少模型文件也报错。

单模型入口新增 `--trace`，默认关闭首回合 TRACE。请求 `speed_mode=96` 保持不变，实际 `getSpeedMode()` 的所有读回值按车保留在每回合和环境元数据中。本机 SUMO 1.20.0 的短程测试读回为 32（请求为 96）；工具原样记录，不将请求值冒充实际值，也不修改控制模式。JSON 同时记录 checkpoint SHA256、训练配置、Git/源码哈希、SUMO 版本、NumPy/PyTorch 版本、路网/路线哈希及关键评估配置。

核心统计仍为 `dual-ego-v1`，扩展输出协议版本为 2；旧 JSON 字段和单位保留。新增联合无碰撞率为没有任一 Ego 碰撞的完整回合比例（不代表成功）。所有回合率提供 `*_stats`，包含分子、分母和 Wilson 95% CI（单位与对应比例一致）；100 个评估回合不是 100 个独立训练重复。

新增只读行为指标：成功驶离时间为 SUMO 终止时刻减实测 depart_time，仅严格成功回合有效；停车定义为有效步前速度 `<0.5 m/s`，报告停车步数/速度有效样本数及 Wilson CI，汇总按有效步池化。换道请求按实际执行动作经控制器解析的非零方向统计，朝向目标车道为负方向且当前 lane>0，远离为正方向；请求可能被道路/边界条件限制，不等于实际 SUMO 换道。物理换道为同 Edge 连续有效实测 lane index 的变化，另报有效观测对数；跨 Edge 全部排除，计数仅代表可观测变化，不能宣称完整运动总数。首次进入 E77 的车道读取有效实测 road/lane；未观测到记 null，分布只使用有观测的回合。无法采集或不适用的时间、行为统计为 null，并保留诊断说明及有效分母。

### 6.3 论文图表与统计方法

```bash
.venv/bin/python scripts/plot_masac_experiments.py \
  --experiment MASAC 42 runs/seed42 \
  --experiment MASAC 43 runs/seed43 \
  --experiment IndependentSAC 42 runs/baseline42 \
  --output runs/comparison/figures
```

每个 `--experiment ALGORITHM TRAINING_SEED DIRECTORY` 明确算法、独立训练 seed 和目录；输入一个 seed 即绘制单 seed。对照算法需导出同一评估 JSON/CSV 协议。绘图完全离线，不调用 SUMO；论文评估曲线拒绝随机或非固定 100 seeds 结果。

生成七组 PNG、PDF 及各自源 CSV：成功率；碰撞与 Missed Exit；Critic loss/Q/target；Actor loss/alpha；速度/Jerk/急刹回合率；换道请求与可观测物理换道；首次 E77 车道分布。横轴统一训练 global_step，探索训练滚动曲线用虚线、checkpoint 正式评估用实线，标签明确区分。

同算法多个训练 seed 在相同训练步上等权取均值及样本标准差（ddof=1），阴影为 ±1 SD；只有全部输入训练 seed 在该步都有观测时才画跨 seed 点。单 seed 不绘制 SD 阴影。训练回合结束步不同不插值、不补点，因此跨 seed 训练滚动曲线可能没有共同横坐标；可分别输入各 seed 查看。失败/缺失 checkpoint 写入 `missing_checkpoints.json`，曲线在已知缺失步断开，不插值；不使用残留失败结果。

`plot_source.csv` 保存逐 seed、逐 global_step 原值、分母和已有 CI；`final_metrics_per_seed.csv` 保存每个 seed 最后实际有效 checkpoint 的各车/联合指标与有效 n；`final_metrics.csv` 按算法、实际训练预算、指标分组，输出有效训练 seed 数、输入训练 seed 数、均值、样本 SD 和有效样本数合计；全部缺失的指标保留 NA、有效 seed 数 0。不同最终预算不合并，单 seed 的 SD 留空。最后有效模型可能不是计划最终模型，应结合缺失索引使用；不得隐去缺失后宣称相同预算。

Wilson CI 描述给定模型的评估回合比例；停车步 CI 按观测步作二项描述，时间相关性意味着它不能作为独立样本推断的证据。跨 seed SD 描述训练重复之间的离散性，不能用回合数替代独立训练 seed 数。本文档不声称因果差异或统计显著性。

## 7. 当前验证记录（非训练性能结论）

截至本版本整理时：

- 模型与经验链路的针对性测试已通过；真实 SUMO 30 步采样和完整回合 Replay 奖励一致性测试已通过。
- 本地 2,500 SUMO 全局步冒烟训练完成 247 次梯度更新、1 个完整 episode，能够重置后继续训练并生成 checkpoint。
- 该 2,500 步 checkpoint 的 5 回合确定性评估：两车均 0/5 Arrived。
- 同 checkpoint 的 10 回合随机采样评估：`ego_o3` 1/10 Arrived、`ego_e74` 1/10 Arrived、联合成功 0/10；随机策略碰撞很多。

**这些记录仅证明训练与评估管线可运行，以及随机采样存在个别成功轨迹；不能据此声称 CTDE-MASAC 已经学会安全驶离或优于单 Ego SAC。** 尚缺多种子训练、正式评估、与独立 SAC 对照及统计检验。

## 8. 测试

```bash
# 单元和集成测试（含需要 SUMO 的测试）
python3 -m pytest tests/ -q

# 训练核心与回放归因
python3 -m pytest \
  tests/test_masac_actor.py \
  tests/test_centralized_critic.py \
  tests/test_joint_replay_buffer.py \
  tests/test_masac_trainer.py \
  tests/test_transition_collector.py -q

# 真实 SUMO 完整回合回放（需要 SUMO）
python3 -m pytest tests/test_full_episode_replay.py -s -v
```

当前 `tests/test_e74_rule_lane.py` 是非学习式诊断，不是 MASAC 训练使用的安全控制器。

## 9. 限制与后续工作

1. 双 Ego 仅实现 Easy；Medium/Hard 目前只有单 Ego 路线文件。
2. 训练入口尚无精确断点恢复或 W&B；已有周期 checkpoint、训练配置与 SHA256 索引、独立固定种子批量评估。多训练 seed 仍须分别启动和归档。
3. 当前连续横向 Actor / 连续 Q 输入与 SUMO 离散换道执行之间存在潜在不匹配，需要后续实验评估。
4. `laneChangeMode=0` 不对请求换道提供常规安全保护；高碰撞率是必须测量的任务结果，不代表策略已安全。
5. 现有历史 E76→D3 单 Ego SAC / 奖励消融代码保留用于研究追溯及后续对照，但不作为当前双 Ego CTDE-MASAC 的运行说明。

## 10. 版本固定建议

在文件一致性修复、测试及 README 更新后，再提交并创建不可变的 Git tag。发布时记录 Git SHA、SUMO 版本、Python/PyTorch 版本、路网与路线 SHA256、训练 CLI 参数、训练 seed 和评估 seed；模型权重与评估 JSON 应放入独立实验档案。

项目基于 DAROM 的延迟感知强化学习思路进行研究性扩展；本分支是实验实现，不应当作原论文算法或结果的官方复现声明。