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
| 速度      | 道路及 Ego 限速 120 km/h（33.333 m/s）；两辆 Ego 的设定出发速度均为 18 m/s |
| Episode   | 每辆 Ego 默认最多 **2,500 个活跃控制步（250 s 仿真时间）**；分别终止，双方结束后重置 |
| Ego 控制  | 纵向加速度 + 横向换道请求；`laneChangeMode=0`；评估可选换道入口门控 |
| 奖励      | 模块化奖励，具体配置见下文 |

**训练目标**：分别提高两辆 Ego 的安全抵达率，最终提高**同一回合两车同时抵达 D3 的联合成功率**。两车均由各自的 Actor 决策，计划出发间隔为 3 s，实际插入时刻由 SUMO 决定。

Easy 双 Ego 路线中的小客车和摩托车采用 `speedFactor="normc(0.75,0.08,0.55,0.9)"` 和 `departSpeed="desired"`。按入口限速计算，名义期望入网速度约为 25 m/s，采样倍率对应的速度范围为 18.33–30 m/s。货车使用 `departSpeed="max"`，车型最高速度分别为 25 和 22 m/s。

## 2. CTDE-MASAC 结构

- **分散 Actor**：每辆 Ego 使用独立的 `GaussianActor`（DAROM `DelayAwareGRUEncoder` + tanh-Gaussian 策略）；仅使用本车观测，默认 136 维。
- **Actor 输入（136）**：当前 Ego 状态及最多 30 辆邻车的 31×3 基础观测（93）＋目标出口编码（2）＋20×2 自身动作历史（40）＋观测年龄（1）。邻车状态带通信延迟；Ego 自身状态实时更新。新动作位于历史开头，按实际观测年龄决定保留数量。
- **前车特征消融（139 维）**：训练时加 `--actor-leader-features`。从 TraCI 采集 120 m 范围内的前车净间距、带符号的闭合速度（Ego 速度减前车速度）和 TTC。TTC 封顶 10 s，无前车时使用 `[120 m, 0 m/s, 10 s]`，有前车但未逼近时 TTC 同样取 10 s。三个量与邻车共用同一数据包、延迟和观测年龄。Critic 和奖励输入沿用原定义。checkpoint 保存该开关，评估自动继承，旧 checkpoint 缺少该字段时为关闭。
- **集中 Critic**：`CentralizedTwinQ` 接收两辆 Ego 的当前真实交通观测与目标出口（每车 95 维）、联合请求动作及 active mask，拼接为 196 维输入；两套 Q 网络分别对两辆车输出 Q 值。**Critic 不使用 Actor 的 256 维 GRU 特征。**
- **联合 Replay**：每个 SUMO 全局步对应一条双 Ego 联合经验，分别保存奖励、done、active/acted/next-active mask。`JointTransitionCollector` 延迟一个全局步提交，以正确归因 pre-step Missed Exit。
- **优化**：SAC Twin-Q 目标、Polyak 目标网络、分 Ego Actor 优化器和独立自动熵系数 α；Critic 使用真实联合状态，执行时只需各自 Actor。

Actor 输出两个连续值，均限制在 `[-5,5]`。纵向值对应加速度（m/s²）；横向值经阈值离散化：小于 `−5/3` 请求朝车道 0 换道，在 `[-5/3, 5/3)` 保持，大于等于 `5/3` 请求反方向换道。当前训练中的 Critic 看到**连续请求动作**，而 SUMO 执行的是离散换道命令；这是后续需要检验的建模限制。

### 评估换道入口门控

在评估命令中加 `--lane-change-gate ego_2` 仅保护 Ego 2，也可选 `ego_1` 或 `both`。默认 `off`，原 checkpoint 可直接用于对照。门控在 TraCI `changeLane` 之前查询目标车道的实时前后车辆。采用恒速近似，计算未来 4 秒内的最小净间隙 `当前净间隙 − 4 s × max(闭合速度, 0)`。任一侧低于 2 米时拦截该请求，纵向动作仍按原有限速规则执行。

评估 JSON 记录启用的 Ego 和阈值，每回合按车记录检查次数、拦截次数及前后侧触发次数。同一次拦截可能同时触发前后两侧。逐步轨迹的 `control.lane_change_gate` 记录预计间隙和对应车辆。该版本检查入口请求，横移期间已在执行的同向换道可继续，训练保持默认关闭。

### 模块化奖励

新训练默认开启任务、汇入、前后车安全间距、换道、效率、舒适度、团队和路线准备模块。一次性路线引导需显式开启。参数采用 `--reward-<模块>` 和 `--no-reward-<模块>`，可独立组合。直接构造环境时传入 `MultiEgoEnv(reward_config=MASACRewardConfig(...))`。

| 模块 | 结算规则 |
| --- | --- |
| `task` | 到达指定出口 +50、碰撞 −500、错过出口 −400、停滞 −400、超时 −400。每车只结算一次终止事件。 |
| `merge` | 需要汇入的 O3 Ego 完全进入 E75 主路车道或进入 E76 时，当步获得 +5，每辆 Ego 每回合一次。之后的出口结算不再补发。 |
| `following` | 每个活跃控制步用步前真实状态计算净间距风险和 TTC 风险，结算 −0.1 × max(C_gap, C_TTC)。安全时距 1.5 s，TTC 阈值 3 s，前车查询前瞻 120 m。 |
| `rear-following` | 按同样的公式和权重独立扣分。用后车速度计算 1.5 s 安全间距，用后车速度减 Ego 速度计算闭合速度。前后车可在同一步各扣最多 0.1。无后车时该项为 0，可用 `--no-reward-rear-following` 单独关闭。 |
| `lane-change` | 实际横移开始后，每步按固定目标车道的前后净间距更新风险峰值，只对峰值的新增部分乘 −0.5 扣分。反应时间 1.5 s，最小缓冲 2 m，参考减速度 4.5 m/s²。车身完成横移前，同向重复动作不会提前发往再下一条车道。 |
| `efficiency` | 每个活跃控制步按步前速度结算 +0.01 × min(1, v/20)，速度单位 m/s。 |
| `comfort` | 有实测步前和步后加速度时，结算 −0.0025 × clip(加速度差的绝对值, 0, 1)。新训练的权重可用 `--reward-comfort-weight` 设置。 |
| `team` | 双车均到达指定出口时，回合一次性记录 +100。环境通过 `env.team_reward` 提供本步值，日志和评估 JSON 用 `team_reward` 记录回合值。独立团队 Twin-Q 学习该回报，每辆 Actor 最大化个人 Q 加团队 Q。Actor 团队权重用 `--team-q-weight` 设置，默认 1.0。 |
| `route-guidance` | 默认 0。加 `--reward-route-guidance` 后，每车首次实际通过任务允许的关键连接时 +5。可用 `--reward-route-guidance-weight` 设置权重。无匹配规则的任务为 0。 |
| `route-preparation` | 默认开启。稳定完成换道且阶段需求创本回合新低时，每减少一次需求奖励 +0.1。安全前进另发按距离和需求分档的巡航奖，基准上限为 +0.005。 |

恢复训练和评估时使用 checkpoint 的效率饱和速度和舒适度权重。旧 checkpoint 缺少效率饱和速度字段时按 20 m/s 结算。

`--terminal-penalty-scale` 缩放个人任务项中的负终止奖励，默认 1.0。设为 0.2 时，碰撞为 −100，错过出口、停滞和超时均为 −80。`--collision-penalty 1000` 可将碰撞单独设为 −1000，显式指定的碰撞值优先于整体缩放。个人到达、汇入和团队奖励继续按上表结算。这些参数随 `reward_protocol` 写入 checkpoint，评估恢复其存档值。旧 checkpoint 缺少这些字段时采用缩放因子 1.0 和原碰撞值。训练更新日志中的 `target_std` 和 `team_target_std` 分别记录个人和团队 Critic 的批内 TD 目标标准差。

舒适度权重也保存在 `reward_protocol` 中。评估和续训恢复存档权重，旧 checkpoint 缺少该字段时使用 0.01。

Actor 与 Critic 均收到各车自己的目标出口编码（D1 为 [0,0]，D2 为 [1,0]，D3 为 [0,1]）。SUMO 启动时关闭等待、高速路错车道、断路、双向车道及等候接载的 teleport，车辆碰撞使用 `remove` 策略。Ego 无故移除时报错。个人回报包括已启用的个人模块及显式开启的旧 USP，可从 `reward_components` 和 USP 统计追溯。逐步信息、`--trace-controls` 训练明细和回合汇总分别记录 `task`、`merge`、`route_guidance`，团队回报单独记账。

路线规则位于 [`DEFAULT_ROUTE_GUIDANCE_RULES`](src/envs/route_lane_plan.py)。以该 Ego 实际 SUMO 路线的完整外部 edge 序列匹配 `route`，因此同一出口的不同进场路线可各自配置。新增规则填写 `route`、关键分流的相邻上游和下游 edge `critical_connection`，以及一个或多个实际可接受的 `[上游车道 ID, 下游车道 ID]` 组成的 `accepted_connections`。现有示例匹配 `O3 → E75 → E76 → E77 → D3`，仅接受 `E76_0 → E77_0`，经 `E76_0 → E77_1` 为 0。由实际经过的 SUMO 内部车道识别连接，保持车道自然通过也计分，每辆车每回合最多一次。直接构造环境时可通过 `route_guidance_rules` 传入规则。

新训练默认使用 `linear_progress_v1`。对走 E77→D3 出口的任务，路线准备奖励为 `0.02 × G(n) × clip(d / 1000 m, 0, 1) × clip(Δs / 1 m, 0, 1)`。`n` 是从步首车道沿规划路线到 D3 所需的最少合法换道次数。目标车道取 1，差一次换道取 0.5，其余为 0。规划路线的上游路段也参与计奖，E76 和 E77 的目标车道均为 0 号车道。

`d` 是步首到 E77 末端的路线距离，忽略 Junction 内部车道长度。内部连接期间使用实际连接的下游车道来确定 `n`。`Δs` 按已验证的步首和步末路线位置计算，跨边界时计入实际经过的内部车道长度。停止、倒退或无法确认位移时奖励为零。每一步只结算当前模式的 `route_preparation`。跟驰、后车风险、换道和团队奖励各自按现有模块结算。

控制 trace 和评估轨迹记录步首距离、合法换道需求、实测前进距离及三个系数。可用 `--no-reward-route-preparation` 关闭该模块。该模式每步前进 1 m 时已达进度上限，因此在相同的一段目标车道上，10 m/s 可能比 20 m/s 累积更多路线奖励。错过出口率、碰撞率和 E77_0 向外实际换道次数用于检查训练后的行为。

旧 checkpoint 的评估与续训按保存的路线准备模式复现。旧模式包括首次推进事件奖及阶梯巡航奖 `first_progress_staircase`、`progress_bonus`、`continuous_bonus` 和 `legacy_penalty`。缺少路线准备配置的旧 checkpoint 关闭该项。可用 `--reward-route-preparation-mode` 显式选择历史模式；历史模式支持 `--route-preparation-stages FILE.json` 配置分区。新模式的三个固定参数随 checkpoint 和评估输出保存。

团队 Q 按共享 SUMO 步折扣。一辆 Ego 先到达时，团队价值继续沿后续状态传递，直至全员到达或任意 Ego 失败。Replay 区分已到达与尚未出发；个人奖励和团队奖励分别写入日志。

例如，追加 `--no-reward-comfort --no-reward-team` 可关闭舒适度和团队奖励。前后车共用 `--reward-ttc-threshold` 和 `--reward-following-lookahead`。新训练的换道风险扣分权重默认为 `0.5`，可用 `--reward-lane-change-weight` 设置。评估和续训复用存档的换道权重，缺少该字段时按历史值 `1.0`。旧 checkpoint 缺少 `rear_following` 开关时，评估按关闭处理。单步信息分别提供 `gap_risk`、`ttc_risk`、`following_reward` 和 `rear_gap_risk`、`rear_ttc_risk`、`rear_following_reward`，训练可用 `--trace-controls` 记录。换道事件写入回合训练日志和评估 JSON，包含开始、完成或中断时刻、目标车道、风险峰值与累计扣分。

每辆 Actor 的首维是到该车任务终点的沿路线剩余米数，沿用编码器固定的 1000 m 缩放。环境从双 Ego 路线文件中各自的 `<trip>` 读取目标 Edge 和 `arrivalPos`，默认取末端，支持 `max`、`center` 和数值位置。可用 `MultiEgoEnv(route_file=...)` 指定同一路网的其他双 Ego 路线。活动 Ego 的无效距离会报告车辆、道路、路线和目标。Critic 仍使用实时累计行驶距离。

Actor 的第二维是沿该车指定路线到目标出口最少还需主动换几条车道，沿用除以 5 的缩放。右为负、左为正，符号表示某条最优走法的第一次主动换道方向。自然车道连接计 0，合法的相邻横移计 1。路口内部车道按已进入的连接对应的下游车道查表。调试字段 `signed_min_lane_changes` 记录当前未延迟值，拓扑不可达时为 `None`，Actor 对仍被采样的不可达状态使用原始值 `+6`。Critic 的第二维仍是实时车道编号。

Missed Exit 使用同一张路线车道图。车辆进入无法通往目标的连接或偏离指定路线时，按错过出口结算。距必须通过的路段末端 10 m 内，若当前车道无法继续沿路线前进、车辆无法在末端前停下，且没有正在横移到可通行的相邻车道，也会触发终止。这条预判沿用制动距离近似。Ego 首次控制时在步后判定，以便 Replay 将终止归给本步动作。后续的制动预判在步前执行。

新训练 checkpoint 默认标记 `masac_env_version=5`。训练配置日志和 checkpoint 保存 `reward_components`、`reward_protocol`、`route_guidance_rules` 和 `stalled_block_grace_s`，评估按 checkpoint 复现。v3–v5 使用相同的路线观测和错出口规则，v2 恢复旧车道编号和 D3 终止语义。v2/v3 的停滞阻挡宽限为 0 秒。v4 已使用 3 秒宽限，v5 进一步核对实时连接是否来自 Ego 当前车道。已有 `reward_protocol` 的旧 checkpoint 缺少 `merge_reward` 时，奖金沿用 +25，发放时机沿用所存的 `merge_on_success`。旧 v2/v3 缺少 `reward_protocol` 时，评估使用汇入当步 +25、效率系数 0.05、路线引导 0。134 维 B0 checkpoint 沿用原评估路径。评估记录实际环境版本与宽限，比较指标时应使用相同终止协议。

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

历史单 Ego YAML 归档在 [`conf/legacy/`](conf/legacy/README.md)。当前 MASAC 训练从 CLI、代码默认值及续训 checkpoint 解析配置。

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

当前入口提供从零训练、联合采样、梯度更新、结构化日志及周期 checkpoint。奖励对照可用 `--init-actors-from CHECKPOINT --init-actors-sha256 SHA256` 仅加载两辆 Actor，其他模型、温度、优化器和 replay 用新运行初始化。入口校验源 checkpoint 的观测语义、特征配置和 Ego ID，并记录源步数和哈希。训练步数按 SUMO 全局控制步计。

```bash
# 已经过本地 2,500 全局步的训练冒烟验证
python3 train_masac.py \
  --total-steps 2500 \
  --learning-starts 32 \
  --batch-size 32 \
  --buffer-capacity 5000 \
  --train-every 10 \
  --log-every 500 \
  --gamma 0.99 \
  --learning-rate 3e-5 \
  --initial-alpha 0.01 \
  --seed 42 \
  --output-dir runs/masac_smoke_2500
```

正式训练默认参数为 1,000,000 个共享 SUMO 全局步、Replay 容量 1,000,000、batch 512、Learning Starts 100、学习率 3e-5、γ=0.99、`--tau=0.005`、seed 42，每 500,000 步保存模型和 Replay Buffer，默认输出 `runs/masac`。折扣因子对照可显式指定 `--gamma 0.995`。两辆 Ego 的 α 独立学习。上述 2,500 步命令显式使用小参数，用于冒烟验证。训练开始的前 `learning-starts` 个全局步使用均匀随机动作，后续使用随机 Actor 策略。

训练时使用 `--decision-interval-steps 2` 可将 Actor 决策周期设为 0.2 s。两车共享决策边界，新车激活则提前开始下一区间。环境仍每 0.1 s 推进，训练步、保存步和更新频率仍按 SUMO 步计数。Replay 按决策区间保存经验，使用逐步折扣累加奖励及实际区间长度对应的 bootstrap 折扣。使用新输出目录从零训练。评估默认读取 checkpoint 中的频率和决策时钟。

训练在首次启动 SUMO 前，将生效配置原子写入 `--output-dir` 根目录的 `resolved_config.json`。其中记录合并后的训练参数、SUMO 启动命令及输入文件哈希、环境与奖励设置、网络结构、优化器和 Replay 设置。该次启动生成的每个 checkpoint 都在 `config` 中保存同一份配置。续训更新目录中的唯一配置文件，既有 checkpoint 保留各自保存时的配置。

训练结束会保存模型 `masac_seed42_step_2500.pt` 和配对的 `masac_seed42_step_2500.replay.npz`。续训默认开启。重复原训练命令，沿用同一个 `--output-dir` 和原训练配置即可从最新 checkpoint 恢复。若要延长训练，把 `--total-steps` 改为新的累计目标。例如从 500,000 步续到 1,000,000 步，设置 `--total-steps 1000000`。从零训练时使用新的输出目录，可显式加 `--no-resume`。`runs/` 由 `.gitignore` 排除，归档续训文件时需保留模型、对应 Replay 和索引。

## 6. 独立评估

使用 `--decision-interval-steps 2` 可显式评估 0.2 s 决策。旧 checkpoint 默认每 0.1 s 决策，显式降低其频率时，每辆 Ego 从激活时开始按自己的步数决策。中间步保持上次的纵向加速度请求并按当前速度限幅，横向请求只在决策步提交。SUMO、奖励、事件判定和轨迹仍按 0.1 s 更新。两种频率使用独立输出目录。报告记录决策间隔与逐车决策数，轨迹的 `actor_decision` 标明本步是否刷新动作。

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
| Teleport | 历史报告保留该统计字段 | 当前 SUMO 配置禁用，评估不采集该事件 |
| 成功 / raw arrival | 均以 Arrived 为准 | 评估报表兼容历史字段 |
| 无末端受阻到达 | 到达且该车没有 SUMO 明确报告「车道末端无下一路段连接」的急停警告 | 与普通到达率并列，补救成功仍计入普通到达 |

上述一致指同一有效输入上的统计规则；双 Ego 的插入时刻、路径和现有终止实现不同，不代表跨场景的成功事件或性能可直接等同。环境终止规则不因指标适配改变。

速度、Jerk、急刹分别记录 `speed_samples`、`jerk_samples`、`hard_braking_samples`；事件记录 `event_samples`，不得共用分母。连续量无有效样本时 JSON 为 `null`、CSV 为空；汇总另记各自有效回合数。急刹等回合率以完整回合数为分母，不以速度/Jerk 样本数为分母。

联合成功要求同回合两车均成功，联合 raw arrival 要求两车均 Arrived；联合碰撞、Timeout、Missed Exit、Teleport、急刹及 emergency stopping 回合率表示任一车发生，步数和样本数相加。联合速度/Jerk 先对同回合有效车辆等权，再跨有效回合等权；CSV 保留各回合有效车辆数。这些联合指标是新增口径，无旧单车严格对应项。**联合成功率不是两个单车成功率的乘积。**

`global_steps` 是评估循环中 SUMO 共享推进次数（不含 reset 的插入预热）；`agent_steps_sum` 是两车实际控制步数之和（与环境 `active_steps` 一致，不含步前 Missed Exit 已终止的车）。兼容字段 `steps` 等于 `global_steps`，不能将控制步数之和当作仿真推进次数。

随机模式每回合固定 Python、NumPy、PyTorch、SUMO 和 delay buffer 种子。Actor 保持 ego_1、ego_2 的固定采样顺序，包括 inactive Actor 采样后掩码；诊断只读且不消耗随机数。确定性和随机模式分别汇报，不混合。复现保证限于相同模型、软件与硬件环境。

缺失/空终止原因、Unknown/UnknownRemoval、环境异常和步数上限导致的未完成评估直接报错，不计完整回合，也不输出本次最终文件。`Other` 仅用于环境正常返回的其他非空合法终止事件，原始原因同时保留。

验证：`.venv/bin/python -m pytest tests/test_evaluation_metrics.py -q` 包括固定提交函数对照、独立掩码/分母、真实 SUMO 诊断开关前后确定性行为和随机状态对照、随机重复性、完整输出重算与中断拒绝。该验证属于小规模本地功能验证，不是正式远程性能评估。

### 6.1 训练记录与周期 checkpoint

省略 `--device` 时使用 `auto`。PyTorch 检测到 CUDA 时选择 `cuda:0` 并用 2 个 PyTorch CPU 算子线程，否则回退到 CPU，线程数为 PyTorch 当前默认值与 8 的较小者。可用 `--device cpu`、`--device cuda:N` 或 `--torch-threads N` 显式覆盖。SUMO 与 Replay 存储在 CPU。训练启动时打印请求设备、实际设备和线程数，并将其写入配置。评估器可在 CPU 读取 CUDA 训练生成的 checkpoint。

默认每 500,000 个共享 SUMO 全局步及训练结束时保存模型和 Replay Buffer，恰逢保存节点只写一次。可用 `--checkpoint-every N` 覆盖。若目录中的最新 checkpoint 已达到目标步数，命令直接退出。`--tensorboard` 默认开启，可用 `--no-tensorboard` 关闭，JSONL 始终开启。动作窗口复用 `--log-every`（默认 40 步），最后不足窗口仍写实际长度。

```text
runs/<experiment_id>/
  train/metrics.jsonl
  train/tb/
  checkpoints/masac_seed42_step_500000.pt
  checkpoints/masac_seed42_step_500000.replay.npz
  checkpoints/index.json
  eval/step_500000/summary.json
  eval/step_500000/episodes.csv
  eval/index.json
  figures/
```

JSONL 以 `event` 区分：

- `config`：完整 CLI、Git 提交及工作树是否有修改，`global_step=0`。
- `resume`：恢复步数、Replay 大小和哈希、累计更新数、下一回合的 SUMO 种子及配置。崩溃后超过恢复点的日志保存在 `train/metrics.before-resume-*.jsonl`，主日志从恢复点继续。
- `episode`：仅两车均完整终止后写入，包含 `episode_index`、`episode_global_steps`、每车终止原因/奖励/严格成功/有效控制步数、联合成功；最近 100 回合的 ego_1 / ego_2 成功、碰撞、Missed Exit、Timeout 百分率和平均奖励，以及联合成功率和两车奖励和的均值。`rolling_episodes` 是实际分母，`policy=exploration_training`，不能冒充正式评估。
- `stalled_shadow`：用 `--stalled-shadow` 记录每辆 Ego 的步前诊断，环境在 `infos[ego]["stalled_shadow"]` 提供同一数据。每车的有效通行滑窗为 10 秒，里程进展小于 20 m 时输出 `would_stall=true`。前车约束仍使用 50 m 内前车的净间隙、最小间距和按当前车速估计的 10 秒前车位移。路口检查仍限于内部连接或距车道末端 20 m 内。
- v4 起检测到外部阻挡时暂停窗口，持续达到 3 秒后清空一次。连续畅通达到 3 秒才确认恢复，并补计这段连续畅通的步间增量。更短的畅通闪烁并回同一阻挡段。跨阻挡边界的步间增量不计。无效里程、不连续样本、TraCI 不可用和传送信号立即清空。3 秒是未校准的初值。使用宽限规则并启用终止或影子诊断时，每回合按车输出 `stalled_diagnostics`，包括实际窗口重置数及各原因次数、暂停段数，以及原始和合并闪烁后的阻挡时长直方图，单位为 0.1 秒步数。因采样中断或离开而截尾的合并段单列，未结束段另记已观测时长。
- v5 在路口用路线索引和实际车道查询通往任务下一路段的合法连接。只有 TraCI 的目标车道和内部连接能唯一对应当前车道时，才使用信号、开放状态、优先权和冲突信息。内部车道上的让行检查沿用路线图。存在合法连接但实时项无法确认时，记为 `route_link_unknown` 并中断窗口。当前车道缺少任务连接时，前车约束或计划停车仍算外部阻挡，否则继续按原进展阈值判定。v2–v4 保留原来的路口检查。
- `Emergency Braking`：训练和评估始终启用。每个 SUMO 仿真步后，读取 TraCI 的 `getEmergencyStoppingVehiclesIDList()`，终止并移除名单中的 Ego。该接口返回 SUMO 原生 Emergency Stop，常见于车道末端强制停车；限值急减速产生的 `performs emergency braking` 日志按原评估口径另行记录。急刹终止固定给该 Ego −400 任务奖励。若同一步发生碰撞，优先结算碰撞。旧 checkpoint 可用新规则评估；继续训练须使用采用新规则的 checkpoint。
- `--stalled-terminate`：新 MASAC 训练默认启用。用 `--no-stalled-terminate` 关闭，可加 `--stalled-shadow` 单独记录诊断。步前检测后，在 SUMO 步完成时按 Collision、Emergency Braking、Missed Exit、Arrived、Stalled、Timeout 的顺序结算。Stalled 仅结束触发的 Ego，单次任务分为 −400。正常步进奖励按原规则结算。团队目标在首次失败时结束。旧 USP 训练默认关闭停滞终止，显式同时开启会报错。评估默认继承 checkpoint 的开关，缺少该设置时按关闭处理。可用 `--stalled-terminate` 或 `--no-stalled-terminate` 覆盖。训练滚动指标与评估报告均含 `stalled_rate`。
- `control_trace`：使用 `--trace-controls` 时，每个实际控制步记录前车 ID 和实际净间隙、Ego 道路和车道位置、步前和步后速度、策略请求加速度、TraCI 提交加速度、SUMO 步后实测加速度和原始到达标志。前车由 TraCI 沿当前路线搜索，搜索最短前瞻 100 m。查无前车或车辆已在步后移除时，相应字段为 `null`。
- `update`：第一轮、每隔 `--update-log-every` 轮（默认 40）以及最后一个可执行更新步记录一次。设置为 1 可记录每轮。每条记录保留该轮的原始 `critic_loss`、双 Q 均值 `mean_q`、`mean_target`、`valid_agents`、`update_index`；各车 `actor_loss`、`alpha`、`alpha_loss`、`mean_log_prob`、`valid_samples`。log-prob 是当前 actor 更新中已有采样的实际动作 log-prob，未再额外前向；无有效样本的损失为 null、样本数为 0。
- `runtime`：窗口实际步数、速度（SUMO 全局训练步/墙钟秒）、Replay 大小、累计更新数、各车动作样本数/有效动作占比、加速度均值与总体标准差、横向动作均值、换道请求比例。只统计真正执行动作的样本；换道比例直接调用当前控制器 `parse_action()`，不另定义阈值。

TensorBoard 的标量标签统一带 `exploration_training/` 前缀，以 `global_step` 为横轴；例如 `tensorboard --logdir runs/<experiment_id>/train/tb`。更新曲线展示实际记录轮次的 minibatch 值，横轴使用真实全局步。运行窗口仍记录累计更新总数。

模型 checkpoint 保存 Actor、个人及团队 Critic 和目标网络、α、优化器、累计进度及训练随机状态。配对的 `.replay.npz` 保存有效样本、容量、环形写入位置和采样随机状态，评估只需模型 `.pt`。文件先写 `.partial`，flush/fsync 后原子替换，模型发布时 Replay 已写完，索引记录两者的 SHA-256。续训检查哈希和关键训练配置。旧模型没有配对 Replay 时无法完整续训，仍可评估，或在满足观测兼容条件时用 `--init-actors-from` 初始化新训练。若首次保存前中断且目录已有日志，请用新输出目录重跑。

续训时 SUMO 用下一回合种子启动。中断时若有一条等待下一步事件归因的 transition，会在恢复时舍弃。恢复前已提交到 Replay 的样本和累计进度会保留。此方式支持继续学习，轨迹不会与未中断训练逐步一致。满容量的 1,000,000 条双 Ego Replay 每个保存点约 3.5 GiB，保存体积随有效样本数增长，已有保存点会保留。使用 `--replay-final-only` 时，中间节点只保存可评估的模型权重。最后一个节点保存模型、优化器和 Replay，中间节点无法完整续训。

### 6.2 固定种子评估与只读行为诊断

```bash
# 以下是正式评估入口示例；本次开发验证没有执行 100 回合评估
.venv/bin/python scripts/evaluate_masac_checkpoints.py runs/<experiment_id>
# 配置不一致或损坏时默认报错；明确重算时加 --rerun
```

入口按 checkpoint 内部 `global_step` 排序；每个模型固定 Easy、两车、确定性 Actor、100 回合、种子 `200001–200100`，不提供修改这些设置的选项。`--max-steps` 默认 4500；只限制评估器等待回合结束的共享步数。环境按 checkpoint 的 `max_active_steps` 分别判定每辆 Ego 超时。新训练可用 `--max-active-steps` 设置，默认 2,500。旧 checkpoint 缺少该项时按原 4,000 步评估。批量评估不在训练过程中自动运行。

复用结果前核对 JSON/CSV 所有回合字段、分组汇总重算、checkpoint SHA256、种子序列、协议版本、Git 提交、源码哈希、路网/路线和 SUMO 二进制哈希、环境配置及实际 speed mode 读回记录。任何失败记录到 `eval/index.json` 后报错，后续项保留 pending，不能当作已评估；索引缺少模型文件也报错。

单模型和批量评估默认保存 `eval_trajectory.jsonl.gz`。逐种子、逐步记录两辆 Ego 的位置、车道与横移、动作、间距、奖励分项、路线准备和停滞诊断，以及终止事件。完整评估结束后发布文件，`summary.json` 记录轨迹 SHA-256 和步数，可用 `--no-trajectory` 在单模型评估中关闭采集。`--trace` 独立控制首回合的控制台打印。请求 `speed_mode=96`，实际 `getSpeedMode()` 的所有读回值按车保留在每回合和环境元数据中。本机 SUMO 1.20.0 的短程测试读回为 32。JSON 同时记录 checkpoint SHA256、训练配置、Git/源码哈希、SUMO 版本、NumPy/PyTorch 版本、路网/路线哈希及关键评估配置。

已发布的评估轨迹可通过[交互式轨迹诊断工具](scripts/trajectory_dashboard/README.md)按 seed 和 Ego 浏览。页面用同一仿真步联动车道、速度与动作、奖励三张图，并校验轨迹与评估报告。

核心统计仍为 `homogeneous-dual-ego-v1`，扩展输出协议为 v4。普通到达、联合无碰撞及其他历史指标保留既有字段与单位。回合率的 `*_stats` 提供分子、分母和 Wilson 95% CI。100 个评估回合不构成 100 个独立训练重复。

v4 从每回合的 SUMO 警告日志提取错车道末端受阻。仅识别 `performs emergency stop at the end of lane ... because there is no connection to the next edge`，无需速度或减速度门槛。同车同车道且相隔至多一个仿真步的警告合并为一次事件，逐种子 JSON/CSV 保留车道、首次和末次时间、原始警告数、每条的减速度与偏移。普通 `emergency braking` 警告单列，背景车事件另列在 `traffic_sumo_warnings`。

每辆 Ego 和双车汇总均提供普通到达率 `raw_arrival_rate`、无末端受阻到达率 `lane_end_block_free_arrival_rate` 以及受阻后到达数 `arrivals_after_lane_end_block`。双车无末端受阻到达要求各车都到达且都无该事件。日志缺失时，已到达但无法分类的结果记为未知，汇总显示未知回合数，相关率为 `null`。旧报告的普通到达统计继续可读。

制动对照通过 `--actor-leader-features` 分组。两组保持相同的环境版本、训练配置和评估种子。与 v2 历史模型做同口径对照时，用 `--masac-env-version 2` 训练新组，沿用 v2 的 Actor 车道索引和出口终止规则。评估 JSON/CSV 逐车记录首次请求制动时间、真实步前 TTC 风险首次出现时间，以及首次连续 TTC 风险窗口内从风险起点到 Actor 请求制动和 SUMO 实测制动的时间。制动阈值为加速度 ≤ −0.1 m/s²。未观察到对应事件时输出 `null`，比较时分别报告风险回合数和无制动响应的回合数。

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
2. 训练入口支持从模型与 Replay checkpoint 续训，SUMO 从新回合开始。训练记录使用本地结构化日志和 TensorBoard，尚未接入 W&B。多训练 seed 仍须分别启动和归档。
3. 当前连续横向 Actor / 连续 Q 输入与 SUMO 离散换道执行之间存在潜在不匹配，需要后续实验评估。
4. `laneChangeMode=0` 不对请求换道提供常规安全保护；高碰撞率是必须测量的任务结果，不代表策略已安全。
5. 现有历史 E76→D3 单 Ego SAC / 奖励消融代码保留用于研究追溯及后续对照，但不作为当前双 Ego CTDE-MASAC 的运行说明。

## 10. 版本固定建议

在文件一致性修复、测试及 README 更新后，再提交并创建不可变的 Git tag。发布时记录 Git SHA、SUMO 版本、Python/PyTorch 版本、路网与路线 SHA256、训练 CLI 参数、训练 seed 和评估 seed；模型权重与评估 JSON 应放入独立实验档案。

项目基于 DAROM 的延迟感知强化学习思路进行研究性扩展；本分支是实验实现，不应当作原论文算法或结果的官方复现声明。
