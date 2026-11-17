# DAROM-TPR — Dual-Ego CTDE-MASAC for Highway D3 Exit

> **状态：研究原型 / v0.1 训练管线基线（尚未证明策略收敛）。** 适用分支：`exp/o3-d3-ctde-v1`。本分支的主要任务是让两辆独立决策的 Ego 在同一个 SUMO 仿真中驶离 D3，而不是人为制造车辆冲突或同步出发时差。

## 1. 研究任务

使用 DAROM 的延迟感知 GRU 观测编码器，在 Antwerp 路网构建双 Ego 驶离场景，采用集中训练、分散执行（CTDE）的多智能体 SAC（MASAC）：

| 项目      | 当前实现                                                     |
| --------- | ------------------------------------------------------------ |
| 场景      | Antwerp **Easy**（双 Ego；Medium/Hard 双 Ego 尚未接入）      |
| `ego_o3`  | O3 → D3；计划 120 s 出发；`O3_0` 起步                        |
| `ego_e74` | E74 → D3；计划 120 s 出发；`E74_2` 起步                      |
| 仿真      | 一个 SUMO 进程、一个 0.1 s 全局时钟；各车实际插入时刻可不同  |
| 速度      | 道路及 Ego 限速 120 km/h（33.333 m/s）；Ego 出发速度 10 m/s  |
| Episode   | 每辆 Ego 最多 **4,000 个活跃控制步**；分别终止，双方结束后重置 |
| Ego 控制  | 纵向加速度 + 横向换道请求；`laneChangeMode=0`、无自定义安全拦截 |
| 奖励      | B0 纯事件：Arrived +50；Collision/Missed Exit/Ego Teleported/Timeout −100；非终止 0 |

**训练目标**：分别提高两辆 Ego 的安全抵达率，最终提高**同一回合两车同时抵达 D3 的联合成功率**。当前没有规则换道器替模型决策，也没有人为设置出发时间差 Δt。

## 2. CTDE-MASAC 结构

- **分散 Actor**：每辆 Ego 使用独立的 `GaussianActor`（DAROM `DelayAwareGRUEncoder` + tanh-Gaussian 策略）；仅使用自己的 134 维延迟策略输入。
- **Actor 输入（134）**：当前 Ego 状态及最多 30 辆邻车的 31×3 基础观测（93）＋20×2 自身动作历史（40）＋观测年龄（1）。邻车状态带通信延迟；Ego 自身状态实时更新。新动作位于历史开头，按实际观测年龄决定保留数量。
- **集中 Critic**：`CentralizedTwinQ` 接收两辆 Ego 的当前真实 93 维观测、联合请求动作及 active mask，拼接为 192 维输入；两套 Q 网络分别对两辆车输出 Q 值。**Critic 不使用 Actor 的 256 维 GRU 特征。**
- **联合 Replay**：每个 SUMO 全局步对应一条双 Ego 联合经验，分别保存奖励、done、active/acted/next-active mask。`JointTransitionCollector` 延迟一个全局步提交，以正确归因 pre-step Missed Exit。
- **优化**：SAC Twin-Q 目标、Polyak 目标网络、分 Ego Actor 优化器和独立自动熵系数 α；Critic 使用真实联合状态，执行时只需各自 Actor。

Actor 输出两个连续值，均限制在 `[-5,5]`。纵向值对应加速度（m/s²）；横向值经阈值离散化：小于 `−5/3` 请求朝车道 0 换道，在 `[-5/3, 5/3)` 保持，大于等于 `5/3` 请求反方向换道。当前训练中的 Critic 看到**连续请求动作**，而 SUMO 执行的是离散换道命令；这是后续需要检验的建模限制。

## 3. 目录结构

```text
train_masac.py                     双 Ego CTDE-MASAC 训练（当前为基础训练入口）
evaluate_masac.py                  双 Ego 独立评估（deterministic / --stochastic）
src/agents/
  masac.py                         两个独立 DAROM-GRU Gaussian Actor
  centralized_critic.py            集中式 Twin-Q，当前输入 192 维
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
src/rewards/events.py               事件奖励映射（由单 Ego 环境复用）
sumo_new/
  Antwerp.net.xml                   Antwerp 路网
  Antwerp_easy_o3_d3_dual.rou.xml   当前双 Ego Easy 路线
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

当前入口提供**从零训练**、联合采样、梯度更新及训练结束时的 checkpoint 保存，不提供中途定期 checkpoint 或精确断点续训。训练步数是 SUMO 全局控制步，不是两车活跃步数之和。

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

默认配置还包括 `tau=0.005`，两辆 Ego 的 α 独立学习。训练开始的前 `learning-starts` 个全局步使用均匀随机动作，后续使用随机 Actor 策略。

训练结束保存 `runs/masac_smoke_2500/masac_step_2500.pt`，其中包含 Actor、Critic、Target Critic、α 和优化器状态；**不包含 Replay Buffer、SUMO 物理状态或随机数全状态，不能精确续训**。`runs/` 由 `.gitignore` 排除，模型文件需另行归档。

## 6. 独立评估

```bash
# 确定性策略（正式汇报优先使用）
python3 evaluate_masac.py \
  --checkpoint runs/masac_smoke_2500/masac_step_2500.pt \
  --episodes 10 \
  --seed-start 200001 \
  --output runs/masac_eval_2500/deterministic_10.json

# 随机采样策略（探索行为诊断，不与确定性评估混为一谈）
python3 evaluate_masac.py \
  --checkpoint runs/masac_smoke_2500/masac_step_2500.pt \
  --episodes 10 \
  --seed-start 200001 \
  --stochastic \
  --output runs/masac_eval_2500/stochastic_10.json
```

评估器加载已冻结的 Actor，不更新参数。随机评估对每个 episode 设置 PyTorch / NumPy 种子。JSON 包含各 Ego 的事件分布、分别的 Arrived 成功率及同一回合两车同时 Arrived 的联合成功率。**两车联合成功率不是两个单车成功率的乘积。**

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
2. 训练入口尚无断点恢复、周期 checkpoint、W&B、规范化实验 manifest 与多种子自动评估。
3. 当前连续横向 Actor / 连续 Q 输入与 SUMO 离散换道执行之间存在潜在不匹配，需要后续实验评估。
4. `laneChangeMode=0` 不对请求换道提供常规安全保护；高碰撞率是必须测量的任务结果，不代表策略已安全。
5. **发布前应同步修复** `sumo_new/Antwerp.sumocfg` 的旧路线名，以及 `scripts/build_dual_ego_routes.py` 无法重建当前 E74_2 路线的问题。不要把 `conf/` 的空配置文件描述为已启用训练配置。
6. 现有历史 E76→D3 单 Ego SAC / 奖励消融代码保留用于研究追溯及后续对照，但不作为当前双 Ego CTDE-MASAC 的运行说明。

## 10. 版本固定建议

在文件一致性修复、测试及 README 更新后，再提交并创建不可变的 Git tag。发布时记录 Git SHA、SUMO 版本、Python/PyTorch 版本、路网与路线 SHA256、训练 CLI 参数、训练 seed 和评估 seed；模型权重与评估 JSON 应放入独立实验档案。

项目基于 DAROM 的延迟感知强化学习思路进行研究性扩展；本分支是实验实现，不应当作原论文算法或结果的官方复现声明。