# DAROM 复现实验计划

状态（2026-10-17）：用户已选择只做 D 组的 7M 长训练；配置为 `experiments/d_long.json`，独立入口为 `python -B -m experiments.long_run`，CPU/CUDA 的短程完整断点恢复验证均已通过，已于 2026-10-17 10:22（北京时间）后台启动正式 7M。完整状态按用户要求每 2M 步保存一次，另保存 7M 最终状态。原首轮使用 `experiments/first_round.json` 与 `experiments.run`，保留其对照结果及已有队列。当前方案是基于官方代码、修正两项网络结构的复现，并非所有环境和算法细节均与论文完全一致。短验证结果不能当作复现结果。

## 目标与证据

最新目标：仅选择 D、训练种子 42，从随机初始化开始连续训练至累计 7,000,000 environment steps，并支持从完整断点恢复。前期 2×2 结构对照作为选择依据，原有多组长训练预算不再执行。比较训练回报、评估成功率、失败类型和 loss 趋势；不能仅因 loss 更低就认定复现更好，也不能保证不同随机种子的曲线逐点一致。

- 上游基线：`amin-tabrizian/DAROM`，commit `eabd572fc0aece579e5b553bdc7af27a8b81d03a`。
- 论文：[arXiv 2403.11852 v5](https://arxiv.org/html/2403.11852v5)。训练曲线与训练预算对照 Fig. 4；Table III 属于独立评估，不能拿其平均奖励直接要求训练曲线达到同一数值。
- 既有审计：`../DAROM-reproduction-audit/configuration-audit.csv`、`audit-evidence.json`、`2403.11852v5.pdf`。
- 官方已发布 final checkpoint 为 10M steps、best 为 9.1M steps，不能直接作为论文 7M 预算的公平对照。所有候选从随机初始化开始。

## 第一轮：结构 2×2 对照

| 组 | GRU hist_len | features_dim | 编码器划分：状态 / 动作历史 / delay | 含义 |
|---|---:|---:|---|---|
| A | 21 | 128 | 91 / 42 / 1 | 保留上游训练入口的结构 |
| B | 20 | 128 | 93 / 40 / 1 | 仅修正动作历史长度 |
| C | 21 | 256 | 91 / 42 / 1 | 仅修改融合输出维度 |
| D | 20 | 256 | 93 / 40 / 1 | 两项同时修改 |

Wrapper 的真实输入始终为 134 维：31×3 状态 + 20×2 动作历史 + 1 延迟。A/C 有意保留原始错分：最后两个状态分量被算入动作历史。这是受控对照，不是推荐实现。A 也采用公共的路径、随机性和评估隔离修正，因此不等于逐字复跑未经修改的原入口。

每组 seeds = 42、43，共 8 次运行；每次 100,000 steps，总计 800,000 steps，最多同时运行 2 个独立训练进程。队列顺序为 A42、B42、C42、D42、A43、B43、C43、D43；有任务完成后立即补位。

| 公共参数 | 配置 |
|---|---|
| 场景 | Hard；活动 `mergingP.rou.xml` 必须与 `mergingP_hard.rou.xml` 完全相同 |
| 仿真 / 感知 | 步长 0.1 s；半径 50 m；ego + 最多 30 邻车 |
| 延迟 | uniform，采样范围 0–20 steps；保留原 Wrapper 收包与历史处理 |
| 训练安全控制器 | 关闭；评估分别关闭和启用原版 SC |
| 状态分支 | 64→64，LayerNorm + ReLU |
| GRU | hidden_size = 64，单层 |
| 融合分支 | 129→128→features_dim，LayerNorm + Tanh |
| Actor / Critic | 各 256→256，沿用 SAC 默认激活及其余未覆盖参数 |
| learning_rate / batch_size | 3e-5 / 512 |
| replay buffer / learning_starts | 1,000,000 / 100 |
| gamma / tau | 0.99 / 0.005 |
| train_freq / gradient_steps | 1 / 1 |
| target_update_interval | 1 |
| ent_coef / target_entropy / use_sde | auto / auto / false |
| 设备 / 线程 | 训练 CUDA；评估 CPU；每进程 torch 线程 2，interop 线程 1；首轮最多 2 次并发 |

首轮保留原奖励分支、截断后排序方式、交通行为和原版 SC。已知后续待检验差异：合作车辆 tau 代码为 N(0.7, 0.1²)、论文为 U(0.6, 0.8)；部分车流仍使用 XML maxSpeed=60；论文报告流量不一定等于 XML 输入需求；原版 SC 只读取 ego 和 4 个邻车。不能同时修正这些差异后将效果归因于网络结构。

## 随机性与评估

训练 seed 同时控制 SAC 初始化/采样；交通行为、SUMO 和延迟使用按 episode 派生的独立随机流。评估在新 Python 进程中运行，训练等待评估结束，避免全局 TraCI 连接和 NumPy RNG 相互覆盖。显式相同 episode seed 可复用同一场景起点；策略行为变化仍会改变后续交通演化。

每次训练在 25k、50k、100k 完成该步优化后保存模型并评估。每个节点对 no_sc、original_sc 各运行 32 episodes，双方使用同一组 seeds 100001–100032，确定性动作，共 192 episodes/run、1,536 episodes/首轮。它们是用于筛选的验证集，后续正式测试必须另选未使用 seeds。32 episodes 的成功率分辨率只有 3.125%，不能用微小差异做强结论。

记录每回合奖励、长度、终止原因、是否完成合流和 SC 干预次数；汇总成功、碰撞、超时、紧急停车比例。选择 best checkpoint 使用 original_sc 成功率，再以平均奖励打破平局；final 模型另外保存。跨组比较必须同时检查 no_sc，避免 SC 掩盖策略差异。

训练记录 `train/episode_reward`、`train/reward_mean_100`、actor/critic loss、熵系数及其 loss、已完成梯度更新数、实际速度和评估耗时。横轴统一为环境步数，不能混用 optimizer updates 或 W&B 内部 `_step`。保留原始曲线，展示时使用相同平滑设置；先比较学习起点、增长阶段、平台和波动，再检查最终效果。论文没有公开的逐步 loss 数据时，只能评估合理性和现有基线的一致性，不能声称精确重现论文 loss。

## 原分阶段预算与筛选（历史方案，已由单组 D 长训练替代）

1. 短验证：128 CPU steps、buffer 1024，在 64/128 steps 各评估每种模式 2 episodes。仅检验网络、日志、优化和进程隔离，产物放 `/tmp`，不计入研究结果。
2. 第一轮：4 组×2 seeds×100k。正式运行前检查 10k 内有无 NaN、崩溃或明显错误；前 10k 包含在 100k 内。100k 只判定运行有效性，不因回报低而淘汰。
3. 初筛：有效组都达到 500k，总累计目标 4M steps。Hard GRU 可能在 0.5–1M 才开始明显改善，应保持耐心。
4. 复核：2–3 组达到 1M，总累计目标 6–7M steps；如果差异不清楚，全部达到 1M，总计 8M。综合两种评估、跨种子稳定性、回报和 loss 诊断选候选。
5. 正式：2–3 组各训练至 7M；建议最终至少 3 个训练 seeds。最终评估采用独立的 3 组 seeds×500 episodes，并区分训练种子重复与评估种子重复。置信区间按论文口径单独计算，不将 std 冒充 CI。

这些历史“累计”预算以能连续推进或可靠续训为前提。原首轮入口 `experiments.run` 不支持完整恢复：虽然保存模型、最终 replay buffer 与部分 RNG 快照，但没有完整环境与回调状态。不得把首轮 `final.zip` 当成长训练的完整断点。新 D 长训练入口及恢复协议见文末；两种入口的断点能力不可混淆。

结构确定后再逐项测试 tau 分布；超过 30 邻车时再针对排序/截断检查；奖励分支、完整邻车 SC、exponential 和 bursty 的定义差异另列对照。流量输入与实测口径先核实，不直接凭表格替换 XML。

## 时间估算

历史证据：旧项目 `runs/GRU-uniform-delay-full/SAC_1/events.out.tfevents.1789817725.server-0-Z790-EAGLE-AX.216395.0`，有效记录从 step 254 到 9,999,587，跨时 121,106.29 s（33.64 h），约 82.57 steps/s。旧短测试从 267 到 13,041，265.4 s，约 48.13 steps/s。这不是新入口或目标依赖环境的实测速度。

共享机器及额外评估影响未知，暂以 **35–60 有效 steps/s** 做预算假设，实际要用首次运行的总 wall time（包含评估）校准；不要假设并行就翻倍加速。

| 规模 | 按上述速度估算 |
|---|---|
| 单次 100k | 0.46–0.79 h |
| 首轮 800k | 3.7–6.35 h；建议预留 4–8 h，不含环境安装 |
| 全组达到 500k，累计 4M | 18.5–31.7 h |
| 复核累计 6–7M | 27.8–55.6 h |
| 全组达到 1M，累计 8M | 37.0–63.5 h |
| 单次正式 7M | 32.4–55.6 h |
| 2 组×2 seeds×7M（预算下限） | 5.4–9.3 天 |
| 3 组×3 seeds×7M | 12.2–20.8 天 |

阶段时间不含重跑、安装、最终 3×500 episodes 的评估；依赖环境和代码改动后的速度仍待实测。1M 容量的 134 维 replay buffer 约 1.1 GB/run，8 次保存约 9 GB，另有模型和日志；不会自动删除产物。

## 环境、输出和操作

正式环境是本项目 `.venv`。目标核心版本来自发布 checkpoint 的 `system_info.txt`：torch 2.8.0（checkpoint 为 +cu128）、stable-baselines3 2.7.0、numpy 2.2.6、gymnasium 1.2.1、cloudpickle 3.1.1。SUMO 固定本机 1.18.0，traci/sumolib 目标同为 1.18.0；原训练 SUMO 版本未得到证实。Python 当前 3.12.3，而发布 checkpoint 为 3.10.12，因此这里是受控复现环境，并非作者环境完全复制。首轮检查包版本及 SUMO，不自动安装、改 PATH 或设置环境变量。

当前本项目 `.venv` 已安装全部 7 项目标依赖及其传递依赖，版本预检与 `pip check` 均通过；W&B/TensorBoard 保留原已安装版本。早期曾借用旧项目解释器验证入口，那些结果属于验证记录；后续实验与测试使用本项目解释器，不混用旧环境。

用户已确认并完成的安装范围（仅本项目 `.venv`，包含传递依赖；以下为重建命令，不会在启动实验时自动执行）：

```bash
.venv/bin/python -m pip install --index-url https://pypi.org/simple --timeout 120 --cache-dir /tmp/darom-reproduction-pip-cache 'torch==2.8.0' 'stable-baselines3==2.7.0' 'numpy==2.2.6' 'gymnasium==1.2.1' 'cloudpickle==3.1.1' 'traci==1.18.0' 'sumolib==1.18.0'
```

本次已将 numpy 2.5.4 调整为 2.2.6。默认镜像连接失败后，仅在安装命令中指定官方 PyPI，未修改系统环境变量或 pip 配置。大包因网络超时改用临时脚本分段续传，并核对官方 SHA-256；完整安装日志和包清单保留在 `/tmp/darom-reproduction-install.log`、`/tmp/darom-reproduction-packages-after.json`。实际 torch 为 2.8.0+cu128，W&B 0.30.0、TensorBoard 2.21.0。目标环境验证记录见文末。

从项目根目录执行：

```bash
# 不训练，只列出 8 个任务与检查依赖
.venv/bin/python -B -m experiments.run list
.venv/bin/python -B -m experiments.run check

# 短验证，产物仅放临时目录
.venv/bin/python -B -m experiments.run train --group A --seed 42 --smoke --output /tmp/darom-reproduction-verification

# 正式单组或全套的启动方式；已有后台队列运行，请勿重复启动
.venv/bin/python -B -m experiments.run train --group A --seed 42
.venv/bin/python -B -m experiments.run suite

# 本地查看所有组曲线
.venv/bin/tensorboard --logdir runs/reproduction --host 127.0.0.1 --port 6006
```

按用户要求，正式实验默认 `tracking.mode=online`，同时保留 TensorBoard 和 `metrics.jsonl`。当前已使用本机已有凭据验证项目访问和真实指标上传；密钥不进入配置、源码或输出。`--tracking offline` 可改为 W&B 离线记录，`--tracking disabled` 仅写本地日志。普通 `--smoke` 仍默认关闭上传，只有显式传入 `--tracking online` 才会上传验证数据。具体曲线和入口见文末 Forge 配置。

正式产物在已被 Git 忽略的 `runs/reproduction/`，每次唯一目录包括实际配置、环境版本、源码快照及哈希、Git diff、训练/评估日志、逐回合评估 JSON、模型和运行状态。临时探针和短验证在 `/tmp` 或已忽略的 `tmp/ai/`；不加入 Git。原 `train_sac_delay.py` 保留为上游参考入口，四组实验使用新入口，避免其原有训练/评估连接问题。

## 配置阶段验证记录（2026-10-16）

- `tests/test_reproduction.py` 的四项测试通过：四组输入边界与梯度、显式 seed 重复性、隐式 reset 随机流推进、交通行为参数及独立 RNG。
- 用旧项目现有解释器完成 A/seed42 的 128-step CPU 集成验证：28 次梯度更新，64/128 steps 两个节点各完成 no_sc/original_sc × 2 episodes，共 8 个评估回合；检查到 TensorBoard actor/critic loss、熵系数和评估标签。
- 重复同一短测试后，训练标量及两次评估的全部逐回合结果完全一致；最终速度统计已包含最后一次评估。最后一次验证为 `/tmp/darom-reproduction-verification/smoke-hard-uniform-A-seed42-20261016T074559Z-7ba38cd7/`，耗时 33.75 s，其中评估 26.72 s。这种极短 CPU 测试主要受启动与评估影响，不能外推正式训练速度。
- 上述早期测试解释器实际为 torch 2.14.0+cu130、SB3 2.9.0；不是目标版本，该次成功只证明旧环境下入口可运行。目标环境验证另记于下方。
- 短测试曾因沙箱本地端口限制失败，获准在沙箱外重跑后完成。SUMO 碰撞/紧急制动提示来自初始随机策略，已作为终止类型记录，不用于判断候选优劣。
- 安装前，本项目 `.venv` 的正式预检曾按预期拒绝运行：核心依赖缺失且 numpy 版本不匹配。该依赖阻塞现已解除；当时未开始 100k 实验或启用 W&B 上传，后续 Forge 接入另记于下方。
- 临时输出位于 `/tmp/darom-reproduction-verification/`；`tmp/ai/` 和 `runs/` 已验证被 Git 忽略；没有暂存或提交这些产物。

## 目标环境验证记录（2026-10-16，安装完成后）

全部操作使用 `${WORKSPACE}/DAROM-reproduction/.venv/bin/python`，没有借用旧项目环境：

- `pip check`：`No broken requirements found.`；7 项目标依赖逐项匹配，SUMO 为 1.18.0。
- `python -B -m unittest discover -s tests -v`：4 项回归测试全部通过，包含四组网络前向/反向与随机性检查。
- A/seed42，CPU：128 steps、28 次梯度更新、8 个独立评估回合，约 36.32 s，其中评估 30.08 s。
- D/seed42，CUDA：128 steps、28 次梯度更新、8 个独立评估回合，约 73.02 s，其中评估 68.93 s；checkpoint 记录 `GPU Enabled: True`、`PyTorch: 2.8.0+cu128`。GPU 为本机 RTX 4090。
- 两次运行均核对了：全部指标为有限数，actor/critic loss 和熵系数写入 TensorBoard，64/128 steps 的两种评估模式均完成，终止原因比例和为 1，最终耗时包含最后一次评估，模型正常保存。
- 上述时间属于短验证，受启动、独立评估和机器共享负载影响，不能据此判断 CPU/GPU 正式训练速度或四组性能。

验证汇总：`/tmp/darom-reproduction-target-verification/validation-summary.json`。CPU 产物：`/tmp/darom-reproduction-target-verification/cpu/smoke-hard-uniform-A-seed42-20261016T105036Z-b8640fa5/`；GPU 产物：`/tmp/darom-reproduction-target-verification/gpu/smoke-hard-uniform-D-seed42-20261016T105407Z-9f3b3391/`。均标记 `verification_only=true`，W&B disabled，不纳入正式研究结果。

依赖安装阶段结束时，尚未启动 100k 首轮或线上 W&B 同步。精确断点续训仍未实现，Python 3.12.3 与发布 checkpoint 的 3.10.12 的版本差异仍然存在。未更改系统环境或删除临时下载缓存，临时产物不进入 Git。

## 论文一致性复核（2026-10-16）

结论：**核心参数已对齐，不能称为环境和四组配置与论文完全一致。** 以下区分论文明确给出的参数、发布 checkpoint 的线索，以及有意保留的对照变量。核对来源为本地 v5 PDF 的 V-A/V-B/V-C/V-E、现有 34 项审计、官方 checkpoint 元数据、当前源码与目标环境预检。

| 项目 | 论文或发布 checkpoint | 当前配置及结论 |
|---|---|---|
| SAC | lr=3e-5、batch=512、actor/critic 两层各 256、Adam；其余使用 SB3 默认 | 一致；gamma=.99、tau=.005、buffer=1M、learning_starts=100 等按发布模型和 SB3 2.7.0 固定 |
| 观测与延迟 | ego+30 邻车，每车 3 特征；步长 .1 s；最大采样延迟 20 步 | 容量、特征数、步长和 uniform 延迟参数一致；超量邻车选择方式仍有差异 |
| GRU 公共分支 | hidden=64；状态编码 64→64，LayerNorm/ReLU | 四组一致 |
| 融合输出 | 128→256 | C/D 一致；A/B 输出 128，作为有意保留的对照 |
| 动作历史切分 | 状态与动作历史分开编码；当前 wrapper 实际为 93+40+1 | B/D 为 93+40+1；A/C 保留 91+42+1 错分。D 同时对齐这两项结构，但不代表全部环境已对齐 |
| 训练预算 | 论文 7M steps；发布 final 为 10M | 首轮每次 100k，是有意缩小的试验；后续正式目标才是 7M |
| 软件 | 发布 checkpoint：torch 2.8.0+cu128、SB3 2.7.0、NumPy 2.2.6、Gymnasium 1.2.1、cloudpickle 3.1.1 | 这些版本已匹配。论文没有完整软件版本锁，不能据此证明所有论文实验使用同一环境 |
| Python / SUMO | 发布 checkpoint Python 3.10.12；原训练 SUMO 版本未证实 | 本机 Python 3.12.3 不同；SUMO/traci/sumolib 固定 1.18.0，属于本地可复现选择 |
| 硬件 | 论文 V-A：AMD 16-Core CPU、RTX 5090 | 本机 Intel i9-14900K、RTX 4090，不一致；不能保证速度或逐点数值完全相同 |
| 合作车辆 tau | U(0.6,0.8) s | 仍保留代码 N(0.7,0.1²)，明确不同；后续独立对照 |
| 交通流 / 背景车辆 | Hard 表格为 1394/1460/1390/1374/1490 vph；描述两类驾驶员 | 保留原 Hard 路由；名义输入流量与表格不对应，实测/输入口径待核实；部分车流未应用两类行为设置 |
| 邻车、奖励和安全控制 | 最近/关键邻车、事件奖励加和、冲突车辆安全检查 | 原代码先截断再排序、事件 if/elif、SC 仅查看 4 个邻车等差异仍在；奖励系数和训练期 SC 开关也未被论文完整说明 |
| 评估协议 | 最终每组 500 episodes、3 组评估种子 | 首轮每节点每种 SC 模式 32 episodes，属于筛选；最终 3×500 另行执行，不混用 |

首轮仍遵守已选定的 2×2 对照设计，没有把交通、奖励或 SC 的修改混入网络对照。更详细的 34 项差异及不确定项见 `../DAROM-reproduction-audit/configuration-audit.csv`。

## Forge 实时曲线配置及验证

- 项目：[DAROM-reproduction](https://forge.coreweave.com/wandb/arlikache-anhui-university/DAROM-reproduction)。浏览器使用 Forge 地址；SDK 的公共 API 地址固定为 `https://api.wandb.ai`，不是项目网页 URL。
- 正式运行名称为 `hard-uniform-A-seed42` 等，按 A/B/C/D 和 seed 标记。验证运行名称带 `verification-`，分组带 `-verification`，配置写入 `verification_only=true`，避免与正式结果混淆。
- 正式训练每 100 个环境步记录当前 SAC 指标；回合结束记录奖励；25k/50k/100k 记录独立评估结果。SDK 请求每 2 s 发送已有记录，实际页面更新仍受网络和服务器缓冲影响，不能保证硬实时。
- 所有指标使用 `env_steps` 作横轴。首 100 步是 SAC warmup，loss 在开始优化后的记录节点出现；32 回合评估期间训练暂时等待，评估指标不会每 100 步更新。
- 上传实验参数和标量指标，关闭代码、Git diff、模型、自动终端输出和机器信息采集；本地仍保存完整实验产物。每次运行的 `tracking.json` 保存 run ID 和 Forge 链接；在线初始化失败会明确报错，不静默转为离线。

建议查看的面板（横轴全部为 `env_steps`）：

| 曲线 | 指标 |
|---|---|
| 优化状态 | `train/actor_loss`、`train/critic_loss`、`train/ent_coef`、`train/ent_coef_loss` |
| 训练回报 | `train/episode_reward`、`train/reward_mean_100`、`train/episode_length` |
| 训练进度 | `train/completed_updates`、`time/effective_steps_per_second`、`time/evaluation_seconds` |
| 评估效果 | `eval/no_sc/success_rate`、`eval/original_sc/success_rate`、各模式 `mean_reward` 及失败类型比例 |

按 [W&B 官方自定义横轴说明](https://docs.coreweave.com/products/wandb/track/log/customize-logging-axes) 使用 `define_metric`；指标通过 SDK 日志同步，机制见 [官方记录说明](https://docs.coreweave.com/products/wandb/track/log)。如果已有手动布局未显示新增指标，在项目 Workspace 添加对应 Line plot，X 轴选择 `env_steps`。

联通验证命令：

```bash
.venv/bin/python -B -m experiments.run train --group D --seed 42 --smoke --smoke-steps 512 --tracking online --output /tmp/darom-forge-verification
```

已完成真实 CPU 短训练 512 steps、412 次优化和 8 个评估回合。云端回读验证 run `t3mc5mjp` 为 `finished`，有 13 个 actor/critic loss 与熵系数数据点，对应 env_steps 128、160、…、512。远端文件列表只有 `config.yaml` 和 `wandb-summary.json`，没有模型或源码附件。这次验证日志每 32 步采样，仅用于确认联通，不代表正式训练结果。

[直接打开验证曲线](https://forge.coreweave.com/wandb/arlikache-anhui-university/DAROM-reproduction/runs/t3mc5mjp)。本地证据：`/tmp/darom-forge-remote-verification.json`；运行产物：`/tmp/darom-forge-verification/smoke-hard-uniform-D-seed42-20261016T110314Z-fb7fe334/`。包含新增日志协议检查在内的 6 项回归测试通过。此记录对应正式训练之前的联通验证；正式 100k 首轮启动情况见下方。

## 正式首轮启动（2026-10-16）

北京时间 19:17 启动后台队列，监督进程 PID `1797308`。启动前 RTX 4090 显存占用约 6.6/24 GB、GPU 利用率 0–1%，内存可用约 37 GB；32 个 CPU 逻辑线程已有约 13 的系统负载，短时采样没有持续换页。综合共享服务器负载，选择 `runtime.parallel_runs=2`，每个训练进程固定 2 个 PyTorch 线程、1 个 interop 线程。评估仍使用独立 CPU 子进程，对应训练进程等待评估结束，避免共享 TraCI 连接。

正式输出根目录：`runs/reproduction/suite-20261016T111748Z-16d42973/`。其中 `queue.json` 保存任务状态、PID 和日志位置，`config.json` 固化本轮配置；每项任务的 `process.log`、独立训练目录和本地曲线在对应的 `A-seed42/` 等子目录中。后台启动探针和监督进程日志在 `/tmp/darom-first-round-*`，均不加入 Git。

| 首批任务 | PID | Forge 曲线 |
|---|---:|---|
| A / seed 42 | 1797310 | [hard-uniform-A-seed42](https://forge.coreweave.com/wandb/arlikache-anhui-university/DAROM-reproduction/runs/hdn922lz) |
| B / seed 42 | 1797311 | [hard-uniform-B-seed42](https://forge.coreweave.com/wandb/arlikache-anhui-university/DAROM-reproduction/runs/4u48aorn) |

本轮只执行 8 × 100k，不自动扩展到 500k 或 7M。`experiments/suite.py` 实现并发上限、相同输出根目录的队列锁、每任务日志与状态记录。检测到任务失败后不再启动后续任务，已经运行的任务正常完成，等待人工检查。队列锁不阻止直接执行独立 `train`，因此不要在现有队列之外重复发起训练。配置在启动时冻结，源码仍从项目读取；本轮期间不要修改训练、环境或编码器实现。强制终止监督进程后的自动恢复未实现，也不保证训练精确断点续跑。

启动前完成 6 项回归测试及严格环境预检；另用 `/tmp/darom-verify-suite.py` 验证正常情况下 8 个任务按顺序完成、并发峰值为 2，及首批失败后其余 6 个任务停止派发。

运行约 6 分钟后，从 Forge API 分别抽样回读到两组各 100 个 loss 数据点，在线最大步数为 A 10,700 / B 12,300，本地分别已到 10,800 / 12,300；均为 `running`、`verification_only=false`、`device=cuda`、正式预算 100k，已检查本地 actor/critic loss 和熵系数均为有限数。GPU 进程表确认两个训练进程各使用约 650 MiB 显存；整卡占用 7,914 MiB、利用率约 65%，机器内存可用约 33 GiB。验证证据保存在 `/tmp/darom-first-round-remote-verification.json`。

首批实测 A 约 30.4 steps/s、B 约 34.8 steps/s，此时尚未包含第一次 25k 评估。若后续吞吐相近，800k 总训练部分约 3.4 小时；计入独立评估、保存及共享机器波动，本轮暂估 4–6 小时，须以实际评估耗时继续校准。前期 loss 仅用于检查训练稳定性，不能据此宣称与论文一致或筛掉某组。

## 唯一长训练方案：D / seed 42 / 7M（2026-10-17）

用户已授权后台启动，2026-10-17 10:22（北京时间）已启动正式 7M；已有首轮任务继续运行。使用 `experiments/d_long.json` 与独立的 `experiments/long_run.py`，不修改现有 `src/` 中交通、奖励、邻车截断或安全控制实现。D 仍为 `hist_len=20`、`features_dim=256`、GRU hidden 64、actor/critic 256×256，其他 SAC 参数沿用首轮。训练预算为累计 7,000,000 步，CUDA，2 个 torch 线程，最多 1 个 D 长训练进程。

### 保存及恢复协议

- 每 **2,000,000 步**保存完整断点，并保存最终状态，即 **2,000,000、4,000,000、6,000,000、7,000,000** 步节点。默认保留这 4 个完整断点，不循环覆盖、不自动删除。另支持在指定绝对步数保存并暂停。
- 保存发生在该步 transition 已进入 replay buffer、该步优化器更新已完成之后。保存 SAC policy/target 网络、actor/critic/entropy 优化器、熵系数、完整 replay buffer（包含环形位置与 timeout）、环境步数及优化次数、Python/NumPy/PyTorch CPU/CUDA 随机状态、环境和延迟随机状态、Wrapper 历史、Monitor/VecEnv 状态、奖励窗口、评估进度与 best 选择记录。
- SUMO 恢复使用当前回合的独立种子及已执行动作，重新建立当前回合，并逐步重放到断点。恢复时核对整段观测/奖励/终止信息的摘要、所有车辆的位置/速度/车道及仿真时间，再恢复保存的 Python 状态；重放不计入训练步数，也不做梯度更新。若核对失败则拒绝继续，不静默重置回合。由原回合时限限制，重放只涉及当前回合，不是重放此前百万步训练。
- 断点先写 `.partial` 目录，所有文件写完、fsync、计算 SHA-256 后原子发布正式目录，再更新 `latest-checkpoint.json`。不完整目录、缺少文件或校验失败的断点不可恢复。
- 恢复时核对实际软件版本、Python、设备/线程配置及训练源码/场景文件哈希。每个运行段保存源码快照；代码不一致时应先恢复对应版本，不允许无提示混用。相同环境的短验证用于检验恢复正确性，不能保证换机器、换驱动或任意 CUDA 算子下都逐位一致。
- 从 2M 恢复后目标仍为 7M，剩余 5M；从 4M 恢复剩余 3M。不会重新学习 warmup，也不会把步数归零。
- 原生 `--resume` 入口每次创建新的本地目录和 Forge run，并记录父断点及相同 lineage 分组，适用于从旧断点重新分支。用户在 2026-10-17 要求本次正常续训保持同一条 Forge 曲线：本次运维恢复使用原 run ID `118fna3s` 和 W&B `resume='must'`，本地仍按运行段分别保存。必须先确认旧进程退出、积压数据上传完成且云端进度不超过断点，才能接续；从更早断点重跑属于分支，不应写入该 run。
- 用户要求：后续涉及训练暂停/重启、断点恢复、评估调度或 Forge 记录方式的变更，执行前先说明是否需要中断训练、数据同步与曲线连续性风险、可能回退的步数及额外存储，并获得确认。优先保持运行中的训练与记录稳定。
- 对训练进程发送 SIGINT/SIGTERM，会在安全边界保存后退出；若正在评估，将等待评估返回。强制 kill、断电或进程崩溃只能回退到最后一个已完整发布的断点，不承诺保存崩溃瞬间。按当前保存间隔，异常中断最多可能损失接近 2M 步进度；首个 2M 断点前异常中断则需从头开始。

模型文件仅占几 MB；主要空间来自容量 100 万条的 replay buffer，保存训练时采集的状态、动作、奖励等，单份约 1.1 GB。按每 2M 步保存一次及 7M 保存最终状态计算，4 个计划断点的 replay 数据合计约 4.4 GB，另计模型、日志和可能的中断断点；恢复分支会额外占用空间。只保存模型可用于推理或重新积累经验训练，但无法恢复原经验回放状态。原首轮的模型文件没有上述全套状态，不能用此入口把它伪装成完整续训起点。

### 运行命令

在 `DAROM-reproduction` 项目目录执行。当前正式训练已经在后台运行；以下命令用于后续操作参考，不要重复启动。

```bash
# 只检查配置及依赖
.venv/bin/python -B -m experiments.long_run check

# 从随机初始化开始，累计目标 7M，自动保存全部计划断点
.venv/bin/python -B -m experiments.long_run train

# 或计划在恰好 2M 保存并暂停；运行目录会打印到终端
.venv/bin/python -B -m experiments.long_run train --stop-at 2000000

# 从 2M 的完整目录恢复，剩余 5M；替换 RUN_NAME 为实际运行目录
.venv/bin/python -B -m experiments.long_run train \
  --resume runs/D-long/RUN_NAME/checkpoints/step-002000000

# 从 4M 重跑到总计 7M，或同时加 --stop-at 6000000 在 6M 暂停
.venv/bin/python -B -m experiments.long_run train \
  --resume runs/D-long/RUN_NAME/checkpoints/step-004000000
```

必须保留整个 `step-XXXXXXXXX/` 目录：`model.zip`、`replay-buffer.pkl`、`state.pkl`、`config.json`、`manifest.json`；不要只复制模型。恢复仅加载自己生成、可信的本地断点。

### 评估与验证范围

验证评估原计划在 100k 及每 250k 进行；2026-10-17 用户授权后续改为每 1M，默认配置现为 1M、2M、3M、4M、5M、6M、7M。当前运行已完成的 100k、250k、500k 评估保留在历史记录及恢复配置中，下一次为 1M。两种 SC 模式各 32 回合，固定配对种子，使用独立 CPU 子进程。这些节点只保存轻量模型用于评估，不额外保存 replay buffer；完整状态每 2M 步保存一次，另保存 7M 最终状态。训练 loss/熵系数仍每 100 步记录，训练奖励仍按回合记录，评估间隔变化不改变这些曲线的采样频率。它们仍是选模型的验证集。配置同时记录最终独立测试协议（3 个种子段×500 回合/模式），当前不自动执行此最终测试，不能把周期验证成绩称为论文规模的最终测试。

集成验证命令为 `.venv/bin/python -B -m tests.test_long_resume --device cpu`（另验证 `cuda`）。仅在 `/tmp/` 生成临时文件，不上传指标。连续 512 步与第 73 步（warmup）及第 273 步（回合中途）两次保存、恢复至 512 步对照；使用小 replay buffer 验证环形覆盖，包含真实优化、独立评估及恢复后的新回合；比较权重/优化器、随机状态、replay 内容、环境轨迹、回调进度和最终评估，另检查损坏文件及源码漂移会被拒绝。

2026-10-17 实测：CPU 与 CUDA 均通过上述对照，各完成 512 步、412 次梯度更新；连续训练与两次恢复后的权重、优化器、随机状态及 replay 数据逐位一致，最终评估一致，损坏文件及源码漂移均被拒绝。结果分别保存在 `tmp/ai/darom-long-resume-cpu-gy9v8qra/result.json` 与 `tmp/ai/darom-long-resume-cuda-46twsgpr/result.json`。这是本机同环境的短程验证，尚未执行百万步恢复测试，也不代表跨硬件逐位一致。原有 6 项复现测试通过。

恢复断点用于恢复训练状态。若还要保留此前选出的 best 推理模型及完整历史曲线，应同时保留原运行目录；恢复段可能继续沿用此前 best 的评分和文件引用。终止时推荐向训练 Python 的单个 PID 发送 SIGTERM；不要向整个进程组发送信号，否则可能先终止 SUMO 或评估子进程，届时只能使用此前完整断点。


### 正式 D 长训练启动记录（2026-10-17）

- 启动时间：北京时间 10:22:29；独立后台会话，训练 PID `1969400`，seed 42，从随机初始化训练至累计 7,000,000 步。
- 运行目录：`runs/D-long/D-seed42-from0-20261017T022231Z-7c46cd52`，内含冻结配置、源码快照、环境信息、状态、TensorBoard 与 JSONL 指标。
- 实时曲线：[Forge D 长训练 118fna3s](https://forge.coreweave.com/wandb/arlikache-anhui-university/DAROM-reproduction/runs/118fna3s)，lineage 为 `D-long-5f31457fe8c0`。
- 北京时间 10:27:54 的本地状态为 `running`，已训练 9,600 步、完成 9,500 次梯度更新；云端回读状态为 `running`，抽样 actor/critic loss 已包含 9,100 步的数据点。该记录仅证明启动及曲线同步正常，不代表训练效果结论。
- 完整状态保存于 2M、4M、6M、7M；首次独立评估在 100k，此前成功率评估曲线尚无数据。每 100 步记录 loss 等训练指标。
- 临时后台启动器、启动记录及控制台日志位于 `tmp/ai/darom-d-long-launch/`，不进入 Git；正式训练产物位于已忽略的 `runs/`。


2026-10-17 临时产物整理：上述启动、曲线分析及 CPU/CUDA 恢复测试目录已迁入项目根目录 `tmp/ai/`，该目录已被 Git 忽略。迁移采用同一文件系统内的目录重命名，保留正在写入的启动日志 inode；未重启训练。历史 JSON、测试日志和断点元数据中的原始绝对路径作为执行记录保留，实际查找时应使用迁移后的目录。

### 与论文 Fig. 4 的首次直接曲线比较（500k，2026-10-17）

结论：当前 D 训练有学习进展，但与论文 Hard/GRU 的早期轨迹明显不同，尚不能称为复现论文训练曲线。此前 91.8k 的逐点一致结论只针对本地首轮 D/seed42，不代表与论文一致。本次只读分析，未修改或中断正在运行的训练。

来源为 [arXiv v5 PDF 第 9 页 Fig. 4](https://arxiv.org/pdf/2403.11852v5#page=9) 右侧 Hard 列的蓝色 GRU 曲线。将本地同版本 PDF 页面转为 SVG，按颜色、位置和图例选择三条实线，用坐标轴刻度还原环境步数与显示数值。下表论文数值均为图示近似值，并非作者原始日志；论文未明确图中平滑窗口和训练种子聚合方式。当前训练回报使用最近 100 个完整回合均值，loss 使用截至该节点最近 10k 步的 100 个日志样本均值，因此不做严格点对点误差或统计显著性结论。

| 环境步数 | 论文回报≈ | 当前百回合奖励 | 论文 critic≈ | 当前 critic（10k 均值） | 论文 actor≈ | 当前 actor（10k 均值） |
|---|---:|---:|---:|---:|---:|---:|
| 100k | -227 | 64.8 | 7.6 | 27.8 | 61.7 | -1.7 |
| 250k | -221 | 82.3 | 7.7 | 20.7 | 52.4 | -13.7 |
| 500k | -180 | 92.1 | 9.0 | 14.5 | 40.6 | -20.1 |

当前奖励提升、actor loss 转负均明显早于论文；当前 critic loss 在这些节点更高。更快或更高不能直接解释为方法更优，也不能据此确认训练发生错误。训练种子、统计和平滑口径，以及已记录的交通分布、奖励实现和训练期 SC 等差异，都可能影响曲线，尚未分离各因素的作用。

论文后期 Hard/GRU 图示训练回报约 130–160，actor loss 约 -34 至 -38；critic loss 后期也有明显尖峰，并非严格单调下降。当前只有 0.5M/7M（约 7.1%）进度，不能将当前值与论文最终平台直接作等预算结论。论文未展示熵/熵系数训练曲线，无法证明当前 α 轨迹与论文一致。

500k 的独立验证已完成：无 SC 成功 30/32（93.75%）、碰撞 2/32，平均回报 148.04；原版 SC 成功 31/32（96.875%）、碰撞 1/32，平均回报 162.67。验证回报与训练回报分开报告，这些 32 回合样本也不能替代论文规模的最终评估。当前 500k 内日志无 NaN/Inf；保持配置继续训练，在已有 1M、2M 评估节点继续检查趋势及性能，不能仅为了曲线外形相似而改动正在运行的实验。

分析产物均位于已忽略的 `tmp/ai/paper-curve-comparison/`：`paper-vs-current-500k.png` / `.pdf` 为全预算及 0–0.5M 放大对比图；`paper-hard-gru-*-digitized.csv` 为论文曲线近似数据；`current-metrics-through-500k.json` 为本地日志快照；`comparison-summary.json` 为汇总，`compare.py` 为重建脚本。

### Forge 曲线补传与调度变更（2026-10-17）

- 用户授权后续减少评估至每 1M；运行配置不支持热更新，因此于北京时间 15:30 保存暂停，完整断点为原目录下 `checkpoints/step-000657971`，包含 657,971 步、657,871 次更新及完整 replay/环境/RNG/优化器状态。本次额外保存一份约 1.1 GB 的恢复状态，原计划 2M/4M/6M/7M 保存节点不变。
- 原 run ID 为 `118fna3s`，名称 `D-seed42-from0-7c46cd52`。从约 11:29 起，旧进程继承的代理 `127.0.0.1:17890` 连接失败；本地训练继续推进，但云端停留在约 100k。当前本地 `metrics.jsonl` 含 10,610 条记录，其中 6,578 个 actor/critic/熵系数采样点（200 至 657,900 步，每 100 步）及 4,028 条回合奖励，另有 3 次独立评估。不能把云端曲线末端当作实际训练进度。
- 初始自动交接等待旧进程退出超时，没有启动第二个训练进程。旧进程已完成 checkpoint 保存，仍在 W&B `finish()` 同步阶段。经用户授权，临时回环转发仅允许 W&B API 和其 Google 文件存储域名，不解密 TLS，不修改系统代理配置；旧进程退出或 30 分钟后自动关闭。
- 北京时间 15:50，配置与摘要文件的上传已推进，但历史数据接口仍返回 HTTP 500 / `context deadline exceeded`，尚不能宣布补传成功。已逐文件核对 checkpoint SHA-256 和当前源码 SHA-256，全部匹配；未修改训练算法源码或原始指标。
- 恢复脚本为已忽略目录 `tmp/ai/evaluation-frequency/handoff.py`。`continue` 模式最多等待旧进程 180 秒，要求旧 writer 完全退出、云端状态为 finished、全部云端指标与本地逐项一致后，才启动恢复进程并使用 `resume='must'` 接续同一 run。失败则保持暂停。记录位于同目录 `handoff.json`、`checkpoint-audit.json`；仅成功时才生成 `before-resume-cloud-audit.json`、`schedule-change.json`、`resumed-training.log`。直接执行原生 `--resume` 会新建云端 run，不满足本次同一曲线要求。
- 今后涉及此类暂停、恢复或记录方式变更，必须先向用户说明风险并确认。恢复成功需同时验证本地步数增长和原 Forge run 出现断点之后的新数据；仅本地状态正常不足以宣称云端同步正常。
- 北京时间 15:51，第二次恢复协调也因旧 writer 未退出而超时，未启动新训练进程，没有待执行的自动续训。用户表示无法恢复则先停止扩大操作，本次保留完整断点与原 run 等待后续处理；旧 writer 的既有上传重试仍在进行，临时转发按上述时限退出。
- 用户随后明确要求“先恢复训练”。告知云端仍可能缺失、先保留本地曲线后，15:59 再次核对断点及源码 SHA-256 全部匹配，只向已卡在同步收尾的 W&B 子进程 `1969452` 发送 SIGTERM，未使用 SIGKILL。旧训练 Python `1969400` 随后退出、释放单训练锁；原始 JSONL、TensorBoard、W&B spool 与断点均保留。
- 后台恢复 PID `2395485`，新本地运行段 `runs/D-long/D-seed42-from657971-20261017T075908Z-435a663a`，保持原 lineage `D-long-5f31457fe8c0`。实际仅调整 `evaluation.steps`（保留已完成的 100k/250k/500k，后续 1M 间隔）与 `tracking.mode='disabled'`（暂时关闭 W&B 上传，JSONL/TensorBoard 继续记录）；模型、训练环境、优化器、replay、随机状态与其他训练参数按原断点恢复。16:00:58 状态为 running，661,200 步/661,100 次梯度更新，恢复时的环境重放及计数校验已通过。下一评估 1M，下一计划完整断点 2M，累计目标仍为 7M。
- 运维脚本与记录位于 `tmp/ai/evaluation-frequency/resume_training.py`、`training-first-resume.json`、`training-first-config-change.json`、`training-first-resume.log`。本次没有创建新的 Forge run，也没有补齐旧 run；后续重新接入云端前须说明具体方案和风险并确认，不能将本地恢复成功表述为 Forge 已恢复。以后产生的完整断点会保存当前本地记录模式，原生恢复不会自行开启在线上传。
- 用户要求的风险告知规则已写入全局 `/home/cyl/.codex/AGENTS.md`：涉及运行中任务的暂停/恢复/调度/记录及网络环境变更，必须先说明具体影响、进度和曲线风险、成本与替代方案，再获得明确确认；既有授权不覆盖新增风险。

### 不中断训练的 Forge 周期同步（2026-10-17）

用户要求训练不中断、Forge 每隔一段时间能看到曲线变化。采用独立低优先级同步进程（PID `2440549`），只读原始与恢复段的 `metrics.jsonl`，使用原 run `118fna3s` / `D-seed42-from0-7c46cd52`。训练 PID `2395485` 未变，未修改训练源码、配置或单训练锁，也未重启训练。同步器不加载模型、不使用 CUDA、不上传模型或代码；只上传已有标量指标。

- 首次全量回读证明云端 1,766 条记录与本地前缀逐项一致，末端为 100,000 步。首次恢复校验因 SDK 的 `run.step` 返回 0 而主动停止，未追加指标。进一步核实恢复接口的 `historyLineCount=1766` 及 SDK 的 `starting_step=1766` 一致；改用实际恢复起点校验，每条日志显式指定递增的 `_step`，指标横轴保留实际累计 `env_steps`。未调用 rewind、删除或覆盖历史。
- 先补齐积压，分批最多 250 条，限制传输请求大小；每批先回读确认服务器记录位置推进，再发送下一批。补齐当前快照后每 300 秒检查新增日志；网络或服务端失败只等待/重试同步，不重新训练，也不重复追加状态不明的批次。正常情况下约 5 分钟加上传延迟可看到新增数据，不能保证服务器故障期间也准时更新。
- loss、熵系数、回合奖励、奖励均值及已有评估指标沿用原值与原采样频率，不插值、不补造未执行的评估。周期同步改变上传时机，不改变训练或评估频率；wall-clock 上传时间不应当作原始采样时间，曲线使用 `env_steps`。
- 操作入口为已忽略的 `tmp/ai/evaluation-frequency/periodic_forge.py`；运行记录在同级 `forge-periodic/`：`launch.json`、`status.json`、`sync-*.log`、`initial-audit.json`、`verified-history.json`。`status.json` 区分已被云端确认的进度与排队中的进度，不能只凭同步进程存活判断同步成功。可用 `.venv/bin/python -B tmp/ai/evaluation-frequency/periodic_forge.py verify` 只读回查云端前缀与本地逐项一致性。
- 此同步器只跟随已指定的两个本地运行段；后续恢复到其他目录时须先验证 lineage、断点位置和云端前缀再接续。训练自身的 `tracking.mode` 仍为 `disabled`，不要同时启动另一个写入同 run 的训练 SDK 或上传器。同步器独立终止不应影响训练；不向训练进程发送信号。
- 北京时间 16:14:05 首轮历史补传完成：云端已确认 10,903 条记录，推进至 676,300 步（本轮开始时的日志快照），包含 657,971 步恢复前后两段。随后完整回读的全部 10,903 个唯一记录与本地逐项匹配。下一轮约 16:19 执行；训练在补传期间继续推进，16:14:34 本地已到 686,100 步，因此页面末端允许落后本地当前步数。本地源码哈希和训练 PID 均未变化。
- 回读过程中发现 SDK 扫描在分页与已有 parquet 片段边界重复返回 766 条相同记录；校验器对每个返回副本都先与本地值核对，再按 `_step` 统计唯一覆盖，不将 API 重复返回误判为新增记录，也不因此重复上传。核查记录包括 `verified-history.json`、`verified-snapshot.json` 和 `isolation-verification.json`。
