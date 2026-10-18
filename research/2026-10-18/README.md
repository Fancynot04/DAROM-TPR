# DAROM 研究成果归档 · 2026-10-18

本目录将原本分散在两个项目中的研究成果整理为正式、可核验的版本化资料。原临时文件保留原位；本目录不被正在运行的训练或同步程序使用。以后新增实验使用新的日期目录，避免覆盖本次快照。

## 内容与状态

| 目录 | 内容 | 证据状态 |
|---|---|---|
| `antwerp/evaluation/` | 六组共 9000 回合的 TABLE III、精简逐回合数据、模型/场景/源码哈希、评估器源码参考 | 已完成；三种密度各一个 1M checkpoint，每种安全设置 3 批 × 500 回合 |
| `antwerp/training/` | Antwerp 与旧原场景训练的标量 CSV、统计和曲线图 | 历史训练记录；不同实验、预算与日志频率不能混为同一条件 |
| `antwerp/loss_audit/` | SAC 的 actor 分解、critic/TD 异常及回放池诊断结果 | 冻结模型离线检查；不是历史尖峰发生时的精确批次重演 |
| `antwerp/action_modes/` | 确定性/随机动作类别的离线概率分析 | 已完成离线采样；配对 SUMO 对照尚未运行 |
| `reproduction/source/` | 独立 D 复现项目的源码、ABCD/D 配置、测试、场景与实验计划 | 当前工作区快照，含当时未提交的改动；不是对 Antwerp 主代码的替换 |
| `reproduction/run_metadata/` | 原始段及恢复段的实际配置、依赖和源码哈希 | 解释配置文件默认值与实际恢复状态的差别 |
| `reproduction/d_metrics.csv` | D 训练两段日志的固定前缀，保留原始指标值和段名 | 截止 6,332,700 环境步；不是 7M 最终结果，也不是云端同步证明 |
| `reproduction/paper_comparison/` | 论文 Fig.4 数字化曲线、早期 500k 对比图及摘要 | 论文显示曲线的近似提取值，非作者原始日志；原文平滑方式未知 |
| `methods/` | 当时用于分析、评估调度和 Forge 同步的源码文本 | 历史方法证据；`.py.txt` 不作为可直接执行的入口 |
| `tools/` | 不依赖本机原临时目录的校验与绘图入口 | 校验只读；绘图仅写本项目已忽略的 `tmp/ai/` |

优先查看 [最终评估表](antwerp/evaluation/table3.md)、[研究结论与限制](FINDINGS.md)、[复现实验计划](reproduction/source/docs/reproduction-experiment-plan.md)。各结果的完整 JSON/CSV 保留精度，Markdown 表用于阅读。

## 离线复核

在仓库根目录执行，无需运行 SUMO、连接 Forge 或加载模型：

```bash
python3 -B research/2026-10-18/tools/verify.py
```

该命令校验归档文件 SHA-256、9000 个任务的唯一性及完整配对、六组指标与批间标准差、成功率 Wilson 区间、D 指标步数顺序，以及实际训练源码哈希和快照的一致性。

如当前 Python 已安装 numpy/matplotlib，可以用现有环境重新绘制 D 与论文的曲线比较：

```bash
python3 -B research/2026-10-18/tools/plot_curves.py --window 50000
```

图保存至仓库 `tmp/ai/research-archive/d-paper-comparison.png`，不修改归档。loss 使用按环境步数划定的滑动窗口均值；训练 reward 使用原日志中的 100 回合均值；熵系数 `ent_coef` 不等同于策略熵。工具不会安装依赖或启动网络请求。

## 溯源、缺失项与使用边界

- `provenance.json` 记录来源及原文件哈希；`checksums.json` 校验归档文件。文本中的本机工作区路径统一写成 `${WORKSPACE}`，无需设置此变量，它只是出处占位符。
- `.gitattributes` 保留源码快照与参考脚本原有字节，不为清理历史空白而改变哈希；CSV 保留原有 CRLF 行分隔。新增工具与说明仍进行常规空白检查。
- `episodes.jsonl` 每行保留任务、结果、运动统计、SUMO seed、场景与模型哈希和原报告哈希。为降低体积，没有复制重复的启动命令和整份逐回合报告；不是完整交通轨迹。
- `reproduction/metric_snapshot.json` 记录正在增长的日志被读取时的固定完整前缀及其哈希。CSV 保留各日志行和同一步上的多类事件，未插值、未用云端记录补造数据。
- 源码快照与实际两个 D 运行段的 `source-hashes.json` 已逐项比对。`experiments/d_long.json` 是入口默认配置；恢复后实际配置以 `run_metadata/*/config.json` 为准，尤其是本地日志模式与独立 Forge 上传器。
- 模型、回放池、W&B 原始文件、缓存、凭据、论文全文和重复运行日志不入库。模型/场景哈希可以确认版本，但不能代替缺失的模型和回放池；仅凭此归档无法重做所有模型离线诊断。
- 论文来源为 [DAROM arXiv v5](https://arxiv.org/abs/2403.11852v5)，上游实现为 [amin-tabrizian/DAROM](https://github.com/amin-tabrizian/DAROM)，D 配置固定上游提交 `eabd572fc0aece579e5b553bdc7af27a8b81d03a`。保留作者归属，未宣称重新许可上游代码或论文图线。
- `methods/` 和评估器源码参考保留原研究方法，可能依赖未入库的模型、固定 run 或历史临时目录。尤其 `periodic_forge.py.txt` 会写云端，`paired_eval.py.txt` 会启动仿真：本次仅归档，不将它们当作已通用化、已完成验证的工具。
- `reproduction/source/` 不会自动安装依赖或启动任务。需要新训练时，应先在独立工作目录准备环境并复核配置；其中的历史文档可能含旧 PID、旧状态或先前临时目录策略，不能当作当前状态或操作授权。

后续维护：新增证据时保留实验配置、来源哈希、种子与指标定义；只将经过检查的正式成果纳入日期目录。原始日志、模型及回放池继续另行保存，不因不提交 Git 而删除。
