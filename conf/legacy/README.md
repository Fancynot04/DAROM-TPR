# 单 Ego 历史配置

- `e76_d3/` 保留提交 `ca64f77` 的父提交中的 E76 → D3 配置组，包括基础实验和 B1–B3 奖励配置。`experiment/baseline_1m.yaml` 保留 `0b5c667` 移除前最后一版基础实验预算。
- `o3_d3/task/o3_d3.yaml` 来自提交 `0b5c667` 的父提交，记录 O3 → D3 单 Ego 任务设置。

归档文件保留历史内容和组内路径。当前双 Ego 训练入口 `train_masac.py` 的参数来自 CLI、代码默认值及续训 checkpoint，运行配置保存在训练输出目录的 `resolved_config.json`。
