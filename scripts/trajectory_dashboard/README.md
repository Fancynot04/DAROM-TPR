# 双 Ego 轨迹诊断

读取同一次已发布评估的 `eval_trajectory.jsonl.gz` 和 `summary.json`。可换用其他 checkpoint，只需提供对应的一对文件。本工具使用 Streamlit 和 Plotly，可在独立的分析环境运行。本次验证使用 Python 3.12、Streamlit 1.67.0 和 Plotly 7.1.0。

在新机器上，可在独立虚拟环境中安装页面依赖。

```bash
python3.12 -m venv tmp/ai/trajectory-dashboard/.venv
source tmp/ai/trajectory-dashboard/.venv/bin/activate
python -m pip install streamlit==1.67.0 plotly==7.1.0
```

### Mac 一键打开

双击同目录的 `打开轨迹分析.command`，打开 `http://127.0.0.1:8501`。默认发现
`~/Experiments/darom` 的评估和 `~/Library/Caches/darom-trajectories` 的已下载批次。
侧栏的「刷新远程列表」读取 4090-2 上 `~/.local/share/darom-trajectories` 的清单。
选中远程批次后点击「下载轨迹」，SHA-256 校验成功后可离线打开。SSH 密码提示出现在启动终端。

切换到「并排对照」，左右分别选实验、训练检查点和评估批次。共用一个 Ego、车道横轴口径
和双方都有的评估 seed。左右各自保留回合步游标，可查看长度不同的回合各自的终止步。
每侧显示各自的训练 seed、训练步数、结局和三张轨迹诊断图。奖励图图例位于绘图区上方，
窄列中可在图例区域内滚动查看更多分项。

已有评估可在评估服务器运行以下命令发布到统一目录。发布器先校验轨迹与报告，再发布清单。

```bash
python -m scripts.trajectory_dashboard.catalog \
  --summary /path/to/summary.json --experiment-id <完整实验ID> --batch <批次标签>
```

### 从终端直接打开指定轨迹

在项目根目录校验并建立 seed 索引。

```bash
python -m scripts.trajectory_dashboard.trace --trace /path/to/eval_trajectory.jsonl.gz
```

启动页面后打开 `http://127.0.0.1:8501`。默认读取轨迹同目录的 `summary.json`，也可用 `--summary FILE` 指定。默认索引写入 `tmp/ai/trajectory-dashboard/index/`，可通过 `--cache-dir DIR` 调整。

```bash
python -m streamlit run scripts/trajectory_dashboard/app.py \
  --server.address 127.0.0.1 --server.port 8501 \
  -- --trace /path/to/eval_trajectory.jsonl.gz
```

若同一实验有多个检查点，传入其评估根目录即可在页面切换。该目录下的
`step_<步数>/` 各放同次评估的 `summary.json` 与 `eval_trajectory.jsonl.gz`。
页面默认选取训练步数最大的节点，显示该节点的双车到达数，并在切换时保留仍可用的评估 seed。

```bash
python -m streamlit run scripts/trajectory_dashboard/app.py \
  --server.address 127.0.0.1 --server.port 8501 \
  -- --eval-dir /path/to/eval
```

### 图表口径

- **车道与距离**。默认横轴是每个有效首末 XY 位置之间的累计位移估算。缺少首末坐标、未执行动作或终止时已移除的位移不补值。也可切换到实测步末任务剩余距离。实线按道路和连续步分段，内部连接区以灰色标示。
- **变道节点**。同一道路上步首与步末的车道编号发生变化时，以步末位置标出菱形。悬停可查看该步存档的各项加权个人奖励，沿用单步奖励分项的顺序。按带正负号的原始数值比较，最大项标绿，并列时标绿顺序最靠前的一项。
- **时间与动作**。三图使用同一个回合步游标，可切换步首与步末。请求和实际提交使用 SUMO 步首时刻，步末速度、实测加速度和换道事件采用对应存档时刻。步首已判定的 Missed Exit 对齐步首，详情保留 Actor 原始输出和动作执行标志。
- **奖励**。稠密分项是单步值。累计回报包含所有个人分项和终止步，折扣选项使用 checkpoint 中的 γ，以该 Ego 首个计入回报的回合步为起点。团队奖励使用共享仿真步记录，独立展示。路线准备详情显示阶段、连接、局部换道需求、安全门槛、首次前进距离和实发奖金。

索引建立时检查文件哈希、种子与步编号连续性、每车终止结局、每车逐步奖励与回合分项以及团队奖励。索引按轨迹和报告的 SHA-256 唯一标识，页面切换 seed 时只读取所在回合。源文件始终以只读方式打开。

奖励图默认关闭汇入、个人累计回报、终止奖励、团队单步奖励和团队实发。汇入可在「单步奖励分项」中选中，其余可点击图例开启。当前页面会话内，调整检查点、Ego、结局、评估 Seed、车道图横轴、共享回合步或游标时刻时，分项选择和图例开关都会保留。某项在当前轨迹中缺少记录时仍会保留选择，有记录时按原开关状态显示。

### 轨迹归档

评估发布后，将同一 checkpoint 的原始 `eval_trajectory.jsonl.gz`、`summary.json`、`episodes.csv` 和现有的分区分析一并保存到 `~/Experiments/darom/<完整实验 ID>/evidence/evaluations/step_<步数>/`。核对副本与远端的 SHA-256，在同目录的 `manifest.json` 记录来源、模型哈希与评估协议，并将清单登记到实验档案。

后续打开页面时，将 `--trace` 指向归档中的原始轨迹。SQLite 索引可以用归档文件重建，项目 `tmp/ai/` 下的副本和索引供临时分析使用。
