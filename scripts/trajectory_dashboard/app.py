"""Read-only DAROM-TPR trajectory dashboard."""

import sqlite3
import subprocess
import tarfile
from pathlib import Path

import streamlit as st

from scripts.trajectory_dashboard import catalog
from scripts.trajectory_dashboard.figures import TERMS, lane_figure, reward_figure, telemetry_figure
from scripts.trajectory_dashboard.series import cursor_point, episode_series
from scripts.trajectory_dashboard.trace import EGO_IDS, argument_parser, build_index, published_evaluations


reward_legend_state = st.components.v2.component(
    "trajectory_reward_legend",
    css=":host { display: none; }",
    js="""
    export default function ({ data, setStateValue }) {
        const container = document.querySelector('.st-key-' + data.chartKey);
        if (!container) return;
        let graph;
        const saveVisibility = () => {
            setStateValue('visibility', Object.fromEntries(graph.data.map(trace => [
                trace.meta ?? trace.name, trace.visible ?? true,
            ])));
        };
        const detach = () => graph?.removeListener?.('plotly_restyle', saveVisibility);
        const attach = () => {
            const next = container.querySelector('.js-plotly-plot');
            if (!next?.on || next === graph) return;
            detach();
            graph = next;
            graph.on('plotly_restyle', saveVisibility);
        };
        // Streamlit can replace the Plotly graph when the cursor changes.
        const observer = new MutationObserver(attach);
        observer.observe(container, { childList: true, subtree: true });
        attach();
        return () => {
            observer.disconnect();
            detach();
        };
    }
    """,
)


@st.cache_data(show_spinner=False, max_entries=4)
def open_archive(trace, summary, cache_dir, signatures):
    return build_index(trace, summary, cache_dir)


def signature(path):
    attrs = path.stat()
    return (attrs.st_dev, attrs.st_ino, attrs.st_size, attrs.st_mtime_ns)


@st.cache_data(show_spinner=False, max_entries=12)
def load_series(archive, seed, ego):
    return episode_series(archive.episode(seed), ego, archive.metadata["step_length_s"])


def shown(value, digits=2):
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def show_selected(point):
    a, b, c, d = st.columns(4)
    a.metric("SUMO 时间 (s)", shown(point["cursor_time"]))
    road = point["road"] or "—"
    b.metric("道路 · 车道", f"{road} · {shown(point['lane'])}")
    c.metric("当步个人回报", shown(point["reward"] if point["acted"] or point["terminal"] else None, 4))
    d.metric("距任务终点 (m)", shown(point["remaining"]))
    if point["terminal"]:
        timing = "步末" if point["acted"] else "步首，未执行动作"
        st.caption(f"终止 {point['terminal']} · {timing} · 任务奖励 {shown(point['components'].get('task'))}")
    elif not point["acted"]:
        st.caption("所选 Ego 此步没有执行动作。")
    if point["time_inferred"]:
        st.caption("此步时间按同回合的实测时钟和步长推算。")
    prep = point["preparation"]
    if prep is not None:
        local_changes = shown(prep.get("local_lane_changes"))
        normalization = shown(prep.get("normalization"))
        forward_m = shown(prep.get("first_progress_m"), 3)
        eligible = shown(prep.get("safety"))
        st.caption(
            f"路线准备 {prep['phase']} · n/N {local_changes}/{normalization} · "
            f"安全 {eligible} · 首次前进 {forward_m} m · 实发 {shown(prep.get('reward'), 5)}"
        )
    with st.expander("当步原始字段"):
        st.json({
            "step": point["step"], "pre_time": point["pre_time"], "post_time": point["post_time"],
            "terminal_time": point["terminal_time"], "position_source": point["position_source"],
            "pre_road": point["pre_road"], "pre_lane": point["pre_lane"],
            "actor_output": point["raw_action"], "action_executed": point["acted"],
            "submitted_direction": point["submitted_direction"],
            "lane_change_events": point["events"], "route_preparation": prep,
            "leader_gap_m": point["leader_gap_m"], "components": point["components"],
            "team_reward": point["team_reward"], "terminal": point["terminal"],
        })


def sources(args):
    if args.trace:
        trace = args.trace.resolve()
        summary = Path(args.summary or trace.with_name(catalog.SUMMARY)).resolve()
        return {"trace": ({"experiment": trace.parent.name, "step": None, "batch": "指定轨迹"}, (trace, summary))}
    if args.eval_dir:
        return {str(step): ({"experiment": args.eval_dir.name, "step": step, "batch": "本机评估"}, paths)
                for step, paths in published_evaluations(args.eval_dir).items()}

    local = catalog.local_entries(args.bundle_cache, args.local_archive)
    entries = {key: (row, (directory / catalog.TRACE, directory / catalog.SUMMARY))
               for key, (row, directory) in local.items()}
    if args.remote and st.sidebar.button("刷新远程列表"):
        try:
            with st.spinner("正在读取远程列表。如需 SSH 密码，请在启动终端输入。"):
                st.session_state["remote_evaluations"] = catalog.remote_entries(
                    args.remote, args.ssh_port, args.remote_archive
                )
        except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
            st.warning(f"远程刷新失败：{error}")
    for key, row in st.session_state.get("remote_evaluations", {}).items():
        entries.setdefault(key, (row, None))
    return entries


def choose(entries, panel, preferred=None, step_offset=0):
    experiments = sorted({row["experiment"] for row, _ in entries.values()}, reverse=True)
    experiment = st.selectbox("实验", experiments, key=f"{panel}-experiment",
                              index=experiments.index(preferred) if preferred in experiments else 0)
    matching = {key: (row, paths) for key, (row, paths) in entries.items() if row["experiment"] == experiment}
    steps = sorted({row["step"] for row, _ in matching.values()}, reverse=True)
    step = st.selectbox("训练检查点", steps, index=min(step_offset, len(steps) - 1),
                        format_func=lambda value: f"{value:,} 步" if value is not None else "指定轨迹",
                        key=f"{panel}-step-{experiment}")
    keys = sorted(key for key, (row, _) in matching.items() if row["step"] == step)

    def label(key):
        row, paths = matching[key]
        digest = row.get("sha256", {}).get(catalog.SUMMARY)
        batch = f"{row['batch']} · {digest[:10]}" if digest else row["batch"]
        return f"{batch} · {'本机' if paths else '远程'}"

    key = st.selectbox("评估批次", keys, format_func=label, key=f"{panel}-batch-{experiment}-{step}")
    return matching[key]


def load_choice(args, choice, panel):
    row, paths = choice
    try:
        if paths is None:
            if not st.button("下载轨迹", key=f"{panel}-download"):
                return None
            with st.spinner("下载评估文件并校验 SHA-256。如需 SSH 密码，请查看启动终端。"):
                catalog.download(args.remote, args.ssh_port, args.remote_archive, args.bundle_cache, row)
            st.rerun()
        trace, summary = paths
        with st.spinner("校验轨迹并读取 seed 索引"):
            archive = open_archive(trace, summary, args.cache_dir, (signature(trace), signature(summary)))
        if row["step"] is not None and row["step"] != archive.metadata["checkpoint"]:
            raise ValueError("检查点目录与评估报告步数不一致")
        return archive
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, tarfile.TarError,
            subprocess.SubprocessError) as error:
        st.error(f"无法打开轨迹：{error}")
        return None


def describe(archive):
    meta = archive.metadata
    episodes = meta["episodes"]
    both_arrived = sum(all(ep["outcomes"][ego] == "Arrived" for ego in EGO_IDS) for ep in episodes)
    mode = "共享 Actor · " if meta.get("shared_actor") else ""
    seed_label = f"训练 Seed {meta['training_seed']} · " if meta.get("training_seed") is not None else ""
    st.caption(
        f"{mode}{seed_label}Checkpoint {meta['checkpoint']:,} 步 · 双车均到达 {both_arrived}/{len(episodes)} · "
        f"{meta['step_records']:,} 条步记录 · Δt {meta['step_length_s']:g} s"
    )



def trajectory(archive, seed, ego, panel):
    meta = archive.metadata
    ep = next(ep for ep in meta["episodes"] if ep["seed"] == seed)
    st.caption(f"Ego 1 · {ep['outcomes']['ego_1']}　　Ego 2 · {ep['outcomes']['ego_2']}")
    points = load_series(archive, seed, ego)
    cursor, timing = st.columns([4, 1])
    if len(points) > 1:
        step = cursor.slider("回合步", 1, len(points), 1, key=f"{panel}-cursor-{meta['trace_sha256']}-{seed}")
    else:
        step = 1
    at = timing.radio("游标时刻", ["步首", "步末"], horizontal=True, key=f"{panel}-timing")
    return points, step, at


def draw_charts(archive, seed, ego, x_key, panel):
    meta = archive.metadata
    points, step, at = trajectory(archive, seed, ego, panel)
    selected = cursor_point(points[step - 1], at == "步首")
    show_selected(selected)

    revision = f"{meta['trace_sha256']}-{seed}-{ego}"
    st.subheader("车道—距离轨迹")
    st.caption("累计位移由每步可用的首末 XY 估算。连接区单独标注，末步缺少坐标时只显示实测位置。"
               " 悬停菱形变道节点可查看当步个人奖励，数值最高的一项标绿。")
    st.plotly_chart(lane_figure(points, selected, x_key, revision + x_key), width="stretch", key=f"{panel}-lane")

    st.subheader("速度与动作时序")
    st.caption("提交方向 −1 朝较小车道编号，+1 朝较大车道编号。稳定完成和中断使用事件记录。")
    st.plotly_chart(telemetry_figure(points, selected, revision), width="stretch", key=f"{panel}-telemetry")

    st.subheader("逐步与累计奖励")
    filters, discount = st.columns([3, 1])
    if f"{panel}-reward_terms" not in st.session_state:
        st.session_state[f"{panel}-reward_terms"] = [
            term for term in TERMS if term != "merge" and any(p["components"].get(term, 0) != 0 for p in points)
        ]
    terms = filters.multiselect("单步奖励分项", list(TERMS), format_func=TERMS.get, key=f"{panel}-reward_terms")
    plotted_terms = [term for term in terms if any(term in p["components"] for p in points)]
    discounted = discount.checkbox(f"折扣累计 γ={meta['gamma']:g}", key=f"{panel}-reward_discounted")
    gamma = meta["gamma"] if discounted else 1.0
    chart_key = f"{panel}-reward-chart"
    with st.container(key=chart_key, gap=None):
        chart = st.empty()
        legend = reward_legend_state(key=f"{panel}-reward-legend", data={"chartKey": chart_key}, default={"visibility": {}},
                                     on_visibility_change=lambda: None, height=0)
        # A curve can disappear in one episode and reappear in another.
        remembered = st.session_state.get(f"{panel}-reward_visibility", {}) | legend.visibility
        st.session_state[f"{panel}-reward_visibility"] = remembered
        view = (revision, step, at, tuple(terms), gamma)
        # Keep the chart mounted on legend-only reruns so Plotly can finish handling clicks.
        previous_view, visibility = st.session_state.get(f"{panel}-reward_chart_view", (None, {}))
        if view != previous_view:
            visibility = remembered
            st.session_state[f"{panel}-reward_chart_view"] = (view, visibility)
        fig = reward_figure(points, selected, plotted_terms, gamma, revision)
        for trace in fig.data:
            trace.visible = visibility.get(trace.meta or trace.name, trace.visible)
        chart.plotly_chart(fig, width="stretch", key=f"{panel}-reward")



def single_view(args, entries):
    choice = next(iter(entries.values())) if args.trace else choose(entries, "single")
    archive = load_choice(args, choice, "single")
    if archive is None:
        return
    describe(archive)
    a, b, c, d = st.columns([1, 1.3, 1.4, 1.7])
    ego = a.selectbox("Ego", EGO_IDS, format_func=lambda e: "Ego 1" if e == "ego_1" else "Ego 2")
    outcomes = sorted({ep["outcomes"][ego] for ep in archive.metadata["episodes"]})
    outcome = b.selectbox("所选 Ego 的结局", ["全部", *outcomes])
    seeds = [ep["seed"] for ep in archive.metadata["episodes"] if outcome in ("全部", ep["outcomes"][ego])]
    preferred = st.session_state.get("current_seed", 200002)
    default = seeds.index(preferred) if preferred in seeds else 0
    seed = c.selectbox("评估 Seed", seeds, index=default, key=f"seed-{ego}-{outcome}")
    st.session_state["current_seed"] = seed
    x_mode = d.selectbox("车道图横轴", ["累计已观测 XY 位移", "距任务终点剩余距离"])
    x_key = "remaining" if x_mode == "距任务终点剩余距离" else "distance"

    draw_charts(archive, seed, ego, x_key, "single")


def comparison(args, entries):
    left, right = st.columns(2, gap="large")
    checkpoints = {}
    for row, _ in entries.values():
        checkpoints.setdefault(row["experiment"], set()).add(row["step"])
    preferred = max((experiment for experiment, steps in checkpoints.items() if len(steps) > 1), default=None)
    with left:
        st.subheader("左侧评估")
        left_choice = choose(entries, "left", preferred, step_offset=1)
        left_archive = load_choice(args, left_choice, "left")
    with right:
        st.subheader("右侧评估")
        right_choice = choose(entries, "right", left_choice[0]["experiment"])
        right_archive = load_choice(args, right_choice, "right")
    if left_archive is None or right_archive is None:
        return
    shared_seeds = sorted({ep["seed"] for ep in left_archive.metadata["episodes"]}
                          & {ep["seed"] for ep in right_archive.metadata["episodes"]})
    if not shared_seeds:
        st.warning("这两个评估批次没有共同的 seed。")
        return
    seed_column, ego_column, axis_column = st.columns([1, 1, 2])
    previous = st.session_state.get("comparison_seed", 200001)
    index = shared_seeds.index(previous) if previous in shared_seeds else 0
    seed = seed_column.selectbox("共同评估 Seed", shared_seeds, index=index, key="compare-seed")
    st.session_state["comparison_seed"] = seed
    ego = ego_column.selectbox("共同 Ego", EGO_IDS, format_func=lambda e: "Ego 1" if e == "ego_1" else "Ego 2")
    x_mode = axis_column.selectbox("共同车道图横轴", ["累计已观测 XY 位移", "距任务终点剩余距离"])
    x_key = "remaining" if x_mode == "距任务终点剩余距离" else "distance"
    left_charts, right_charts = st.columns(2, gap="large")
    for column, archive, panel in ((left_charts, left_archive, "left"), (right_charts, right_archive, "right")):
        with column:
            describe(archive)
            draw_charts(archive, seed, ego, x_key, panel)


def main():
    st.set_page_config(page_title="DAROM-TPR 轨迹诊断", layout="wide")
    parser = argument_parser(__doc__, allow_eval_dir=True, allow_catalog=True)
    parser.add_argument("--local-archive", type=Path, default=Path.home() / "Experiments/darom")
    parser.add_argument("--bundle-cache", type=Path, default=Path.home() / "Library/Caches/darom-trajectories")
    parser.add_argument("--remote")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--remote-archive", default=catalog.ROOT)
    args = parser.parse_args()
    if not args.trace and args.summary:
        parser.error("--summary requires --trace")
    st.title("DAROM-TPR 轨迹诊断")
    if args.trace:
        page = "单次查看"
    else:
        page = st.radio("查看方式", ["单次查看", "并排对照"], horizontal=True)
    try:
        entries = sources(args)
    except (OSError, ValueError) as error:
        st.error(f"无法读取评估目录：{error}")
        return
    if not entries:
        st.info("暂无本机评估。可通过侧栏刷新远程列表。")
    elif page == "并排对照":
        comparison(args, entries)
    else:
        single_view(args, entries)


if __name__ == "__main__":
    main()
