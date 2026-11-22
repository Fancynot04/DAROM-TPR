"""Read-only DAROM-TPR trajectory dashboard."""

import sqlite3
from pathlib import Path

import streamlit as st

from scripts.trajectory_dashboard.figures import TERMS, lane_figure, reward_figure, telemetry_figure
from scripts.trajectory_dashboard.series import cursor_point, episode_series
from scripts.trajectory_dashboard.trace import EGO_IDS, argument_parser, build_index


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


def main():
    st.set_page_config(page_title="DAROM-TPR 轨迹诊断", layout="wide")
    args = argument_parser(__doc__).parse_args()
    trace = args.trace.resolve()
    summary = Path(args.summary or trace.with_name("summary.json")).resolve()
    st.title("DAROM-TPR 轨迹诊断")
    try:
        with st.spinner("校验轨迹并读取 seed 索引"):
            archive = open_archive(trace, summary, args.cache_dir, (signature(trace), signature(summary)))
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        st.error(f"无法打开轨迹：{error}")
        st.stop()

    meta = archive.metadata
    st.caption(
        f"Checkpoint {meta['checkpoint']:,} 步 · {len(meta['episodes'])} 个评估回合 · "
        f"{meta['step_records']:,} 条步记录 · Δt {meta['step_length_s']:g} s"
    )

    a, b, c, d = st.columns([1, 1.3, 1.4, 1.7])
    ego = a.selectbox("Ego", EGO_IDS, format_func=lambda e: "Ego 1" if e == "ego_1" else "Ego 2")
    outcomes = sorted({ep["outcomes"][ego] for ep in meta["episodes"]})
    outcome = b.selectbox("所选 Ego 的结局", ["全部", *outcomes])
    seeds = [ep["seed"] for ep in meta["episodes"] if outcome in ("全部", ep["outcomes"][ego])]
    preferred = st.session_state.get("current_seed", 200002)
    default = seeds.index(preferred) if preferred in seeds else 0
    seed = c.selectbox("评估 Seed", seeds, index=default, key=f"seed-{ego}-{outcome}")
    st.session_state["current_seed"] = seed
    x_mode = d.selectbox("车道图横轴", ["累计已观测 XY 位移", "距任务终点剩余距离"])
    x_key = "remaining" if x_mode == "距任务终点剩余距离" else "distance"

    ep = next(ep for ep in meta["episodes"] if ep["seed"] == seed)
    st.caption(f"Ego 1 · {ep['outcomes']['ego_1']}　　Ego 2 · {ep['outcomes']['ego_2']}")
    points = load_series(archive, seed, ego)
    cursor, timing = st.columns([4, 1])
    if len(points) > 1:
        step = cursor.slider("共享回合步", 1, len(points), 1, key=f"cursor-{meta['trace_sha256']}-{seed}")
    else:
        step = 1
    at = timing.radio("游标时刻", ["步首", "步末"], horizontal=True)
    selected = cursor_point(points[step - 1], at == "步首")
    show_selected(selected)

    revision = f"{meta['trace_sha256']}-{seed}-{ego}"
    st.subheader("车道—距离轨迹")
    st.caption("累计位移由每步可用的首末 XY 估算。连接区单独标注，末步缺少坐标时只显示实测位置。")
    st.plotly_chart(lane_figure(points, selected, x_key, revision + x_key), width="stretch", key="lane")

    st.subheader("速度与动作时序")
    st.caption("提交方向 −1 朝较小车道编号，+1 朝较大车道编号。稳定完成和中断使用事件记录。")
    st.plotly_chart(telemetry_figure(points, selected, revision), width="stretch", key="telemetry")

    st.subheader("逐步与累计奖励")
    filters, discount = st.columns([3, 1])
    available = [term for term in TERMS if any(term in p["components"] for p in points)]
    enabled = [term for term in available if any(p["components"].get(term, 0) != 0 for p in points)]
    terms = filters.multiselect("单步奖励分项", available, default=enabled, format_func=TERMS.get,
                                key=f"terms-{revision}")
    discounted = discount.checkbox(f"折扣累计 γ={meta['gamma']:g}")
    gamma = meta["gamma"] if discounted else 1.0
    st.plotly_chart(reward_figure(points, selected, terms, gamma, revision), width="stretch", key="reward")


if __name__ == "__main__":
    main()
