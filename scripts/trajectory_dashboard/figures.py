"""Plotly views linked to the same selected episode step."""

from html import escape

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.trajectory_dashboard.series import cumulative_rewards, road_segments


ROAD_COLORS = {"O3": "#6B7280", "E75": "#B4691C", "E76": "#2563AE",
               "E77": "#16806D", "D3": "#598137"}
TERMS = {
    "merge": "汇入", "efficiency": "速度", "following": "前车风险", "rear_following": "后车风险",
    "lane_change": "换道风险", "comfort": "舒适度", "route_preparation": "路线准备",
    "route_guidance": "一次性路线引导",
}
COLORS = ["#9C6D28", "#227958", "#6152AD", "#7855A0", "#B34034", "#C0641C", "#12638D", "#6F7C89"]
CURSOR = "#D1741F"
TERMINAL = "#A8273D"


def _lane_change_hover(point):
    components = point["components"]
    labels = {**TERMS, "task": "任务奖励"}
    ordered = dict.fromkeys([*labels, *components])
    rewards = [(term, components[term]) for term in ordered if components.get(term) is not None]
    lines = [f"{escape(point['road'])} · 车道 {point['pre_lane']} → {point['lane']}",
             f"第 {point['step']} 步 · {point['position_time']:.2f} s"]
    if not rewards:
        return "<br>".join([*lines, "当步无个人奖励分项记录"])

    highest, _ = max(rewards, key=lambda reward: reward[1])
    for term, value in rewards:
        label = escape(labels.get(term, term))
        entry = f"{label} {value:+.6g}" if value else f"{label} 0"
        if term == highest:
            entry = f'<span style="color:#116B45"><b>{entry}（最高）</b></span>'
        lines.append(entry)
    return "<br>".join(lines)


def _layout(fig, height, revision):
    fig.update_layout(
        height=height, template="plotly_white", uirevision=revision, hovermode="closest",
        margin=dict(l=16, r=16, t=76, b=30),
        legend=dict(orientation="h", y=1.16, x=0),
        font=dict(color="#203349"),
    )
    return fig


def lane_figure(points, selected, x_key, revision):
    fig = go.Figure()
    shown = set()
    annotations = set()
    for group in road_segments(points, x_key):
        road = group[0]["road"]
        x = [p[x_key] for p in group]
        if road.startswith(":"):
            name = road.split("_", 1)[0].lstrip(":")
            shape = dict(fillcolor="#A9B4C5", opacity=0.24, line_width=0, layer="below")
            if min(x) == max(x):
                fig.add_vline(x=x[0], line_width=1, line_dash="dot", line_color="#7C899B")
            else:
                fig.add_vrect(x0=min(x), x1=max(x), **shape)
            if name not in annotations:
                fig.add_annotation(x=x[0], y=1, yref="paper", text=name, showarrow=False,
                                   yshift=10, font=dict(size=10, color="#495B70"))
                annotations.add(name)
            continue
        fig.add_trace(go.Scatter(
            x=x, y=[p["lane"] for p in group], name=road, legendgroup=road,
            showlegend=road not in shown, mode="lines+markers",
            line=dict(color=ROAD_COLORS.get(road, "#704C98"), width=2.2), marker=dict(size=3),
            customdata=[[p["step"], p["position_time"], p["position_source"]] for p in group],
            hovertemplate=road + " · 车道 %{y}<br>距离 %{x:.2f} m<br>第 %{customdata[0]} 步 · "
                          "%{customdata[1]:.1f} s · %{customdata[2]}<extra></extra>",
        ))
        shown.add(road)
    x = selected[x_key]
    if x is not None:
        fig.add_vline(x=x, line_color=CURSOR, line_dash="dash")
        if selected["road"] and not selected["road"].startswith(":") and selected["lane"] is not None:
            fig.add_trace(go.Scatter(x=[x], y=[selected["lane"]], name="当前步", mode="markers",
                                     marker=dict(color=CURSOR, size=12, symbol="circle-open")))
    changes = [p for p in points if p[x_key] is not None and p["position_source"] == "步末"
               and p["position_time"] is not None and p["road"] == p["pre_road"]
               and p["road"] and not p["road"].startswith(":")
               and p["pre_lane"] is not None and p["lane"] is not None and p["pre_lane"] != p["lane"]]
    if changes:
        # Plotly favors smaller markers for close hover ties with the dense road samples.
        fig.add_trace(go.Scatter(
            x=[p[x_key] for p in changes], y=[p["lane"] for p in changes],
            name="变道节点", mode="markers",
            marker=dict(color="#203349", size=6, symbol="diamond", line=dict(color="#203349", width=1)),
            text=[_lane_change_hover(p) for p in changes], hovertemplate="%{text}<extra></extra>",
            hoverlabel=dict(bgcolor="white", bordercolor="#203349", font=dict(color="#203349")),
        ))
    fig.update_xaxes(title_text="距任务终点剩余距离 (m)" if x_key == "remaining" else "累计已观测 XY 位移 (m)",
                     autorange="reversed" if x_key == "remaining" else True)
    fig.update_yaxes(title_text="SUMO 车道编号", dtick=1)
    return _layout(fig, 365, revision)


def _line(fig, points, row, time_key, key, name, color, shape="linear", dash=None, visible=None):
    fig.add_trace(go.Scatter(
        x=[p[time_key] for p in points], y=[p[key] for p in points], name=name,
        mode="lines", visible=visible, line=dict(color=color, width=1.6, shape=shape, dash=dash),
        customdata=[p["step"] for p in points],
        hovertemplate="%{x:.2f} s · 第 %{customdata} 步<br>%{y:.4f}<extra>%{fullData.name}</extra>",
    ), row=row, col=1)


def _add_cursor(fig, selected, points):
    timestamp = selected["cursor_time"]
    if timestamp is not None:
        fig.add_vline(x=timestamp, line_color=CURSOR, line_dash="dash", row="all", col=1)
    terminal = next((p for p in points if p["terminal"] and p["terminal_time"] is not None), None)
    if terminal is not None:
        fig.add_vline(x=terminal["terminal_time"], line_color=TERMINAL, line_dash="dot", row="all", col=1)


def telemetry_figure(points, selected, revision):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, specs=[[{}], [{}], [{"secondary_y": True}]],
                        vertical_spacing=0.12, subplot_titles=("速度", "加速度", "横向动作与实际换道"))
    _line(fig, points, 1, "pre_time", "pre_speed", "步首速度", "#12638D")
    _line(fig, points, 1, "post_time", "post_speed", "步末速度", "#16806D", dash="dot")
    for key, time_key, name, color in (
        ("requested_accel", "action_time", "Actor 请求加速度", "#6F7C89"),
        ("commanded_accel", "action_time", "提交加速度", "#2563AE"),
        ("actual_accel", "post_time", "实测加速度", "#C0641C"),
    ):
        _line(fig, points, 2, time_key, key, name, color)
    _line(fig, points, 3, "action_time", "raw_lateral", "Actor 横向输出", "#654D8E")
    fig.add_trace(go.Scatter(
        x=[p["action_time"] for p in points], y=[p["submitted_direction"] for p in points],
        name="实际提交方向", mode="lines", line=dict(color="#2563AE", shape="hv"),
    ), row=3, col=1, secondary_y=True)
    for event_name, label, symbol, color in (
        ("started", "横移开始", "triangle-up", "#197FA0"),
        ("completed", "稳定换道完成", "diamond", "#16806D"),
        ("interrupted", "换道中断", "x", TERMINAL),
    ):
        events = [(p["step"], e) for p in points for e in p["events"]
                  if e["event"] == event_name and e.get("time") is not None]
        if events:
            fig.add_trace(go.Scatter(
                x=[e["time"] for _, e in events], y=[e.get("direction") for _, e in events],
                customdata=[[s, e.get("road"), e.get("source_lane"), e.get("target_lane"), e.get("reason")]
                            for s, e in events],
                hovertemplate="%{x:.2f} s · 第 %{customdata[0]} 步<br>%{customdata[1]} "
                              "%{customdata[2]}→%{customdata[3]}<br>%{customdata[4]}<extra>%{fullData.name}</extra>",
                name=label, mode="markers", marker=dict(color=color, size=9, symbol=symbol),
            ), row=3, col=1, secondary_y=True)
    _add_cursor(fig, selected, points)
    fig.update_yaxes(title_text="m/s", row=1, col=1)
    fig.update_yaxes(title_text="m/s²", row=2, col=1)
    fig.update_yaxes(title_text="Actor 输出", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="方向", tickvals=[-1, 0, 1], range=[-1.25, 1.25],
                     row=3, col=1, secondary_y=True)
    fig.update_xaxes(title_text="SUMO 仿真时间 (s)", row=3, col=1)
    return _layout(fig, 630, revision)


def reward_figure(points, selected, terms, gamma, revision):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.43, .34, .23],
                        vertical_spacing=.13, subplot_titles=("单步奖励分项", "个人累计回报", "双车团队奖励"))
    cum = cumulative_rewards(points, gamma)
    paid = [p for p, _ in cum]
    for term in terms:
        values = [{**p, term: p["components"].get(term)} for p in paid]
        _line(fig, values, 1, "reward_time", term, TERMS[term], COLORS[list(TERMS).index(term)])
    name = "个人累计回报" if gamma == 1 else f"个人折扣累计回报 (γ={gamma:g})"
    fig.add_trace(go.Scatter(x=[p["reward_time"] for p, _ in cum], y=[v for _, v in cum],
                             name=name, meta="personal-return", visible="legendonly",
                             mode="lines", line=dict(color="#203349", width=2)), row=2, col=1)
    endings = [(p, v) for p, v in cum if p["terminal"]]
    if endings:
        fig.add_trace(go.Scatter(
            x=[p["terminal_time"] for p, _ in endings], y=[v for _, v in endings],
            customdata=[[p["terminal"], p["components"].get("task"), p["step"]] for p, _ in endings],
            hovertemplate="%{customdata[0]} · 第 %{customdata[2]} 步<br>任务项 %{customdata[1]}"
                          "<br>累计 %{y:.3f}<extra>终止奖励</extra>",
            name="终止奖励", visible="legendonly",
            mode="markers", marker=dict(size=11, color=TERMINAL, symbol="diamond"),
        ), row=2, col=1)
    _line(fig, points, 3, "clock", "team_reward", "团队单步奖励（未折扣）", "#16806D",
          shape="hv", visible="legendonly")
    paid_team = [p for p in points if p["team_reward"]]
    if paid_team:
        fig.add_trace(go.Scatter(x=[p["clock"] for p in paid_team], y=[p["team_reward"] for p in paid_team],
                                 mode="markers", name="团队实发", visible="legendonly",
                                 marker=dict(color="#16806D", size=9)),
                      row=3, col=1)
    _add_cursor(fig, selected, points)
    fig.update_xaxes(title_text="SUMO 仿真时间 (s)", row=3, col=1)
    _layout(fig, 620, revision)
    # Anchor the wrapped legend above the first subplot and let Plotly reserve its margin.
    fig.update_layout(legend=dict(y=1.115, yanchor="bottom", maxheight=115))
    return fig
