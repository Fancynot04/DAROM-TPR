"""SUMO GUI 双 Ego 自动定位。"""


EGO_IDS = ("ego_1", "ego_2")


def update_gui_camera(
    conn,
    focus="all",
    margin=60.0,
):
    """
    focus:
        all   - 两辆 Ego 同屏
        ego_1 - 跟踪 ego_1
        ego_2 - 跟踪 ego_2
    """

    views = conn.gui.getIDList()

    if not views:
        return

    view_id = views[0]

    # SUMO 中当前实际存在的车辆
    present = set(conn.vehicle.getIDList())

    active_egos = [
        ego_id
        for ego_id in EGO_IDS
        if ego_id in present
    ]

    # 尚无 Ego 出发
    if not active_egos:
        return

    # ---------- 单车跟踪 ----------
    if focus in active_egos:
        tracked = conn.gui.getTrackedVehicle(
            view_id
        )

        if tracked != focus:
            conn.gui.trackVehicle(
                view_id,
                focus,
            )

        return

    # ---------- 双车同屏 ----------
    # 同时用于 focus=all，或指定 Ego 尚未出发时
    tracked = conn.gui.getTrackedVehicle(view_id)

    if tracked:
        conn.gui.trackVehicle(view_id, "")

    positions = [
        conn.vehicle.getPosition(ego_id)
        for ego_id in active_egos
    ]

    xs = [position[0] for position in positions]
    ys = [position[1] for position in positions]

    xmin = min(xs) - margin
    xmax = max(xs) + margin
    ymin = min(ys) - margin
    ymax = max(ys) + margin

    conn.gui.setBoundary(
        view_id,
        xmin,
        ymin,
        xmax,
        ymax,
    )