
from pathlib import Path

import traci
from sumolib import checkBinary


# 项目根目录下的 SUMO 文件
SUMO_DIR = Path(__file__).resolve().parents[2] / "sumo_new"


class SumoWorld:
    """管理一个 SUMO 仿真世界，不负责具体 Ego 的决策。"""

    def __init__(self, scenario="easy", seed=42, gui=False, route_file=None):
        if scenario not in ("easy", "medium", "hard"):
            raise ValueError(f"Unknown scenario: {scenario}")

        self.scenario = scenario
        self.seed = seed
        self.gui = gui

        self.step_length = 0.1
        self.label = "o3_d3_world"

        self.net_file = SUMO_DIR / "Antwerp.net.xml"
        self.route_file = (
            Path(route_file)
            if route_file is not None
            else SUMO_DIR / f"Antwerp_{scenario}_o3_d3.rou.xml"
        )

        self.conn = None

    def start(self):
        """启动一个新的 SUMO 仿真。"""

        if self.conn is not None:
            raise RuntimeError("SUMO is already running")

        if not self.net_file.is_file():
            raise FileNotFoundError(self.net_file)

        if not self.route_file.is_file():
            raise FileNotFoundError(self.route_file)

        binary = checkBinary("sumo-gui" if self.gui else "sumo")

        command = [
            binary,
            "--net-file", str(self.net_file),
            "--route-files", str(self.route_file),
            "--begin", "0",
            "--step-length", str(self.step_length),
            "--lateral-resolution", "0.8",
            "--seed", str(self.seed),
            "--no-step-log",
            "--collision.check-junctions",
            "--collision.action", "remove",
            "--time-to-teleport", "-1",
            "--time-to-teleport.highways", "-1",
            "--time-to-teleport.disconnected", "-1",
            "--time-to-teleport.bidi", "-1",
            "--time-to-teleport.ride", "-1",
            "-Q",
        ]

        # GUI 启动后自动运行，无需手动点击播放按钮
        if self.gui:
            command.append("--start")

        traci.start(command, label=self.label)
        self.conn = traci.getConnection(self.label)

        return self.conn

    def step(self):
        """统一推进 SUMO 一个仿真步。"""

        if self.conn is None:
            raise RuntimeError("SUMO has not started")

        self.conn.simulationStep()

        return self.conn.simulation.getTime()

    def close(self):
        """关闭 SUMO 连接。"""

        if self.conn is not None:
            conn = self.conn
            self.conn = None
            conn.close()
