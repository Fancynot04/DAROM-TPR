
from pathlib import Path

import traci
from sumolib import checkBinary

from src.experiment_tracking import sha256
from .sumo_detectors import prepare_detector_input


# 项目根目录下的 SUMO 文件
SUMO_DIR = Path(__file__).resolve().parents[2] / "sumo_new"
STEP_LENGTH_S = 0.1
DETECTOR_TEMPLATE = SUMO_DIR / "Antwerp.add.xml"
DUAL_EGO_ROUTES = {
    scenario: SUMO_DIR / f"Antwerp_{scenario}_o3_d3_homogeneous_dual.rou.xml"
    for scenario in ("easy", "easy_speed18")
}


EMPIRICAL_ROUTES = {
    f"empirical_{name}": SUMO_DIR / "empirical_v1" / f"Antwerp_empirical_v1_{name}.rou.xml"
    for name in ("easy", "medium", "hard")
}


def checkpoint_scenario(config):
    if config.get("scenario") is not None:
        return config["scenario"]
    route_hash = config.get("environment", {}).get("sumo", {}).get("route_sha256")
    if route_hash and route_hash == sha256(DUAL_EGO_ROUTES["easy_speed18"]):
        return "easy_speed18"
    return "easy"


class SumoWorld:
    """管理一个 SUMO 仿真世界，不负责具体 Ego 的决策。"""

    def __init__(self, scenario="easy", seed=42, gui=False, route_file=None, warning_log=None,
                 lanechange_output=None, detector_output_dir=None):
        if scenario not in ("easy", "medium", "hard", "easy_speed18") and scenario not in EMPIRICAL_ROUTES:
            raise ValueError(f"Unknown scenario: {scenario}")

        self.scenario = scenario
        self.seed = seed
        self.gui = gui
        self.warning_log = warning_log
        self.lanechange_output = lanechange_output
        self.detector_additional_file = (
            Path(detector_output_dir).resolve() / DETECTOR_TEMPLATE.name if detector_output_dir is not None else None
        )

        self.step_length = STEP_LENGTH_S
        self.label = "o3_d3_world"

        self.net_file = SUMO_DIR / "Antwerp.net.xml"
        if route_file is not None:
            self.route_file = Path(route_file)
        elif scenario in EMPIRICAL_ROUTES:
            self.route_file = EMPIRICAL_ROUTES[scenario]
        elif scenario == "easy_speed18":
            self.route_file = DUAL_EGO_ROUTES[scenario]
        else:
            self.route_file = SUMO_DIR / f"Antwerp_{scenario}_o3_d3.rou.xml"

        self.conn = None

    def launch_command(self):
        """Return the command for this world's SUMO process."""
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
        if self.warning_log is not None:
            command.extend(("--error-log", str(self.warning_log), "--aggregate-warnings", "-1"))
        if self.detector_additional_file is not None:
            command.extend(("--additional-files", str(self.detector_additional_file)))
        if self.lanechange_output is not None:
            command.extend((
                "--lanechange-output", str(self.lanechange_output),
                "--lanechange-output.started", "--lanechange-output.ended",
            ))
        return command

    def start(self):
        """启动一个新的 SUMO 仿真。"""
        if self.conn is not None:
            raise RuntimeError("SUMO is already running")
        if not self.net_file.is_file():
            raise FileNotFoundError(self.net_file)
        if not self.route_file.is_file():
            raise FileNotFoundError(self.route_file)

        if self.detector_additional_file is not None:
            prepare_detector_input(DETECTOR_TEMPLATE, self.detector_additional_file.parent)
        traci.start(self.launch_command(), label=self.label)
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
