
"""对比背景车流的出发速度和出发车道配置。"""

from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SUMO_DIR = ROOT / "sumo_new"

SOURCE = SUMO_DIR / "Antwerp_easy_o3_d3.rou.xml"
NETWORK = SUMO_DIR / "Antwerp.net.xml"

OUTPUT = (
    Path(tempfile.gettempdir())
    / "darom_o3_d3_traffic_audit"
)

SEEDS = (42, 43, 44)

VARIANTS = ("A0", "A1", "A2", "A3")
HEAVY_TYPES = {"rigid_lorry", "long_vehicle"}


def make_variant(name):
    """生成临时路线文件，不覆盖原始文件。"""

    tree = ET.parse(SOURCE)
    root = tree.getroot()

    changed_speed = 0
    changed_lane = 0

    for flow in root.findall("flow"):
        if flow.get("type") not in HEAVY_TYPES:
            continue

        # A1、A3：货车出发速度
        if name in ("A1", "A3"):
            flow.set("departSpeed", "max")
            changed_speed += 1

        # A2、A3：O1 -> D2/D3 的货车
        if name in ("A2", "A3"):
            if (
                flow.get("from") == "O1"
                and flow.get("to") in ("D2", "D3")
            ):
                flow.set("departLane", "best")
                changed_lane += 1

    path = OUTPUT / f"{name}.rou.xml"

    tree.write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )

    print(
        f"{name}: speed changes={changed_speed}, "
        f"lane changes={changed_lane}"
    )

    return path


def percentile(values, percent):
    if not values:
        return 0.0

    values = sorted(values)
    index = int((len(values) - 1) * percent / 100)

    return values[index]


def inspect_tripinfo(path):
    """统计实际出发车辆和插入延迟。"""

    root = ET.parse(path).getroot()

    departed = 0
    arrived = 0
    undeparted = 0
    delays = []

    # 排除受控任务中的 Ego
    for trip in root.findall("tripinfo"):
        if trip.get("id") == "ego":
            continue

        depart = float(trip.get("depart", "-1"))
        arrival = float(trip.get("arrival", "-1"))

        if depart < 0:
            undeparted += 1
            continue

        departed += 1

        if arrival >= 0:
            arrived += 1

        delay = float(trip.get("departDelay", "-1"))

        if delay >= 0:
            delays.append(delay)

    return {
        "departed": departed,
        "arrived": arrived,
        "undeparted": undeparted,
        "delay_mean": (
            sum(delays) / len(delays) if delays else 0.0
        ),
        "delay_p95": percentile(delays, 95),
    }


def run_case(name, route_path, seed):
    """同一 SUMO 配置运行至 500 秒。"""

    prefix = OUTPUT / f"{name}_seed{seed}"

    error_file = Path(f"{prefix}.errors.log")
    trip_file = Path(f"{prefix}.tripinfo.xml")
    summary_file = Path(f"{prefix}.summary.xml")

    cmd = [
        "sumo",
        "--net-file", str(NETWORK),
        "--route-files", str(route_path),
        "--begin", "0",
        "--end", "500",
        "--step-length", "0.1",
        "--lateral-resolution", "0.8",
        "--seed", str(seed),
        "--no-step-log",
        "--collision.check-junctions",
        "--collision.action", "remove",
        "--error-log", str(error_file),
        "--tripinfo-output", str(trip_file),
        "--tripinfo-output.write-unfinished", "true",
        "--tripinfo-output.write-undeparted", "true",
        "--summary-output", str(summary_file),
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"{name} seed={seed} SUMO failed:\n"
            f"{proc.stderr[-3000:]}"
        )

    messages = (
        error_file.read_text(errors="replace")
        if error_file.exists()
        else proc.stderr
    )

    emergency = sum(
        "performs emergency braking" in line
        for line in messages.splitlines()
    )

    lane_errors = sum(
        "too many lane changes required" in line
        for line in messages.splitlines()
    )

    collision_messages = sum(
        "collision" in line.lower()
        for line in messages.splitlines()
    )

    metrics = inspect_tripinfo(trip_file)

    return {
        "variant": name,
        "seed": seed,
        "emergency": emergency,
        "lane_errors": lane_errors,
        "collision_messages": collision_messages,
        **metrics,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("SUMO version:")
    subprocess.run(["sumo", "--version"], check=True)

    print("\nOutput directory:", OUTPUT)

    results = []

    for name in VARIANTS:
        route_path = make_variant(name)

        for seed in SEEDS:
            result = run_case(name, route_path, seed)
            results.append(result)

            print(
                f"{name} seed={seed} "
                f"braking={result['emergency']} "
                f"lane_errors={result['lane_errors']} "
                f"departed={result['departed']} "
                f"undeparted={result['undeparted']} "
                f"delay_p95={result['delay_p95']:.2f}s"
            )

    print("\n===== Mean results =====")

    fields = (
        "emergency",
        "lane_errors",
        "departed",
        "undeparted",
        "collision_messages",
        "delay_mean",
        "delay_p95",
    )

    for name in VARIANTS:
        subset = [
            r for r in results
            if r["variant"] == name
        ]

        print(f"\n{name}")

        for field in fields:
            mean = sum(
                r[field] for r in subset
            ) / len(subset)

            print(f"  {field}: {mean:.2f}")


if __name__ == "__main__":
    main()
