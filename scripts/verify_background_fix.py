
"""验证 Medium / Hard 背景车流 A1 修复。"""

from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.audit_background_fix import (
    OUTPUT,
    run_case,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
SCENARIOS = ("medium", "hard")


def check_xml(before, after):
    """确保只修改了 18 条背景货车的 departSpeed。"""

    old = ET.parse(before).getroot()
    new = ET.parse(after).getroot()

    old_flows = {
        f.get("id"): f
        for f in old.findall("flow")
    }

    new_flows = {
        f.get("id"): f
        for f in new.findall("flow")
    }

    assert old_flows.keys() == new_flows.keys()

    changed = 0

    for flow_id, original in old_flows.items():
        updated = new_flows[flow_id]

        old_attrs = original.attrib.copy()
        new_attrs = updated.attrib.copy()

        if original.get("type") in (
            "rigid_lorry",
            "long_vehicle",
        ):
            assert old_attrs["departSpeed"] == "speedLimit"
            assert new_attrs["departSpeed"] == "max"

            old_attrs["departSpeed"] = "max"
            changed += 1

        assert old_attrs == new_attrs, flow_id

    assert changed == 18

    # 所有 Ego trip 保持一致
    old_trips = [
        t.attrib for t in old.findall("trip")
    ]
    new_trips = [
        t.attrib for t in new.findall("trip")
    ]

    assert old_trips == new_trips

    print(f"XML check PASS: {changed} changes")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for scenario in SCENARIOS:
        before = Path("/tmp") / (
            f"Antwerp_{scenario}_o3_d3."
            "rou_before_a1.rou.xml"
        )

        after = ROOT / "sumo_new" / (
            f"Antwerp_{scenario}_o3_d3.rou.xml"
        )

        assert before.is_file(), before
        assert after.is_file(), after

        print(f"\n========== {scenario.upper()} ==========")

        check_xml(before, after)

        results = {}

        for label, route in (
            ("A0", before),
            ("A1", after),
        ):
            rows = []

            for seed in SEEDS:
                result = run_case(
                    f"{scenario}_{label}",
                    route,
                    seed,
                )

                rows.append(result)

                print(
                    f"{label} seed={seed} "
                    f"braking={result['emergency']} "
                    f"lane_errors={result['lane_errors']} "
                    f"departed={result['departed']} "
                    f"delay_p95={result['delay_p95']:.2f}s"
                )

            results[label] = rows

        print(f"\n{scenario.upper()} SUMMARY")

        for field in (
            "emergency",
            "lane_errors",
            "departed",
            "delay_mean",
            "delay_p95",
            "collision_messages",
        ):
            before_avg = sum(
                r[field] for r in results["A0"]
            ) / len(SEEDS)

            after_avg = sum(
                r[field] for r in results["A1"]
            ) / len(SEEDS)

            print(
                f"{field:20s} "
                f"A0={before_avg:8.2f} "
                f"A1={after_avg:8.2f} "
                f"delta={after_avg-before_avg:+8.2f}"
            )


if __name__ == "__main__":
    main()
