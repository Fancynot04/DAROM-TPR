
"""根据现有 Easy 场景生成双 Ego 路线文件。"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMO_DIR = ROOT / "sumo_new"

SOURCE = SUMO_DIR / "Antwerp_easy_o3_d3.rou.xml"
TARGET = SUMO_DIR / "Antwerp_easy_o3_d3_dual.rou.xml"


def main():
    text = SOURCE.read_text(encoding="utf-8")

    # 找到原来的单 Ego trip
    ego_lines = [
        line for line in text.splitlines()
        if '<trip id="ego"' in line
    ]

    assert len(ego_lines) == 1
    original = ego_lines[0]

    # Ego A：入口 O3 -> D3
    ego_a = original.replace(
        'id="ego"',
        'id="ego_o3"',
        1,
    )

    # Ego B：主线 E74 -> D3
    ego_b = original.replace(
        'id="ego"',
        'id="ego_e74"',
        1,
    ).replace(
        'from="O3"',
        'from="E74"',
        1,
    ).replace(
        'color="red"',
        'color="blue"',
        1,
    )

    updated = text.replace(
        original,
        ego_a + "\n" + ego_b,
        1,
    )

    # 避免覆盖已有的其他版本
    if TARGET.exists():
        assert TARGET.read_text(encoding="utf-8") == updated, (
            f"Existing route differs: {TARGET}"
        )
    else:
        TARGET.write_text(updated, encoding="utf-8")

    print("Dual Ego route created:", TARGET.name)
    print(ego_a.strip())
    print(ego_b.strip())


if __name__ == "__main__":
    main()
