
"""验证 SUMO 原生驾驶能否完成 O3 -> D3 路线。"""

from src.envs.sumo_world import SumoWorld


EGO_ID = "ego"
EXPECTED_ROUTE = ("O3", "E75", "E76", "E77", "D3")


def main():
    world = SumoWorld(scenario="easy", seed=42)

    departed = False
    departure_time = None
    visited_edges = []
    last_road = None

    try:
        conn = world.start()

        print("SUMO version:", conn.getVersion())

        # 最多模拟 600 秒
        for _ in range(6000):
            current_time = world.step()

            simulation = conn.simulation
            vehicle = conn.vehicle

            # 读取当前仿真步的事件
            departures = simulation.getDepartedIDList()
            arrivals = simulation.getArrivedIDList()
            collisions = simulation.getCollidingVehiclesIDList()
            teleports = simulation.getStartingTeleportIDList()

            if EGO_ID in departures:
                departed = True
                departure_time = current_time
                print(f"Ego departed at {current_time:.1f}s")

            # 首先检查失败事件
            if EGO_ID in collisions:
                raise RuntimeError(
                    f"Ego collision at {current_time:.1f}s"
                )

            if EGO_ID in teleports:
                raise RuntimeError(
                    f"Ego teleport at {current_time:.1f}s"
                )

            # 记录车辆实际经过的道路
            if EGO_ID in vehicle.getIDList():
                road = vehicle.getRoadID(EGO_ID)

                if road != last_road:
                    print(
                        f"{current_time:.1f}s: "
                        f"{last_road} -> {road}"
                    )
                    last_road = road

                    if road in EXPECTED_ROUTE:
                        visited_edges.append(road)

            # 检查正常到达
            if EGO_ID in arrivals:
                print("\nEgo arrived!")
                print("Departure:", departure_time)
                print("Arrival:", current_time)
                print("Travel time:", round(
                    current_time - departure_time, 2
                ))
                print("Visited edges:", visited_edges)

                assert visited_edges == list(EXPECTED_ROUTE), (
                    "Ego did not follow the expected route"
                )

                print("\nPASS: O3 -> D3 route completed")
                return

            # 出发后异常消失
            if departed and EGO_ID not in vehicle.getIDList():
                raise RuntimeError(
                    f"Ego disappeared at {current_time:.1f}s"
                )

        raise RuntimeError("Ego did not arrive within 600s")

    finally:
        world.close()


if __name__ == "__main__":
    main()
