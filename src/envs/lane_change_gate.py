"""Check the projected clearance of an adjacent lane before a SUMO request."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class LaneChangeGate:
    horizon_s: float = 4.0
    min_gap_m: float = 2.0

    def __post_init__(self):
        for value in (self.horizon_s, self.min_gap_m):
            if not isfinite(value) or value <= 0:
                raise ValueError("Lane-change gate horizon and minimum gap must be finite and positive")

    def check(self, vehicle, ego_id, direction, ego_speed, target_lane):
        """Return the most restrictive projected gap on each side of the Ego."""
        if direction > 0:
            leaders, followers = vehicle.getLeftLeaders, vehicle.getLeftFollowers
        else:
            leaders, followers = vehicle.getRightLeaders, vehicle.getRightFollowers

        def nearest(neighbors, *, rear):
            weakest = None
            for other, traci_gap in neighbors:
                other_speed = vehicle.getSpeed(other)
                min_gap = vehicle.getMinGap(other if rear else ego_id)
                if (not all(isfinite(v) for v in (other_speed, min_gap, traci_gap))
                        or other_speed < 0 or min_gap < 0):
                    raise ValueError(f"Invalid lane-change gap observation for {ego_id} and {other}")
                gap = traci_gap + min_gap
                closing = other_speed - ego_speed if rear else ego_speed - other_speed
                projected = gap - max(0.0, closing) * self.horizon_s
                if weakest is None or projected < weakest["projected_gap_m"]:
                    weakest = {
                        "vehicle_id": other, "gap_m": gap, "closing_speed_mps": closing,
                        "projected_gap_m": projected,
                    }
            return weakest

        front = nearest(leaders(ego_id), rear=False)
        rear = nearest(followers(ego_id), rear=True)
        blocked_by = [side for side, sample in (("front", front), ("rear", rear))
                      if sample is not None and sample["projected_gap_m"] < self.min_gap_m]
        return {"target_lane": target_lane, "blocked_by": blocked_by, "front": front, "rear": rear}
