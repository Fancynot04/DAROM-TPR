"""Single-ego E76-to-D3 experiment profile."""

import math

TASK_ID = 'exit_e76_d3'
ORIGIN = 'E76'
DESTINATION = 'D3'
DEPART_LANE = 'random'
POTENTIAL_BETA = 10.0
POTENTIAL_GAMMA = 0.99
REWARD_PROFILE = 'no_lane_command_penalty_v1'
CONTROL_PROFILE = 'tracked_lane_change_v2'
DISTANCE_SCALE_EDGES = ('E76', 'E77', 'D3')
# A same-seed TraCI probe needed 3.3 s from first lateral motion until the
# vehicle was fully inside the adjacent lane. Reserve 4 s plus body clearance.
LANE_CHANGE_TIME_SECONDS = 4.0
MANEUVER_MAX_SECONDS = 6.0
MANEUVER_STALL_SECONDS = 2.0
MANEUVER_STABLE_STEPS = 3
LANE_END_BUFFER_METERS = 5.0
MIN_MANEUVER_SPEED_MPS = 1.0
LATERAL_SETTLED_SPEED_MPS = 0.05
# SUMO's sublane solver may settle a fully-contained 1.8 m vehicle up to
# about 0.5 mm beyond the analytic 0.7 m margin of a 3.2 m lane.  Keep this
# tolerance far below the 0.8 m lateral resolution and apply it only after the
# target lane matches and lateral motion has stopped.
LATERAL_CONTAINMENT_EPSILON_METERS = 0.001
LANE_END_EPSILON_METERS = 0.01
MISSED_EXIT_PENALTY = 1000.0  # Before the environment's existing 0.1 scale.

WANDB_ENTITY = 'arlikache-anhui-university'
WANDB_PROJECT = 'DAROM-OUT'

PHASES = {
    'easy_pilot': {'scenario': 'easy', 'total_timesteps': 20_000,
                   'checkpoint_freq': 10_000, 'gamma': POTENTIAL_GAMMA},
    'easy_1m': {'scenario': 'easy', 'total_timesteps': 1_000_000,
                'checkpoint_freq': 250_000, 'gamma': POTENTIAL_GAMMA},
    'exp005_gamma0999_200k': {
        'scenario': 'easy', 'total_timesteps': 200_000,
        'checkpoint_freq': 100_000, 'gamma': 0.999,
    },
    'gamma099_baseline_800k': {
        'scenario': 'easy', 'total_timesteps': 800_000,
        'checkpoint_freq': 200_000, 'gamma': POTENTIAL_GAMMA,
    },
}


def potential(distance_to_exit, distance_scale, lane_index):
    """Potential in final reward units, with lane zero connected to D3."""
    if (not math.isfinite(distance_scale) or not math.isfinite(distance_to_exit)
            or distance_scale <= 0 or distance_to_exit < 0):
        return 0.0
    distance_weight = max(0.0, min(1.0, 1.0 - distance_to_exit / distance_scale))
    lane_weight = 1.0 - min(max(int(lane_index), 0), 5) / 5.0
    return POTENTIAL_BETA * distance_weight * lane_weight
