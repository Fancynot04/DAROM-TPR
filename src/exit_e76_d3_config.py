"""Single-ego E76-to-D3 experiment profile."""

TASK_ID = 'exit_e76_d3'
ORIGIN = 'E76'
DESTINATION = 'D3'
DEPART_LANE = 'random'
POTENTIAL_BETA = 10.0
POTENTIAL_GAMMA = 0.99
REWARD_PROFILE = 'no_lane_command_penalty_v1'
ABLATION_PROFILES = {
    'A0': {'lane_command': -0.5, 'progress_per_meter': 0.15},
    'A1': {'lane_command': 0.0, 'progress_per_meter': 0.15},
    'A2': {'lane_command': -0.5, 'progress_per_meter': 0.0375},
    'A3': {'lane_command': 0.0, 'progress_per_meter': 0.0375},
}
COMPONENT_PROFILES = {
    'B1': {'lane_command': 0.0, 'progress_per_meter': 0.15, 'removed': 'behind_gap'},
    'B2': {'lane_command': 0.0, 'progress_per_meter': 0.15, 'removed': 'action_change'},
    'B3': {'lane_command': 0.0, 'progress_per_meter': 0.15, 'removed': 'stage'},
}
ALL_REWARD_PROFILES = {**ABLATION_PROFILES, **COMPONENT_PROFILES}
ABLATION_PHASE = 'basic_reward_200k'
DISTANCE_SCALE_EDGES = ('E76', 'E77', 'D3')

WANDB_ENTITY = 'arlikache-anhui-university'
WANDB_PROJECT = 'DAROM-OUT'

PHASES = {
    'easy_pilot': {'scenario': 'easy', 'total_timesteps': 20_000,
                   'checkpoint_freq': 10_000},
    'easy_1m': {'scenario': 'easy', 'total_timesteps': 1_000_000,
                'checkpoint_freq': 250_000},
    ABLATION_PHASE: {'scenario': 'easy', 'total_timesteps': 200_000,
                     'checkpoint_freq': 50_000},
}


def potential(distance_to_exit, distance_scale, lane_index):
    """Potential in final reward units, with lane zero connected to D3."""
    if distance_scale <= 0 or distance_to_exit < 0:
        return 0.0
    distance_weight = max(0.0, min(1.0, 1.0 - distance_to_exit / distance_scale))
    lane_weight = 1.0 - min(max(int(lane_index), 0), 5) / 5.0
    return POTENTIAL_BETA * distance_weight * lane_weight
