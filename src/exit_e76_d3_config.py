"""Single-ego E76-to-D3 experiment profile."""

TASK_ID = 'exit_e76_d3'
ORIGIN = 'E76'
DESTINATION = 'D3'
DEPART_LANE = 'random'
POTENTIAL_BETA = 10.0
POTENTIAL_GAMMA = 0.99
REWARD_PROFILE = 'no_lane_command_penalty_v1'
DISTANCE_SCALE_EDGES = ('E76', 'E77', 'D3')

WANDB_ENTITY = 'arlikache-anhui-university'
WANDB_PROJECT = 'DAROM-OUT'

PHASES = {
    'easy_pilot': {'scenario': 'easy', 'total_timesteps': 20_000,
                   'checkpoint_freq': 10_000},
    'easy_1m': {'scenario': 'easy', 'total_timesteps': 1_000_000,
                'checkpoint_freq': 250_000},
}


def potential(distance_to_exit, distance_scale, lane_index):
    """Potential in final reward units, with lane zero connected to D3."""
    if distance_scale <= 0 or distance_to_exit < 0:
        return 0.0
    distance_weight = max(0.0, min(1.0, 1.0 - distance_to_exit / distance_scale))
    lane_weight = 1.0 - min(max(int(lane_index), 0), 5) / 5.0
    return POTENTIAL_BETA * distance_weight * lane_weight
