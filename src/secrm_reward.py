"""SECRM-2D Eq. (2), (5), 4.4.1 and 4.4.4, reward-only adaptation.

The target computed from pre-action s_t is the next-step safe-speed bound;
its deviation from v(s_t) is paid on that transition. No action filter is added.
https://doi.org/10.1049/itr2.70013
"""
import math

EFFICIENCY_PROFILE = 'secrm_efficiency_events_v1'
ROUTE_PROFILE = 'secrm_efficiency_route_events_v1'
PROFILES = (EFFICIENCY_PROFILE, ROUTE_PROFILE)
FORMULA_VERSION = 'secrm_eq2_prestate_v1'


def efficiency(speed, limit, reaction, gap=None, leader_speed=None,
               ego_decel=None, leader_decel=None, stopped_gap=None, epsilon=0.1):
    values = [speed, limit, reaction, epsilon]
    if any(not math.isfinite(x) for x in values) or min(values) < 0 or min(limit, reaction, epsilon) <= 0:
        raise ValueError('Invalid SECRM speed/limit/reaction/epsilon')
    safe = None
    effective_decel = None
    if gap is not None:
        params = [gap, leader_speed, ego_decel, leader_decel, stopped_gap]
        if any(x is None or not math.isfinite(x) or x < 0 for x in params) or min(ego_decel, leader_decel) <= 0:
            raise ValueError('Invalid SECRM leader data')
        # Defensive assumption d_E <= d_L in the paper, without changing SUMO.
        effective_decel = min(ego_decel, leader_decel)
        d = effective_decel
        radicand = (reaction*d/2)**2 - 2*d*(reaction*speed/2 - leader_speed**2/(2*leader_decel) - gap + stopped_gap)
        if radicand < 0 or not math.isfinite(radicand):
            raise ValueError(f'Invalid SECRM safe-speed radicand: {radicand}; gap={gap}, speed={speed}')
        safe = max(0.0, -reaction*d/2 + math.sqrt(radicand))
    target = limit if safe is None else min(safe, limit)
    value = -abs(target-speed)/max(target, epsilon)
    return value, {'formula_version': FORMULA_VERSION, 'speed': speed, 'limit': limit,
                   'safe_speed': safe, 'target_speed': target, 'gap': gap,
                   'leader_speed': leader_speed, 'ego_decel': ego_decel,
                   'leader_decel': leader_decel, 'effective_decel': effective_decel,
                   'stopped_gap': stopped_gap, 'reaction_seconds': reaction}


def route_penalty(required, remaining):
    if (type(required) is not int or required < 0 or not math.isfinite(remaining)
            or remaining < 0):
        raise ValueError('Invalid settled route requirement/distance')
    return -required/(1.0+remaining)


def minimum_changes(network, route, index, lane, change_edges):
    """Topology only: no gap/urgency/time feasibility assumptions."""
    if index < 0 or index >= len(route) or lane not in network.lanes[route[index]]:
        raise ValueError('Invalid stable lane/route index')
    costs={lane:0}
    for edge,next_edge in zip(route[index:],route[index+1:]):
        following={}
        for start,cost in costs.items():
            exits=network.lanes[edge] if edge in change_edges else (start,)
            for end in exits:
                if any(i not in network.lanes[edge] for i in range(min(start,end),max(start,end)+1)):continue
                for target,_ in network.connections.get((edge,next_edge,end),()):
                    value=cost+abs(end-start)
                    following[target]=min(value,following.get(target,value))
        costs=following
    if not costs:raise ValueError('No continuation to route endpoint')
    return min(costs.values())


def stamp_replay(buffer, experiment):
    from .exit_progress_reward import experiment_identity
    buffer.secrm_experiment_identity = experiment_identity(experiment)
    buffer.secrm_formula_version = FORMULA_VERSION


def validate_replay(buffer, experiment):
    from .exit_progress_reward import experiment_identity
    if (getattr(buffer, 'secrm_experiment_identity', None) != experiment_identity(experiment)
            or getattr(buffer, 'secrm_formula_version', None) != FORMULA_VERSION):
        raise ValueError('Replay buffer SECRM configuration/formula identity mismatch')
