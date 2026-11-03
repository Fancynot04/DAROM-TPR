"""SECRM-2D Eq. (2), (5), 4.4.1 and 4.4.4, reward-only adaptation.

The target computed from pre-action s_t is the next-step safe-speed bound;
its deviation from v(s_t) is paid on that transition. No action filter is added.
https://doi.org/10.1049/itr2.70013
"""
import math

EFFICIENCY_PROFILE = 'secrm_efficiency_events_v1'
ROUTE_PROFILE = 'secrm_efficiency_route_events_v1'
PROFILES = (EFFICIENCY_PROFILE, ROUTE_PROFILE)
FORMULA_VERSION = 'secrm_eq2_prestate_geometry_v2'


def lateral_relation(ego_position, ego_angle, ego_length, ego_width, ego_lateral_speed,
                     other_position, other_angle, other_length, other_width,
                     other_lateral_speed, reaction):
    """Classify projected body overlap in the ego driving frame for one step.

    This is a SUMO sublane adaptation, not part of SECRM Eq. (2). Position and
    width uncertainty is handled conservatively by a 0.1 m clearance margin.
    """
    values = (*ego_position, ego_angle, ego_length, ego_width, ego_lateral_speed,
              *other_position, other_angle, other_length, other_width,
              other_lateral_speed, reaction)
    if (any(not math.isfinite(float(x)) for x in values)
            or min(ego_length, other_length, ego_width, other_width, reaction) <= 0):
        raise ValueError(f'Invalid SECRM lateral geometry: {values!r}')
    theta = math.radians(ego_angle)
    other_theta = math.radians(other_angle)
    left = (-math.cos(theta), math.sin(theta))
    ego_center = (ego_position[0]-math.sin(theta)*ego_length/2,
                  ego_position[1]-math.cos(theta)*ego_length/2)
    other_center = (other_position[0]-math.sin(other_theta)*other_length/2,
                    other_position[1]-math.cos(other_theta)*other_length/2)
    lateral = ((other_center[0]-ego_center[0])*left[0]
               +(other_center[1]-ego_center[1])*left[1])
    relative_heading = other_theta-theta
    relative_motion = (other_lateral_speed*math.cos(relative_heading)-ego_lateral_speed)*reaction
    other_half_span = (abs(math.cos(relative_heading))*other_width/2
                       +abs(math.sin(relative_heading))*other_length/2)
    clearance = ego_width/2 + other_half_span + 0.1
    separated = min(abs(lateral), abs(lateral+relative_motion)) > clearance
    if lateral*(lateral+relative_motion) <= 0:
        separated = False
    return 'separated' if separated else 'overlap_or_uncertain'


def efficiency(speed, limit, reaction, gap=None, leader_speed=None,
               ego_decel=None, leader_decel=None, stopped_gap=None, epsilon=2.0):
    values = [speed, limit, reaction, epsilon]
    if any(not math.isfinite(x) for x in values) or min(values) < 0 or min(reaction, epsilon) <= 0:
        raise ValueError('Invalid SECRM speed/limit/reaction/epsilon')
    safe = None
    effective_decel = None
    if gap is not None:
        params = [gap, leader_speed, ego_decel, leader_decel, stopped_gap]
        if (any(x is None or not math.isfinite(x) for x in params)
                or leader_speed < 0 or ego_decel <= 0 or leader_decel <= 0
                or stopped_gap < 0):
            raise ValueError(f'Invalid SECRM leader data: {dict(zip(("gap", "leader_speed", "ego_decel", "leader_decel", "stopped_gap"), params))!r}')
        effective_decel = ego_decel
        d = ego_decel
        radicand = (reaction*d/2)**2 - 2*d*(reaction*speed/2 - leader_speed**2/(2*leader_decel) - gap + stopped_gap)
        if not math.isfinite(radicand):
            raise ValueError(f'Nonfinite SECRM safe-speed radicand: {radicand}; gap={gap}, speed={speed}')
        status = 'infeasible' if radicand < 0 or -reaction*d/2 + math.sqrt(max(0.0, radicand)) < 0 else 'feasible'
        if status == 'feasible':
            safe = -reaction*d/2 + math.sqrt(radicand)
    else:
        status = 'no_leader'
        radicand = None
    # Engineering convention: an infeasible braking inequality has no safe
    # positive solution. Target zero is a conservative reward target, not Eq. 2.
    target = 0.0 if status == 'infeasible' else limit if safe is None else min(safe, limit)
    value = -abs(target-speed)/max(target, epsilon)
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite SECRM efficiency reward: target={target}, speed={speed}')
    return value, {'formula_version': FORMULA_VERSION, 'speed': speed, 'limit': limit,
                   'safe_speed': safe, 'target_speed': target, 'gap': gap,
                   'leader_speed': leader_speed, 'ego_decel': ego_decel,
                   'leader_decel': leader_decel, 'effective_decel': effective_decel,
                   'stopped_gap': stopped_gap, 'reaction_seconds': reaction,
                   'radicand': radicand, 'constraint_status': status,
                   'paper_braking_assumption': (None if gap is None else ego_decel <= leader_decel),
                   'denominator_floor': epsilon}


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
