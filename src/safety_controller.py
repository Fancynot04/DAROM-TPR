import numpy as np
import math
from .merging import LANE_CHANGE_EDGES, parseAction, traci


def lane_centerline(ego_pos, ego_angle, shape):
    """Target lane center as lateral = intercept + slope * longitudinal."""
    angle = math.radians(ego_angle)
    forward = (math.sin(angle), math.cos(angle))
    left = (-forward[1], forward[0])
    best = None
    for start, end in zip(shape, shape[1:]):
        segment = (end[0] - start[0], end[1] - start[1])
        length_sq = segment[0] ** 2 + segment[1] ** 2
        if length_sq == 0:
            continue
        offset = (ego_pos[0] - start[0], ego_pos[1] - start[1])
        fraction = np.clip((offset[0] * segment[0] + offset[1] * segment[1]) / length_sq, 0, 1)
        nearest = (start[0] + fraction * segment[0], start[1] + fraction * segment[1])
        delta = (nearest[0] - ego_pos[0], nearest[1] - ego_pos[1])
        distance_sq = delta[0] ** 2 + delta[1] ** 2
        along = segment[0] * forward[0] + segment[1] * forward[1]
        if abs(along) < 1e-6:
            continue
        if best is None or distance_sq < best[0]:
            slope = (segment[0] * left[0] + segment[1] * left[1]) / along
            x = delta[0] * forward[0] + delta[1] * forward[1]
            y = delta[0] * left[0] + delta[1] * left[1]
            best = (distance_sq, y - slope * x, slope)
    return None if best is None else best[1:]


def target_lane_line(ego_id, direction):
    """Return the selected lane centerline in the ego's current local frame."""
    if direction:
        edge = traci.vehicle.getRoadID(ego_id)
        if edge not in LANE_CHANGE_EDGES:
            return None
        target = traci.vehicle.getLaneIndex(ego_id) + direction
        if not 0 <= target < traci.edge.getLaneNumber(edge):
            return None
        lane_id = f'{edge}_{target}'
    else:
        lane_id = traci.vehicle.getLaneID(ego_id)
    return lane_centerline(traci.vehicle.getPosition(ego_id),
                           traci.vehicle.getAngle(ego_id),
                           traci.lane.getShape(lane_id))


def safetyCheck(action, state, delayed, mode, target_line=None):
    min_gap = 10.0
    min_acc = -5
    dt = 0.1  # time step in seconds (from SUMO config)
    parsed_action = parseAction(action)
    delay_steps = 0
    if delayed:
        num_entities = 31
        feature_dim = 3
        obs_size = num_entities * feature_dim
        if mode == 'all':
            observation = np.array(state[:obs_size]).reshape(num_entities, feature_dim)
            # Extract delay value (last element of state)
            delay_steps = int(state[-1])
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
    else:
        observation = state
    ego_speed = observation[0][2]
    target_lane = int(observation[0][1]) + parsed_action[1]
    target_lateral = parsed_action[1] * 3.2
    # Initialize info dictionary to store safety check details
    info = {
        'front_gap': None,
        'back_gap': None,
        'front_stopping_dist': None,
        'back_stopping_dist': None,
        'front_approach_speed': None,
        'back_approach_speed': None,
        'front_relative_speed': None,
        'back_relative_speed': None,
        'front_speed': None,
        'back_speed': None,
        'delay_steps': delay_steps,
        'target_lane': None,
        'safe': True
    }
    
    # find closest front and back vehicles in the same lane from the observation
    front_vehicle_state  = None
    back_vehicle_state = None
    potential_front_vehicle_state = None
    potential_back_vehicle_state = None
    min_front_dist = float('inf')
    min_back_dist = float('inf')
    for veh_state in observation[1:]:
        expected_lateral = (target_lateral if target_line is None
                            else target_line[0] + target_line[1] * veh_state[0])
        if abs(veh_state[1] - expected_lateral) <= 1.6:
            if veh_state[0] > 0:
                if veh_state[0] < min_front_dist:
                    min_front_dist = veh_state[0]
                    front_vehicle_state = veh_state
            elif veh_state[0] < 0:
                if abs(veh_state[0]) < min_back_dist:
                    min_back_dist = abs(veh_state[0])
                    back_vehicle_state = veh_state
    
    # Predict front and back vehicle states based on delay
    if delayed and delay_steps > 0:
        # Predict front vehicle position after delay
        if front_vehicle_state is not None:
            # veh_state format: [rel_x, rel_y, rel_velocity]
            # rel_velocity = vehicle_velocity - ego_velocity
            # After delay_steps, the relative x position changes by: rel_velocity * delay_steps * dt
            predicted_rel_x = front_vehicle_state[0] + front_vehicle_state[2] * delay_steps * dt
            if predicted_rel_x < 0:
                potential_back_vehicle_state = np.array([
                    predicted_rel_x,
                    front_vehicle_state[1],
                    front_vehicle_state[2]
                ])
                potential_back_dist = abs(predicted_rel_x)
                front_vehicle_state = None
            else:
                front_vehicle_state = np.array([
                    predicted_rel_x,
                    front_vehicle_state[1],
                    front_vehicle_state[2]
                ])
                min_front_dist = predicted_rel_x

        
        # Predict back vehicle position after delay
        if back_vehicle_state is not None:
            # Same prediction for back vehicle
            predicted_rel_x = back_vehicle_state[0] + back_vehicle_state[2] * delay_steps * dt
            back_vehicle_state = np.array([
                predicted_rel_x,
                back_vehicle_state[1],  # y position doesn't change
                back_vehicle_state[2]   # relative velocity remains the same
            ])
            min_back_dist = abs(predicted_rel_x)
            if predicted_rel_x > 0:
                potential_front_vehicle_state = np.array([
                    predicted_rel_x,
                    back_vehicle_state[1],
                    back_vehicle_state[2]
                ])
                back_vehicle_state = None
                potential_front_dist = predicted_rel_x
        if potential_back_vehicle_state is not None:
            if potential_back_dist < min_back_dist:
                back_vehicle_state = potential_back_vehicle_state
                min_back_dist = potential_back_dist
        if potential_front_vehicle_state is not None:
            if potential_front_dist < min_front_dist:
                front_vehicle_state = potential_front_vehicle_state
                min_front_dist = potential_front_dist
    info['target_lane'] = target_lane
    
    if front_vehicle_state is not None:
        # front_vehicle_state[2] is relative velocity: vehVel_front - egoVel
        # Absolute front vehicle speed:
        front_speed = ego_speed + front_vehicle_state[2]
        # By definition: relative_speed = ego_speed - veh_speed
        # Positive when ego is faster and is closing in on the front vehicle.
        relative_speed = ego_speed - front_speed
        approach_speed = max(relative_speed, 0.0)

        gap = min_front_dist 
        if approach_speed > 0:
            stopping_dist = (approach_speed ** 2) / (2 * abs(min_acc))
        else:
            stopping_dist = 0
        
        # Store front vehicle info
        info['front_gap'] = gap
        info['front_stopping_dist'] = stopping_dist
        info['front_approach_speed'] = approach_speed
        info['front_relative_speed'] = relative_speed
        info['front_speed'] = front_speed
        
        if gap < stopping_dist + min_gap and parsed_action[1] != 0:
            print('Unsafe lane change becuase of front vehicle')
            info['safe'] = False
            return [action[0], 0], info
        elif gap < stopping_dist + min_gap and parsed_action[1] == 0:
            print('Unsafe lane lane keeping becuase of front vehicle')
            info['safe'] = False
            return [-5, 0], info

    if back_vehicle_state is not None:
        # back_vehicle_state[2] is relative velocity: vehVel_back - egoVel
        # Absolute back vehicle speed:
        back_speed = ego_speed + back_vehicle_state[2]
        # By definition: relative_speed = ego_speed - veh_speed
        # For a vehicle behind, we care when it is faster than ego,
        # i.e. when (veh_speed - ego_speed) > 0  <=>  relative_speed < 0.
        relative_speed = ego_speed - back_speed
        approach_speed = max(-relative_speed, 0.0)  # positive when back vehicle is catching up

        gap = min_back_dist 
        if approach_speed > 0:
            stopping_dist = (approach_speed ** 2) / (2 * abs(min_acc))
        else:
            stopping_dist = 0
        
        # Store back vehicle info
        info['back_gap'] = gap
        info['back_stopping_dist'] = stopping_dist
        info['back_approach_speed'] = approach_speed
        info['back_relative_speed'] = relative_speed
        info['back_speed'] = back_speed
        
        if gap < stopping_dist + min_gap and parsed_action[1] != 0:
            print('Unsafe lane change becuase of back vehicle')
            info['safe'] = False
            return [action[0], 0], info
    return action, info
