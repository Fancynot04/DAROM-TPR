import os, sys
import hashlib
import math
import re
import numpy as np
from pathlib import Path
# from inference import model
import gymnasium as gym
from gymnasium import spaces
from .utils import *
from .geometry import TrafficSnapshot, driving_axes, world_velocity
from . import exit_e76_d3_config as exit_config
from . import exit_progress_reward, secrm_reward, road_boundary
from .rewards.registry import RewardContext, settle
from .route_urgency import load_route_network, observe_route_urgency, route_urgency


maxSteps = 6000
ANTWERP_DIR = Path(__file__).resolve().parents[1] / 'sumo_new'
EGO_DEPART_TIME = 120.0
EGO_DEPARTURE_DEADLINE = 1200.0
SUMO_STEP_LENGTH = 0.1
EGO_SPEED_CAP_MPS = 32.0
LANE_CHANGE_EDGES = frozenset(('E75', 'E76', 'E77', 'E78', 'E82'))
ANTWERP_ENV_VERSION = 3


def curriculum_route_file(source, lane, output_dir):
    """Materialize a lane-specific route without changing any traffic entry."""
    if lane not in range(5):
        raise ValueError(f'Invalid E76 departure lane: {lane!r}')
    source = Path(source)
    output_dir = Path(output_dir)
    original = source.read_bytes()
    ego = re.search(rb'<trip\s+id="ego"(?=\s)[^>]*/>', original)
    if ego is None or ego.group().count(b'departLane="random"') != 1:
        raise ValueError('Expected exactly one random-lane ego trip in source route')
    selected = ego.group().replace(b'departLane="random"',
                                   f'departLane="{lane}"'.encode('ascii'))
    variant = original[:ego.start()] + selected + original[ego.end():]
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f'{source.stem}_ego_lane_{lane}.rou.xml'
    if target.exists():
        if target.read_bytes() != variant:
            raise FileExistsError(f'Existing curriculum route differs: {target}')
    else:
        target.write_bytes(variant)
    return target


def e09_evaluation_route_file(source, origin, lane, output_dir):
    """Change only ego origin and lane in the existing easy traffic route."""
    lane_counts = {'E76': 5, 'E77': 6, 'O1': 3}
    if origin not in lane_counts or type(lane) is not int or not 0 <= lane < lane_counts[origin]:
        raise ValueError('Invalid E09 evaluation origin or departure lane')
    source, output_dir = Path(source), Path(output_dir)
    original = source.read_bytes()
    matches = list(re.finditer(rb'<trip\s+id="ego"(?=\s)[^>]*/>', original))
    if len(matches) != 1:
        raise ValueError('Expected exactly one ego trip')
    ego = matches[0]
    trip = ego.group()
    if trip.count(b'from="E76"') != 1 or trip.count(b'departLane="random"') != 1:
        raise ValueError('Unexpected ego trip contract')
    selected = trip.replace(b'from="E76"', f'from="{origin}"'.encode('ascii'))
    selected = selected.replace(b'departLane="random"', f'departLane="{lane}"'.encode('ascii'))
    variant = original[:ego.start()] + selected + original[ego.end():]
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f'{source.stem}_e09_{origin}_{lane}.rou.xml'
    if target.exists():
        if target.read_bytes() != variant:
            raise FileExistsError(f'Existing E09 evaluation route differs: {target}')
    else:
        target.write_bytes(variant)
    return target


ANTWERP_TASKS = {
    'legacy_o3_d1': ('O3', 'D1', 'merge'),
    'entry': ('O3', 'E76', 'merge'),
    'exit': ('O1', 'D3', 'exit'),
    exit_config.TASK_ID: (exit_config.ORIGIN, exit_config.DESTINATION, 'exit'),
}

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

def isEmergencyBraking():
    for vehID in traci.vehicle.getIDList():
        if traci.vehicle.getAcceleration(vehID) <- 4.5:
            return True
    return False

def parseAction(action):
    acc = np.clip(action[0], -5, 5)
    lc = np.clip(action[1], -5, 5)
    if lc < -5 + 10/3:
        lc = -1
    elif lc > -5 + 10/3 and lc < -5 + 20/3:
        lc = 0
    elif lc > -5 + 20/3:
        lc = 1
    return [acc, lc]

def set_ego_lane_mode(vehicle_name):
    traci.vehicle.setLaneChangeMode(vehicle_name, 0)



class Merging(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 10}
    def __init__(self, seed, mode='all', radius = 50, render_mode= None, runSafetyCheck=False,
                 scenario='medium', task='legacy_o3_d1', potential_gamma=None,
                 exit_reward_profile=None, e09_pending_mode=None,
                 progress_experiment=None):
        if scenario not in ('easy', 'medium', 'hard'):
            raise ValueError(f'Unknown Antwerp scenario: {scenario}')
        if task not in ANTWERP_TASKS:
            raise ValueError(f'Unknown Antwerp task: {task}')
        if e09_pending_mode not in (None, 'on', 'off'):
            raise ValueError('Invalid E09 pending mode')
        if e09_pending_mode is not None and (task != exit_config.TASK_ID or runSafetyCheck):
            raise ValueError('E09 requires the E76-D3 task with custom safety disabled')
        self.e09_pending_mode = e09_pending_mode
        self.scenario = scenario
        self.task = task
        self.exit_reward_profile = (exit_config.REWARD_PROFILE if exit_reward_profile is None
                                    else exit_reward_profile)
        self.progress_experiment = (exit_progress_reward.validate_experiment(progress_experiment)
                                    if progress_experiment is not None else None)
        if self.exit_reward_profile in (exit_config.PROGRESS_REWARD_PROFILE,
                                        exit_config.DIRECTIONAL_REWARD_PROFILE, *secrm_reward.PROFILES):
            if task != exit_config.TASK_ID or self.progress_experiment is None:
                raise ValueError('Progress reward requires an E76-D3 experiment configuration')
            if exit_progress_reward.reward_profile(self.progress_experiment) != self.exit_reward_profile:
                raise ValueError('Progress configuration and reward profile disagree')
        elif self.progress_experiment is not None:
            raise ValueError('Progress experiment configuration requires its reward profile')
        self.episode_limit_steps = (int(round(self.progress_experiment['timeout_seconds']
                                               / SUMO_STEP_LENGTH))
                                    if self.progress_experiment is not None else maxSteps)
        if (self.progress_experiment is not None and
                (self.episode_limit_steps < 1 or
                 not math.isclose(self.episode_limit_steps * SUMO_STEP_LENGTH,
                                  self.progress_experiment['timeout_seconds'], abs_tol=1e-9))):
            raise ValueError('Progress timeout must be a multiple of the SUMO step length')
        if task == exit_config.TASK_ID:
            exit_config.potential_beta_for_profile(self.exit_reward_profile)
        elif exit_reward_profile is not None:
            raise ValueError('An E76-to-D3 reward profile requires the E76-to-D3 task')
        self.route_network = (load_route_network(ANTWERP_DIR / 'Antwerp.net.xml')
                              if self.exit_reward_profile in (exit_config.URGENCY_REWARD_PROFILE,
                                                              exit_config.DIRECTIONAL_REWARD_PROFILE, *secrm_reward.PROFILES)
                              and task == exit_config.TASK_ID else None)
        self.potential_gamma = (exit_config.POTENTIAL_GAMMA
                                if potential_gamma is None else float(potential_gamma))
        self.origin, self.destination, self.stage_name = ANTWERP_TASKS[task]
        self.ego_id = None
        self.ego_depart_time = None
        self._seed_initialized = False
        self.done = False
        self.observation = []
        self.reward = 0
        self.state = []
        self.ego_inserted = False
        self.egoIdx = 0
        self.seed = seed
        self.radius = radius
        self.size = 31
        self.merged = False
        self.stage_reached = False
        self.last_stage_reward = 0.0
        self.ego_teleported = False
        self.ego_teleporting = False
        self.missed_exit = False
        self.exit_maneuver = None
        self.ego_on_road = False
        self.teleport_transition = False
        self.observation_snapshot = None
        self.lane_id = ''
        self.lane_name = ''
        self.lane_idx = 0
        self.parsed_action = None
        self.exit_actor_request_direction = 0
        self.ego_start_lane = None
        self.curriculum_depart_lane = None
        self.curriculum_route_dir = None
        self.e09_evaluation_origin = None
        self.e09_evaluation_lane = None
        self.e09_evaluation_route_dir = None
        self.potential_distance_scale = None
        self.potential_previous = 0.0
        self.initial_potential = 0.0
        self.progress_initial_remaining = None
        self.progress_before_remaining = None
        self.progress_before_odometer = None
        self.progress_after_remaining = None
        self.progress_after_odometer = None
        self.terminal_reason = ''
        self.secrm_before = None
        self.boundary_diagnostic = None
        self.network_path = ANTWERP_DIR / 'Antwerp.net.xml'
        self.directional_settled_lane = None
        self.directional_pending_request = None
        self.directional_seen_lateral_motion = False
        self.directional_lane_event = None
        self.directional_lane_reward = 0.0
        self.last_valid_urgency = None
        self.last_settled_urgency_edge = None
        self.last_settled_urgency_lane = None
        self.action = None
        self.runSafetyCheck = runSafetyCheck
        self.last_acc = 0
        self._ahead_state = []
        self._behind_state = []
        if not render_mode == 'human':
            self.sumoBinary = checkBinary('sumo')
        else:
            self.sumoBinary = checkBinary('sumo-gui')

        self.num_entities = self.size          # 31 = ego + 30 neighbors
        self.feature_dim = 3                   # [x, y, vel]

        self.observation_space = spaces.Box(
            low=-5e3, high=5e3,
            shape=(self.num_entities, self.feature_dim),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0], dtype=np.float32),
            high=np.array([5.0, 5.0], dtype=np.float32),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self._connection = None

    @property
    def traci(self):
        # Single-ego callers retain the default connection and existing test mocks.
        return traci if self._connection is None else self._connection

    

    def reset_episode_state(self):
        """Clear per-vehicle state without owning or advancing a simulation."""
        self.reward = 0.0
        self.reward_components = {}
        self.merged = False
        self.stage_reached = False
        self.last_stage_reward = 0.0
        self.ego_teleported = False
        self.ego_teleporting = False
        self.missed_exit = False
        self.exit_maneuver = None
        self.ego_on_road = False
        self.teleport_transition = False
        self.observation_snapshot = None
        self.given_merge_reward = False
        self.done = False
        self.ego_inserted = False
        self.ego_id = None
        self.ego_depart_time = None
        self.counter = 0
        self.last_acc = 0
        self._ahead_state = []
        self._behind_state = []
        self.action = None
        self.parsed_action = None
        self.lane_id = self.lane_name = ""
        self.lane_idx = 0
        self.observation = np.zeros((31, 3), dtype=np.float32)
        self.state = self.observation.copy()
        self.ego_start_lane = None
        self.potential_distance_scale = None
        self.potential_previous = 0.0
        self.initial_potential = 0.0
        self.progress_initial_remaining = None
        self.progress_before_remaining = None
        self.progress_before_odometer = None
        self.progress_after_remaining = None
        self.progress_after_odometer = None
        self.terminal_reason = ''
        self.secrm_before = None
        self.boundary_diagnostic = None
        self.network_path = ANTWERP_DIR / 'Antwerp.net.xml'
        self.directional_settled_lane = None
        self.directional_pending_request = None
        self.directional_seen_lateral_motion = False
        self.directional_lane_event = None
        self.directional_lane_reward = 0.0
        self.last_valid_urgency = None
        self.last_settled_urgency_edge = None
        self.last_settled_urgency_lane = None

    def _remaining_to_d3_end(self):
        """Physical route distance to the end of D3, including travel on D3."""
        if not self.ego_on_road:
            return None
        destination_position = float(self.traci.lane.getLength('D3_0'))
        if self.traci.vehicle.getRoadID(self.ego_id) == 'D3':
            distance = destination_position - float(self.traci.vehicle.getLanePosition(self.ego_id))
        else:
            distance = float(self.traci.vehicle.getDrivingDistance(
                self.ego_id, 'D3', destination_position))
        return distance if math.isfinite(distance) and distance >= 0 else None

    def _secrm_snapshot(self):
        v, e = self.traci.vehicle, self.ego_id
        leader = v.getLeader(e, 1000000.0)
        # The action can command at most 5 m/s² of braking. SUMO emergencyDecel
        # is the physical maximum; decel is comfortable, apparentDecel is what
        # other drivers expect. The paper's d_E must respect both limits.
        ego_braking = {'comfortable': float(v.getDecel(e)),
                       'apparent': float(v.getApparentDecel(e)),
                       'emergency': float(v.getEmergencyDecel(e)),
                       'action_limit': -float(self.action_space.low[0])}
        ego_braking['used'] = min(ego_braking['emergency'], ego_braking['action_limit'])
        if (any(not math.isfinite(x) or x <= 0 for x in ego_braking.values())
                or not math.isfinite(float(v.getMinGap(e))) or float(v.getMinGap(e)) < 0):
            raise ValueError(f'Invalid SECRM ego braking/minGap: {ego_braking!r}')
        params = dict(ego_decel=ego_braking['used'], stopped_gap=float(v.getMinGap(e)))
        leader_info = {'status': 'none', 'id': None}
        ego_pose = v.getPosition(e)
        ego_angle = float(v.getAngle(e))
        ego_road = v.getRoadID(e)
        ego_lane_position = float(v.getLanePosition(e))
        def relation(other):
            return secrm_reward.lateral_relation(
                ego_pose, ego_angle, float(v.getLength(e)), float(v.getWidth(e)),
                float(v.getLateralSpeed(e)), v.getPosition(other),
                float(v.getAngle(other)), float(v.getLength(other)), float(v.getWidth(other)),
                float(v.getLateralSpeed(other)), SUMO_STEP_LENGTH)
        # During a continuous lane change getLeader may refer to one nominal
        # lane, or return no vehicle, while another body occupies the swept ego
        # path. Compare the native leader with all same-section path occupants.
        native_id = leader[0] if leader and leader[0] else None
        candidates = []
        if native_id is not None and relation(native_id) != 'separated':
            candidates.append((native_id, float(leader[1]) + params['stopped_gap'],
                               'traci_leader'))
        for other in v.getIDList():
            if other in (e, native_id) or v.getRoadID(other) != ego_road:
                continue
            longitudinal = float(v.getLanePosition(other))-ego_lane_position
            if (-float(v.getLength(other)) <= longitudinal <= 150
                    and relation(other) != 'separated'):
                candidates.append((other, longitudinal-float(v.getLength(other)),
                                   'lateral_path'))
        selected = min(candidates, key=lambda item: item[1]) if candidates else None
        if selected is None and native_id is not None:
            leader_info = {'status': 'laterally_separated', 'id': native_id,
                           'raw_gap': float(leader[1])}
        if selected is not None:
            lid, gap, source = selected
            leader_braking = {'comfortable': float(v.getDecel(lid)),
                              'apparent': float(v.getApparentDecel(lid)),
                              'emergency': float(v.getEmergencyDecel(lid))}
            if any(not math.isfinite(x) or x <= 0 for x in leader_braking.values()):
                raise ValueError(f'Invalid SECRM leader braking: {lid!r}, {leader_braking!r}')
            params.update(gap=gap, leader_speed=float(v.getSpeed(lid)),
                          leader_decel=leader_braking['emergency'])
            leader_info = {'status': source, 'id': lid, 'braking': leader_braking,
                           'physical_gap': gap}
        allowed_speed = float(v.getAllowedSpeed(e))
        if not math.isfinite(allowed_speed) or allowed_speed < 0:
            raise ValueError(f'Invalid SECRM allowed speed: {allowed_speed!r}')
        eff, diagnostic = secrm_reward.efficiency(
            float(v.getSpeed(e)), min(allowed_speed, EGO_SPEED_CAP_MPS), SUMO_STEP_LENGTH,
            epsilon=self.progress_experiment['reward']['speed_epsilon'], **params)
        diagnostic.update(ego_braking=ego_braking, leader_selection=leader_info)
        road = v.getRoadID(e)
        route_value = 0.0
        route_info = {'status': 'connector_excluded', 'required_changes': None,
                      'remaining_section_m': None}
        if road == 'D3':
            remaining = self._remaining_to_d3_end()
            if remaining is None or not math.isfinite(remaining):
                raise ValueError(f'Invalid SECRM D3 remaining distance: {remaining!r}')
            route_info = {'status': 'at_destination', 'required_changes': 0,
                          'remaining_section_m': max(0.0, remaining)}
        elif not road.startswith(':'):
            stable = self.exit_maneuver is None and self._exit_laterally_settled()
            route = tuple(v.getRoute(e))
            route_index = int(v.getRouteIndex(e))
            lane = int(v.getLaneIndex(e))
            lane_id = v.getLaneID(e)
            if (route_index < 0 or route_index >= len(route) or route[route_index] != road
                    or lane_id != f'{road}_{lane}'):
                raise ValueError(f'SECRM road/route/lane mismatch: {road!r}, {route!r}, {route_index}, {lane_id!r}, {lane}')
            required = secrm_reward.minimum_changes(
                self.route_network, route, route_index, lane, LANE_CHANGE_EDGES)
            remaining = float(self.traci.lane.getLength(lane_id)) - float(v.getLanePosition(e))
            if not math.isfinite(remaining):
                raise ValueError(f'Invalid SECRM section distance: road={road!r}, lane={lane}, distance={remaining!r}')
            remaining = max(0.0, remaining)
            route_value = secrm_reward.route_penalty(required, remaining)
            route_info = {'road': road, 'lane': lane, 'required_changes': required,
                          'status': 'settled' if stable else 'pending_current_lane',
                          'remaining_section_m': remaining}
        return {'efficiency': eff, 'route_reward': route_value, 'target': diagnostic,
                'route': route_info, 'road': road}

    def _reward_urgency(self):
        """Use only settled lane-route estimates for the opt-in reward."""
        if self.done or not self.ego_on_road or self.ego_teleported:
            return 0.0, False
        estimate = observe_route_urgency(
            self, self.route_network, LANE_CHANGE_EDGES,
            lane_change_seconds=exit_config.LANE_CHANGE_TIME_SECONDS)
        if estimate['status'] == 'at_destination':
            return 0.0, False
        if estimate['status'] != 'ok':
            raise ValueError(f'Invalid route urgency during reward: {estimate["status"]}')
        if estimate['valid_for_threshold']:
            self.last_valid_urgency = estimate['urgency']
            self.last_settled_urgency_edge = estimate['projected_edge']
            self.last_settled_urgency_lane = estimate['projected_lane']
            return self.last_valid_urgency, False
        if self.last_valid_urgency is None:
            raise ValueError('No settled route urgency before provisional movement')
        if estimate['road'] == self.last_settled_urgency_edge:
            vehicle = self.traci.vehicle
            route = tuple(vehicle.getRoute(self.ego_id))
            edge_index = int(vehicle.getRouteIndex(self.ego_id))
            if (0 <= edge_index < len(route)
                    and route[edge_index] == self.last_settled_urgency_edge):
                stable = route_urgency(
                    self.route_network, route, edge_index,
                    self.last_settled_urgency_lane,
                    float(vehicle.getLanePosition(self.ego_id)),
                    float(vehicle.getSpeed(self.ego_id)),
                    float(vehicle.getLength(self.ego_id)), LANE_CHANGE_EDGES,
                    lane_change_seconds=exit_config.LANE_CHANGE_TIME_SECONDS)
                if stable['status'] != 'ok':
                    raise ValueError('Settled-lane urgency route became unreachable')
                return stable['urgency'], True
        return self.last_valid_urgency, True

    def _exit_potential(self):
        if self.task != exit_config.TASK_ID or not self.ego_on_road or self.ego_teleported:
            return 0.0
        road = self.traci.vehicle.getRoadID(self.ego_id)
        if road == 'D3':
            distance = 0.0
            lane_index = 0
        else:
            distance = self.traci.vehicle.getDrivingDistance(self.ego_id, 'D3', 0.0)
            if road in ('E76', 'E77'):
                lane_index = self.traci.vehicle.getLaneIndex(self.ego_id)
            else:
                lane_index = self.lane_idx
        if not math.isfinite(distance):
            return 0.0
        return exit_config.potential(distance, self.potential_distance_scale, lane_index)

    def exit_navigation(self):
        """Live route information for the opt-in E76-to-D3 observation only."""
        if self.task != exit_config.TASK_ID:
            raise ValueError('Exit navigation is only defined for E76-to-D3')
        if self.done or not self.ego_on_road or self.ego_teleported:
            return np.zeros(2, dtype=np.float32)
        road = self.traci.vehicle.getRoadID(self.ego_id)
        lane_id = self.traci.vehicle.getLaneID(self.ego_id)
        if road == 'D3' or lane_id == ':J106_0_0':
            distance, required = 0.0, 0
        else:
            distance = float(self.traci.vehicle.getDrivingDistance(self.ego_id, 'D3', 0.0))
            if road in ('E76', 'E77'):
                required = int(self.traci.vehicle.getLaneIndex(self.ego_id))
            elif lane_id in exit_config.J105_TARGET_LANES:
                required = exit_config.J105_TARGET_LANES[lane_id]
            else:
                raise ValueError(f'No D3 navigation mapping for nonterminal lane {lane_id!r}')
        if not math.isfinite(distance) or distance < 0:
            raise ValueError(f'Invalid nonterminal D3 driving distance: {distance!r}')
        if not 0 <= required <= 5:
            raise ValueError(f'Invalid required lane-change count: {required!r}')
        return np.array([min(distance / exit_config.NAV_DISTANCE_SCALE_METERS, 1.0),
                         required / 5.0], dtype=np.float32)

    def _e77_remaining(self):
        lane_id = self.traci.vehicle.getLaneID(self.ego_id)
        return max(0.0, self.traci.lane.getLength(lane_id)
                   - self.traci.vehicle.getLanePosition(self.ego_id))

    def _e77_exit_lane_ready(self):
        if self.traci.vehicle.getLaneIndex(self.ego_id) != 0:
            return False
        return self._exit_laterally_settled()

    def _exit_laterally_settled(self):
        """Return whether SUMO has stopped with the body inside its lane."""
        lane_id = self.traci.vehicle.getLaneID(self.ego_id)
        clearance = (self.traci.lane.getWidth(lane_id)
                     - self.traci.vehicle.getWidth(self.ego_id)) / 2.0
        return (clearance >= 0
                and abs(self.traci.vehicle.getLateralLanePosition(self.ego_id))
                <= clearance + exit_config.LATERAL_CONTAINMENT_EPSILON_METERS
                and abs(self.traci.vehicle.getLateralSpeed(self.ego_id))
                <= exit_config.LATERAL_SETTLED_SPEED_MPS)

    def _exit_change_has_room(self):
        speed = max(self.traci.vehicle.getSpeed(self.ego_id),
                    exit_config.MIN_MANEUVER_SPEED_MPS)
        required = (speed * exit_config.LANE_CHANGE_TIME_SECONDS
                    + self.traci.vehicle.getLength(self.ego_id)
                    + exit_config.LANE_END_BUFFER_METERS)
        lane_id = self.traci.vehicle.getLaneID(self.ego_id)
        remaining = self.traci.lane.getLength(lane_id) - self.traci.vehicle.getLanePosition(self.ego_id)
        return remaining > required

    def _exit_maneuver_target_matches(self):
        maneuver = self.exit_maneuver
        road = self.lane_name
        target = maneuver['target']
        if road == maneuver['edge']:
            return self.lane_idx == target
        if maneuver['edge'] == 'E76' and (road == 'E77' or road.startswith(':J105')):
            # J105 adds a lane on the D3 side. E76_0 also has a D3 connector.
            return self.lane_idx == target + 1 or (target == 0 and self.lane_idx == 0)
        if maneuver['edge'] == 'E77' and (road == 'D3' or road.startswith(':J106')):
            return target == 0 and self.lane_idx == 0
        return False

    def _update_exit_maneuver(self):
        maneuver = self.exit_maneuver
        if maneuver is None:
            return
        lateral_speed = abs(self.traci.vehicle.getLateralSpeed(self.ego_id))
        now = self.traci.simulation.getTime()
        if lateral_speed > exit_config.LATERAL_SETTLED_SPEED_MPS:
            maneuver['last_progress_time'] = now
        settled = (self._exit_maneuver_target_matches()
                   and self._exit_laterally_settled())
        maneuver['stable_steps'] = maneuver['stable_steps'] + 1 if settled else 0
        if maneuver['stable_steps'] >= exit_config.MANEUVER_STABLE_STEPS:
            self.exit_maneuver = None
        elif (now - maneuver['start_time'] >= exit_config.MANEUVER_MAX_SECONDS
              and lateral_speed <= exit_config.LATERAL_SETTLED_SPEED_MPS):
            # An unfulfilled command expires; a vehicle still moving laterally
            # remains locked until it settles or the exit is missed.
            self.exit_maneuver = None

    def _update_directional_lane_reward(self):
        """Settle a physical same-edge change once; ignore junction remapping."""
        self.directional_lane_reward = 0.0
        self.directional_lane_event = None
        if self.exit_reward_profile not in (exit_config.DIRECTIONAL_REWARD_PROFILE, *secrm_reward.PROFILES):
            return
        if (not self.ego_on_road or self.ego_teleported or self.teleport_transition
                or self.terminal_reason in ('Collision', 'Ego Teleported')
                or self.lane_name not in LANE_CHANGE_EDGES):
            self.directional_settled_lane = None
            self.directional_pending_request = None
            self.directional_seen_lateral_motion = False
            return
        if (self.directional_pending_request is not None
                and abs(float(self.traci.vehicle.getLateralSpeed(self.ego_id)))
                > exit_config.LATERAL_SETTLED_SPEED_MPS):
            self.directional_seen_lateral_motion = True
        estimate = observe_route_urgency(
            self, self.route_network, LANE_CHANGE_EDGES,
            lane_change_seconds=exit_config.LANE_CHANGE_TIME_SECONDS)
        if not estimate.get('valid_for_threshold', False):
            previous = self.directional_settled_lane
            if (estimate.get('status') != 'ok' or not estimate.get('pending')
                    or (previous is not None and estimate.get('road') != previous['road'])):
                self.directional_settled_lane = None
                self.directional_pending_request = None
                self.directional_seen_lateral_motion = False
            return
        current = {'road': estimate['road'], 'lane': estimate['projected_lane'],
                   'minimum_lane_changes': estimate['minimum_lane_changes']}
        previous = self.directional_settled_lane
        self.directional_settled_lane = current
        if previous is None or previous['road'] != current['road']:
            self.directional_pending_request = None
            self.directional_seen_lateral_motion = False
            return
        value, direction = exit_progress_reward.directional_lane_reward(
            previous, current, (self.progress_experiment['reward']
                                if self.exit_reward_profile == exit_config.DIRECTIONAL_REWARD_PROFILE
                                else {'alpha': 0.0, 'beta': 0.0}))
        if (direction is not None and self.directional_pending_request is not None
                and self.directional_seen_lateral_motion):
            self.directional_lane_reward = value
            self.directional_lane_event = {
                'before': previous, 'after': current,
                'request_direction': self.directional_pending_request,
                'actual_direction': current['lane'] - previous['lane'],
                'route_direction': direction, 'reward': value,
            }
        if not estimate['pending']:
            self.directional_pending_request = None
            self.directional_seen_lateral_motion = False

    def _e77_maneuver_can_finish(self):
        maneuver = self.exit_maneuver
        if maneuver is None or maneuver['target'] != 0:
            return False
        now = self.traci.simulation.getTime()
        return (now - maneuver['start_time'] < exit_config.MANEUVER_MAX_SECONDS
                and now - maneuver['last_progress_time'] < exit_config.MANEUVER_STALL_SECONDS)

    def reset(self, seed=None, options=None):
        # Seed the environment RNG per Gymnasium API
        print('Resetting environment', 'radius', self.radius)
        initial_seed = self.seed if not self._seed_initialized and seed is None else seed
        super().reset(seed=initial_seed)
        self._seed_initialized = True
        sumo_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.reset_episode_state()

        # Close any existing SUMO connection
        try:
            self.traci.close()
        except Exception:
            pass

        # Load the Antwerp network and one scenario route without detector output files.
        route_suffix = '' if self.task == 'legacy_o3_d1' else f'_{self.task}'
        source_routes = ANTWERP_DIR / f'Antwerp_{self.scenario}{route_suffix}.rou.xml'
        if self.e09_evaluation_origin is not None:
            if self.e09_pending_mode is None or self.e09_evaluation_route_dir is None:
                raise ValueError('E09 evaluation origin requires an explicit E09 mode and route directory')
            self.origin = self.e09_evaluation_origin
            routes = e09_evaluation_route_file(source_routes, self.origin,
                                                self.e09_evaluation_lane,
                                                self.e09_evaluation_route_dir)
        elif self.curriculum_depart_lane is not None:
            if self.task != exit_config.TASK_ID or self.curriculum_route_dir is None:
                raise ValueError('Curriculum departure requires E76-to-D3 and a route output directory')
            routes = curriculum_route_file(source_routes, self.curriculum_depart_lane,
                                           self.curriculum_route_dir)
        else:
            routes = source_routes
        scene_paths = {'network': ANTWERP_DIR / 'Antwerp.net.xml',
                       'routes': routes}
        scene_files = {name: {'path': (str(path) if name == 'routes' and routes != source_routes
                                      else str(path.relative_to(ANTWERP_DIR.parent))),
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                       for name, path in scene_paths.items()}
        sumo_cmd = [
            self.sumoBinary,
            '--net-file', str(scene_paths['network']),
            '--route-files', str(scene_paths['routes']),
            '--begin', '0', '--end', '3600',
            '--step-length', str(SUMO_STEP_LENGTH), '--lateral-resolution', '0.8',
            '--seed', str(sumo_seed),
            '--no-step-log', '--collision.check-junctions',
            '--collision.action', 'remove', '-Q',
        ]
        self.traci.start(sumo_cmd)

        # Stop one step before the scheduled departure so TraCI observes its ID.
        self.traci.simulationStep(EGO_DEPART_TIME - SUMO_STEP_LENGTH)
        while not self.ego_inserted and self.traci.simulation.getTime() < EGO_DEPARTURE_DEADLINE:
            self.traci.simulationStep()
            for vehicle_name in self.traci.simulation.getDepartedIDList():
                if vehicle_name == 'ego':
                    self.traci.vehicle.setLaneChangeMode(vehicle_name, 0)
                    if self.traci.vehicle.getRoadID(vehicle_name) != self.origin:
                        self.traci.close()
                        raise RuntimeError(f'The scheduled ego did not enter on {self.origin}')
                    self.ego_id = vehicle_name
                    self.ego_depart_time = self.traci.simulation.getTime()
                    self.ego_inserted = True
                    self.ego_on_road = True
                    if self.task == exit_config.TASK_ID:
                        self.ego_start_lane = self.traci.vehicle.getLaneIndex(self.ego_id)
                        if (self.curriculum_depart_lane is not None
                                and self.ego_start_lane != self.curriculum_depart_lane):
                            self.traci.close()
                            raise RuntimeError('SUMO ego lane differs from selected curriculum lane')
                        self.potential_distance_scale = sum(
                            self.traci.lane.getLength(f'{edge}_0')
                            for edge in exit_config.DISTANCE_SCALE_EDGES)
                        if self.exit_reward_profile == exit_config.REWARD_PROFILE:
                            self.potential_previous = self._exit_potential()
                            self.initial_potential = self.potential_previous
                        elif self.exit_reward_profile == exit_config.URGENCY_REWARD_PROFILE:
                            self._reward_urgency()
                        else:
                            self.progress_initial_remaining = self._remaining_to_d3_end()
                            if self.progress_initial_remaining is None or self.progress_initial_remaining <= 0:
                                raise ValueError('Invalid initial route distance to the D3 endpoint')
                    self.observation = np.asarray(self._get_obs(), dtype=np.float32)
        if not self.ego_inserted:
            self.traci.close()
            raise RuntimeError(f'The scheduled ego did not enter {self.origin} before '
                               f'{EGO_DEPARTURE_DEADLINE}s in {self.scenario}')
        
        self.state = self.observation
        return self.state, {
            'ego_id': self.ego_id, 'ego_depart_time': self.ego_depart_time,
            'task': ('antwerp-o3-d1' if self.task == 'legacy_o3_d1'
                     else f'antwerp-{self.origin.lower()}-{self.destination.lower()}'),
            'task_id': self.task, 'origin': self.origin, 'destination': self.destination,
            'stage_name': self.stage_name, 'scenario': self.scenario,
            'antwerp_env_version': ANTWERP_ENV_VERSION,
            'sumo_version': list(self.traci.getVersion()), 'sumo_seed': sumo_seed,
            'sumo_command': sumo_cmd, 'scene_files': scene_files,
            'step_length': self.traci.simulation.getDeltaT(),
            'episode_limit_steps': self.episode_limit_steps,
            'episode_clock': 'steps_since_takeover',
            'ego_route': list(self.traci.vehicle.getRoute(self.ego_id)),
            'ego_speed_mode': self.traci.vehicle.getSpeedMode(self.ego_id),
            'ego_lane_change_mode': self.traci.vehicle.getLaneChangeMode(self.ego_id),
            'ego_start_lane': self.ego_start_lane,
            'initial_potential': self.initial_potential,
            'potential_distance_scale': self.potential_distance_scale,
        }

    def getReward(self):

        if (self.exit_reward_profile in secrm_reward.PROFILES
                or (self.exit_reward_profile in (exit_config.PROGRESS_REWARD_PROFILE,
                                                 exit_config.DIRECTIONAL_REWARD_PROFILE)
                    and self.task == exit_config.TASK_ID)):
            context = RewardContext(
                outcome=self.terminal_reason, config=self.progress_experiment['reward'],
                teleported=self.teleport_transition or self.ego_teleported,
                progress_before=self.progress_before_remaining,
                progress_after=self.progress_after_remaining,
                odometer_before=self.progress_before_odometer,
                odometer_after=self.progress_after_odometer,
                initial_remaining=self.progress_initial_remaining,
                lane_change=self.directional_lane_reward,
                secrm_before=self.secrm_before)
            total, self.reward_components = settle(self.exit_reward_profile, context)
            self.last_stage_reward = 0.0
            return total

        # print(self.egoIdx)
        # print('action', action, 'safe_action', safe_action)
        # print(self.state[self.egoIdx])
        # Teleport jumps and time outside the road must not earn progress.
        distance_traveled = float(self.observation[self.egoIdx, 0] - self.state[self.egoIdx, 0])
        if self.teleport_transition or not self.ego_on_road:
            distance_traveled = 0.0
        # print(distance_traveled)
        r = -0.30
        base_penalty = -0.30

        jerk = np.abs(self.parsed_action[0] - self.last_acc)
        jerk_penalty = 1.5 * jerk
        r -= jerk_penalty
        # r = 0
        try:
            progress_reward = 1.5 * distance_traveled
            r += progress_reward
        except Exception:
            progress_reward = 0.0
            pass

        is_safe = safe_gap(self._behind_state, self._ahead_state)

        penalty_behind = 0.0
        penalty_ahead = 0.0

        if not is_safe[0]:
            # Use tanh penalty for behind; use distance to behind vehicle
            behind_dist = abs(self._behind_state[0])
            penalty_behind = 10 * np.tanh(1.0 / (behind_dist + 1e-3))  # strong penalty for small gaps, saturates at 10
            r -= penalty_behind

        if not is_safe[1]:
            # Use tanh penalty for ahead; use distance to ahead vehicle
            ahead_dist = abs(self._ahead_state[0]) 
            penalty_ahead = 10 * np.tanh(1.0 / (ahead_dist + 1e-3))
            r -= penalty_ahead

        collision_penalty = 0.0
        arrived_reward = 0.0
        emergency_stopping_penalty = 0.0
        merge_reward = 0.0
        lane_command_penalty = 0.0
        max_steps_penalty = 0.0
        missed_exit_penalty = 0.0

        if self.ego_id in self.traci.simulation.getCollidingVehiclesIDList():
            print("Colliding")
            collision_penalty = 1000
            r -= collision_penalty
        elif self.missed_exit:
            missed_exit_penalty = exit_config.MISSED_EXIT_PENALTY
            r -= missed_exit_penalty
        elif self.ego_id in self.traci.simulation.getArrivedIDList():
            print("Arrived")
            arrived_reward = 0 if self.ego_teleported else 500
            r += arrived_reward
        elif self.ego_id in self.traci.simulation.getEmergencyStoppingVehiclesIDList():
            print("Emergency Stopping")
            emergency_stopping_penalty = 1000
            r -= emergency_stopping_penalty
        elif self.stage_reached and not self.given_merge_reward and not self.ego_teleported:
            print(f"{self.stage_name} reward given.")
            self.given_merge_reward = True
            merge_reward = 500
            r += merge_reward
        elif self.counter >= maxSteps:
            print("Max Steps")
            max_steps_penalty = 1000
            r -= max_steps_penalty
        elif self.task != exit_config.TASK_ID and self.parsed_action[1] != 0:
            lane_command_penalty = 5
            r -= lane_command_penalty

        # print(f"Reward components:"
        #       f"\n  base_penalty: {base_penalty}"
        #       f"\n  progress_reward: {progress_reward}"
        #       f"\n  jerk_penalty: {-jerk_penalty}"
        #       f"\n  penalty_behind: {-penalty_behind}"
        #       f"\n  penalty_ahead: {-penalty_ahead}"
        #       f"\n  collision_penalty: {-collision_penalty}"
        #       f"\n  arrived_reward: {arrived_reward}"
        #       f"\n  emergency_stopping_penalty: {-emergency_stopping_penalty}"
        #       f"\n  merge_reward: {merge_reward}"
        #       f"\n  lane_command_penalty: {-lane_command_penalty}"
        #       f"\n  max_steps_penalty: {-max_steps_penalty}"
        #       f"\n  TOTAL (BEFORE SCALING): {r}")

        # print("reward", r, "action", action, "lane", self.lane_id, "merged", self.merged)
        self.last_stage_reward = merge_reward * 0.1
        self.reward_components = {
            'base': base_penalty * 0.1, 'progress': progress_reward * 0.1,
            'action_change': -jerk_penalty * 0.1,
            'behind_gap': -penalty_behind * 0.1, 'ahead_gap': -penalty_ahead * 0.1,
            'collision': -collision_penalty * 0.1, 'arrival': arrived_reward * 0.1,
            'emergency': -emergency_stopping_penalty * 0.1, 'stage': merge_reward * 0.1,
            'timeout': -max_steps_penalty * 0.1,
            'missed_exit': -missed_exit_penalty * 0.1,
            'lane_command': -lane_command_penalty * 0.1,
        }
        self.reward_components = {key: float(value) for key, value in self.reward_components.items()}
        return 0.1*r    
    
    def _get_obs(self):

        # Express surrounding vehicles in the ego's local driving direction.
        egoPos = self.traci.vehicle.getPosition(self.ego_id)
        egoVel = self.traci.vehicle.getSpeed(self.ego_id)
        ego_angle = self.traci.vehicle.getAngle(self.ego_id)
        forward, left = driving_axes(ego_angle)
        ego = [[self.traci.vehicle.getDistance(self.ego_id),
                self.traci.vehicle.getLaneIndex(self.ego_id), egoVel]]

        # Get surrounding vehicles pos and vel
        intruders = []
        self._ahead_state = [] # closest ahead vehicle state
        self._behind_state = [] # closest behind vehicle state
        closest_ahead = None
        closest_behind = None
        min_ahead_dist = float('inf')
        min_behind_dist = float('inf')
        for vehID in self.traci.vehicle.getIDList():
            if vehID == self.ego_id:
                continue
            vehPos = self.traci.vehicle.getPosition(vehID)
            distance = getDistance(vehPos, egoPos) # distance between vehicle and ego everything is absolute coordinates.
            if distance <= self.radius:
                vehVel = self.traci.vehicle.getSpeed(vehID)
                velocity = world_velocity(vehVel, self.traci.vehicle.getAngle(vehID))
                dx, dy = vehPos[0] - egoPos[0], vehPos[1] - egoPos[1]
                relState = [dx * forward[0] + dy * forward[1],
                            dx * left[0] + dy * left[1], float(velocity @ forward) - egoVel]
                # print('Distance is less than radius', distance, 'for vehicle', vehID, 'relState', relState)
                intruders.append((relState, vehPos, velocity))
                # Determine if ahead or behind by x position relative to ego
                if relState[0] > 0 and abs(relState[1]) <= 1.6:  # same lane, ahead
                    if relState[0] < min_ahead_dist:
                        min_ahead_dist = relState[0]
                        closest_ahead = relState
                elif relState[0] < 0 and abs(relState[1]) <= 1.6:  # same lane, behind
                    if abs(relState[0]) < min_behind_dist:
                        min_behind_dist = abs(relState[0])
                        closest_behind = relState
        self._ahead_state = closest_ahead if closest_ahead is not None else [0.0, 0.0, 0.0]
        self._behind_state = closest_behind if closest_behind is not None else [0.0, 0.0, 0.0]
        # Keep the nearest neighbors before applying the existing spatial order.
        intruders = sorted(intruders, key=lambda x: x[0][0] ** 2 + x[0][1] ** 2)[:self.num_entities - 1]
        intruders = sorted(intruders, key=lambda x: (x[0][1], x[0][0]))
        positions = np.zeros((self.num_entities - 1, 2))
        velocities = np.zeros_like(positions)
        valid = np.zeros(self.num_entities - 1, dtype=bool)
        for index, (_, position, velocity) in enumerate(intruders):
            positions[index] = position
            velocities[index] = velocity
            valid[index] = True
        self.observation_snapshot = TrafficSnapshot(
            self.traci.simulation.getTime(), np.asarray(egoPos), ego_angle,
            positions, velocities, valid,
        )
        intruders = [entry[0] for entry in intruders]
        
    
        while len(intruders) < self.num_entities - 1:
            intruders.append([0.0, 0.0, 0.0])
        # print('Intruders', intruders, 'ego', ego)
        obs = np.array(ego + intruders, dtype=np.float32)
        return obs
        

    def _has_merged(self):
        return ((self.lane_name == 'E75' and self.lane_idx > 0)
                or self.lane_name in ('E76', 'E77', 'E78', 'E82', 'D1'))

    def _apply_lane_change(self, direction):
        if direction and self.lane_name in LANE_CHANGE_EDGES:
            if (self.task == exit_config.TASK_ID and self.exit_maneuver is not None
                    and self.e09_pending_mode != 'off'):
                return
            target = self.lane_idx + direction
            if 0 <= target < self.traci.edge.getLaneNumber(self.lane_name):
                if self.task == exit_config.TASK_ID and self.lane_name in ('E76', 'E77'):
                    if self.e09_pending_mode is None and not self._exit_change_has_room():
                        return
                    now = self.traci.simulation.getTime()
                    self.traci.vehicle.changeLane(
                        self.ego_id, target, exit_config.LANE_CHANGE_TIME_SECONDS)
                    # In E09 off mode this is non-blocking bookkeeping, retained
                    # for the existing urgency and exit-terminal contracts. Do
                    # not restart its clock on a repeated identical request.
                    current = self.exit_maneuver
                    if (current is None or current['edge'] != self.lane_name
                            or current['target'] != target):
                        self.exit_maneuver = {'edge': self.lane_name, 'target': target,
                                              'start_time': now, 'last_progress_time': now,
                                              'stable_steps': 0}
                        if self.exit_reward_profile in (exit_config.DIRECTIONAL_REWARD_PROFILE, *secrm_reward.PROFILES):
                            self.directional_pending_request = int(direction)
                            self.directional_seen_lateral_motion = False
                    return
                self.target_lane_idx = target
                self.traci.vehicle.changeLane(self.ego_id, target, 0)

    def prepare_step(self, action):
        """Apply this vehicle's command without advancing the shared clock."""
        if self.done:
            raise RuntimeError('Call reset() before stepping a finished episode')
        was_on_road = self.ego_on_road
        if self.progress_experiment is not None:
            self.progress_before_remaining = self._remaining_to_d3_end() if was_on_road else None
            self.progress_before_odometer = (float(self.traci.vehicle.getDistance(self.ego_id))
                                             if was_on_road else None)
            self.progress_after_remaining = None
            self.progress_after_odometer = None
        if was_on_road:
            self.lane_id = self.traci.vehicle.getLaneID(self.ego_id)
            self.lane_name = self.traci.vehicle.getRoadID(self.ego_id)
            self.lane_idx = self.traci.vehicle.getLaneIndex(self.ego_id)
            if (self.exit_reward_profile in (exit_config.DIRECTIONAL_REWARD_PROFILE, *secrm_reward.PROFILES)
                    and self.directional_settled_lane is None):
                self._update_directional_lane_reward()
        if self.exit_reward_profile in secrm_reward.PROFILES:
            self.secrm_before = self._secrm_snapshot() if was_on_road else None
            self.boundary_diagnostic = road_boundary.observe(self)
        parsed_action = parseAction(action)
        self.exit_actor_request_direction = (
            int(parsed_action[1]) if was_on_road and self.task == exit_config.TASK_ID
            and self.lane_name in ('E76', 'E77') else 0)
        self.action = action
        if self.runSafetyCheck and was_on_road:
            from .safety_controller import safetyCheck, target_lane_line
            target_line = target_lane_line(self.ego_id, parsed_action[1], connection=self.traci)
            self.action, _ = safetyCheck(action, self.observation, False, 'all', target_line)
            parsed_action = parseAction(self.action)

        if was_on_road:
            self._apply_lane_change(parsed_action[1])
            self.traci.vehicle.setAcceleration(self.ego_id, parsed_action[0], 1)
            if self.traci.vehicle.getSpeed(self.ego_id) > EGO_SPEED_CAP_MPS:
                self.traci.vehicle.setSpeed(self.ego_id, EGO_SPEED_CAP_MPS)
        self.counter += 1
        self.parsed_action = parsed_action
        return was_on_road

    def collect_step(self, was_on_road, close_on_done=True):
        """Read one already-advanced transition; never advance the clock here."""
        terminated = False
        truncated = False
        info = {"message": ""}
        started = self.ego_id in self.traci.simulation.getStartingTeleportIDList()
        ended = self.ego_id in self.traci.simulation.getEndingTeleportIDList()
        self.ego_teleported |= started or ended
        if started:
            self.ego_teleporting = True
        if ended:
            self.ego_teleporting = False
        self.ego_on_road = (self.ego_id in self.traci.vehicle.getIDList()
                            and not self.ego_teleporting)
        self.teleport_transition = started or ended or not was_on_road
        self.missed_exit = False
        previous_road = self.lane_name
        if self.ego_on_road:
            self.lane_id = self.traci.vehicle.getLaneID(self.ego_id)
            self.lane_name = self.traci.vehicle.getRoadID(self.ego_id)
            self.lane_idx = self.traci.vehicle.getLaneIndex(self.ego_id)
            if self.teleport_transition:
                self.traci.vehicle.setLaneChangeMode(self.ego_id, 0)
                self.exit_maneuver = None
            elif self.task == exit_config.TASK_ID:
                self._update_exit_maneuver()
            if (self.task == exit_config.TASK_ID and self.lane_name == 'E77'
                    and self._e77_remaining() <= exit_config.LANE_END_EPSILON_METERS
                    and not self._e77_exit_lane_ready()
                    and not self._e77_maneuver_can_finish()):
                self.missed_exit = True
            if (self.task == exit_config.TASK_ID and previous_road == 'E77'
                    and (self.lane_name == 'E78'
                         or (self.lane_name.startswith(':J106')
                             and self.lane_id != ':J106_0_0'))):
                self.missed_exit = True
            if self.stage_name == 'exit':
                if self.lane_name == 'D3':
                    self.stage_reached = True
            elif self._has_merged():
                self.merged = self.stage_reached = True
        else:
            self._ahead_state = self._behind_state = [0.0, 0.0, 0.0]
            self.exit_maneuver = None

        if self.ego_id in self.traci.simulation.getCollidingVehiclesIDList():
            info['message'] = 'Collision'
            terminated = True
        elif self.progress_experiment is not None and (started or ended or self.ego_teleported):
            info['message'] = 'Ego Teleported'
            terminated = True
        elif self.missed_exit:
            info['message'] = 'Missed Exit'
            terminated = True
        elif self.ego_id in self.traci.simulation.getArrivedIDList():
            info['message'] = 'Arrived'
            terminated = True
        elif self.counter >= self.episode_limit_steps:
            info['message'] = 'Max Steps'
            truncated = True
        elif self.ego_id in self.traci.simulation.getEmergencyStoppingVehiclesIDList():
            info['message'] = 'Emergency Stopping'
            terminated = True
        self.done = terminated or truncated
        self.terminal_reason = info['message'] if self.done else ''
        self._update_directional_lane_reward()
        if (self.progress_experiment is not None and self.ego_on_road
                and not self.teleport_transition and info['message'] != 'Collision'):
            self.progress_after_remaining = self._remaining_to_d3_end()
            self.progress_after_odometer = float(self.traci.vehicle.getDistance(self.ego_id))
        if self.ego_on_road and info['message'] != 'Collision':
            self.observation = self._get_obs()
        else:
            # Teleport buffers have no physical position to query or control.
            self.observation = self.state.copy()
            self.observation_snapshot = None
        self.reward = self.getReward()
        potential_reward = 0.0
        urgency_reward = 0.0
        urgency_value = None
        urgency_provisional = False
        if self.task == exit_config.TASK_ID:
            if self.exit_reward_profile == exit_config.REWARD_PROFILE:
                next_potential = 0.0 if self.done else self._exit_potential()
                potential_reward = (self.potential_gamma * next_potential
                                    - self.potential_previous)
                self.potential_previous = next_potential
                self.reward += potential_reward
            elif self.exit_reward_profile == exit_config.URGENCY_REWARD_PROFILE:
                urgency_value, urgency_provisional = self._reward_urgency()
                urgency_reward = (0.0 if self.done else
                                  exit_config.urgency_penalty(urgency_value))
                self.reward += urgency_reward
                self.reward_components['urgency'] = float(urgency_reward)
            self.reward_components['potential'] = float(potential_reward)
        self.state = self.observation.copy()
        self.last_acc = self.parsed_action[0]
        if self.done and close_on_done:
            try:
                self.traci.close()
            except Exception:
                pass
        
        info.update({'lane': self.lane_id, 'action': self.action, 'merged': bool(self.merged),
                     'reward_components': self.reward_components.copy(),
                     'stage_reached': bool(self.stage_reached), 'stage_name': self.stage_name,
                     'stage_reward': self.last_stage_reward,
                     'stage_reward_paid': bool(self.given_merge_reward),
                     'ego_teleported': bool(self.ego_teleported),
                     'ego_teleporting': bool(self.ego_teleporting),
                     'ego_on_road': bool(self.ego_on_road),
                     'ego_start_lane': self.ego_start_lane,
                     'missed_exit': self.missed_exit,
                     'exit_maneuver_active': self.exit_maneuver is not None,
                     'initial_potential': self.initial_potential,
                     'potential': self.potential_previous,
                     'potential_reward': potential_reward,
                     'urgency_reward': urgency_reward,
                     'urgency_value': urgency_value,
                     'urgency_provisional': urgency_provisional,
                     'exit_reward_profile': (self.exit_reward_profile
                                             if self.task == exit_config.TASK_ID else None),
                     'exit_actor_request_direction': self.exit_actor_request_direction,
                     'directional_lane_change_event': self.directional_lane_event,
                     'secrm_diagnostic': getattr(self, 'secrm_before', None),
                     'road_boundary': getattr(self, 'boundary_diagnostic', None),
                     'teleport_started': started, 'teleport_ended': ended,
                     'is_success': info['message'] == 'Arrived' and not self.ego_teleported})
        return self.observation, float(self.reward), terminated, truncated, info


    def step(self, action):
        was_on_road = self.prepare_step(action)
        self.traci.simulationStep()
        return self.collect_step(was_on_road)

    def render(self):
        # Gymnasium render contract: return frame for rgb_array, None for human
        if self.render_mode == "human":
            # SUMO-GUI handles visualization when using sumo-gui; nothing to draw here
            return None
        return None

    def close(self):
        if self.traci.isLoaded():
            self.traci.close()
