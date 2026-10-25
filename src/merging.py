import os, sys
import hashlib
import math
import numpy as np
from pathlib import Path
# from inference import model
import gymnasium as gym
from gymnasium import spaces
from .utils import *
from .geometry import TrafficSnapshot, driving_axes, world_velocity
from . import exit_e76_d3_config as exit_config


maxSteps = 6000
ANTWERP_DIR = Path(__file__).resolve().parents[1] / 'sumo_new'
EGO_DEPART_TIME = 120.0
EGO_DEPARTURE_DEADLINE = 1200.0
SUMO_STEP_LENGTH = 0.1
LANE_CHANGE_EDGES = frozenset(('E75', 'E76', 'E77', 'E78', 'E82'))
ANTWERP_ENV_VERSION = 2
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
                 scenario='medium', task='legacy_o3_d1', potential_gamma=None):
        if scenario not in ('easy', 'medium', 'hard'):
            raise ValueError(f'Unknown Antwerp scenario: {scenario}')
        if task not in ANTWERP_TASKS:
            raise ValueError(f'Unknown Antwerp task: {task}')
        self.scenario = scenario
        self.task = task
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
        self.ego_start_lane = None
        self.potential_distance_scale = None
        self.potential_previous = 0.0
        self.initial_potential = 0.0
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

    def _e77_remaining(self):
        lane_id = self.traci.vehicle.getLaneID(self.ego_id)
        return max(0.0, self.traci.lane.getLength(lane_id)
                   - self.traci.vehicle.getLanePosition(self.ego_id))

    def _e77_exit_lane_ready(self):
        if self.traci.vehicle.getLaneIndex(self.ego_id) != 0:
            return False
        lane_id = self.traci.vehicle.getLaneID(self.ego_id)
        clearance = (self.traci.lane.getWidth(lane_id)
                     - self.traci.vehicle.getWidth(self.ego_id)) / 2.0
        return (clearance >= 0
                and abs(self.traci.vehicle.getLateralLanePosition(self.ego_id)) <= clearance
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
        lane_id = self.traci.vehicle.getLaneID(self.ego_id)
        clearance = (self.traci.lane.getWidth(lane_id)
                     - self.traci.vehicle.getWidth(self.ego_id)) / 2.0
        settled = (self._exit_maneuver_target_matches() and clearance >= 0
                   and abs(self.traci.vehicle.getLateralLanePosition(self.ego_id)) <= clearance
                   and lateral_speed <= exit_config.LATERAL_SETTLED_SPEED_MPS)
        maneuver['stable_steps'] = maneuver['stable_steps'] + 1 if settled else 0
        if maneuver['stable_steps'] >= exit_config.MANEUVER_STABLE_STEPS:
            self.exit_maneuver = None
        elif (now - maneuver['start_time'] >= exit_config.MANEUVER_MAX_SECONDS
              and lateral_speed <= exit_config.LATERAL_SETTLED_SPEED_MPS):
            # An unfulfilled command expires; a vehicle still moving laterally
            # remains locked until it settles or the exit is missed.
            self.exit_maneuver = None

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
        scene_paths = {'network': ANTWERP_DIR / 'Antwerp.net.xml',
                       'routes': ANTWERP_DIR / f'Antwerp_{self.scenario}{route_suffix}.rou.xml'}
        scene_files = {name: {'path': str(path.relative_to(ANTWERP_DIR.parent)),
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
                        self.potential_distance_scale = sum(
                            self.traci.lane.getLength(f'{edge}_0')
                            for edge in exit_config.DISTANCE_SCALE_EDGES)
                        self.potential_previous = self._exit_potential()
                        self.initial_potential = self.potential_previous
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
            'episode_limit_steps': maxSteps, 'episode_clock': 'steps_since_takeover',
            'ego_route': list(self.traci.vehicle.getRoute(self.ego_id)),
            'ego_speed_mode': self.traci.vehicle.getSpeedMode(self.ego_id),
            'ego_lane_change_mode': self.traci.vehicle.getLaneChangeMode(self.ego_id),
            'ego_start_lane': self.ego_start_lane,
            'initial_potential': self.initial_potential,
            'potential_distance_scale': self.potential_distance_scale,
        }

    def getReward(self):

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
            if self.task == exit_config.TASK_ID and self.exit_maneuver is not None:
                return
            target = self.lane_idx + direction
            if 0 <= target < self.traci.edge.getLaneNumber(self.lane_name):
                if self.task == exit_config.TASK_ID and self.lane_name in ('E76', 'E77'):
                    if not self._exit_change_has_room():
                        return
                    now = self.traci.simulation.getTime()
                    self.traci.vehicle.changeLane(
                        self.ego_id, target, exit_config.LANE_CHANGE_TIME_SECONDS)
                    self.exit_maneuver = {'edge': self.lane_name, 'target': target,
                                          'start_time': now, 'last_progress_time': now,
                                          'stable_steps': 0}
                    return
                self.target_lane_idx = target
                self.traci.vehicle.changeLane(self.ego_id, target, 0)

    def prepare_step(self, action):
        """Apply this vehicle's command without advancing the shared clock."""
        if self.done:
            raise RuntimeError('Call reset() before stepping a finished episode')
        was_on_road = self.ego_on_road
        if was_on_road:
            self.lane_id = self.traci.vehicle.getLaneID(self.ego_id)
            self.lane_name = self.traci.vehicle.getRoadID(self.ego_id)
            self.lane_idx = self.traci.vehicle.getLaneIndex(self.ego_id)
        parsed_action = parseAction(action)
        self.action = action
        if self.runSafetyCheck and was_on_road:
            from .safety_controller import safetyCheck, target_lane_line
            target_line = target_lane_line(self.ego_id, parsed_action[1], connection=self.traci)
            self.action, _ = safetyCheck(action, self.observation, False, 'all', target_line)
            parsed_action = parseAction(self.action)

        if was_on_road:
            self._apply_lane_change(parsed_action[1])
            self.traci.vehicle.setAcceleration(self.ego_id, parsed_action[0], 1)
            if self.traci.vehicle.getSpeed(self.ego_id) > 32:
                self.traci.vehicle.setSpeed(self.ego_id, 32)
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
        elif self.missed_exit:
            info['message'] = 'Missed Exit'
            terminated = True
        elif self.ego_id in self.traci.simulation.getArrivedIDList():
            info['message'] = 'Arrived'
            terminated = True
        elif self.counter >= maxSteps:
            info['message'] = 'Max Steps'
            truncated = True
        elif self.ego_id in self.traci.simulation.getEmergencyStoppingVehiclesIDList():
            info['message'] = 'Emergency Stopping'
            terminated = True
        self.done = terminated or truncated
        if self.ego_on_road and info['message'] != 'Collision':
            self.observation = self._get_obs()
        else:
            # Teleport buffers have no physical position to query or control.
            self.observation = self.state.copy()
            self.observation_snapshot = None
        self.reward = self.getReward()
        potential_reward = 0.0
        if self.task == exit_config.TASK_ID:
            next_potential = 0.0 if self.done else self._exit_potential()
            potential_reward = (self.potential_gamma * next_potential
                                - self.potential_previous)
            self.potential_previous = next_potential
            self.reward += potential_reward
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
