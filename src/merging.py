import math
import os, sys
import numpy as np
from pathlib import Path
# from inference import model
import gymnasium as gym
from gymnasium import spaces
from .utils import *


maxSteps = 6000
ANTWERP_DIR = Path(__file__).resolve().parents[1] / 'sumo_new'
EGO_DEPART_TIME = 120.0
EGO_DEPARTURE_DEADLINE = 1200.0
SUMO_STEP_LENGTH = 0.1
LANE_CHANGE_EDGES = frozenset(('E75', 'E76', 'E77', 'E78', 'E82'))

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

def SetTrafficBehavior(vehicle_name):
    if 'f_1' in vehicle_name:
            traci.vehicle.setTau(vehicle_name, np.random.uniform(0.1, 0.7))
            traci.vehicle.setMaxSpeed(vehicle_name, np.random.uniform(10, 13))
            traci.vehicle.setSpeedMode(vehicle_name, 32)
    elif 'f_2' in vehicle_name:
        traci.vehicle.setTau(vehicle_name, np.random.normal(0.7, 0.1))
        traci.vehicle.setMaxSpeed(vehicle_name, np.random.uniform(8, 11))
    elif vehicle_name == 'ego':
        traci.vehicle.setLaneChangeMode(vehicle_name, 0)



class Merging(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 10}
    def __init__(self, seed, mode='all', radius = 50, render_mode= None, runSafetyCheck=False, scenario='medium'):
        if scenario not in ('easy', 'medium', 'hard'):
            raise ValueError(f'Unknown Antwerp scenario: {scenario}')
        self.scenario = scenario
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
        self.lane_id = ''
        self.lane_name = ''
        self.lane_idx = 0
        self.parsed_action = None
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

    

    def reset(self, seed=None, options=None):
        # Seed the environment RNG per Gymnasium API
        print('Resetting environment', 'radius', self.radius)
        initial_seed = self.seed if not self._seed_initialized and seed is None else seed
        super().reset(seed=initial_seed)
        self._seed_initialized = True
        sumo_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.merged = False
        self.given_merge_reward = False
        self.done = False
        self.ego_inserted = False
        self.ego_id = None
        self.ego_depart_time = None
        self.counter = 0
        self.last_acc = 0
        self._ahead_state = []
        self._behind_state = []
       

        # Close any existing SUMO connection
        try:
            traci.close()
        except Exception:
            pass

        # Load the Antwerp network and one scenario route without detector output files.
        sumo_cmd = [
            self.sumoBinary,
            '--net-file', str(ANTWERP_DIR / 'Antwerp.net.xml'),
            '--route-files', str(ANTWERP_DIR / f'Antwerp_{self.scenario}.rou.xml'),
            '--begin', '0', '--end', '3600',
            '--step-length', '0.1', '--lateral-resolution', '0.8',
            '--seed', str(sumo_seed),
            '--no-step-log', '--collision.check-junctions',
            '--collision.action', 'remove', '-Q',
        ]
        traci.start(sumo_cmd)

        # Stop one step before the scheduled departure so TraCI observes its ID.
        traci.simulationStep(EGO_DEPART_TIME - SUMO_STEP_LENGTH)
        while not self.ego_inserted and traci.simulation.getTime() < EGO_DEPARTURE_DEADLINE:
            traci.simulationStep()
            for vehicle_name in traci.simulation.getDepartedIDList():
                SetTrafficBehavior(vehicle_name)
                if vehicle_name == 'ego':
                    if traci.vehicle.getRoadID(vehicle_name) != 'O3':
                        traci.close()
                        raise RuntimeError('The scheduled ego did not enter on O3')
                    self.ego_id = vehicle_name
                    self.ego_depart_time = traci.simulation.getTime()
                    self.ego_inserted = True
                    self.observation = np.asarray(self._get_obs(), dtype=np.float32)
        if not self.ego_inserted:
            traci.close()
            raise RuntimeError(f'The scheduled ego did not enter O3 before {EGO_DEPARTURE_DEADLINE}s in {self.scenario}')
        
        self.state = self.observation
        return self.state, {'ego_id': self.ego_id, 'ego_depart_time': self.ego_depart_time}

    def getReward(self):

        # print(self.egoIdx)
        # print('action', action, 'safe_action', safe_action)
        # print(self.state[self.egoIdx])
        # Use ego x-position delta only to produce a scalar distance traveled
        distance_traveled = float(self.observation[self.egoIdx, 0] - self.state[self.egoIdx, 0])
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
        wrong_lane_change_penalty = 0.0
        max_steps_penalty = 0.0

        if self.ego_id in traci.simulation.getCollidingVehiclesIDList():
            print("Colliding")
            collision_penalty = 1000
            r -= collision_penalty
        elif self.ego_id in traci.simulation.getArrivedIDList():
            print("Arrived")
            arrived_reward = 500
            r += arrived_reward
        elif self.ego_id in traci.simulation.getEmergencyStoppingVehiclesIDList():
            print("Emergency Stopping")
            emergency_stopping_penalty = 1000
            r -= emergency_stopping_penalty
        elif self.merged and not self.given_merge_reward:
            print("Merge reward given.")
            self.given_merge_reward = True
            merge_reward = 500
            r += merge_reward
        elif self.counter >= maxSteps:
            print("Max Steps")
            max_steps_penalty = 1000
            r -= max_steps_penalty
        elif self.parsed_action[1] != 0:
            # print("Wrong lane change")
            wrong_lane_change_penalty = 5
            r -= wrong_lane_change_penalty

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
        #       f"\n  wrong_lane_change_penalty: {-wrong_lane_change_penalty}"
        #       f"\n  max_steps_penalty: {-max_steps_penalty}"
        #       f"\n  TOTAL (BEFORE SCALING): {r}")

        # print("reward", r, "action", action, "lane", self.lane_id, "merged", self.merged)
        return 0.1*r    
    
    def _get_obs(self):

        # Express surrounding vehicles in the ego's local driving direction.
        egoPos = traci.vehicle.getPosition(self.ego_id)
        egoVel = traci.vehicle.getSpeed(self.ego_id)
        angle = math.radians(traci.vehicle.getAngle(self.ego_id))
        forward = (math.sin(angle), math.cos(angle))
        left = (-forward[1], forward[0])
        ego = [[traci.vehicle.getDistance(self.ego_id),
                traci.vehicle.getLaneIndex(self.ego_id), egoVel]]

        # Get surrounding vehicles pos and vel
        intruders = []
        self._ahead_state = [] # closest ahead vehicle state
        self._behind_state = [] # closest behind vehicle state
        closest_ahead = None
        closest_behind = None
        min_ahead_dist = float('inf')
        min_behind_dist = float('inf')
        for vehID in traci.vehicle.getIDList():
            if vehID == self.ego_id:
                continue
            vehPos = traci.vehicle.getPosition(vehID)
            distance = getDistance(vehPos, egoPos) # distance between vehicle and ego everything is absolute coordinates.
            if distance <= self.radius:
                vehVel = traci.vehicle.getSpeed(vehID)
                dx, dy = vehPos[0] - egoPos[0], vehPos[1] - egoPos[1]
                relState = [dx * forward[0] + dy * forward[1],
                            dx * left[0] + dy * left[1], vehVel - egoVel]
                # print('Distance is less than radius', distance, 'for vehicle', vehID, 'relState', relState)
                intruders.append(relState)
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
        intruders = sorted(intruders, key=lambda x: x[0] ** 2 + x[1] ** 2)[:self.num_entities - 1]
        intruders = sorted(intruders, key=lambda x: (x[1], x[0]))
        
    
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
            target = self.lane_idx + direction
            if 0 <= target < traci.edge.getLaneNumber(self.lane_name):
                self.target_lane_idx = target
                traci.vehicle.changeLane(self.ego_id, target, 0)

    def step(self, action):
        terminated = False
        truncated = False
        info = {'message': ''}
        self.lane_id = traci.vehicle.getLaneID(self.ego_id)
        self.lane_name = traci.vehicle.getRoadID(self.ego_id)
        self.lane_idx = traci.vehicle.getLaneIndex(self.ego_id)


        parsed_action = parseAction(action)
        self.action = action
        if self.runSafetyCheck:
            from .safety_controller import safetyCheck, target_lane_line
            target_line = target_lane_line(self.ego_id, parsed_action[1])
            self.action, _ = safetyCheck(action, self.observation, False, 'all', target_line)
            parsed_action = parseAction(self.action)

        if not self.done:
            self._apply_lane_change(parsed_action[1])

            for vehicle_name in traci.simulation.getDepartedIDList():
                SetTrafficBehavior(vehicle_name)
            traci.vehicle.setAcceleration(self.ego_id, parsed_action[0], 1)
            if traci.vehicle.getSpeed(self.ego_id) > 32:
                traci.vehicle.setSpeed(self.ego_id, 32)
            self.counter += 1
            traci.simulationStep()

            if self.ego_id in traci.vehicle.getIDList():
                self.lane_id = traci.vehicle.getLaneID(self.ego_id)
                self.lane_name = traci.vehicle.getRoadID(self.ego_id)
                self.lane_idx = traci.vehicle.getLaneIndex(self.ego_id)
            self.parsed_action = parsed_action
            if self._has_merged() and not self.merged:
                print('Merged')
                self.merged = True

            if self.ego_id in traci.simulation.getCollidingVehiclesIDList():
                info['message'] = 'Collision'
                terminated = True
                self.done = True
                self.observation = self.state
            elif self.ego_id in traci.simulation.getArrivedIDList():
                terminated = True
                self.done = True
                self.observation = self.state
                info['message'] = 'Arrived'
            elif self.counter >= maxSteps:
                print('Terminating episode since it exceeded maximum time.')
                info['message'] = 'Max Steps'
                truncated = True
                self.done = True
                self.observation = np.asarray(self._get_obs(), dtype=np.float32)
                
            elif self.ego_id in traci.simulation.getEmergencyStoppingVehiclesIDList():
                self.done = True
                terminated = True
                info['message'] = 'Emergency Stopping'
                self.observation = np.asarray(self._get_obs(), dtype=np.float32)
                
            else:
                self.observation = np.asarray(self._get_obs(), dtype=np.float32)
               
        self.reward = self.getReward()
        self.state = self.observation.copy()
        self.last_acc = self.parsed_action[0]
        if self.done:
            try:
                traci.close()
            except Exception:
                pass
        
        info.update({'lane': self.lane_id, 'action': action, 'merged': bool(self.merged)})
        return self.observation, float(self.reward), terminated, truncated, info


    def render(self):
        # Gymnasium render contract: return frame for rgb_array, None for human
        if self.render_mode == "human":
            # SUMO-GUI handles visualization when using sumo-gui; nothing to draw here
            return None
        return None

    def close(self):
        if traci.isLoaded():
            traci.close()
