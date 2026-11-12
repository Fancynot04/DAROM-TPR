"""Read-only per-step trajectory recorder for E76->D3 evaluation.

One ``trajectory_steps.csv`` row and one ``trajectory_transitions.csv`` row are
written per ``env.step`` call. The recorder only READS TraCI / environment
state; it never calls a setter, never touches an RNG and never mutates the
arrays it is given. Unreadable quantities are written as empty cells.

Time convention for one row (transition t):
  * ``*_before`` / un-suffixed state fields are read BEFORE ``env.step`` (state_t)
  * ``*_after`` fields describe state_{t+1}; they are read from TraCI on
    non-terminal steps. ``Merging.collect_step`` closes the TraCI connection on
    the terminal step, so on that step the after-state comes from the
    environment's own cached observation when the Ego is still on the road and
    is empty when the Ego was removed (Arrived / Collision / Teleport).
"""

import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .route_urgency import observe_route_urgency

SCHEMA_VERSION = 'trajectory-csv-v1'
OBS_DIM = 134
ACTION_DIM = 2
TTC_MIN_CLOSING_SPEED_MPS = 0.1
NEIGHBOR_SEARCH_M = 150.0
STEPS_FILE = 'trajectory_steps.csv'
TRANSITIONS_FILE = 'trajectory_transitions.csv'
SUMMARY_FILE = 'episode_summary.csv'
MANIFEST_FILE = 'trajectory_manifest.json'
COLUMNS_FILE = 'trajectory_columns.json'

OBS_LAYOUT = {
    'entities_0_92': '31 entities x [a, b, c]; entity 0 = Ego [odometer_m, lane_index, speed_mps]; '
                     '1..30 = neighbours [rel_forward_m, rel_left_m, rel_speed_mps] (delayed packet)',
    'action_history_93_132': '20 x [acc_cmd, lane_cmd_raw] executed-action history, masked beyond the delay',
    'delay_133': 'delay of the received observation packet [control steps]',
}

# (name, unit, description) for the fixed part of trajectory_steps.csv
FIXED_COLUMNS = [
    ('trajectory_id', '', 'unique id: <ckpt8>_<scenario>_tr<train_seed>_ev<eval_seed>'),
    ('model_label', '', 'user label of the checkpoint'),
    ('checkpoint_sha256_12', '', 'first 12 hex chars of checkpoint SHA-256'),
    ('training_seed', '', 'training seed stored in the checkpoint'),
    ('eval_seed', '', 'Merging.reset seed of this episode (run seed + episode index)'),
    ('scenario', '', 'easy|medium|hard'),
    ('step', 'index', '0-based index of the env.step call (transition t -> t+1)'),
    ('sim_time_s', 's', 'SUMO time before the step'),
    ('sim_time_after_s', 's', 'SUMO time after the step; empty on the terminal step (TraCI closed)'),
    ('ego_on_road', 'bool', 'Ego present and controllable before the step'),
    ('ego_road_id', '', 'SUMO edge id before the step (":..." = junction connector)'),
    ('ego_lane_id', '', 'SUMO lane id before the step'),
    ('ego_lane_index', 'index', 'lane index (0 = rightmost / exit side) before the step'),
    ('ego_lane_pos_m', 'm', 'longitudinal position along the lane (Ego front reference) before the step'),
    ('ego_x_m', 'm', 'SUMO network x of Ego reference point before the step'),
    ('ego_y_m', 'm', 'SUMO network y of Ego reference point before the step'),
    ('ego_heading_deg', 'deg', 'SUMO angle (0=north, clockwise) before the step'),
    ('ego_lateral_pos_m', 'm', 'lateral offset from lane centre before the step'),
    ('ego_lateral_speed_mps', 'm/s', 'lateral speed before the step'),
    ('ego_odometer_m', 'm', 'distance driven since insertion before the step'),
    ('ego_speed_mps', 'm/s', 'speed before the step'),
    ('ego_accel_mps2', 'm/s^2', 'SUMO-measured acceleration over the PREVIOUS step (value at state_t)'),
    ('route_remaining_to_d3_end_m', 'm', 'route driving distance to the end of D3 (env._remaining_to_d3_end)'),
    ('route_remaining_to_d3_start_m', 'm', 'route driving distance to the D3 entrance; empty on D3'),
    ('route_status', '', 'route_urgency status: ok|unreachable|at_destination|unmapped_connector|off_route|error'),
    ('exit_reachable', 'bool', '1 if a lane-connection path to D3 exists from the current lane (net.xml connections); empty if not determinable'),
    ('min_lane_changes_required', 'count', 'fewest lane changes to reach D3 along the Ego route (lane graph); empty if not determinable'),
    ('direct_exit_lane', 'bool', '1 if the current lane connects directly to D3 (needs 0 further changes at the edge end)'),
    ('route_state_settled', 'bool', '1 if lateral motion settled and no pending maneuver (estimate valid for thresholds)'),
    ('route_urgency', 'ratio', 'lane_graph_minimax_v1 urgency; optimistic, not a probability'),
    ('action_policy_acc_raw', 'raw[-5,5]', 'SAC output component 0 (before any wrapper)'),
    ('action_policy_lane_raw', 'raw[-5,5]', 'SAC output component 1 (before any wrapper)'),
    ('cmd_acc_mps2', 'm/s^2', 'parseAction acceleration (clipped to [-5,5])'),
    ('cmd_lane_direction', '-1|0|1', 'parseAction lane decision on the policy action; -1 = toward lane idx-1'),
    ('action_exec_acc_raw', 'raw', 'action that reached Merging.prepare_step (info["action"]; differs only with --safety)'),
    ('action_exec_lane_raw', 'raw', 'idem, component 1'),
    ('action_modified_by_wrapper', 'bool', '1 if executed action != policy action'),
    ('lc_request_e76_e77', '-1|0|1', 'info.exit_actor_request_direction: lane request counted on E76/E77 only'),
    ('lc_command_sent', '-1|0|1', 'info.exit_applied_lane_change_direction: changeLane actually issued (E76/E77, target lane exists)'),
    ('lc_target_lane', 'index', 'target lane index of the issued command; empty if none'),
    ('exit_maneuver_active_after', 'bool', 'env bookkeeping: lane-change maneuver unsettled after the step'),
    ('lane_change_physical', 'bool', '1 if road id is unchanged and lane index changed between before/after state'),
    ('leader_id', '', 'TraCI getLeader id within 150 m'),
    ('leader_gap_m', 'm', 'bumper-to-bumper gap to leader (TraCI gap + ego minGap)'),
    ('leader_speed_mps', 'm/s', 'leader speed'),
    ('follower_id', '', 'TraCI getFollower id within 150 m'),
    ('follower_gap_m', 'm', 'bumper-to-bumper gap to follower (TraCI gap + follower minGap)'),
    ('follower_speed_mps', 'm/s', 'follower speed'),
    ('ttc_leader_s', 's', 'gap/(v_ego-v_leader) when closing > 0.1 m/s and gap>=0; else empty'),
    ('ttc_leader_status', '', 'ok|no_leader|not_closing|overlap|unavailable'),
    ('ttc_follower_s', 's', 'gap/(v_follower-v_ego) when closing > 0.1 m/s and gap>=0; else empty'),
    ('ttc_follower_status', '', 'ok|no_follower|not_closing|overlap|unavailable'),
    ('ego_road_id_after', '', 'edge id after the step'),
    ('ego_lane_index_after', 'index', 'lane index after the step'),
    ('ego_speed_after_mps', 'm/s', 'speed after the step'),
    ('ego_odometer_after_m', 'm', 'odometer after the step'),
    ('ego_accel_after_mps2', 'm/s^2', 'SUMO-measured acceleration of this step; empty on the terminal step'),
    ('jerk_measured_mps3', 'm/s^3', '(accel_after - accel_before)/0.1; empty on step 0 and terminal step'),
    ('jerk_cmd_mps3', 'm/s^3', '(cmd_acc_t - cmd_acc_{t-1})/0.1; empty on step 0'),
    ('ego_on_road_after', 'bool', 'env.ego_on_road after the step (0 = removed or teleporting)'),
    ('post_state_source', '', 'traci|env_cache|unavailable'),
    ('hard_braking_this_step', 'bool', 'post-step accel <= -4.5 m/s^2 (env diagnostic)'),
    ('sumo_emergency_stopping_this_step', 'bool', 'SUMO emergency-stopping event for Ego'),
    ('teleport_started', 'bool', 'info.teleport_started'),
    ('teleport_ended', 'bool', 'info.teleport_ended'),
    ('ego_teleported_cum', 'bool', 'Ego teleported at any time so far'),
    ('terminated', 'bool', 'value returned by env.step (NOTE: Timeout is reported as terminated, not truncated)'),
    ('truncated', 'bool', 'value returned by env.step'),
    ('is_terminal_step', 'bool', 'terminated or truncated'),
    ('terminal_reason', '', 'info.message on the terminal step, empty otherwise'),
    ('ego_removed_from_sim', 'bool', 'Ego was on the road before the step and not after it'),
    ('reward_paid', 'reward', 'reward returned by env.step (checkpoint reward configuration)'),
    ('return_cum', 'reward', 'running sum of reward_paid in this episode'),
    ('reward_terms_sum_abs_err', 'reward', '|sum(weighted terms) - reward_paid|'),
    ('reward_terms_json', '', 'full info.reward_terms list'),
    ('route_diag_error', '', 'non-empty if a diagnostic read failed (field left empty)'),
]
TERM_SUFFIXES = (('raw', 'reward'), ('weighted', 'reward'), ('active', 'bool'))

SUMMARY_FIELDS = [
    'trajectory_id', 'model_label', 'checkpoint_sha256_12', 'training_seed', 'eval_seed', 'scenario',
    'steps', 'terminal_reason', 'success', 'terminated', 'truncated', 'policy_return',
    'route_remaining_to_d3_end_initial_m', 'route_remaining_to_d3_end_last_m',
    'route_remaining_to_d3_start_initial_m', 'route_remaining_to_d3_start_last_m',
    'exit_reachable_known_steps', 'exit_reachable_true_steps', 'exit_reachable_false_steps',
    'min_lane_changes_required_min', 'min_lane_changes_required_max', 'direct_exit_lane_steps',
    'route_urgency_max', 'route_diag_error_rows',
    'lane_requests_e76_e77', 'lane_change_commands_sent', 'physical_lane_changes',
    'hard_braking_steps', 'sumo_emergency_stopping_steps',
    'leader_ttc_min_s', 'follower_ttc_min_s', 'leader_ttc_below_2s_steps',
    'follower_ttc_below_2s_steps',
    'efficiency_raw_mean', 'efficiency_weighted_sum',
    'route_raw_mean', 'route_weighted_sum',
]


def _fmt(value):
    if value is None:
        return ''
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return '' if not math.isfinite(value) else format(value, '.10g')
    return value


def _vec(values):
    return [format(float(x), '.9g') for x in values]  # round-trips float32


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def ttc(gap_m, closing_mps):
    """Effective TTC: finite only for a non-overlapping, actually closing pair."""
    if gap_m is None or closing_mps is None:
        return None, 'unavailable'
    if gap_m < 0:
        return None, 'overlap'
    if closing_mps <= TTC_MIN_CLOSING_SPEED_MPS:
        return None, 'not_closing'
    return gap_m / closing_mps, 'ok'


def _git(args, cwd):
    try:
        return subprocess.check_output(['git', *args], cwd=cwd, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


class TrajectoryRecorder:
    def __init__(self, out_dir, *, model_label, checkpoint, checkpoint_sha256,
                 training_seed, scenario, reward_config, route_network,
                 lane_change_edges, parse_action, destination='D3',
                 lane_change_seconds=4.0, step_length=0.1,
                 hard_braking_threshold=-4.5, config=None, run_label=None):
        self.observation_mode = (config or {}).get('observation_mode', 'legacy')
        dimensions = {'legacy': 134, 'ego_lateral_v1': 136, 'ego_spatial_v2': 137, 'ego_spatial_onehot_v3': 143}
        if self.observation_mode not in dimensions:
            raise ValueError(f'Unknown observation mode: {self.observation_mode}')
        self.obs_dim = dimensions[self.observation_mode]
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=False)  # never reuse a directory
        self.model_label, self.checkpoint = model_label, str(checkpoint)
        self.checkpoint_sha256 = checkpoint_sha256
        self.training_seed, self.scenario = training_seed, scenario
        self.network, self.lane_change_edges = route_network, frozenset(lane_change_edges)
        self.parse_action, self.destination = parse_action, destination
        self.lane_change_seconds, self.dt = lane_change_seconds, step_length
        self.hard_braking_threshold = hard_braking_threshold
        self.config, self.run_label = dict(config or {}), run_label
        self.reward_config = reward_config
        self.term_names = list(reward_config['terms'])
        self.term_columns = [(f'rt_{name}_{suffix}', unit)
                             for name in self.term_names for suffix, unit in TERM_SUFFIXES]
        self.step_fields = [c[0] for c in FIXED_COLUMNS] + [c[0] for c in self.term_columns]
        self.obs_fields = [f'obs_{i:03d}' for i in range(self.obs_dim)]
        self.next_obs_fields = [f'next_obs_{i:03d}' for i in range(self.obs_dim)]
        self.transition_fields = (
            ['trajectory_id', 'step'] + self.obs_fields
            + ['action_policy_0', 'action_policy_1', 'action_exec_0', 'action_exec_1']
            + self.next_obs_fields
            + ['reward', 'terminated', 'truncated', 'is_terminal_step', 'terminal_reason',
               'next_obs_valid', 'next_obs_note'])
        self._steps_file = (self.out_dir / STEPS_FILE).open('x', newline='')
        self._trans_file = (self.out_dir / TRANSITIONS_FILE).open('x', newline='')
        self._summary_file = (self.out_dir / SUMMARY_FILE).open('x', newline='')
        self._steps = csv.DictWriter(self._steps_file, fieldnames=self.step_fields, restval='')
        self._trans = csv.DictWriter(self._trans_file, fieldnames=self.transition_fields, restval='')
        self._summary = csv.DictWriter(self._summary_file, fieldnames=SUMMARY_FIELDS, restval='')
        self._steps.writeheader()
        self._trans.writeheader()
        self._summary.writeheader()
        self.episodes = []
        self.errors = Counter()
        self.unknown_terms = Counter()
        self._episode = None
        self._started = time.time()
        self._closed = False

    # ---------------------------------------------------------------- episode
    def begin_episode(self, *, run_seed, episode_index, eval_seed, reset_info):
        if self._episode is not None:
            raise RuntimeError('Previous trajectory episode was not ended')
        tid = (f'{(self.checkpoint_sha256 or "nockpt")[:8]}_{self.scenario}'
               f'_tr{self.training_seed}_ev{eval_seed}')
        self._episode = {
            'trajectory_id': tid, 'eval_seed': eval_seed, 'run_seed': run_seed,
            'episode_index': episode_index, 'rows': 0, 'return': 0.0, 'prev_cmd_acc': None,
            'last_row': None, 'prev_obs_hash': None, 'chain_breaks': 0,
            'summary_stats': _summary_stats(),
            'sumo_seed': reset_info.get('sumo_seed'), 'sumo_version': reset_info.get('sumo_version'),
            'ego_route': reset_info.get('ego_route'), 'ego_start_lane': reset_info.get('ego_start_lane'),
            'scene_files': reset_info.get('scene_files'),
            'ego_speed_mode': reset_info.get('ego_speed_mode'),
            'ego_lane_change_mode': reset_info.get('ego_lane_change_mode'),
        }

    def end_episode(self, outcome, steps):
        ep = self._episode
        if ep is None or ep['last_row'] is None:
            raise RuntimeError('No trajectory rows recorded for this episode')
        last = ep['last_row']
        if ep['rows'] != steps or last['terminal_reason'] != outcome or not last['is_terminal_step']:
            raise RuntimeError(f'Trajectory/evaluation mismatch: rows={ep["rows"]} steps={steps} '
                               f'last_reason={last["terminal_reason"]!r} outcome={outcome!r}')
        self._steps_file.flush()
        self._trans_file.flush()
        summary = _episode_summary(ep, self.model_label, self.checkpoint_sha256, self.training_seed,
                                   self.scenario)
        self._summary.writerow({key: _fmt(summary.get(key)) for key in SUMMARY_FIELDS})
        self._summary_file.flush()
        self.episodes.append({k: ep[k] for k in (
            'trajectory_id', 'run_seed', 'episode_index', 'eval_seed', 'sumo_seed',
            'sumo_version', 'ego_route', 'ego_start_lane', 'scene_files', 'ego_speed_mode',
            'ego_lane_change_mode', 'chain_breaks')}
            | {'rows': ep['rows'], 'outcome': outcome, 'return_paid': ep['return'], 'complete': True})
        self._episode = None

    # ------------------------------------------------------------- read state
    def _try(self, row, name, fn):
        try:
            return fn()
        except Exception as error:  # diagnostics must never alter evaluation
            self.errors[name] += 1
            row['route_diag_error'] = (row.get('route_diag_error', '') + f'{name}:{type(error).__name__};')
            return None

    def capture_pre(self, base, observation, action):
        """Read state_t. Call immediately before env.step(action); read-only."""
        ep = self._episode
        pre = {'obs': np.array(observation, dtype=np.float32, copy=True),
               'action': np.array(action, dtype=np.float32, copy=True),
               'step': ep['rows'], 'on_road': bool(base.ego_on_road and not base.ego_teleporting),
               'row': {}}
        if pre['obs'].shape != (self.obs_dim,) or pre['action'].shape != (ACTION_DIM,):
            raise ValueError(f'Unexpected obs/action shape: {pre["obs"].shape}, {pre["action"].shape}')
        row = pre['row']
        parsed = self.parse_action(pre['action'])
        row['cmd_acc_mps2'], row['cmd_lane_direction'] = float(parsed[0]), int(parsed[1])
        if not pre['on_road']:
            return pre
        t, ego = base.traci, base.ego_id
        v = t.vehicle

        def read(name, fn):
            return self._try(row, name, fn)
        row['sim_time_s'] = read('time', t.simulation.getTime)
        row['ego_road_id'] = read('road', lambda: v.getRoadID(ego))
        row['ego_lane_id'] = read('lane_id', lambda: v.getLaneID(ego))
        row['ego_lane_index'] = read('lane_idx', lambda: int(v.getLaneIndex(ego)))
        row['ego_lane_pos_m'] = read('lane_pos', lambda: float(v.getLanePosition(ego)))
        xy = read('position', lambda: v.getPosition(ego))
        if xy is not None:
            row['ego_x_m'], row['ego_y_m'] = float(xy[0]), float(xy[1])
        row['ego_heading_deg'] = read('angle', lambda: float(v.getAngle(ego)))
        row['ego_lateral_pos_m'] = read('lat_pos', lambda: float(v.getLateralLanePosition(ego)))
        row['ego_lateral_speed_mps'] = read('lat_speed', lambda: float(v.getLateralSpeed(ego)))
        row['ego_odometer_m'] = read('odometer', lambda: float(v.getDistance(ego)))
        row['ego_speed_mps'] = read('speed', lambda: float(v.getSpeed(ego)))
        row['ego_accel_mps2'] = read('accel', lambda: float(v.getAcceleration(ego)))
        row['route_remaining_to_d3_end_m'] = read('rem_end', base._remaining_to_d3_end)
        road = row.get('ego_road_id')
        if road is not None and road != self.destination:
            row['route_remaining_to_d3_start_m'] = read('rem_start', lambda: _finite_nonneg(
                v.getDrivingDistance(ego, self.destination, 0.0)))
        estimate = read('route_urgency', lambda: observe_route_urgency(
            base, self.network, self.lane_change_edges,
            lane_change_seconds=self.lane_change_seconds))
        if estimate is None:
            row['route_status'] = 'error'
        else:
            status = estimate.get('status')
            row['route_status'] = status
            if status == 'ok':
                row['exit_reachable'] = 1
            elif status == 'unreachable':
                row['exit_reachable'] = 0
            elif status == 'at_destination':
                row['exit_reachable'] = 1
            row['min_lane_changes_required'] = estimate.get('minimum_lane_changes')
            row['route_state_settled'] = estimate.get('valid_for_threshold')
            row['route_urgency'] = estimate.get('urgency')
        if road in self.network.lanes and row.get('ego_lane_index') is not None:
            row['direct_exit_lane'] = bool(self.network.connections.get(
                (road, self.destination, row['ego_lane_index'])))
        self._neighbors(row, v, ego)
        return pre

    def _neighbors(self, row, v, ego):
        ego_speed = row.get('ego_speed_mps')
        leader = self._try(row, 'leader', lambda: v.getLeader(ego, NEIGHBOR_SEARCH_M))
        if leader and leader[0]:
            row['leader_id'] = leader[0]
            min_gap = self._try(row, 'ego_min_gap', lambda: float(v.getMinGap(ego)))
            speed = self._try(row, 'leader_speed', lambda: float(v.getSpeed(leader[0])))
            gap = None if min_gap is None else float(leader[1]) + min_gap
            row['leader_gap_m'], row['leader_speed_mps'] = gap, speed
            closing = None if None in (ego_speed, speed) else ego_speed - speed
            row['ttc_leader_s'], row['ttc_leader_status'] = ttc(gap, closing)
        else:
            row['ttc_leader_status'] = 'no_leader' if leader is not None else 'unavailable'
        follower = self._try(row, 'follower', lambda: v.getFollower(ego, NEIGHBOR_SEARCH_M))
        if follower and follower[0]:
            row['follower_id'] = follower[0]
            min_gap = self._try(row, 'follower_min_gap', lambda: float(v.getMinGap(follower[0])))
            speed = self._try(row, 'follower_speed', lambda: float(v.getSpeed(follower[0])))
            gap = None if min_gap is None else float(follower[1]) + min_gap
            row['follower_gap_m'], row['follower_speed_mps'] = gap, speed
            closing = None if None in (ego_speed, speed) else speed - ego_speed
            row['ttc_follower_s'], row['ttc_follower_status'] = ttc(gap, closing)
        else:
            row['ttc_follower_status'] = 'no_follower' if follower is not None else 'unavailable'

    # ------------------------------------------------------------ write a row
    def record_step(self, pre, base, next_observation, reward, terminated, truncated, info):
        """Write the row for the transition just executed. Call right after env.step."""
        ep = self._episode
        row = pre['row']
        terminal = bool(terminated or truncated)
        next_obs = np.asarray(next_observation, dtype=np.float32)
        if next_obs.shape != (self.obs_dim,):
            raise ValueError(f'Unexpected next_obs shape: {next_obs.shape}')
        obs_hash = hashlib.sha1(pre['obs'].tobytes()).hexdigest()
        if ep['prev_obs_hash'] is not None and ep['prev_obs_hash'] != obs_hash:
            ep['chain_breaks'] += 1  # obs_t must equal next_obs_{t-1}
        ep['prev_obs_hash'] = hashlib.sha1(next_obs.tobytes()).hexdigest()

        row.update(trajectory_id=ep['trajectory_id'], model_label=self.model_label,
                   checkpoint_sha256_12=(self.checkpoint_sha256 or '')[:12],
                   training_seed=self.training_seed, eval_seed=ep['eval_seed'],
                   scenario=self.scenario, step=pre['step'], ego_on_road=pre['on_road'])
        raw = pre['action']
        row['action_policy_acc_raw'], row['action_policy_lane_raw'] = float(raw[0]), float(raw[1])
        executed = info.get('action')
        if executed is not None:
            executed = np.asarray(executed, dtype=np.float32).reshape(-1)
            row['action_exec_acc_raw'], row['action_exec_lane_raw'] = float(executed[0]), float(executed[1])
            row['action_modified_by_wrapper'] = not np.array_equal(executed[:2], raw)
        sent = int(info.get('exit_applied_lane_change_direction', 0))
        row['lc_request_e76_e77'] = int(info.get('exit_actor_request_direction', 0))
        row['lc_command_sent'] = sent
        if sent and row.get('ego_lane_index') is not None:
            row['lc_target_lane'] = row['ego_lane_index'] + sent
        row['exit_maneuver_active_after'] = info.get('exit_maneuver_active')

        # ---- after-state
        alive = bool(info.get('ego_on_road')) and info.get('message') != 'Collision'
        source = 'unavailable'
        if alive and not terminal:
            t, v, ego = base.traci, base.traci.vehicle, base.ego_id
            row['sim_time_after_s'] = self._try(row, 'time_after', t.simulation.getTime)
            row['ego_road_id_after'] = self._try(row, 'road_after', lambda: v.getRoadID(ego))
            row['ego_lane_index_after'] = self._try(row, 'lane_after', lambda: int(v.getLaneIndex(ego)))
            row['ego_speed_after_mps'] = self._try(row, 'speed_after', lambda: float(v.getSpeed(ego)))
            row['ego_odometer_after_m'] = self._try(row, 'odo_after', lambda: float(v.getDistance(ego)))
            row['ego_accel_after_mps2'] = self._try(row, 'accel_after', lambda: float(v.getAcceleration(ego)))
            source = 'traci'
        elif alive and terminal:
            ego_state = base.observation[0]  # [odometer, lane index, speed] cached before TraCI closed
            row['ego_road_id_after'] = base.lane_name
            row['ego_lane_index_after'] = int(ego_state[1])
            row['ego_speed_after_mps'] = float(ego_state[2])
            row['ego_odometer_after_m'] = float(ego_state[0])
            source = 'env_cache'
        row['post_state_source'] = source
        row['ego_on_road_after'] = bool(info.get('ego_on_road'))
        row['ego_removed_from_sim'] = bool(pre['on_road'] and not info.get('ego_on_road'))
        if (pre['on_road'] and row.get('ego_road_id') is not None
                and row.get('ego_road_id_after') is not None):
            row['lane_change_physical'] = (row['ego_road_id'] == row['ego_road_id_after']
                                           and row['ego_lane_index'] != row.get('ego_lane_index_after'))
        before, after = row.get('ego_accel_mps2'), row.get('ego_accel_after_mps2')
        if pre['step'] > 0 and before is not None and after is not None:
            row['jerk_measured_mps3'] = (after - before) / self.dt
        if ep['prev_cmd_acc'] is not None:
            row['jerk_cmd_mps3'] = (row['cmd_acc_mps2'] - ep['prev_cmd_acc']) / self.dt
        ep['prev_cmd_acc'] = row['cmd_acc_mps2']

        for key in ('hard_braking_this_step', 'sumo_emergency_stopping_this_step',
                    'teleport_started', 'teleport_ended'):
            row[key] = info.get(key)
        row['ego_teleported_cum'] = info.get('ego_teleported')
        row.update(terminated=bool(terminated), truncated=bool(truncated),
                   is_terminal_step=terminal, terminal_reason=info.get('message', '') if terminal else '')

        # ---- reward ledger (as paid)
        terms = info.get('reward_terms', [])
        reward = float(reward)
        ep['return'] += reward
        row['reward_paid'], row['return_cum'] = reward, ep['return']
        row['reward_terms_sum_abs_err'] = abs(sum(float(x['weighted']) for x in terms) - reward)
        row['reward_terms_json'] = json.dumps(terms, allow_nan=False, sort_keys=True)
        for term in terms:
            if term['name'] in self.term_names:
                row[f'rt_{term["name"]}_raw'] = term['raw']
                row[f'rt_{term["name"]}_weighted'] = term['weighted']
                row[f'rt_{term["name"]}_active'] = term['active']
            else:
                self.unknown_terms[term['name']] += 1  # still preserved in reward_terms_json

        _update_summary_stats(ep['summary_stats'], row)

        self._steps.writerow({k: _fmt(row.get(k)) for k in self.step_fields})
        valid = alive and not (terminal and self.observation_mode in ('ego_spatial_v2', 'ego_spatial_onehot_v3'))
        note = ('' if valid else 'synthetic: ego unavailable or terminal spatial state is zero')
        trans = {'trajectory_id': ep['trajectory_id'], 'step': pre['step'],
                 'action_policy_0': _fmt(float(raw[0])), 'action_policy_1': _fmt(float(raw[1])),
                 'action_exec_0': _fmt(row.get('action_exec_acc_raw')),
                 'action_exec_1': _fmt(row.get('action_exec_lane_raw')),
                 'reward': _fmt(reward), 'terminated': int(terminated), 'truncated': int(truncated),
                 'is_terminal_step': int(terminal), 'terminal_reason': row['terminal_reason'],
                 'next_obs_valid': int(valid), 'next_obs_note': note}
        trans.update(zip(self.obs_fields, _vec(pre['obs'])))
        trans.update(zip(self.next_obs_fields, _vec(next_obs)))
        self._trans.writerow(trans)
        ep['rows'] += 1
        ep['last_row'] = row

    # ------------------------------------------------------------------ close
    def close(self, status='ok'):
        if self._closed:
            return
        self._closed = True
        if self._episode is not None:
            self.episodes.append({'trajectory_id': self._episode['trajectory_id'],
                                  'rows': self._episode['rows'], 'complete': False})
            status = 'aborted'
        self._steps_file.close()
        self._trans_file.close()
        self._summary_file.close()
        root = Path(__file__).resolve().parents[1]
        porcelain = _git(['status', '--porcelain'], root)
        manifest = {
            'schema_version': SCHEMA_VERSION, 'status': status,
            'created_unix': self._started, 'wall_seconds': time.time() - self._started,
            'run_label': self.run_label, 'model_label': self.model_label,
            'checkpoint': self.checkpoint, 'checkpoint_sha256': self.checkpoint_sha256,
            'training_seed': self.training_seed, 'scenario': self.scenario,
            'git_head': _git(['rev-parse', 'HEAD'], root),
            'git_branch': _git(['rev-parse', '--abbrev-ref', 'HEAD'], root),
            'git_dirty': None if porcelain is None else bool(porcelain),
            'git_status_porcelain': porcelain,
            'evaluation_config': self.config,
            'reward_config': self.reward_config,
            'python': platform.python_version(), 'numpy': np.__version__,
            'step_length_s': self.dt, 'hard_braking_threshold_mps2': self.hard_braking_threshold,
            'ttc_min_closing_speed_mps': TTC_MIN_CLOSING_SPEED_MPS,
            'neighbor_search_m': NEIGHBOR_SEARCH_M,
            'obs_dim': self.obs_dim, 'action_dim': ACTION_DIM,
            'observation_mode': self.observation_mode,
            'obs_layout': (OBS_LAYOUT if self.observation_mode == 'legacy' else {
                'ego': ('0:5: odometer, lane, speed, lateral position, lateral speed'
                        if self.observation_mode == 'ego_lateral_v1' else
                        '0:6: x, y, speed, lateral position, acceleration, distance to D3 entrance'),
                'neighbors': ('5:95' if self.observation_mode == 'ego_lateral_v1' else '6:96'),
                'action_history': ('95:135' if self.observation_mode == 'ego_lateral_v1' else '96:136'),
                'delay': ('135' if self.observation_mode == 'ego_lateral_v1' else '136'),
            }),
            'episodes': self.episodes, 'diagnostic_read_errors': dict(self.errors),
            'reward_terms_not_in_config': dict(self.unknown_terms),
            'files': {name: hashlib.sha256((self.out_dir / name).read_bytes()).hexdigest()
                      for name in (STEPS_FILE, TRANSITIONS_FILE, SUMMARY_FILE)},
        }
        columns = ([{'name': n, 'unit': u, 'description': d} for n, u, d in FIXED_COLUMNS]
                   + [{'name': n, 'unit': u, 'description': 'reward term field (see reward_terms_json)'}
                      for n, u in self.term_columns])
        with (self.out_dir / COLUMNS_FILE).open('x') as handle:
            json.dump(columns, handle, indent=2, ensure_ascii=False)
        with (self.out_dir / MANIFEST_FILE).open('x') as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False, default=str)
            handle.write('\n')


def _finite_nonneg(value):
    value = _finite(value)
    return value if value is not None and value >= 0 else None


def _summary_stats():
    return {'first': {}, 'last': {}, 'counts': Counter(), 'numeric': {}}


def _summary_number(stats, name, value):
    value = _finite(value)
    if value is None:
        return
    values = stats['numeric'].setdefault(name, [])
    values.append(value)


def _update_summary_stats(stats, row):
    for key in ('route_remaining_to_d3_end_m', 'route_remaining_to_d3_start_m'):
        value = _finite(row.get(key))
        if value is not None:
            stats['first'].setdefault(key, value)
            stats['last'][key] = value
    for key in ('min_lane_changes_required', 'route_urgency', 'ttc_leader_s', 'ttc_follower_s',
                'rt_efficiency_raw', 'rt_efficiency_weighted', 'rt_route_raw', 'rt_route_weighted'):
        _summary_number(stats, key, row.get(key))
    counts = stats['counts']
    for key in ('exit_reachable', 'direct_exit_lane', 'lc_request_e76_e77', 'lc_command_sent',
                'lane_change_physical', 'hard_braking_this_step', 'sumo_emergency_stopping_this_step'):
        value = row.get(key)
        if value not in (None, '', 0, False, '0'):
            counts[key] += 1
    if row.get('exit_reachable') not in (None, ''):
        counts['exit_reachable_known'] += 1
    if row.get('route_diag_error'):
        counts['route_diag_error'] += 1
    for key in ('ttc_leader_s', 'ttc_follower_s'):
        value = _finite(row.get(key))
        if value is not None and value < 2.0:
            counts[f'{key}_below_2s'] += 1


def _mean(values):
    return float(sum(values) / len(values)) if values else None


def _episode_summary(ep, model_label, checkpoint_sha256, training_seed, scenario):
    stats, last = ep['summary_stats'], ep['last_row']
    numeric, counts = stats['numeric'], stats['counts']
    return {
        'trajectory_id': ep['trajectory_id'], 'model_label': model_label,
        'checkpoint_sha256_12': (checkpoint_sha256 or '')[:12], 'training_seed': training_seed,
        'eval_seed': ep['eval_seed'], 'scenario': scenario, 'steps': ep['rows'],
        'terminal_reason': last['terminal_reason'], 'success': last['terminal_reason'] == 'Arrived',
        'terminated': last['terminated'], 'truncated': last['truncated'], 'policy_return': ep['return'],
        'route_remaining_to_d3_end_initial_m': stats['first'].get('route_remaining_to_d3_end_m'),
        'route_remaining_to_d3_end_last_m': stats['last'].get('route_remaining_to_d3_end_m'),
        'route_remaining_to_d3_start_initial_m': stats['first'].get('route_remaining_to_d3_start_m'),
        'route_remaining_to_d3_start_last_m': stats['last'].get('route_remaining_to_d3_start_m'),
        'exit_reachable_known_steps': counts['exit_reachable_known'],
        'exit_reachable_true_steps': counts['exit_reachable'],
        'exit_reachable_false_steps': counts['exit_reachable_known'] - counts['exit_reachable'],
        'min_lane_changes_required_min': min(numeric['min_lane_changes_required'])
        if numeric.get('min_lane_changes_required') else None,
        'min_lane_changes_required_max': max(numeric['min_lane_changes_required'])
        if numeric.get('min_lane_changes_required') else None,
        'direct_exit_lane_steps': counts['direct_exit_lane'],
        'route_urgency_max': max(numeric['route_urgency']) if numeric.get('route_urgency') else None,
        'route_diag_error_rows': counts['route_diag_error'],
        'lane_requests_e76_e77': counts['lc_request_e76_e77'],
        'lane_change_commands_sent': counts['lc_command_sent'],
        'physical_lane_changes': counts['lane_change_physical'],
        'hard_braking_steps': counts['hard_braking_this_step'],
        'sumo_emergency_stopping_steps': counts['sumo_emergency_stopping_this_step'],
        'leader_ttc_min_s': min(numeric['ttc_leader_s']) if numeric.get('ttc_leader_s') else None,
        'follower_ttc_min_s': min(numeric['ttc_follower_s']) if numeric.get('ttc_follower_s') else None,
        'leader_ttc_below_2s_steps': counts['ttc_leader_s_below_2s'],
        'follower_ttc_below_2s_steps': counts['ttc_follower_s_below_2s'],
        'efficiency_raw_mean': _mean(numeric.get('rt_efficiency_raw', ())),
        'efficiency_weighted_sum': sum(numeric.get('rt_efficiency_weighted', ())),
        'route_raw_mean': _mean(numeric.get('rt_route_raw', ())),
        'route_weighted_sum': sum(numeric.get('rt_route_weighted', ())),
    }
