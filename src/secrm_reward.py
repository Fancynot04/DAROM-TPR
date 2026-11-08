"""SECRM-2D Eq. (2), (5), 4.4.1 and 4.4.4, reward-only adaptation.

The target computed from pre-action s_t is the next-step safe-speed bound;
its deviation from v(s_t) is paid on that transition. No action filter is added.
https://doi.org/10.1049/itr2.70013
"""
import math

EFFICIENCY_PROFILE = 'secrm_efficiency_events_v1'
ROUTE_PROFILE = 'secrm_efficiency_route_events_v1'
DIRECTIONAL_PROFILE = 'secrm_efficiency_directional_events_v1'
ADAPTIVE_PROFILE = 'secrm_efficiency_adaptive_directional_events_v1'
PROFILES = (EFFICIENCY_PROFILE, ROUTE_PROFILE, DIRECTIONAL_PROFILE, ADAPTIVE_PROFILE)
FORMULA_VERSION = 'secrm_eq2_prestate_geometry_v2'


from .rewards.secrm_efficiency import lateral_relation, efficiency
from .rewards.secrm_route import executable_distance, route_penalty, minimum_changes


def stamp_replay(buffer, experiment):
    from .exit_progress_reward import experiment_identity
    buffer.secrm_experiment_identity = experiment_identity(experiment)
    buffer.secrm_formula_version = FORMULA_VERSION


def validate_replay(buffer, experiment):
    from .exit_progress_reward import experiment_identity
    if (getattr(buffer, 'secrm_experiment_identity', None) != experiment_identity(experiment)
            or getattr(buffer, 'secrm_formula_version', None) != FORMULA_VERSION):
        raise ValueError('Replay buffer SECRM configuration/formula identity mismatch')
