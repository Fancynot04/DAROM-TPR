"""Registered reward components for E76 -> D3."""
from dataclasses import dataclass
from . import events
from .terms import RewardTerm

@dataclass(frozen=True)
class RewardContext:
    outcome: str | None
    config: dict
    teleported: bool = False
    efficiency_raw: float | None = None
    route_raw: float | None = None
    diagnostics: dict | None = None
    efficiency_weight: float | None = None
    efficiency_decay_factor: float | None = None

PROFILE_COMPONENTS = {'events_only_v1': ('events',),
                      'e76_d3_dense_v1': ('events', 'efficiency', 'route')}

EVENT_TO_TERM = {'arrival': 'success'}
EVENT_TERMS = ('success', 'collision', 'missed_exit', 'teleport', 'timeout')

def _event(ctx):
    name, value = events.event_reward(ctx.outcome, ctx.config)
    return [RewardTerm(term_name, float(ctx.config['terms'][term_name]['value']) if term_name == name else 0.0,
                       float(ctx.config['terms'][term_name]['weight']),
                       bool(ctx.config['terms'][term_name]['enabled']),
                       active=(term_name == name))
            for term_name in EVENT_TERMS]


def _dense(name, raw, config, effective_weight=None):
    term = config['terms'][name]
    base_weight = float(term['weight'])
    weight = base_weight if effective_weight is None else float(effective_weight)
    if not __import__('math').isfinite(weight) or weight < 0:
        raise ValueError(f'Invalid effective {name} weight: {weight!r}')
    if raw is None:
        return RewardTerm(name, 0.0, weight, bool(term['enabled']), active=False,
                          base_weight=base_weight)
    if not __import__('math').isfinite(float(raw)):
        raise ValueError(f'Non-finite {name} reward before clipping')
    clipped = min(float(term['clip_max']), max(float(term['clip_min']), float(raw)))
    return RewardTerm(name, float(raw), weight, bool(term['enabled']),
                      active=True, clipped=(clipped != float(raw)), clipped_value=clipped,
                      base_weight=base_weight)


def settle_terms(profile, context):
    if profile not in PROFILE_COMPONENTS:
        raise ValueError(f'Unknown configurable reward profile: {profile!r}')
    events_only = _event(context)
    if profile == 'events_only_v1' or bool(context.outcome):
        return events_only
    return events_only + [_dense('efficiency', context.efficiency_raw, context.config,
                                 context.efficiency_weight),
                          _dense('route', context.route_raw, context.config)]

def settle(profile, context):
    if profile not in PROFILE_COMPONENTS:
        raise ValueError(f'Unknown configurable reward profile: {profile!r}')
    terms = settle_terms(profile, context)
    components = {term.name: term.weighted for term in terms}
    return float(sum(components.values())), components
