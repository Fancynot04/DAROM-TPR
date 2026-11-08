"""Registered reward components for E76 -> D3."""
from dataclasses import dataclass
from . import events
from .terms import RewardTerm

@dataclass(frozen=True)
class RewardContext:
    outcome: str | None
    config: dict
    teleported: bool = False

PROFILE_COMPONENTS = {'events_only_v1': ('events',)}

EVENT_TO_TERM = {'arrival': 'success'}

def _event(ctx):
    name, value = events.event_reward(ctx.outcome, ctx.config)
    return [RewardTerm(term_name, float(ctx.config['terms'][term_name]['value']) if term_name == name else 0.0,
                       float(ctx.config['terms'][term_name]['weight']),
                       bool(ctx.config['terms'][term_name]['enabled']),
                       active=(term_name == name))
            for term_name in ctx.config['terms']]


def settle_terms(profile, context):
    if profile not in PROFILE_COMPONENTS:
        raise ValueError(f'Unknown configurable reward profile: {profile!r}')
    return _event(context)

def settle(profile, context):
    if profile not in PROFILE_COMPONENTS:
        raise ValueError(f'Unknown configurable reward profile: {profile!r}')
    terms = settle_terms(profile, context)
    components = {term.name: term.weighted for term in terms}
    return float(sum(components.values())), components
