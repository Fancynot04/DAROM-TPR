"""Registered reward components for E76 -> D3."""
from dataclasses import dataclass
from . import events

@dataclass(frozen=True)
class RewardContext:
    outcome: str | None
    config: dict
    teleported: bool = False

PROFILE_COMPONENTS = {'events_only_v1': ('events',)}

def _event(ctx):
    name, value = events.event_reward(ctx.outcome, ctx.config)
    return {key: value if key == name else 0.0 for key in ctx.config['events']}

def settle(profile, context):
    if profile not in PROFILE_COMPONENTS:
        raise ValueError(f'Unknown configurable reward profile: {profile!r}')
    components = _event(context)
    return float(sum(components.values())), components
