"""Select and settle registered reward components for legacy E76-D3 profiles.

The environment captures pre-action SECRM values and post-action progress and
terminal state. This module only combines those values; it never advances SUMO.
"""
from __future__ import annotations

from dataclasses import dataclass
from . import events, progress


@dataclass(frozen=True)
class RewardContext:
    outcome: str | None
    config: dict
    teleported: bool
    progress_before: float | None = None
    progress_after: float | None = None
    odometer_before: float | None = None
    odometer_after: float | None = None
    initial_remaining: float | None = None
    lane_change: float = 0.0
    secrm_before: dict | None = None


def _event(ctx):
    name, value = events.event_reward(ctx.outcome, ctx.config)
    return {key: value if key == name else 0.0 for key in events.EVENTS.values()}


def _progress(ctx):
    value = 0.0 if ctx.teleported else progress.progress_reward(
        ctx.progress_before, ctx.progress_after, ctx.odometer_before,
        ctx.odometer_after, ctx.initial_remaining, ctx.config)
    return {'progress': float(value)}


def _directional(ctx):
    return {'lane_change': float(ctx.lane_change)}


def _efficiency(ctx):
    active = ctx.secrm_before is not None and not ctx.teleported
    return {'efficiency': ctx.secrm_before['efficiency'] * ctx.config['efficiency_weight'] if active else 0.0}


def _route(ctx):
    active = ctx.secrm_before is not None and not ctx.teleported
    return {'route_following': ctx.secrm_before['route_reward'] * ctx.config.get('route_weight', 0.0) if active else 0.0}


# Explicit registration avoids loading arbitrary code named by experiment files.
COMPONENTS = {'events': _event, 'progress': _progress, 'directional_lane': _directional,
              'secrm_efficiency': _efficiency, 'secrm_route': _route}
PROFILE_COMPONENTS = {
    'route_end_progress_events_v1': ('progress', 'events'),
    'route_end_progress_events_directional_lc_v1': ('progress', 'events', 'directional_lane'),
    'secrm_efficiency_events_v1': ('events', 'secrm_efficiency', 'secrm_route'),
    'secrm_efficiency_route_events_v1': ('events', 'secrm_efficiency', 'secrm_route'),
}


def settle(profile, context):
    """Return the historical total and component keys for a configured profile."""
    try:
        names = PROFILE_COMPONENTS[profile]
    except KeyError as exc:
        raise ValueError(f'Unknown configurable reward profile: {profile!r}') from exc
    parts = {}
    for name in names:
        parts.update(COMPONENTS[name](context))
    event = sum(parts[key] for key in events.EVENTS.values())
    if profile.startswith('secrm_'):
        # Keep the old log schema and arithmetic ordering for checkpoint comparisons.
        components = {'progress': 0.0, 'lane_change': 0.0,
                      'efficiency': parts['efficiency'], 'route_following': parts['route_following'],
                      **{key: parts[key] for key in events.EVENTS.values()}}
        total = parts['efficiency'] + parts['route_following'] + event
    else:
        components = {'progress': parts['progress'], **{key: parts[key] for key in events.EVENTS.values()}}
        if 'directional_lane' in names:
            components['lane_change'] = parts['lane_change']
        # Historically the environment always included this field in the sum.
        total = parts['progress'] + event + context.lane_change
    return float(total), components
