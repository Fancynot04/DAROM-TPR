"""Terminal event names used by the E76 -> D3 reward contract."""
EVENTS = {'Arrived': 'success', 'Collision': 'collision', 'Missed Exit': 'missed_exit',
          'Ego Teleported': 'teleport', 'Timeout': 'timeout'}

def event_reward(outcome, config):
    key = EVENTS.get(outcome)
    values = config.get('terms', config.get('events', {}))
    if key is None:
        return None, 0.0
    value = values[key]['value'] if 'terms' in config else values[key]
    return key, float(value)


# O3 -> D3 / MASAC B0 终止事件奖励
B0_EVENTS = {
    "success": 50.0,
    "collision": -100.0,
    "missed_exit": -100.0,
    "teleport": -100.0,
    "timeout": -100.0,
}

def b0_reward(event):
    name, value = event_reward(
        event, {"events": B0_EVENTS}
    )
    if event is not None and name is None:
        raise ValueError(f"Unknown B0 event: {event}")
    return value
