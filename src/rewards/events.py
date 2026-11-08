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
