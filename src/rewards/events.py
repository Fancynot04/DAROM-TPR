"""Terminal event names used by the E76 -> D3 reward contract."""
EVENTS = {'Arrived': 'success', 'Collision': 'collision', 'Missed Exit': 'missed_exit',
          'Ego Teleported': 'teleport', 'Timeout': 'timeout'}

def event_reward(outcome, config):
    key = EVENTS.get(outcome)
    return (key, float(config['events'][key])) if key is not None else (None, 0.0)
