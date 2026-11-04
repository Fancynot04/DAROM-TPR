"""E76-to-D3 reward component."""
EVENTS = {'Arrived': 'arrival', 'Collision': 'collision', 'Max Steps': 'timeout', 'Missed Exit': 'missed_exit', 'Ego Teleported': 'teleport'}

def event_reward(outcome, reward):
    key = EVENTS.get(outcome)
    return (key, float(reward[key])) if key is not None else (None, 0.0)
