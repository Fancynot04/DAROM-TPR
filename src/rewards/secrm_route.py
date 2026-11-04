"""SECRM route penalty; topology helper remains shared."""
import math

def route_penalty(required, remaining):
    if (type(required) is not int or required < 0 or not math.isfinite(remaining)
            or remaining < 0):
        raise ValueError('Invalid settled route requirement/distance')
    return -required/(1.0+remaining)

def minimum_changes(network, route, index, lane, change_edges):
    """Topology only: no gap/urgency/time feasibility assumptions."""
    if index < 0 or index >= len(route) or lane not in network.lanes[route[index]]:
        raise ValueError('Invalid stable lane/route index')
    costs={lane:0}
    for edge,next_edge in zip(route[index:],route[index+1:]):
        following={}
        for start,cost in costs.items():
            exits=network.lanes[edge] if edge in change_edges else (start,)
            for end in exits:
                if any(i not in network.lanes[edge] for i in range(min(start,end),max(start,end)+1)):continue
                for target,_ in network.connections.get((edge,next_edge,end),()):
                    value=cost+abs(end-start)
                    following[target]=min(value,following.get(target,value))
        costs=following
    if not costs:raise ValueError('No continuation to route endpoint')
    return min(costs.values())
