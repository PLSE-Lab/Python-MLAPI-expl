import pandas as pd
import numpy as np
import networkx as nx


def get_walks_simple(graph, length, walks_per_node, use_time=False):
    walks = []
    for node in graph.keys():
        for walk_number in range(walks_per_node):
            walk = [node]
            current_node = node
            
            avg_time = 0.0
            time = 0.0

            for i in range(length - 1):
                neighbours = graph[current_node]
                
                if use_time:
                    current_node, time = random_neighbour_by_time(neighbours, int(avg_time), intervals=[60, 3600, 86400, 1 << 60])
                else:
                    random_ix = np.random.randint(0, high=len(neighbours))
                    current_node = neighbours[random_ix][0]

                avg_time = 0.8 * avg_time + 0.2 * time if len(walk) > 1 else time
                walk.append(current_node)

            walks.append(walk)
    return walks

def lower_bound(time, nodes):
    lo, hi = -1, len(nodes)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        mid_time = nodes[mid][1]
        if mid_time < time:
            lo = mid
        else:
            hi = mid
            
    assert hi in range(len(nodes) + 1)
    return hi

def upper_bound(time, nodes):
    # assuming time is int
    return lower_bound(time + 1, nodes)

# given timestamp and list of pairs (node, time) sorted by time return random node
# the closer node is by time, the more is probability
def random_neighbour_by_time(nodes, time, intervals):
    non_empty_intervals_bounds = []
    for i, interval in enumerate(intervals):
        prev_interval = 0 if i == 0 else intervals[i - 1]
        left = lower_bound(time - interval, nodes), upper_bound(time - prev_interval, nodes)
        right = lower_bound(time + prev_interval, nodes), upper_bound(time + interval, nodes)
        if left[1] > left[0] or right[1] > right[0]:
            non_empty_intervals_bounds.append((left, right))
            
            
    # sample vertices from next interval with probability twice lower than probability of current one
    assert len(non_empty_intervals_bounds) > 0, str(time) + ' ' + str(nodes[:10]) + ' ' + str(left) + ' ' + str(right)
    high = (1 << len(non_empty_intervals_bounds)) - 1
    rand_interval = np.random.randint(0, high=high) + 1
        
    cur = 1
    for i, (left, right) in enumerate(non_empty_intervals_bounds):
        if rand_interval <= cur:
            left_len = left[1] - left[0]
            right_len = right[1] - right[0]
            rand_ix_in_interval = np.random.randint(0, high=left_len + right_len)
            rand_ix_in_list = left[0] + rand_ix_in_interval if rand_ix_in_interval < left_len \
                                            else right[0] + rand_ix_in_interval - left_len
            return nodes[rand_ix_in_list]
        cur = 2 * cur + 1