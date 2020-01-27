import numpy as np
import itertools as it
from collections import defaultdict
from queue import PriorityQueue, Empty


def _on_segment(p, q, r):
    return q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])


def _orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # colinear
    if val > 0:
        return 1  # clockwise
    return 2  # counterclockwise


def _intersect(p1, q1, p2, q2):
    """Does p1q1 intersect p2q2?"""
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True
    return False


def _visible(a, b, obstacles):
    for c, d in obstacles:
        if a == c or a == d or b == c or b == d:
            continue  # If one of the points is a vertex of the obstacle, it's visible
        if _intersect(a, b, c, d):
            return False
    return True


def calculate_visgraph(vertices, obstacles, bound=None):
    # Obstacles should only be line segments at this point
    result = defaultdict(list)
    points = vertices + list(it.chain(*obstacles))
    if bound is not None:
        # We expect bound to take the form (x_max, y_max)
        # This prevents the graph from taking into account paths that would go outside the bound
        # Implicitly, if there is a bound specified, we also take x_min == y_min == 0
        points = [p for p in points if p[0] <= bound[0] and p[1] <= bound[1] and p[0] >= 0 and p[1] >= 0]
    for a, b in it.combinations(points, 2):
        if _visible(a, b, obstacles):
            result[tuple(a)].append(tuple(b))
            result[tuple(b)].append(tuple(a))
    return result


def _distance(p, q):
    return np.sqrt(pow(p[0] - q[0], 2) + pow(p[1] - q[1], 2))


def shortest_path(a, b, graph, exclude=None):
    a, b = tuple(a), tuple(b)
    q = PriorityQueue()
    visited = set() if exclude is None else set(exclude)
    visited.discard(a)
    visited.discard(b)
    current = [a]
    score = 0
    while True:
        if current[-1] == b:
            return current
        if current[-1] not in visited:
            visited.add(current[-1])
            for p in graph[current[-1]]:
                if p not in visited:
                    q.put((_distance(current[-1], p) + score, current + [p]))
        score, current = q.get(False)  # This will raise an Empty exception if there is no path
