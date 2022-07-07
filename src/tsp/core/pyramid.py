"""Implements a hierarchical clustering ("pyramid") algorithm similar to those used in the
literature to approximate human solutions to the TSP.

This code is a replication and improvement of the "graph pyramid" described in [1]. The improved
algorithm is described in detail in a paper by J. VanDrunen, K. Nam, M. A. Beers, and Z. Pizlo,
currently in preparation.

[1] Y. Haxhimusa, W. G. Kropatsch, Z. Pizlo, and A. Ion. Approximate graph pyramid solution of the
E-TSP. Elsevier, 2009.
"""


from typing import List
from numpy.typing import NDArray
from collections import defaultdict
import itertools as it
import numpy as np

from tsp.core.tsp import N_TSP
from tsp.core.tree_split import citation55


def _cheapest_insertion(centroids, nodes, prev_centroid, next_centroid):
    min_distance = float('inf')
    result = None
    for partial_tour in it.permutations(nodes):
        if prev_centroid is not None:
            partial_tour_centroids = [prev_centroid] + [centroids[i] for i in partial_tour] + [next_centroid]
        else:
            partial_tour_centroids = [centroids[i] for i in partial_tour]
            partial_tour_centroids.append(partial_tour_centroids[0])  # make closed tour!
        distance = 0.
        for i in range(1, len(partial_tour_centroids)):
            distance += np.sqrt(np.sum(np.square(partial_tour_centroids[i] - partial_tour_centroids[i - 1])))
        if distance < min_distance:
            min_distance = distance
            result = partial_tour
    return result


def _solve_level(c, v, level, subcluster, tour, prev_centroid=None, next_centroid=None):
    centroids = c[level]
    nodes = v[level][subcluster]
    # prev_centroid = c[0][tour[-1]] if tour else None
    partial_tour = _cheapest_insertion(centroids, nodes, prev_centroid, next_centroid)
    if level == 0:
        tour.extend(partial_tour)
    else:
        for i, node in enumerate(partial_tour):
            if tour:
                next_prev_centroid = c[0][tour[-1]]
            else:
                next_prev_centroid = c[level][partial_tour[-1]]
            if i + 1 == len(partial_tour):
                _solve_level(c, v, level - 1, node, tour, next_prev_centroid, next_centroid if next_centroid is not None else c[0][tour[0]])
            else:
                _solve_level(c, v, level - 1, node, tour, next_prev_centroid, c[level][partial_tour[i + 1]])


def _do_split(v, e, edges, k):
    return citation55(v, e, edges, k)


def _find_parent(i, parents):
    while parents[i] != i:
        i = parents[i]
    return i


def _cluster_boruvka(nodes: List, k: int):
    c = [nodes]
    v = []
    e = []
    while not v or len(v[-1]) > 1:
        n = len(c[-1])
        v.append(list(map(lambda i: [i], range(n))))

        edges = np.zeros((n, n), dtype=np.float)
        for i in range(n):
            edges[i, i] = np.inf
            for j in range(i + 1, n):
                value = np.sqrt(np.sum(np.square(c[-1][i] - c[-1][j])))
                edges[i, j] = value
        i_lower = np.tril_indices(n, -1)
        edges[i_lower] = edges.T[i_lower]

        minimum_edges = {i : (np.inf, None, None) for i in range(n)}
        for c1, c2 in it.combinations(range(n), 2):
            for edge in it.product(v[-1][c1], v[-1][c2]):
                if edges[edge] < minimum_edges[c1][0]:
                    minimum_edges[c1] = edges[edge], edge, c2
                if edges[edge] < minimum_edges[c2][0]:
                    minimum_edges[c2] = edges[edge], edge, c1

        parents = list(range(n))
        edge_tracker = defaultdict(set)
        for c1, (_, edge, c2) in sorted(minimum_edges.items(), key=lambda t: t[1][0]):
            c1_parent = _find_parent(c1, parents)
            c2_parent = _find_parent(c2, parents)
            if c1_parent != c2_parent:
                parents[c2_parent] = c1_parent
                edge_tracker[c1_parent].add(edge)

        parents = [_find_parent(i, parents) for i in parents]
        vertex_tracker = defaultdict(list)
        centroid_tracker = defaultdict(list)
        for i, p in enumerate(parents):
            vertex_tracker[p].append(i)
            centroid_tracker[p].append(c[-1][i])
            if i != p:
                edge_tracker[p].update(edge_tracker[i])
                del edge_tracker[i]

        c.append([])
        v[-1] = []
        e.append([])
        for p in set(parents):
            if len(vertex_tracker[p]) > k:
                split_v, split_e = _do_split(
                    vertex_tracker[p],
                    list(edge_tracker[p]),
                    edges,
                    k
                )
                split_c = []
                for vertices in split_v:
                    split_c.append(np.mean([c[-2][i] for i in vertices], axis=0))
                c[-1].extend(split_c)
                v[-1].extend(split_v)
                e[-1].extend(split_e)
            else:
                c[-1].append(np.mean(centroid_tracker[p], axis=0))
                v[-1].append(vertex_tracker[p])
                e[-1].append(list(edge_tracker[p]))

    return c, v, e


def pyramid_solve(tsp: N_TSP, k: int = 6, s: int = 1) -> NDArray:
    """Find an approximately-optimal tour using hierarchical clustering algorithm.

    Args:
        nodes (N_TSP): TSP to solve 
        k (int, optional): Cluster size. Defaults to 6.
        s (int, optional): Number of previous cities to account for in partial tour (refines k+s-1 cities, where the extra 1 is the endpoint, for historical reasons). Defaults to 1.
        clustering (str, optional): MST algorithm to use for clustering, either 'boruvka' or 'kruskal'. Defaults to 'boruvka'.

    Returns:
        NDArray: tour
    """
    nodes = list(map(lambda a: np.array(a, dtype=np.float64), tsp.cities))
    c, v, e = _cluster_boruvka(nodes, k)
    level = len(v) - 1
    result = _solve_level(c, v, level, 0)
    while level > 0:
        level -= 1
        new_result = []
        for i, subcluster in enumerate(result):
            if new_result:
                prev_tour = new_result[-s:]
                if len(prev_tour) > 1:
                    new_result = new_result[:-(len(prev_tour)-1)]
                if i + 1 == len(result):
                    new_result.extend(_solve_level(c, v, level, subcluster, c[level][prev_tour[0]], prev_tour[1:], c[level][new_result[0]]))
                else:
                    new_result.extend(_solve_level(c, v, level, subcluster, c[level][prev_tour[0]], prev_tour[1:], c[level + 1][result[(i + 1) % len(result)]]))
            else:
                new_result.extend(_solve_level(c, v, level, subcluster, prev_centroid=c[level + 1][result[i - 1]], next_centroid=c[level + 1][result[(i + 1) % len(result)]]))
        result = new_result
    assert len(result) == len(nodes)
    return np.array(result)
