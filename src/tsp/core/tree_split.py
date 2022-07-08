"""This code implements algorithms for splitting trees to optimal height found in [1,2].

This code is copyright Mark Beers and released under the terms found in the LICENSE file at the
root of this repository (ISC or COIL license).

[1] Kropatsch, W. G., Saib, M., & Schreyer, M. (2002). The Optimal Height of a Graph Pyramid.

[2] Kropatsch, W. G., Haxhimusa, Y., & Pizlo, Z. (2004). Integral Trees: Subtree Depth and Diameter.
"""


import itertools
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix


def _get_child_edges(v, edges, reject):
    """Get children of a particular vertex.
    """
    children = []
    for i,j in edges:
        if v in (i, j):
            if reject not in ((i, j), (j, i)):
                children += [(i,j)] + _get_child_edges(j if i == v else i, edges, (i,j))
    return children


def _get_child_verts(child_edges):
    return np.unique(np.array(child_edges).ravel()).tolist()


def _get_dmax(edges, D):
    """This hideous creation:
    1) finds the distances from each node in the MST to all the leaves.
    2) Computes and return the maximum of all of these numbers. This is
        the greatest distance from one leaf to another leaf in the tree.
    """
    def inner(v, dcur, edges, D):
        child_edges = []
        next_v = []
        edge_lengths = []
        if len(edges) == 0:
            return dcur
        for i, j in edges:
            if v in (i, j):
                child_edges.append(_get_child_edges(j if i == v else i, edges, (i,j)))
                next_v.append(j if i == v else i)
                edge_lengths.append(D[i,j])

        return_list = []
        for k in range(len(child_edges)):
            g = inner(next_v[k], dcur + edge_lengths[k], child_edges[k], D)
            if isinstance(g, list):
                return_list += g
            else:
                return_list += [g]
        return return_list

    verts = _get_child_verts(edges)
    dmax_by_v = []
    for v in verts:
        dmax_by_v += [inner(v, 0, edges, D)]

    dmax_flat = list(itertools.chain(*dmax_by_v))
    if len(dmax_flat) == 0:
        return 0
    return max(dmax_flat)


def do_split(vertices: List, edges: List, D: NDArray, r: int) -> Tuple[List, List]:
    """It should be doable to apply concept of algorithm from [1] for splitting subtrees that are
    too large. The only difference is that we have edges with distances that aren't all 1, whereas
    in [2], "Edge length l(e) = 1 is used in all examples". They made a split by removing the
    "central edge", which was the edge such that the two resulting subtrees had maximally similar
    d_max, where in their case d_max was distance from root of new subtree to the furthest leaf of
    the subtree. For them, because edge length was 1, this just amounted to counting edges. For us,
    we have to add up edge lengths. So how would this algorithm actually work?

    1.   Given a subtree that has size > r,
        1. consider all edges and compute d_max for each of the two new root vertexes if split was
           to occur by removing this edge.
        2. split at the edge where d_max of the two trees is as equal as possible. Doing this
           conceptually will give us, at each split, two trees that could fit in circles of
           approximately the same radius.

    Args:
        vertices (List): List of vertices (defining the cluster) `[i, j, k]`
        edges (List): List of edges `[(i, j), ...]`
        D (NDArray): Distance matrix can be accessed with `D[i,j]`
        r (int): Maximum cluster size

    Returns:
        Tuple[List, List]: Lists of vertices defining split clusters, lists of edges
    """
    if len(vertices) <= r:
        return [vertices], [edges]

    best_split_ij, smallest_delta = np.inf, np.inf
    ci_best, cj_best = np.inf, np.inf
    for i,j in edges:
        # split edge ij
        ci = _get_child_edges(i, edges, (i,j))
        dmax_i = _get_dmax(ci, D)
        cj = _get_child_edges(j, edges, (i,j))
        dmax_j = _get_dmax(cj, D)
        delta = np.abs(dmax_j - dmax_i)
        if delta < smallest_delta:
            best_split_ij = (i,j)
            smallest_delta = delta
            ci_best = ci
            cj_best = cj

    if len(ci_best) == 0:
        vertices_i = [best_split_ij[0]]
    else:
        vertices_i = _get_child_verts(ci_best)

    if len(cj_best) == 0:
        vertices_j = [best_split_ij[1]]
    else:
        vertices_j = _get_child_verts(cj_best)

    vertices_i, edges_i = do_split(vertices_i, ci_best, D, r)
    vertices_j, edges_j = do_split(vertices_j, cj_best, D, r)
    return vertices_i + vertices_j, edges_i + edges_j


if __name__ == '__main__':
    # define a test case
    np.random.seed(124)
    xy = np.random.uniform(0,10, (20,2))

    # pretend 8,0,17,4,16,12,18 is a cluster
    cluster = [8,0,17,4,16,12,18]
    edges = [(18,16), (12,16), (16,4), (4,17), (12, 0), (0,8), (18,11), (11,1), (11,3), (11,10)]
    dist = distance_matrix(xy, xy)

    clusterV, clusterE = do_split(_get_child_verts(edges), edges, dist, 3)
    print("clusterV = \n", clusterV)
    print("clusterE = \n", clusterE)

    ########################################################################
    ####### Make a plot ####################################################
    ########################################################################

    # Hypothetical MST & make a plot
    hypothetical_mst = edges + [(4,6), (6,13), (13,9), (9,19), (19,14),(10,5),
    (5,7), (2,7), (2,15)]

    fig, axs = plt.subplots(2,2, figsize = (10,10))
    colors = ["orange", "green", "red", "cyan", "magenta",
                "yellow", "black", "pink", "olive", "lemonchiffon", "darkslategray"]

    # Full plot
    axs[0,0].scatter(xy[:, 0], xy[:, 1])
    for i in range(xy.shape[0]):
        axs[0,0].text(xy[i, 0], xy[i, 1], i)
    for i,j in hypothetical_mst:
        axs[0,0].plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
        color="orange")
    axs[0,0].set_title("Potential subtree")

    # Plot clusterings based on group size
    for i,j, r in zip([0,1,1], [1,0,1], [3,4,5]):
        clusterV, clusterE = do_split(_get_child_verts(hypothetical_mst),
            hypothetical_mst, dist, r)

        for k in range(xy.shape[0]):
            axs[i,j].text(xy[k, 0], xy[k, 1], k)

        axs[i,j].set_title(f"r = {r}")

        for k, cluster in enumerate(clusterV):
            axs[i,j].scatter(xy[cluster, 0], xy[cluster, 1], color = colors[k])

        for k, edge_lst in enumerate(clusterE):
            if len(edge_lst) > 0:
                for vi,vj in edge_lst:
                    axs[i,j].plot([xy[vi, 0], xy[vj, 0]], [xy[vi, 1], xy[vj, 1]],
                    color=colors[k])
    fig.suptitle('Citation 55 Splitting Rule')
    fig.show()

    #fig.savefig("citation55_cluster.pdf")
