"""This code implements the algorithm for splitting trees found in [1]. Code-named "Citation 55"
because it is citation 55 in [2].

This code is copyright Mark Beers and released under the terms found in the LICENSE file of this
repository (ISC or COIL-1.0 license).

[1] W. G. Kropatsch, Y. Haxhimusa, and Z. Pizlo. Integral Trees: Subtree Depth and Diameter.
Springer, 2004.

[2] Y. Haxhimusa, W. G. Kropatsch, Z. Pizlo, and A. Ion. Approximate graph pyramid solution of the
E-TSP. Elsevier, 2009.
"""

import itertools

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix


# define a test case
np.random.seed(124)
xy = np.random.uniform(0,10, (20,2))

# pretend 8,0,17,4,16,12,18 is a cluster
cluster = [8,0,17,4,16,12,18]
edges = [(18,16), (12,16), (16,4), (4,17), (12, 0), (0,8), (18,11), (11,1), (11,3), (11,10)]
dist = distance_matrix(xy, xy)


# define a function to get children of a particular vertex
def get_child_edges(v, edges, reject):
    children = []
    for i,j in edges:
        if i == v or j == v:
            if (i, j) != reject and (j, i) != reject:
                #print("accepted i,j = ", (i,j))
                children += [(i,j)] + get_child_edges(j if i == v else i, edges, (i,j))
    return children

def get_child_verts(child_edges):
    return np.unique(np.array(child_edges).ravel()).tolist()

def get_dmax(edges, D):
    """
    This hideous creation:
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
            if i == v or j == v:
                child_edges.append(get_child_edges(j if i == v else i, edges, (i,j)))
                next_v.append(j if i == v else i)
                edge_lengths.append(D[i,j])

        return_list = []
        for k in range(len(child_edges)):
            g = inner(next_v[k], dcur + edge_lengths[k], child_edges[k], D)
            if type(g) is list:
                return_list += g
            else:
                return_list += [g]
        return return_list

    verts = get_child_verts(edges)
    dmax_by_v = []
    for v in verts:
        dmax_by_v += [inner(v, 0, edges, D)]

    dmax_flat = list(itertools.chain(*dmax_by_v))
    if len(dmax_flat) == 0:
        return 0
    else:
        return max(dmax_flat)




# dmax_test = get_dmax(edges, dist)
# print(dmax_test)
# ce12 = get_child_edges(12, edges, (12,16))
# print(ce12)
# print(get_child_verts(ce12))

# ce16 = get_child_edges(16, edges, (12,16))
# print(ce16)
# print(get_child_verts(ce16))


def citation55(vertices, edges, D, r):
    if len(vertices) <= r:
        return [vertices], [edges]

    best_split_ij, smallest_delta = np.inf, np.inf
    ci_best, cj_best = np.inf, np.inf
    for i,j in edges:
        # split edge ij
        ci = get_child_edges(i, edges, (i,j))
        dmax_i = get_dmax(ci, D)
        cj = get_child_edges(j, edges, (i,j))
        dmax_j = get_dmax(cj, D)
        delta = np.abs(dmax_j - dmax_i)
        if delta < smallest_delta:
            best_split_ij = (i,j)
            smallest_delta = delta
            ci_best = ci
            cj_best = cj

    #print(best_split_ij, smallest_delta, ci_best, cj_best)
    if len(ci_best) == 0:
        vertices_i = [best_split_ij[0]]
    else:
        vertices_i = get_child_verts(ci_best)

    if len(cj_best) == 0:
        vertices_j = [best_split_ij[1]]
    else:
        vertices_j = get_child_verts(cj_best)

    vertices_i, edges_i = citation55(vertices_i, ci_best, D, r)
    #print("vertices_i = ", list(itertools.chain(*vertices_i)))
    # if len(list(itertools.chain(*vertices_i))) == 0:
    #     print("got here! ")
    #     vertices_i == [[best_split_ij[0]]]
    vertices_j, edges_j = citation55(vertices_j, cj_best, D, r)
    # if len(list(itertools.chain(*vertices_j))) == 0:
    #     vertices_j == [[best_split_ij[1]]]
    return vertices_i + vertices_j, edges_i + edges_j



if __name__ == '__main__':
    clusterV, clusterE = citation55(get_child_verts(edges), edges, dist, 3)
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
        clusterV, clusterE = citation55(get_child_verts(hypothetical_mst),
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
