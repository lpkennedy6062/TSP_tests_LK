"""
1. Generate clusters
2. For each cluster:
   - Generate distance matrix
   - Reconstruct with 2D MDS
   - Calculate Procrustes transformation
   - Project into original space

How easy would it be to do this by operating on a distance matrix instead of on points?
The problem there is in reconstructing the problem piecewise.
"""


import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
from sklearn.manifold import MDS


def get_clusters(cities: [[int]], n_clusters: int) -> [[int]]:
    cities = np.array(cities)
    km = KMeans(n_clusters=n_clusters)
    indices = km.fit_predict(cities)
    cluster_indices = []
    for i in range(n_clusters):
        cluster_indices.append(np.where(indices == i)[0])
    return cluster_indices


def get_local_distances(cluster_indices: [int], edges: [[float]]) -> [[float]]:
    result = np.ndarray((len(cluster_indices), len(cluster_indices)), dtype=np.float)
    for i, a in enumerate(cluster_indices):
        for j, b in enumerate(cluster_indices):
            result[i,j] = edges[a,b]
    return result


def get_reconstruction(distances: [[float]]) -> [[float]]:
    mds = MDS(n_components=2, metric=True, n_init=50, eps=1e-5, dissimilarity='precomputed')
    return mds.fit_transform(distances)


def recover_local(original: [[int]], reconstructed: [[float]]) -> [[int]]:
    o, m, _ = sp.spatial.procrustes(original, reconstructed)
    ox = np.vstack([o[:, 0], np.ones_like(o[:, 0])]).T
    mx = np.vstack([m[:, 0], np.ones_like(m[:, 0])]).T
    oy = np.vstack([o[:, 1], np.ones_like(o[:, 1])]).T
    my = np.vstack([m[:, 1], np.ones_like(m[:, 1])]).T
    x = np.linalg.lstsq(ox, original[:, 0], rcond=None)[0]
    y = np.linalg.lstsq(oy, original[:, 1], rcond=None)[0]
    result = np.ndarray(m.shape, dtype=m.dtype)
    result[:, 0] = mx.dot(x)
    result[:, 1] = my.dot(y)
    return result.astype(np.int)


def local_reconstruct(cities: [[int]], edges: [[float]], n_clusters: int = 4) -> [[int]]:
    """This is the main method of the module. Transforms following the schema outlined above."""
    result = np.ndarray((len(cities), 2), dtype=np.int)
    # clusters = get_clusters(cities, n_clusters)
    clusters = get_clusters(edges, n_clusters)
    for indices in clusters:
        coords = cities[indices]
        dist = get_local_distances(indices, edges)
        mds = get_reconstruction(dist)
        recovered = recover_local(coords, mds)
        for i, c in zip(indices, recovered):
            result[i, :] = c
    return result
