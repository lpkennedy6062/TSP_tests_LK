import numpy as np
import scipy as sp
from sklearn.manifold import Isomap


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


def local_reconstruct(cities: [[int]], edges: [[float]], n_neighbors: int = 5) -> [[int]]:
    im = Isomap(n_neighbors=n_neighbors, n_components=2)
    # Treat the (symmetric) distance matrix as a set of n-dimensional points
    embedding = im.fit_transform(edges)
    return recover_local(cities, embedding)
