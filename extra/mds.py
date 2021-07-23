from typing import Tuple
from numpy.typing import ArrayLike, NDArray
import numpy as np
import scipy as sp
from sklearn.manifold import MDS

from tsp.core.tsp import N_TSP, TSP


def stress(tsp_a: N_TSP, tsp_b: N_TSP) -> float:
    """Compute Kruskal's stress-1 between the distance matrices of two TSPs.

    Args:
        tsp_a (N_TSP): first TSP
        tsp_b (N_TSP): second TSP

    Returns:
        float: stress
    """
    sqsum = 0.
    diffsum = 0.
    for a, b in zip(tsp_a.to_edges(), tsp_b.to_edges()):
        sqsum += a[2] ** 2
        diffsum += (a[2] - b[2]) ** 2
    return np.sqrt(diffsum / sqsum)


def _recover_local(original: ArrayLike, reconstructed: ArrayLike) -> NDArray:
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


def recover_local_scaled(original: N_TSP, reconstructed: ArrayLike) -> TSP:
    """Use Procrustes scaling to make the MDS reconstruction fit the original problem as best as possible.

    Args:
        original (N_TSP): original problem
        reconstructed (ArrayLike): MDS reconstruction

    Returns:
        TSP: reconstructed problem of guaranteed same height and width
    """
    assert original.dimensions == 2 and reconstructed.shape[1] == 2
    t = _recover_local(original.to_matrix(), reconstructed).astype(np.float)
    x_high, y_high = original.w - 1, original.h - 1
    # We want to get everything within coordinates 0, n-1
    # If necessary, shift so that 0 is min
    x_shift = 0 if np.min(t[:,0]) >= 0 else np.min(t[:,0])
    y_shift = 0 if np.min(t[:,1]) >= 0 else np.min(t[:,1])
    t[:,0] -= x_shift
    t[:,1] -= y_shift
    # If necessary, rescale so that n-1 is max
    x_scale = 1. if np.max(t[:,0]) <= x_high else (x_high / np.max(t[:,0]))
    y_scale = 1. if np.max(t[:,1]) <= y_high else (y_high / np.max(t[:,1]))
    t[:,0] *= x_scale
    t[:,1] *= y_scale
    t = TSP.from_cities(t)
    t.w = x_high + 1
    t.h = y_high + 1
    return t


def do_mds(tsp: N_TSP, dimensions: int = 2) -> Tuple[N_TSP, N_TSP, float]:
    """Generate an MDS reconstruction of any TSP problem. If both problems are of dimension 2,
    reconstruction will be scaled to match original as best as possible.

    Args:
        tsp (N_TSP): original problem
        dimensions (int, optional): Dimension of MDS reconstruction. Defaults to 2.

    Returns:
        Tuple[N_TSP, N_TSP, float]: (original problem, reconstructed problem, stress-1)
    """
    mds = MDS(n_components=dimensions, metric=True, dissimilarity='precomputed')
    V = mds.fit_transform(tsp.to_edge_matrix())
    if dimensions == 2 and tsp.dimensions == 2:
        tsp2 = recover_local_scaled(tsp, V)
    else:
        tsp2 = N_TSP.from_cities(V.astype(np.int))
    return tsp, tsp2, stress(tsp, tsp2)
