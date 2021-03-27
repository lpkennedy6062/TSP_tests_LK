import itertools as it
from typing import Iterable
import os
import numpy.random as random
import numpy as np
import scipy as sp

from tsp.tsp import N_TSP, TSP, TSP_O, City

from tsp.pyramid import pyramid_solve
from sklearn.manifold import MDS
from pytsp import dumps_matrix, run as run_concorde


def stress(tsp_a: TSP, tsp_b: TSP) -> float:
    sqsum = 0.
    diffsum = 0.
    for a, b in zip(tsp_a.to_edges(), tsp_b.to_edges()):
        sqsum += a[2] ** 2
        diffsum += (a[2] - b[2]) ** 2
    return np.sqrt(diffsum / sqsum)


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


def recover_local_scaled(original: TSP_O, reconstructed: [[float]]) -> TSP:
    t = recover_local(original.to_matrix(), reconstructed).astype(np.float)
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


def do_mds(tsp: N_TSP, dimensions=2) -> (TSP, TSP, float):
    # mds = MDS(n_components=dimensions, metric=True, n_init=50, eps=1e-5, dissimilarity='precomputed')
    mds = MDS(n_components=dimensions, metric=True, dissimilarity='precomputed')
    V = mds.fit_transform(tsp.to_edge_matrix())
    # V += np.array(tsp.cities[0]) - V[0]
    # tsp2 = N_TSP.from_cities(recover_local(tsp.to_matrix(), V))
    # tsp2 = N_TSP.from_cities(V.astype(np.int))
    if dimensions == 2:
        tsp2 = N_TSP.from_cities(recover_local(tsp.to_matrix(), V))
    else:
        tsp2 = N_TSP.from_cities(V.astype(np.int))
    return tsp, tsp2, stress(tsp, tsp2)
    # return tsp, tsp2, mds.stress_


class Solver:
    """Abstract class for generic TSP solvers"""

    def __init__(self, tsp: TSP):
        """Set up the solver based on the given problem"""
        raise NotImplementedError

    def __call__(self) -> Iterable[int]:
        """Run the solver and produce a tour"""
        raise NotImplementedError


class RandomSolver(Solver):
    """A solver which produces a random tour"""

    def __init__(self, tsp: TSP):
        self.vertices = list(range(len(tsp.cities)))

    def __call__(self) -> Iterable[int]:
        random.shuffle(self.vertices)
        return self.vertices


class ConcordeSolver(Solver):
    """A solver which implements PyTSP (with support for distance matrices"""

    def __init__(self, tsp: TSP):
        E = tsp.to_edge_matrix()
        self.outf = './tsp.temp'
        with open(self.outf, 'w') as dest:
            dest.write(dumps_matrix(E))

    def __call__(self) -> Iterable[int]:
        old_dir = os.getcwd()
        tour = run_concorde(self.outf, start=0, solver="concorde")
        os.unlink(self.outf)
        os.chdir(old_dir)
        return tour['tour']


class PyramidSolver(Solver):
    """A solver which implements a pyramid approximator"""

    def __init__(self, tsp: TSP):
        self.m = tsp.to_matrix()

    def __call__(self) -> Iterable[int]:
        return pyramid_solve(self.m)
