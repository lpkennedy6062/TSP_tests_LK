import itertools as it
from typing import Iterable
import os
import numpy.random as random
import numpy as np
import scipy as sp

from tsp.tsp import N_TSP, TSP, TSP_O, City

from tsp.pyramid import pyramid_solve
from concorde.tsp import TSPSolver
import tsp.christofides as christofides
from sklearn.manifold import MDS, Isomap, SpectralEmbedding, LocallyLinearEmbedding
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from pytsp import dumps_matrix, run as run_concorde


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


# class ConcordeSolver(Solver):
#     """A solver which implements PyConcorde"""

#     def __init__(self, tsp: TSP):
#         m = tsp.to_matrix().T
#         self.solver = TSPSolver.from_data(m[0], m[1], norm='EUC_2D')

#     def __call__(self) -> Iterable[int]:
#         return [int(i) for i in self.solver.solve(verbose=False).tour]


class ConcordeSolver(Solver):
    """A solver which implements PyTSP (with support for distance matrices"""

    def __init__(self, tsp: TSP):
        E = tsp.to_edge_matrix()
        self.outf = '/tmp/tsp.temp'
        with open(self.outf, 'w') as dest:
            dest.write(dumps_matrix(E))

    def __call__(self) -> Iterable[int]:
        old_dir = os.getcwd()
        tour = run_concorde(self.outf, start=0, solver="concorde")
        os.unlink('/tmp/tsp.temp')
        os.chdir(old_dir)
        return tour['tour']


class ChristofidesSolver(Solver):
    """A solver which calls Google's ORTools to run Christofides"""

    def __init__(self, tsp: TSP):
        self.tsp = tsp
        self.E = self.tsp.to_edge_matrix()

    def __call__(self) -> Iterable[int]:
        routing = pywrapcp.RoutingModel(len(self.tsp.cities), 1, 0)
        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES)

        def dist_callback(a, b):
            return int(self.E[a, b])
        routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)

        tour = routing.SolveWithParameters(search_parameters)
        index = routing.Start(0)
        result = []
        while not routing.IsEnd(index):
            result.append(int(routing.IndexToNode(index)))
            # yield int(routing.IndexToNode(index))
            index = tour.Value(routing.NextVar(index))
        return result


class PyramidSolver(Solver):
    """A solver which implements a pyramid approximator"""

    def __init__(self, tsp: TSP):
        self.m = tsp.to_matrix()

    def __call__(self) -> Iterable[int]:
        return pyramid_solve(self.m)


# def edge_matrix(cities: [City]):
#     count = len(cities)
#     result = np.zeros((count, count), dtype=np.float32)
#     for a, b in it.combinations(range(len(cities)), 2):
#         A, B = cities[a], cities[b]
#         x = A[0] - B[0]
#         y = A[1] - B[1]
#         result[a,b] = np.sqrt((x * x) + (y * y))
#     il = np.tril_indices(count)
#     result[il] = result.T[il]  # Make symmetric
#     return result


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


def do_iso(tsp: N_TSP, dimensions=2, n_neighbors=5) -> (TSP, TSP):
    iso = Isomap(n_neighbors=n_neighbors, n_components=dimensions)
    V = iso.fit_transform(tsp.to_edge_matrix())
    if dimensions == 2:
        tsp2 = N_TSP.from_cities(recover_local(tsp.to_matrix(), V))
    else:
        tsp2 = N_TSP.from_cities(V.astype(np.int))
    return tsp, tsp2


def do_lle(tsp: N_TSP, dimensions=2, n_neighbors=5) -> (TSP, TSP):
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=dimensions)
    V = lle.fit_transform(tsp.to_edge_matrix())
    if dimensions == 2:
        tsp2 = N_TSP.from_cities(recover_local(tsp.to_matrix(), V))
    else:
        tsp2 = N_TSP.from_cities(V)
    return tsp, tsp2


def do_le(tsp: N_TSP, dimensions=2) -> (TSP, TSP):
    le = SpectralEmbedding(n_components=dimensions, affinity='precomputed')
    V = le.fit_transform(tsp.to_edge_matrix())
    tsp2 = N_TSP.from_cities(recover_local(tsp.to_matrix(), V))
    return tsp, tsp2


class ConcordeMDSSolver(Solver):
    """A solver which implements PyTSP (with support for distance matrices)"""

    def __init__(self, tsp: TSP_O, dimensions=2):
        self.E = tsp.to_edge_matrix()
        self.mds = MDS(n_components=dimensions, metric=True, dissimilarity='precomputed')

    def __call__(self) -> Iterable[int]:
        old_dir = os.getcwd()
        V = self.mds.fit_transform(self.E)
        tsp2 = N_TSP.from_cities(V)
        E2 = tsp2.to_edge_matrix()
        self.outf = '/tmp/tsp.temp'
        with open(self.outf, 'w') as dest:
            dest.write(dumps_matrix(E2))
        tour = run_concorde(self.outf, start=0, solver="concorde")
        os.unlink('/tmp/tsp.temp')
        os.chdir(old_dir)
        return tour['tour']


class ChristofidesMDSSolver(ChristofidesSolver):
    """A solver which implements Christofides with MDS"""

    def __init__(self, tsp: TSP_O, dimensions=2):
        ChristofidesSolver.__init__(self, tsp)
        self.mds = MDS(n_components=dimensions, metric=True, dissimilarity='precomputed')

    def __call__(self) -> Iterable[int]:
        self.E = edge_matrix(self.mds.fit_transform(self.E))
        return ChristofidesSolver.__call__(self)


class PyramidMDSSolver(Solver):
    """A solver which implements a pyramid approximator on an MDS reconstruction of the problem"""

    def __init__(self, tsp: TSP_O, dimensions: int = 2):
        self.E = tsp.to_edge_matrix()
        self.mds = MDS(n_components=dimensions, metric=True, dissimilarity='precomputed')

    def __call__(self) -> Iterable[int]:
        V = np.array(self.mds.fit_transform(self.E), dtype=np.int)
        return pyramid_solve(V)
