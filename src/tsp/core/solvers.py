"""Implements random, optimal, and human-approximate solvers.

The new version of this API implements solvers as procedures which take in a TSP object as their
single argument (plus additional keyword arguments specific to the model), and returns an array
of integers corresponding to the indices of the vertices ordered as a tour. You can implement a new
solver by following this format.

[OLD DOCUMENTATION: Other solvers can be implemented by extending `Solver`, which functions as an
abstract class. Due to a historical contingency in the depths of the past, the API is somewhat
opaque, and it's questionable whether we might have been better off if the solvers weren't
object-oriented. The basic idea is that the initializer sets up the solver, and then the
`__call__` method is what does the computation and returns the tour. So, `Solver.__init__` and
`Solver.__call__` should be overridden by subclasses - while `Solver.solve` is a newer addition
serving as syntactical sugar for `Solver.__call__`.]

The random solver is exactly as advertised - returning a random permutation of the cities as a
solution.

The optimal solver uses the [Concorde](https://www.math.uwaterloo.ca/tsp/concorde.html) backend.
Sadly, Concorde can be a difficult thing to get working on a machine, but it is the gold standard
in cognitive science research on TSP. Once you have Concorde installed, it is much easier using
this library to find optimal tours than using Concorde directly.

The human-approximate solver uses a hierarchical clustering ("pyramid") algorithm implemented in the
`tsp.core.pyramid` submodule.
"""


import os
import warnings
from numpy.typing import NDArray
import numpy as np
from pytsp import dumps_matrix, run as run_concorde

from tsp.core.tsp import N_TSP
from tsp.core.pyramid import pyramid_solve as pyramid_solve_


def random_solve(tsp: N_TSP, **kwargs) -> NDArray:
    """A solver which produces a random tour.

    Args:
        tsp (N_TSP): TSP to solve

    Returns:
        NDArray: solution as vertex indices
    """
    vertices = np.arange(tsp.cities.shape[0])
    np.random.shuffle(vertices)
    return vertices


def concorde_solve(tsp: N_TSP, **kwargs) -> NDArray:
    """An optimal solver with the Concorde backend.

    Args:
        tsp (N_TSP): TSP to solve

    Returns:
        NDArray: solution as vertex indices
    """
    E = tsp.to_edge_matrix()
    outf = './tsp.temp'
    with open(outf, 'w') as dest:
        dest.write(dumps_matrix(E))

    old_dir = os.getcwd()
    tour = run_concorde(outf, start=0, solver="concorde")
    os.unlink(outf)
    os.chdir(old_dir)
    return np.array(tour['tour'])


def pyramid_solve(tsp: N_TSP, **kwargs) -> NDArray:
    """A solver which implements a pyramid approximator. See `tsp.core.pyramid.pyramid_solve` for
    possible keyword arguments.

    Args:
        tsp (N_TSP): TSP to solve

    Returns:
        NDArray: solution as vertex indices
    """
    return np.array(pyramid_solve_(tsp.cities, **kwargs))


class Solver:
    """[DEPRECATED] Abstract class for generic TSP solvers."""

    def __init__(self, tsp: N_TSP):
        """[DEPRECATED] Set up the solver based on the given problem.

        Args:
            tsp (N_TSP): problem to be solved.
        """
        warnings.warn('Prefer the new solver API (see documentation for details)', DeprecationWarning)

    def __call__(self) -> NDArray:
        """[DEPRECATED] Run the solver and produce a tour.

        Returns:
            NDArray: tour as indices of cities
        """
        raise NotImplementedError

    def solve(self) -> NDArray:
        """[DEPRECATED] Run the solver and produce a tour.

        Returns:
            NDArray: tour as indices of cities
        """
        return self()


class RandomSolver(Solver):
    """[DEPRECATED] A solver which produces a random tour."""

    def __init__(self, tsp: N_TSP):
        Solver.__init__(self, tsp)
        self.vertices = np.arange(tsp.cities.shape[0])

    def __call__(self) -> NDArray:
        np.random.shuffle(self.vertices)
        return self.vertices


class ConcordeSolver(Solver):
    """[DEPRECATED] An optimal solver with the Concorde backend."""

    def __init__(self, tsp: N_TSP):
        Solver.__init__(self, tsp)
        E = tsp.to_edge_matrix()
        self.outf = './tsp.temp'
        with open(self.outf, 'w') as dest:
            dest.write(dumps_matrix(E))

    def __call__(self) -> NDArray:
        old_dir = os.getcwd()
        tour = run_concorde(self.outf, start=0, solver="concorde")
        os.unlink(self.outf)
        os.chdir(old_dir)
        return np.array(tour['tour'])


class PyramidSolver(Solver):
    """[DEPRECATED] A solver which implements a pyramid approximator."""

    def __init__(self, tsp: N_TSP):
        Solver.__init__(self, tsp)
        self.m = tsp.cities

    def __call__(self) -> NDArray:
        return np.array(pyramid_solve_(self.m))
