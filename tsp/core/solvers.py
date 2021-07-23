from typing import Iterable
import os
import numpy.random as random
from pytsp import dumps_matrix, run as run_concorde

from tsp.core.tsp import N_TSP
from tsp.core.pyramid import pyramid_solve


class Solver:
    """Abstract class for generic TSP solvers."""

    def __init__(self, tsp: N_TSP):
        """Set up the solver based on the given problem.

        Args:
            tsp (N_TSP): problem to be solved.
        """

    def __call__(self) -> Iterable[int]:
        """Run the solver and produce a tour.

        Returns:
            Iterable[int]: tour as indices of cities
        """
        raise NotImplementedError


class RandomSolver(Solver):
    """A solver which produces a random tour."""

    def __init__(self, tsp: N_TSP):
        Solver.__init__(self, tsp)
        self.vertices = list(range(len(tsp.cities)))

    def __call__(self) -> Iterable[int]:
        random.shuffle(self.vertices)
        return self.vertices


class ConcordeSolver(Solver):
    """An optimal solver with the Concorde backend."""

    def __init__(self, tsp: N_TSP):
        Solver.__init__(self, tsp)
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
    """A solver which implements a pyramid approximator."""

    def __init__(self, tsp: N_TSP):
        Solver.__init__(self, tsp)
        self.m = tsp.to_matrix()

    def __call__(self) -> Iterable[int]:
        return pyramid_solve(self.m)
