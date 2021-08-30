"""Object-oriented containers for both 2-dimensional and n-dimensional TSPs, including procedures
for randomly generating 2-dimensional TSPs.

`N_TSP` is the superclass from which all TSPs here and in `tsp.extra` inherit. It can handle TSPs
of arbitrary dimension, and implements all the methods needed for a Solver
(tsp.core.solvers.Solver) to be able to generate a tour. The convention is that TSPs are stored as
collections of cities, with distance matrices made available as second-class citizens.

`TSP` extends `N_TSP`, enforcing 2-dimensional cities and storing width (`TSP.w`) and height
(`TSP.h`) attributes. It also implements a constructor (class method) for generating problems with
uniformly-randomly distributed cities (TSP).

To copy a TSP, you can use the `from_cities` class method. Example usage:

```python
# generate a TSP from an N_TSP with 2-dimensional cities
# ... (N_TSP stored as `original_problem`)
problem_2d = TSP.from_cities(original_problem.cities, w=500, h=500)

# convert back
original_problem = N_TSP.from_cities(problem_2d.cities)
```

TSP types `tsp.extra.obstacles.TSP_O` and `tsp.extra.color.TSP_Color` extend `TSP` and `N_TSP`
respectively.
"""


from typing import Callable, Iterable, Iterator, Tuple, Type, Union
import itertools as it
from numpy.typing import NDArray
import numpy.random as random
import numpy as np


def distance(path: Iterable[NDArray]) -> float:
    """Calculate the distance along a path of unspecified length.

    Args:
        path (Iterable[NDArray]): path as [[x1, y1, ...], ...]

    Returns:
        float: distance
    """
    path = np.array(list(path))
    return sum(map(lambda i: np.linalg.norm(path[i - 1] - path[i]), range(1, len(path))))


class N_TSP:
    """Container for a generic TSP instance."""

    @classmethod
    def from_cities(cls, cities: NDArray):
        """Generate object from list/array of cities.

        Args:
            cities (NDArray): cities as [[x1, y1, ...], ...]
        """
        result = cls()
        result.cities = np.array(cities)
        return result

    def __init__(self):
        self.cities = np.array([])

    @property
    def dimensions(self) -> int:
        """Number of dimensions the N_TSP problem is in.

        Returns:
            int: dimensions
        """
        return self.cities.shape[1]

    def add_city(self, *coords: int):
        """Inefficiently adds a city to the problem.

        Args:
            coords ([int]): city coordinates as individual arguments
        """
        assert self.cities.shape[0] == 0 or len(coords) == self.dimensions
        self.cities = list(self.cities)
        self.cities.append(np.array(coords))
        self.cities = np.array(self.cities)

    def edge(self, a: int, b: int) -> float:
        """Edge length between two cities.

        Args:
            a (int): index of first city
            b (int): index of second city

        Returns:
            float: edge length
        """
        return np.linalg.norm(self.cities[a] - self.cities[b])

    def to_edges(self) -> Iterator[Tuple[int, int, float]]:
        """Produces iterable of edges (a, b, d) of distance d between vertices a and b.

        Yields:
            Iterator[int, int, float]: edges
        """
        for a, b in it.combinations(range(len(self.cities)), 2):
            yield a, b, self.edge(a, b)

    def to_edge_matrix(self) -> NDArray:
        """Generate an edge matrix from the problem.

        Returns:
            NDArray: edge matrix
        """
        count = len(self.cities)
        result = np.zeros((count, count), dtype=np.float32)
        for a, b, d in self.to_edges():
            result[a, b] = d
        il = np.tril_indices(count)
        result[il] = result.T[il]  # Make symmetric  # pylint: disable=E1136
        return result

    def solve(self, solver: Union[Callable, Type], **kwargs) -> NDArray:
        """Generate a tour using a Solver.

        Args:
            solver (Union[Callable, Type]): a solver function from `tsp.core.solvers`

        Returns:
            NDArray: tour
        """
        if isinstance(solver, Type):
            return np.array(solver(self)())  # for compatibility with old API
        return np.array(solver(self, **kwargs))

    def tour_segments(self, tour: Iterable[int]) -> Iterator[NDArray]:
        """Convert a tour in index format into a tour in segment format.

        Args:
            tour (Iterable[int]): tour as indices of cities

        Yields:
            Iterator[NDArray]: tour as coordinates of cities
        """
        for c in tour:
            yield self.cities[c]
        yield self.cities[tour[0]]

    def convert_tour_segments(self, tour_segments: Iterable[NDArray]) -> Iterator[int]:
        """Convert a tour in segment format into a tour in index format.

        Args:
            tour_segments (Iterable[NDArray]): tour as coordinates of cities

        Yields:
            Iterator[int]: tour as indices of cities
        """
        cities = list(map(tuple, self.cities))
        tour_segments = list(map(tuple, tour_segments))
        for coord in tour_segments[:-1]:
            yield cities.index(coord)

    def score_indices(self, tour: Iterable[int]) -> float:
        """Calculate tour length (from index format).

        Args:
            tour (Iterable[int]): tour as indices of cities

        Returns:
            float: tour length
        """
        result = 0.
        first = None
        prev = None
        for c in tour:
            if prev is None:
                first = c
                prev = c
                continue
            result += self.edge(prev, c)
            prev = c
        result += self.edge(c, first)
        return result

    def score_tour_segments(self, tour_segments: Iterable[NDArray]) -> float: # pylint: disable=no-self-use
        """Calculate tour length (from segment format).

        Args:
            tour_segments (Iterable[NDArray]): tour as coordinates of cities

        Returns:
            float: tour length
        """
        return distance(tour_segments)

    def score(self, tour: Iterable[Union[int, NDArray]]) -> float:
        """Calculate tour length (automatically detects index or segment format).

        Args:
            tour (Iterable[Union[int, NDArray]]): tour

        Returns:
            float: tour length
        """
        s = list(tour)
        if isinstance(s[0], int):
            return self.score_indices(s)
        return self.score_tour_segments(s)


class TSP(N_TSP):
    """Container for a 2D TSP instance. A wrapper which adds constraints to the N_TSP interface, primarily width and height."""

    @classmethod
    def generate_random(cls, n: int, w: int = 500, h: int = 500, r: int = 10, padding: int = 10):
        """Generate a new problem with uniformly-distributed random cities.

        Args:
            n (int): number of cities
            w (int, optional): Width of problem. Defaults to 500.
            h (int, optional): Height of problem. Defaults to 500.
            r (int, optional): Minimum distance between cities. Defaults to 10.
            padding (int, optional): Minimum distance a city can be from the edge. Defaults to 10.
        """
        while True:
            result = cls(w, h)
            for _ in range(n):
                x = random.randint(padding, w - padding)
                y = random.randint(padding, h - padding)
                result.add_city(x, y)
            if min(d for _, __, d in result.to_edges()) >= r:
                return result

    @classmethod
    def from_cities(cls, cities: NDArray, w: int = 500, h: int = 500):
        """Generate object from list/array of cities.

        Args:
            cities (NDArray): cities as [[x1, y1], ...]
            w (int, optional): Width of problem. Defaults to 500.
            h (int, optional): Height of problem. Defaults to 500.
        """
        result = cls(w, h)
        for x, y in cities:
            result.add_city(int(x), int(y))
        return result

    def __init__(self, w: int = 500, h: int = 500):
        N_TSP.__init__(self)
        self.w, self.h = w, h
