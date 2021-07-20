from typing import Iterable, Iterator, DefaultDict
from numpy.typing import ArrayLike, NDArray
import itertools as it
import numpy.random as random
import numpy as np


def distance(path: Iterable[ArrayLike]) -> float:
    """Calculate the distance along a path of unspecified length.

    Args:
        path (Iterable[ArrayLike]): path as [[x1, y1, ...], ...]

    Returns:
        float: distance
    """
    return sum(map(lambda i: np.linalg.norm(path[i - 1] - path[i]), range(1, len(path))))


class N_TSP:
    """Container for a generic TSP instance."""

    @classmethod
    def from_cities(cls, cities: ArrayLike):
        """Generate object from list/array of cities.

        Args:
            cities (ArrayLike): cities as [[x1, y1, ...], ...]
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
        assert self.dimensions == 0 or len(coords) == self.dimensions
        self.dimensions = len(coords)
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

    def to_edges(self) -> Iterator[int, int, float]:
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

    def tour_segments(self, tour: Iterable[int]) -> Iterator[ArrayLike]:
        """Convert a tour in index format into a tour in segment format.

        Args:
            tour (Iterable[int]): tour as indices of cities

        Yields:
            Iterator[ArrayLike]: tour as coordinates of cities
        """
        for c in tour:
            yield self.cities[c]
        yield self.cities[tour[0]]

    def convert_tour_segments(self, tour_segments: Iterable[ArrayLike]) -> Iterator[int]:
        """Convert a tour in segment format into a tour in index format.

        Args:
            tour_segments (Iterable[ArrayLike]): tour as coordinates of cities

        Yields:
            Iterator[int]: tour as indices of cities
        """
        cities = list(map(tuple, self.cities))
        tour_segments = list(map(tuple, tour_segments))
        for coord in tour_segments[:-1]:
            yield cities.index(coord)

    def score(self, tour: Iterable[int]) -> float:
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

    def score_tour_segments(self, tour_segments: Iterable[ArrayLike]) -> float:
        """Calculate tour length (from segment format).

        Args:
            tour_segments (Iterable[ArrayLike]): tour as coordinates of cities

        Returns:
            float: tour length
        """
        return distance(tour_segments)


class TSP(N_TSP):
    """Container for a TSP instance. A wrapper which adds constraints to the N_TSP interface, primarily width and height."""

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
    def from_cities(cls, cities: ArrayLike, w: int = 500, h: int = 500):
        """Generate object from list/array of cities.

        Args:
            cities (ArrayLike): cities as [[x1, y1], ...]
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


