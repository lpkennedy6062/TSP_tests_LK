"""Container for TSP with obstacles (TSP-O).

In a TSP-O there may be straight-line obstacles occluding the shortest paths between cities. When
this happens, the shortest path will no longer be a straight line between the two vertices, thus
violating an axiom of Euclidean geometry. However, the metric axioms are not violated. As this has
nothing to do with "non-Euclidean geometry" proper, we have taken to calling it a "*not*-Euclidean"
problem rather than a *non*-Euclidean problem.

Note that the generated edge matrix from a `TSP_O` object will use the shortest paths found using
the visibility graph and Dijkstra's algorithm implemented in `tsp.extra.visgraph`.

If you are interested in generating TSP-Os with more complex obstacles made from a bunch of
line segments arranged in some kind of template, the code for that can be found in
`tsp.extra.templates`.
"""


from typing import Iterable, Iterator, DefaultDict, Tuple
import itertools as it
from numpy.typing import NDArray
import numpy.random as random
import numpy as np

from tsp.core.tsp import TSP, distance
from tsp.extra.visgraph import calculate_visgraph, shortest_path
from tsp.extra.templates import Template


def _point_to_line(p: NDArray, L: NDArray) -> float:
    """Calculates distance from point p to line L. Adapted from https://stackoverflow.com/a/1501725.

    Args:
        p (NDArray): point coordinates [x, y, ...]
        L (NDArray): line coordinates [[x1, y1, ...], [x2, y2, ...]]

    Returns:
        float: distance
    """
    p = np.array(p)
    v1, v2 = np.array(L)
    l2 = np.linalg.norm(v1 - v2) ** 2
    if np.isclose(l2, 0):
        return np.linalg.norm(p - v1)
    t = max(0., min(1., np.dot(p - v1, v2 - v1) / l2))
    proj = v1 + t * (v2 - v1)
    return np.linalg.norm(p - proj)


class TSP_O(TSP):
    """Container for a TSP-with-obstacles instance."""

    @staticmethod
    def _check_min_dist(cities: Iterable[NDArray], obstacles: Iterable[NDArray], x: int, y: int, r: int, min_dist: int) -> bool:
        """Check that (x, y) is not within r of any city, or min_dist of any obstacle.

        Args:
            cities (Iterable[NDArray]): cities as [[x1, y1], ...]
            obstacles (Iterable[NDArray]): obstacles as [[[x1, y1], [x2, y2]], ...]
            x (int): city x-coordinate
            y (int): city y-coordinate
            r (int): minimum distance allowed between cities
            min_dist (int): minimum distance allowed between a city and obstacles

        Returns:
            bool: whether city can be safely added
        """
        for p in cities:
            if np.linalg.norm(p - np.array([x, y])) < r:
                return False
        for L in obstacles:
            assert len(L) == 2
            if _point_to_line((x, y), L) < min_dist:
                return False
        return True

    @classmethod
    def generate_random_safe(cls, n: int, w: int = 500, h: int = 500, r: int = 10, padding: int = 10,
                             n_obs: int = 10, edge_length: int = 20, min_dist: int = 10):
        """Generate a random problem in which cities are min_dist from the obstacles.

        Args:
            n (int): number of cities
            w (int, optional): Width of problem. Defaults to 500.
            h (int, optional): Height of problem. Defaults to 500.
            r (int, optional): Minimum distance between cities. Defaults to 10.
            padding (int, optional): Minimum distance a city can be from the edge. Defaults to 10.
            n_obs (int, optional): Number of obstacles. Defaults to 10.
            edge_length (int, optional): Length of obstacles. Defaults to 20.
            min_dist (int, optional): Minimum distance a city can be from an obstacle. Defaults to 10.
        """
        while True:
            result = cls(w, h)
            result.add_random_obstacles(n_obs, edge_length)
            while len(result.cities) < n:
                x = random.randint(padding, w - padding)
                y = random.randint(padding, h - padding)
                if not TSP_O._check_min_dist(result.cities, result.obstacles, x, y, r, min_dist):
                    continue
                result.add_city(x, y)
            try:
                result.vg = None
                result.to_edge_matrix()
            except Exception:
                continue
            if len(result.cities) == n:
                return result

    def __init__(self, w: int = 500, h: int = 500):
        TSP.__init__(self, w, h)
        self.obstacles = np.array([])  # list of "polygons" i.e. lists of two-tuple vertices
        self.vg = None
        self.E = None

    def add_obstacle(self, *vertices: Tuple[int]):
        """Inefficiently add obstacles to the problem.

        Args:
            vertices Tuple[int]: obstacle vertices as [x1, y1], ...
        """
        self.obstacles = list(self.obstacles)
        self.obstacles.append(np.array([tuple(int(i) for i in v) for v in vertices]))
        self.obstacles = np.array(self.obstacles)

    def to_visgraph(self, rebuild: bool = False) -> DefaultDict:
        """Generate and return a visibility graph for the problem.

        Args:
            rebuild (bool, optional): Whether or not to rebuild from scratch. Defaults to False.

        Returns:
            DefaultDict: visibility graph
        """
        if self.vg is None or rebuild:
            self.vg = calculate_visgraph(self.cities, self.obstacles, bound=(self.w, self.h))
        return self.vg

    def edge(self, a: int, b: int) -> float:
        """Calculate shortest path between two cities.

        Args:
            a (int): index of first city
            b (int): index of second city

        Returns:
            float: edge length
        """
        g = self.to_visgraph()
        A, B = self.cities[a], self.cities[b]
        return distance(shortest_path(A, B, g))

    def to_edges(self) -> Iterable[Tuple[int, int, float]]:
        """Produces iterable of edges (a, b, d) of distance d between vertices a and b.

        Yields:
            Iterator[int, int, float]: edges
        """
        g = self.to_visgraph()
        for a, b in it.combinations(range(len(self.cities)), 2):
            A, B = self.cities[a], self.cities[b]
            yield a, b, distance(shortest_path(A, B, g))

    def to_edge_matrix(self) -> NDArray:
        """Generate an edge matrix from the problem.

        Returns:
            NDArray: edge matrix
        """
        if self.E is None:
            self.E = TSP.to_edge_matrix(self)
        return self.E

    def tour_segments(self, tour: Iterable[int]) -> Iterator[NDArray]:
        """Produces iterator of vertices (x, y) that, when connected, make up the tour in obstacle space.

        Args:
            tour (Iterable[int]): tour as indices of cities

        Yields:
            Iterator[NDArray]: tour as coordinates of line segments
        """
        g = self.to_visgraph()
        prev = None
        first = None
        for c in tour:
            if prev is None:
                prev = self.cities[c]
                first = prev
                yield prev
                continue
            A, B = prev, self.cities[c]
            _ = False
            for p in shortest_path(A, B, g):
                if _:
                    yield p
                else:
                    _ = True  # Discard the first point so there are no duplicates
            prev = B
        _ = False
        for p in shortest_path(prev, first, g):
            if _:
                yield p
            else:
                _ = True

    # Helper methods for making obstacles

    def add_random_obstacle(self, edge_length: int = 20):
        """Add an obstacle with random positioning and rotation.

        Args:
            edge_length (int, optional): Obstacle length. Defaults to 20.
        """
        xc, yc = random.randint(0, self.w), random.randint(0, self.h)
        alpha = np.pi * random.rand()
        x1 = xc + int(edge_length * np.cos(alpha) / 2.)
        y1 = yc + int(edge_length * np.sin(alpha) / 2.)
        x2 = xc - int(edge_length * np.cos(alpha) / 2.)
        y2 = yc - int(edge_length * np.sin(alpha) / 2.)
        self.add_obstacle((x1, y1), (x2, y2))

    def add_random_obstacles(self, n: int, edge_length: int = 20):
        """Add multiple random obstacles.

        Args:
            n (int): number of obstacles
            edge_length (int, optional): Obstacle length. Defaults to 20.
        """
        for _ in range(n):
            self.add_random_obstacle(edge_length)
        self.to_visgraph(True)

    def add_random_obstacles_from_template(self, template: Template, n: int = 1, start_offsets = (0, 0)):
        """Add obstacles using a template.

        Args:
            template (Template): template
            n (int, optional): Number of obstacles. Defaults to 1.
            start_offsets (tuple, optional): Bounds for template (see template documentation). Defaults to (0, 0).
        """
        for _ in range(n):
            template(
                self.add_obstacle,
                (start_offsets[0], self.w - start_offsets[0]),
                (start_offsets[1], self.h - start_offsets[1])
            )
        self.to_visgraph(True)
