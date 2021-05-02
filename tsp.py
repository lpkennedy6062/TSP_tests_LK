from typing import Iterable
from collections import namedtuple
import itertools as it
import numpy.random as random
import numpy as np

from tsp.visgraph import calculate_visgraph, shortest_path
from tsp.templates import Template


City = namedtuple('City', ['x', 'y'])


def _euclidean(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def _point_to_line(p, L):
    # see https://stackoverflow.com/a/1501725
    p = np.array(p)
    v1, v2 = np.array(L)
    l2 = _euclidean(v1, v2) ** 2
    if l2 == 0.:
        return _euclidean(p, v1)
    t = max(0., min(1., np.dot(p - v1, v2 - v1) / l2))
    proj = v1 + t * (v2 - v1)
    return _euclidean(p, proj)


def distance(path: [(int, int)]):
    """Calculate the distance along a path of unspecified length"""
    return sum(map(lambda i: _euclidean(path[i - 1], path[i]), range(1, len(path))))


def convert_tour_segments(problem, tour_segments):
    """Expects a list of coordinates, returns a list of cities"""
    cities = list(map(tuple, problem.cities))
    tour_segments = list(map(tuple, tour_segments))
    result = []
    for coord in tour_segments[:-1]:
        result.append(cities.index(coord))
    return result


class N_TSP:
    """Container for an N-dimensional TSP instance"""

    @classmethod
    def from_cities(cls, cities: [[int]]):
        result = cls()
        for coords in cities:
            result.add_city(*coords)
        return result

    def __init__(self):
        self.cities = []
        self.dimensions = 0

    def add_city(self, *args):
        assert self.dimensions == 0 or len(args) == self.dimensions
        self.dimensions = len(args)
        self.cities.append(tuple(args))

    def to_matrix(self) -> np.ndarray:
        return np.array(self.cities)

    def edge(self, a: int, b: int) -> float:
        A, B = self.cities[a], self.cities[b]
        diffsum = 0.
        for i, j in zip(A, B):
            diffsum += (i - j) ** 2
        return np.sqrt(diffsum)

    def to_edges(self) -> ((int, int), (int, int), float):
        """Produces iterable of edges (a, b, d) of distance d between vertices a and b"""
        for a, b in it.combinations(range(len(self.cities)), 2):
            yield a, b, self.edge(a, b)

    def to_edge_matrix(self) -> np.ndarray:
        count = len(self.cities)
        result = np.zeros((count, count), dtype=np.float32)
        for a, b, d in self.to_edges():
            result[a,b] = d
        il = np.tril_indices(count)
        result[il] = result.T[il]  # Make symmetric
        return result

    def score(self, tour: Iterable[int]) -> float:
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

    def tour_segments(self, tour: Iterable[int]) -> ((int, int)):
        return [self.cities[c] for c in tour] + [self.cities[tour[0]]]

    def convert_tour_segments(self, tour_segments):
        """Expects a list of coordinates, returns a list of cities"""
        cities = list(map(tuple, self.cities))
        tour_segments = list(map(tuple, tour_segments))
        result = []
        for coord in tour_segments[:-1]:
            result.append(cities.index(coord))
        return result


class TSP(N_TSP):
    """Container for a TSP instance"""

    @classmethod
    def generate_random(cls, n: int, w: int = 500, h: int = 500, r: int = 10, track_discards = False):
        j = 0
        while True:
            result = cls(w, h)
            for i in range(n):
                x = random.randint(0, w)
                y = random.randint(0, h)
                result.add_city(x, y)
            if min(d for _, __, d in result.to_edges()) >= r:
                return result if not track_discards else (result, j)
            j += 1

    @classmethod
    def from_cities(cls, cities: [[int, int]], w: int = 500, h: int = 500):
        result = cls(w, h)
        for x, y in cities:
            result.add_city(int(x), int(y))
        return result
    
    @classmethod
    def from_tsp(cls, tsp):
        result = cls(tsp.w, tsp.h)
        result.cities = tsp.cities
        return result

    def __init__(self, w: int = 500, h: int = 500):
        N_TSP.__init__(self)
        self.w, self.h = w, h
        self.dimensions = 2

    def add_city(self, x: int, y: int):
        self.cities.append(City(x, y))

    def score_tour_segments(self, tour_segments):
        return distance(tour_segments)


class TSP_Color(N_TSP):

    @classmethod
    def generate_random(cls, n_colors: [int], w: int = 500, h: int = 500, penalty: float = 2.):
        result = cls(w, h, penalty)
        n_total = sum(n_colors)
        while len(set(result.cities)) < n_total:
            result = cls(w, h)
            for c, n in enumerate(n_colors):
                for i in range(n):
                    x = random.randint(10, w - 10)
                    y = random.randint(10, h - 10)
                    result.add_city(x, y, c)
        return result

    @classmethod
    def from_cities(cls, cities: [[int, int], int], w: int = 500, h: int = 500, penalty: float = 2.):
        result = cls(w, h, penalty)
        for xy, c in cities:
            x, y = xy
            result.add_city(int(x), int(y), int(c))
        return result

    def __init__(self, w: int = 500, h: int = 500, penalty: float = 2.):
        N_TSP.__init__(self)
        self.w, self.h = w, h
        self.dimensions = 2
        self.penalty = penalty
        self.colors = []

    def add_city(self, x: int, y: int, color: int):
        self.cities.append(City(x, y))
        self.colors.append(color)

    def edge(self, a: int, b: int) -> float:
        A, B = self.cities[a], self.cities[b]
        Ac, Bc = self.colors[a], self.colors[b]
        penalty = 1 if Ac == Bc else self.penalty
        diffsum = 0.
        for i, j in zip(A, B):
            diffsum += (i - j) ** 2
        return np.sqrt(diffsum) * penalty


class TSP_O(TSP):
    """Container for a TSP-with-obstacles instance"""

    @staticmethod
    def check_min_dist(cities, obstacles, x, y, r, min_dist):
        """Check that (x, y) is not within r of any city, or min_dist of any obstacle"""
        for p in cities:
            if _euclidean(p, (x, y)) < r:
                return False
        for L in obstacles:
            assert len(L) == 2
            if _point_to_line((x, y), L) < min_dist:
                return False
        return True

    @classmethod
    def generate_random_safe(cls, n: int, w: int = 500, h: int = 500, r: int = 10, padding: int = 10,
                             n_obs: int = 10, edge_length: int = 20, min_dist: int = 10, track_discards: bool = False):
        """Generate a random problem in which cities are min_dist from the obstacles"""
        j = 0
        while True:
            result = cls(w, h)
            result.add_random_obstacles(n_obs, edge_length)
            while len(result.cities) < n:
                x = random.randint(padding, w - padding)
                y = random.randint(padding, h - padding)
                if not TSP_O.check_min_dist(result.cities, result.obstacles, x, y, r, min_dist):
                    continue
                result.add_city(x, y)
            try:
                result.vg = None
                result.to_edge_matrix()
            except Exception:
                j += 1
                continue
            if len(result.cities) == n:
                return (result, j) if track_discards else result

    def __init__(self, w: int = 500, h: int = 500):
        TSP.__init__(self, w, h)
        self.obstacles = []  # list of "polygons" i.e. lists of two-tuple vertices
        self.vg = None
        self.E = None

    def add_obstacle(self, *vertices: [(int, int)]):
        """Each vertex is an (x, y) two-tuple"""
        self.obstacles.append([tuple(int(i) for i in v) for v in vertices])

    def to_visgraph(self, rebuild=False):
        if self.vg is None or rebuild:
            self.vg = calculate_visgraph(self.cities, self.obstacles, bound=(self.w, self.h))
            # self.vg = vg.VisGraph()
            # self.vg.build([[vg.Point(*v) for v in o] for o in self.obstacles])
        return self.vg

    def edge(self, a: int, b: int) -> float:
        g = self.to_visgraph()
        A, B = self.cities[a], self.cities[b]
        return distance(shortest_path(A, B, g))

    def to_edges(self) -> ((int, int), (int, int), float):
        """Produces iterable of edges (a, b, d) of distance d between vertices a and b"""
        g = self.to_visgraph()
        for a, b in it.combinations(range(len(self.cities)), 2):
            A, B = self.cities[a], self.cities[b]
            yield a, b, distance(shortest_path(A, B, g))

    def to_edge_matrix(self, rebuild=False):
        if self.E is None or rebuild:
            self.E = TSP.to_edge_matrix(self)
        return self.E

    # def to_edge_matrix(self) -> np.ndarray:
    #     """Compute the distance matrix with the visibility graph"""
    #     count = len(self.cities)
    #     result = np.zeros((count, count), dtype=np.float32)
    #     for a, b, d in self.to_edges():
    #         result[a,b] = d
    #     il = np.tril_indices(count)
    #     result[il] = result.T[il]  # Make symmetric
    #     return result

    def tour_segments(self, tour: Iterable[int]) -> ((int, int)):
        """Produces iterable of vertices (x, y) that, when connected, make up the tour in obstacle space"""
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
            # for p in g.shortest_path(vg.Point(*A), vg.Point(*B)):
            for p in shortest_path(A, B, g):
                if _:
                    yield p
                    # yield int(p.x), int(p.y)
                else:
                    _ = True  # Discard the first point so there are no duplicates
            prev = B
        _ = False
        # for p in g.shortest_path(vg.Point(*prev), vg.Point(*first)):
        for p in shortest_path(prev, first, g):
            if _:
                # yield int(p.x), int(p.y)
                yield p
            else:
                _ = True

    # Helper methods for making obstacles

    def add_random_obstacle(self, edge_length: int = 20):
        xc, yc = random.randint(0, self.w), random.randint(0, self.h)
        alpha = np.pi * random.random()
        x1 = xc + int(edge_length * np.cos(alpha) / 2.)
        y1 = yc + int(edge_length * np.sin(alpha) / 2.)
        x2 = xc - int(edge_length * np.cos(alpha) / 2.)
        y2 = yc - int(edge_length * np.sin(alpha) / 2.)
        self.add_obstacle((x1, y1), (x2, y2))
        # x1, y1 = random.randint(1, self.w - 1), random.randint(1, self.h - 1)
        # x2, y2 = 0, 0
        # while x2 <= 1 or y2 <= 1:
        #     alpha = 2. * np.pi * random.random()
        #     x2 = int((edge_length * np.cos(alpha)) + x1)
        #     y2 = int((edge_length * np.sin(alpha)) + y1)
        # self.add_obstacle((x1, y1), (x2, y2))

    def add_random_obstacles(self, n: int, edge_length: int = 20):
        for i in range(n):
            self.add_random_obstacle(edge_length)
        self.to_visgraph(True)

    def add_random_obstacles_from_template(self, template: Template, n: int = 1, start_offsets = (0, 0)):
        for i in range(n):
            template(
                (start_offsets[0], self.w - start_offsets[0]),
                (start_offsets[1], self.h - start_offsets[1]),
                self.add_obstacle
            )
        self.to_visgraph(True)
