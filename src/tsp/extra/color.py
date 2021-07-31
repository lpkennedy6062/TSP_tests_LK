"""Container for TSP with colors.

In a TSP with color, traveling between cities of different color can incur a proportionally higher
cost than traveling between cities of the same color. This could result in violations of the
triangle inequality, making it a genuinely non-metric TSP.

Note that this is the least developed code in the library.
"""


from typing import Iterable, Tuple
import numpy.random as random
import numpy as np

from tsp.core.tsp import N_TSP


class TSP_Color(N_TSP):
    """Container for a TSP-with-colors instance."""

    @classmethod
    def generate_random(cls, n_colors: Iterable[int], w: int = 500, h: int = 500, penalty: float = 2.):
        """Generate a new problem with uniformly-distributed random cities of different colors.

        Args:
            n_colors (Iterable[int]): number of cities of each color, for example [25, 25] would be a
                50-city problem with two colors evenly distributed
            w (int, optional): Width of problem. Defaults to 500.
            h (int, optional): Height of problem. Defaults to 500.
            penalty (float, optional): Distance multiplier when traveling between colors. Defaults to 2.0.
        """
        result = cls(w, h, penalty)
        n_total = sum(n_colors)
        while len(set(result.cities)) < n_total:
            result = cls(w, h)
            for c, n in enumerate(n_colors):
                for _ in range(n):
                    x = random.randint(10, w - 10)
                    y = random.randint(10, h - 10)
                    result.add_city(x, y, c)
        return result

    @classmethod
    def from_cities(cls, cities: Iterable[Tuple[Tuple[int, int], int]], w: int = 500, h: int = 500, penalty: float = 2.):
        """Generate object from list/array of (colored) cities.

        Args:
            cities (Iterable[Tuple[Tuple[int, int], int]]): colored cities as [((x1, y1), c1), ...]
            w (int, optional): Width of problem. Defaults to 500.
            h (int, optional): Height of problem. Defaults to 500.
            penalty (float, optional): Distance multiplier when traveling between colors. Defaults to 2.0.
        """
        result = cls(w, h, penalty)
        for xy, c in cities:
            x, y = xy
            result.add_city(int(x), int(y), int(c))
        return result

    def __init__(self, w: int = 500, h: int = 500, penalty: float = 2.):
        N_TSP.__init__(self)
        self.w, self.h = w, h
        self.penalty = penalty
        self.colors = np.array([])

    def add_city(self, x: int, y: int, color: int):
        """Inefficiently adds a (colored) city to the problem.

        Args:
            x (int): city x
            y (int): city y
            color (int): city color
        """
        N_TSP.add_city(x, y)
        self.colors = list(self.colors)
        self.colors.append(color)
        self.colors = np.array(self.colors)

    def edge(self, a: int, b: int) -> float:
        """Edge length between two cities, taking into account penalties for switching colors.

        Args:
            a (int): index of first city
            b (int): index of second city

        Returns:
            float: edge length
        """
        A, B = self.cities[a], self.cities[b]
        Ac, Bc = self.colors[a], self.colors[b]
        penalty = 1 if Ac == Bc else self.penalty
        diffsum = 0.
        for i, j in zip(A, B):
            diffsum += (i - j) ** 2
        return np.sqrt(diffsum) * penalty
