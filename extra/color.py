import numpy.random as random
import numpy as np
from PIL import Image, ImageDraw

from ..tsp import City, N_TSP


def draw_cities_color(im, tsp):
    draw = ImageDraw.Draw(im)
    for i, p in enumerate(tsp.cities):
        x, y = p
        c = 'red' if tsp.colors[i] == 0 else 'blue'
        draw.ellipse([(x*2 - 8, y*2 - 8), (x*2 + 8, y*2 + 8)], c, outline=c)


def draw_tour_color(im, tsp, tour):
    draw = ImageDraw.Draw(im)
    draw.line([(x*2, y*2) for x, y in tsp.tour_segments(tour)], fill='black', width=3)


def visualize_color(t, s, path):
    im = Image.new('RGB', (t.w * 2, t.h * 2), color = 'white')
    if s:
        draw_tour_color(im, t, s)
    draw_cities_color(im, t)
    im.thumbnail((t.w, t.h))
    im.save(path)


class TSP_Color(N_TSP):

    @classmethod
    def generate_random(cls, n_colors: [int], w: int = 500, h: int = 500, penalty: float = 2.):
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