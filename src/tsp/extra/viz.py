"""Procedures for visualizing TSP-Os and TSPs with color using the PIL and MatPlotLib backends.
"""


from typing import Iterable, Union
from numpy.typing import NDArray
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.axes import SubplotBase

from tsp.core.viz import _draw_cities_pil, _draw_tour_pil, visualize_tsp_plt
from tsp.extra.obstacles import TSP_O
from tsp.extra.color import TSP_Color


def _draw_obstacles_pil(im: Image, tsp: TSP_O):
    draw = ImageDraw.Draw(im)
    for a, b in tsp.obstacles:
        draw.line([(a[0]*2, a[1]*2), (b[0]*2, b[1]*2)], fill='black', width=4)


def visualize_obstacles_pil(tsp: TSP_O, tour: Iterable[Union[int, NDArray]], path: str):
    """Generate and save visualization of a TSP_O using PIL backend.

    Args:
        tsp (TSP_O): the problem
        tour (Iterable[Union[int, NDArray]]): tour either as indices of vertices or as segments
        path (str): path to save
    """
    im = Image.new('RGB', (tsp.w * 2, tsp.h * 2), color = 'white')
    if len(tour):
        _draw_tour_pil(im, tsp, tour)
    _draw_cities_pil(im, tsp)
    _draw_obstacles_pil(im, tsp)
    im.thumbnail((tsp.w, tsp.h))
    im.save(path)


def _draw_obstacles_plt(ax: SubplotBase, tsp: TSP_O):
    for a, b in tsp.obstacles:
        a, b = list(a), list(b)  # copy objects
        a[1] = tsp.h - a[1]
        b[1] = tsp.h - b[1]
        ax.plot(*zip(a, b), 'k-')


def visualize_obstacles_plt(tsp: TSP_O, tour: Iterable[Union[int, NDArray]], ax: SubplotBase = None):
    """Generate visualization of a TSP_O using MatPlotLib backend.

    Args:
        tsp (TSP_O): the problem
        tour (Iterable[Union[int, NDArray]]): tour either as indices of vertices or as segments
        ax (SubplotBase): Matplotlib axes to plot on. Defaults to None.
    """
    if ax is None:
        ax = plt.subplot(111)

    visualize_tsp_plt(tsp, tour, ax)
    _draw_obstacles_plt(ax, tsp)


def _draw_cities_color_pil(im: Image, tsp: TSP_Color):
    draw = ImageDraw.Draw(im)
    for i, p in enumerate(tsp.cities):
        x, y = p
        c = 'red' if tsp.colors[i] == 0 else 'blue'
        draw.ellipse([(x*2 - 8, y*2 - 8), (x*2 + 8, y*2 + 8)], c, outline=c)


def _draw_tour_color_pil(im: Image, tsp: TSP_Color, tour: Iterable[int]):
    draw = ImageDraw.Draw(im)
    draw.line([(x*2, y*2) for x, y in tsp.tour_segments(tour)], fill='black', width=3)


def visualize_color_pil(tsp: TSP_Color, tour: Iterable[int], path: str):
    """Generate and save visualization of a TSP_Color using PIL backend.
    Due to the fact that the TSP_Color code is underdeveloped, this is not as feature rich.
    Only supports two colors, and tours as indices of vertices.

    Args:
        tsp (TSP_Color): the problem
        tour (Iterable[int]): tour either as indices of vertices
        path (str): path to save
    """
    im = Image.new('RGB', (tsp.w * 2, tsp.h * 2), color = 'white')
    if len(tour):
        _draw_tour_color_pil(im, tsp, tour)
    _draw_cities_color_pil(im, tsp)
    im.thumbnail((tsp.w, tsp.h))
    im.save(path)
