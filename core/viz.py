from math import sin, cos, radians
from typing import Iterable, Union
from numpy.typing import ArrayLike
import numpy as np
from PIL import Image, ImageDraw
from cv2 import cv2
import matplotlib.pyplot as plt
from matplotlib.axes import SubplotBase

from tsp.core.tsp import N_TSP, TSP


def _draw_edges_pil(im: Image, tsp: TSP, edges: Iterable[ArrayLike]):
    draw = ImageDraw.Draw(im)
    for e1, e2 in edges:
        e1 = tuple(np.array(tsp.cities[e1]) * 2)
        e2 = tuple(np.array(tsp.cities[e2]) * 2)
        draw.line([e1, e2], fill='blue', width=6)


def _draw_cities_pil(im: Image, tsp: TSP):
    draw = ImageDraw.Draw(im)
    for x, y in tsp.cities:
        draw.ellipse([(x*2 - 8, y*2 - 8), (x*2 + 8, y*2 + 8)], fill='red', outline='red')


def _draw_tour_pil(im: Image, tsp: TSP, tour: Iterable[Union[int, ArrayLike]]):
    draw = ImageDraw.Draw(im)
    s = list(tour)
    if isinstance(s[0], int):
        draw.line([(x*2, y*2) for x, y in tsp.tour_segments(s)], fill='blue', width=6)
    else:
        draw.line([(x*2, y*2) for x, y in s], fill='blue', width=6)


def visualize_tsp_pil(tsp: TSP, tour: Iterable[Union[int, ArrayLike]], path: str):
    """Generate and save visualization of a TSP using PIL backend.

    Args:
        tsp (TSP): the problem
        tour (Iterable[Union[int, ArrayLike]]): tour either as indices of vertices or as segments
        path (str): path to save
    """
    im = Image.new('RGB', (tsp.w * 2, tsp.h * 2), color = 'white')
    if tour:
        _draw_tour_pil(im, tsp, tour)
    _draw_cities_pil(im, tsp)
    im.thumbnail((tsp.w, tsp.h))
    im.save(path)


def visualize_mst_pil(tsp: TSP, edges: Iterable[ArrayLike], path: str):
    """Generate and save visualization of an MST using PIL backend.

    Args:
        tsp (TSP): the problem
        edges (Iterable[ArrayLike]): edges in MST as [[[x1, y1], [x2, y2]], ...]
        path (str): path to save
    """
    im = Image.new('RGB', (tsp.w * 2, tsp.h * 2), color = 'white')
    _draw_edges_pil(im, tsp, edges)
    _draw_cities_pil(im, tsp)
    im.thumbnail((tsp.w, tsp.h))
    im.save(path)


def visualize_3d(tsp: N_TSP, tour: Iterable[Union[int, ArrayLike]], path: str, step: int = 1, time: int = 12):
    """Generate and save visualization of 3D motion of a 3D TSP as an mp4.

    Args:
        tsp (N_TSP): the problem
        tour (Iterable[Union[int, ArrayLike]]): tour either as indices of vertices or as segments
        path (str): path to save
        step (int, optional): Degrees to rotate per frame. Defaults to 1.
        time (int, optional): Duration of generated video. Defaults to 12.
    """
    assert tsp.dimensions == 3
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    for alpha in range(0, 360, step):
        for x, y, z in tsp.cities:
            x_ = x*cos(radians(alpha)) + z*sin(radians(alpha))
            y_ = y
            min_x = min(min_x, x_)
            min_y = min(min_y, y_)
            max_x = max(max_x, x_)
            max_y = max(max_y, y_)
    min_x -= 10  # padding
    min_y -= 10
    max_x -= min_x - 10  # padding on max side
    max_y -= min_y - 10
    min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)

    video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), int(round((360 / step) / time)), (max_x, max_y))
    for alpha in range(0, 360, step):
        new_coords = []
        for x, y, z in tsp.cities:
            new_coords.append((
                x*cos(radians(alpha)) + z*sin(radians(alpha)),
                y
            ))
        new_coords = np.array(new_coords, dtype=np.int)
        new_coords -= np.array([min_x, min_y])

        new_t = TSP.from_cities(new_coords, w=max_x, h=max_y)
        visualize_mst_pil(new_t, tour, '/tmp/viz.png')
        video.write(cv2.imread('/tmp/viz.png'))
    video.release()


def _draw_edges_plt(ax: SubplotBase, tsp: TSP, edges: Iterable[ArrayLike]):
    for e1, e2 in edges:
        e1[1] = tsp.h - e1[1]
        e2[1] = tsp.h - e2[1]
        ax.plot(*zip(e1, e2), 'b-')


def _draw_cities_plt(ax: SubplotBase, tsp: TSP):
    cities = np.array(list(zip(*tsp.cities)))
    cities[1] = tsp.h - cities[1]
    ax.plot(*cities, 'ro')


def _draw_tour_plt(ax: SubplotBase, tsp: TSP, tour: Iterable[Union[int, ArrayLike]]):
    s = list(tour)
    if isinstance(s, int):
        edges = np.array(list(zip(*list(tsp.tour_segments(s)))))
    else:
        edges = np.array(list(zip(*s)))
    edges[1] = tsp.h - edges[1]
    ax.plot(*edges, 'b-')


def visualize_tsp_plt(tsp: TSP, tour: Iterable[Union[int, ArrayLike]], ax: SubplotBase = None):
    """Generate visualization of a TSP using MatPlotLib backend.

    Args:
        tsp (TSP): the problem
        tour (Iterable[Union[int, ArrayLike]]): tour either as indices of vertices or as segments
        ax (SubplotBase): Matplotlib axes to plot on. Defaults to None.
    """
    if ax is None:
        ax = plt.subplot(111)

    ax.set_xlim((0, tsp.w))
    ax.set_ylim((0, tsp.h))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', 'box')

    _draw_tour_plt(ax, tsp, tour)
    _draw_cities_plt(ax, tsp)


def visualize_mst_plt(tsp: TSP, edges: Iterable[ArrayLike], ax: SubplotBase = None):
    """Generate visualization of an MST using MatPlotLib backend.

    Args:
        tsp (TSP): the problem
        edges (Iterable[ArrayLike]): edges in MST as [[[x1, y1], [x2, y2]], ...]
        ax (SubplotBase): Matplotlib axes to plot on. Defaults to None.
    """
    if ax is None:
        ax = plt.subplot(111)

    ax.set_xlim((0, tsp.w))
    ax.set_ylim((0, tsp.h))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', 'box')

    _draw_edges_plt(ax, tsp, edges)
    _draw_cities_plt(ax, tsp)
