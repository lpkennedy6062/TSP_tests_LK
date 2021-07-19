import numpy as np
import matplotlib.pyplot as plt


def draw_cities(ax, tsp):
    cities = np.array(list(zip(*tsp.cities)))
    cities[1] = tsp.h - cities[1]
    ax.plot(*cities, 'ro')


def draw_tour(ax, tsp, tour, segments=False):
    if segments:
        edges = np.array(list(zip(*tour)))
    else:
        edges = np.array(list(zip(*list(tsp.tour_segments(tour)))))
    edges[1] = tsp.h - edges[1]
    ax.plot(*edges, 'b-')


def visualize(t, s, segments=False, ax=None):
    if ax is None:
        ax = plt.subplot(111)

    draw_tour(ax, t, s, segments)
    draw_cities(ax, t)

    ax.set_xlim((0, t.w))
    ax.set_ylim((0, t.h))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', 'box')
