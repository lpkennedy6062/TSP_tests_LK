"""Visualization procedures that only work with the old pyramid.
"""


from typing import Iterable, Tuple
from matplotlib import pyplot as plt
from matplotlib.axes import SubplotBase

from tsp.core.pyramid import DSNode
from tsp.core.tsp import TSP
from tsp.core.viz import _draw_cities_plt, _draw_edges_plt


def _isolate_edges(mst: Iterable[Tuple[int, int]], clusters: Iterable[DSNode]):
    for cluster in clusters:
        cities = set(cluster.values())
        result = []
        for e0, e1 in mst:
            if e0 in cities and e1 in cities:
                result.append((e0, e1))
        yield result


def visualize_clusters_plt(tsp: TSP, mst: Iterable[Tuple[float, Tuple[int, int]]], clusters: Iterable[DSNode], ax: SubplotBase = None):
    """Generate visualization of clusters using MatPlotLib backend.

    Args:
        tsp (TSP): the problem
        mst (Iterable[NDArray]): edges in MST as [[[x1, y1], [x2, y2]], ...]
        clusters (Iterable[DSNode]): forest of clusters
        ax (SubplotBase): Matplotlib axes to plot on. Defaults to None.
    """
    if ax is None:
        ax = plt.subplot(111)

    ax.set_xlim((0, tsp.w))
    ax.set_ylim((0, tsp.h))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', 'box')

    for edges in _isolate_edges(list(zip(*mst))[1], clusters):
        _draw_edges_plt(ax, tsp, edges)
    _draw_cities_plt(ax, tsp)