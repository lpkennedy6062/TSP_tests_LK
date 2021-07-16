from math import sin, cos, radians
import numpy as np
from PIL import Image, ImageDraw
from cv2 import cv2

from tsp.tsp import TSP


def draw_edges(im, tsp, edges):
    draw = ImageDraw.Draw(im)
    for e1, e2 in edges:
        e1 = tuple(np.array(tsp.cities[e1]) * 2)
        e2 = tuple(np.array(tsp.cities[e2]) * 2)
        draw.line([e1, e2], fill='blue', width=6)


def draw_cities(im, tsp):
    draw = ImageDraw.Draw(im)
    for x, y in tsp.cities:
        draw.ellipse([(x*2 - 8, y*2 - 8), (x*2 + 8, y*2 + 8)], fill='red', outline='red')


def draw_cities_color(im, tsp):
    draw = ImageDraw.Draw(im)
    for i, p in enumerate(tsp.cities):
        x, y = p
        c = 'red' if tsp.colors[i] == 0 else 'blue'
        draw.ellipse([(x*2 - 8, y*2 - 8), (x*2 + 8, y*2 + 8)], c, outline=c)


def draw_obstacles(im, tsp):
    if 'obstacles' in tsp.__dict__:
        draw = ImageDraw.Draw(im)
        for a, b in tsp.obstacles:
                draw.line([(a[0]*2, a[1]*2), (b[0]*2, b[1]*2)], fill='black', width=4)


def draw_tour(im, tsp, tour, segments=False):
    draw = ImageDraw.Draw(im)
    if segments:
        draw.line([(x*2, y*2) for x, y in tour], fill='blue', width=6)
    else:
        draw.line([(x*2, y*2) for x, y in tsp.tour_segments(tour)], fill='blue', width=6)


def draw_tour_color(im, tsp, tour):
    draw = ImageDraw.Draw(im)
    draw.line([(x*2, y*2) for x, y in tsp.tour_segments(tour)], fill='black', width=3)


def visualize(t, s, path, segments=False):
    im = Image.new('RGB', (t.w * 2, t.h * 2), color = 'white')
    if s:
        draw_tour(im, t, s, segments)
    draw_cities(im, t)
    draw_obstacles(im, t)
    im.thumbnail((t.w, t.h))
    im.save(path)


def visualize_mst(t, edges, path):
    im = Image.new('RGB', (t.w * 2, t.h * 2), color = 'white')
    draw_edges(im, t, edges)
    draw_cities(im, t)
    im.thumbnail((t.w, t.h))
    im.save(path)


def visualize_color(t, s, path):
    im = Image.new('RGB', (t.w * 2, t.h * 2), color = 'white')
    if s:
        draw_tour_color(im, t, s)
    draw_cities_color(im, t)
    im.thumbnail((t.w, t.h))
    im.save(path)


def visualize_3d(t, s, path, segments=False, step=1, time=12):
    """
    x’=xcos(alpha)+zsin(alpha)
    y’=y
    z’=-xsin(alpha)+zcos(alpha)
    """
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    for alpha in range(0, 360, step):
        for x, y, z in t.cities:
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
        for x, y, z in t.cities:
            new_coords.append((
                x*cos(radians(alpha)) + z*sin(radians(alpha)),
                y
            ))
        new_coords = np.array(new_coords, dtype=np.int)
        new_coords -= np.array([min_x, min_y])

        new_t = TSP.from_cities(new_coords, w=max_x, h=max_y)
        visualize(new_t, s, '/tmp/viz.png', segments)
        video.write(cv2.imread('/tmp/viz.png'))
    video.release()



# def draw_visgraph(im, tsp):
#     draw = ImageDraw.Draw(im)
#     for a in tsp.vg:
#         for b in tsp.vg[a]:
#             draw.line([a, b], fill='blue', width=3)


if __name__ == '__main__':
    from tsp.batch import load_batch, load_obstacles, load_tour

    # batch = load_batch('./simulations/data/50cities_300_O/problems', load_obstacles)
    # t = batch[0]
    # s = load_tour('./simulations/data/50cities_300_O/concorde_mds_4/001.sol')

    t = load_obstacles('./simulations/toy/10_250.tsp')
    s = load_tour('./simulations/toy/001.sol')

    visualize(t, s, 'viz.png')
