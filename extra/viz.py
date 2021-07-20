from typing import Iterable
from PIL import Image, ImageDraw

from tsp.extra.color import TSP_Color


def draw_cities_color(im: Image, tsp: TSP_Color):
    draw = ImageDraw.Draw(im)
    for i, p in enumerate(tsp.cities):
        x, y = p
        c = 'red' if tsp.colors[i] == 0 else 'blue'
        draw.ellipse([(x*2 - 8, y*2 - 8), (x*2 + 8, y*2 + 8)], c, outline=c)


def draw_tour_color(im, tsp: TSP_Color, tour: Iterable[int]):
    draw = ImageDraw.Draw(im)
    draw.line([(x*2, y*2) for x, y in tsp.tour_segments(tour)], fill='black', width=3)


def visualize_color(t: TSP_Color, s: Iterable[int], path: str):
    im = Image.new('RGB', (t.w * 2, t.h * 2), color = 'white')
    if s:
        draw_tour_color(im, t, s)
    draw_cities_color(im, t)
    im.thumbnail((t.w, t.h))
    im.save(path)
