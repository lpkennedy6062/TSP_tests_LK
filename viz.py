from PIL import Image, ImageDraw


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


def visualize_color(t, s, path):
    im = Image.new('RGB', (t.w * 2, t.h * 2), color = 'white')
    if s:
        draw_tour_color(im, t, s)
    draw_cities_color(im, t)
    im.thumbnail((t.w, t.h))
    im.save(path)


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
