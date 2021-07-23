import argparse
import json
import os
import numpy as np

from tsp.batch import generate_batch, generate_batch_obstacles, save_batch, save_obstacles, load_obstacles, load_batch, save_tour, save_tour_segments, save_tour_times

import bottle
from bottle import route, request, response


tour_dir = None
batch = None

OBSTACLES = False
MAPPING = []


# Utilities

UI_ROOT = os.path.join(os.path.dirname(__file__), 'batch_ui')

def static_file(path):
    return bottle.static_file(path, root=UI_ROOT)

def run():
    bottle.run(host='', port=8080)


# API

@route('/api/<id:int>/tour', method='POST')
def get_tour(id):
    id = MAPPING[id]
    if not OBSTACLES:
        tour = json.loads(request.forms.get('data'))
        save_tour(tour, tour_dir, id + 1)
        return ['Done']
    else:
        tour_edges, tour_edge_times = json.loads(request.forms.get('data'))
        tour_segments = [tour_edges[0][0]] + [edge[1] for edge in tour_edges]
        tour_times = [tour_edge_times[i + 1] - tour_edge_times[i] for i in range(len(tour_edge_times) - 1)]
        assert len(tour_segments) == len(tour_times) + 1
        save_tour_segments(tour_segments, tour_dir, id + 1)
        save_tour_times(tour_times, tour_dir, id + 1)
        return ['Done']

@route('/api/<id:int>/cities')
def send_cities(id):
    id = MAPPING[id]
    response.content_type = 'application/json'
    if not OBSTACLES:
        return [json.dumps(batch[id].cities)]
    else:
        return [json.dumps({
            'cities': batch[id].cities,
            'obstacles': batch[id].obstacles,
            'height': batch[id].h,
            'width': batch[id].w
        })]

@route('/api/<id:int>/visgraph', method='POST')
def get_visgraph(id):
    id = MAPPING[id]
    if OBSTACLES:
        response.content_type = 'application/json'
        vertex = tuple(json.loads(request.forms.get('data')))
        return [json.dumps(batch[id].to_visgraph()[vertex])]


# Static

@route('/')
def serve_main():
    return static_file('index.html')

@route('/<path:path>')
def serve_static(path):
    return static_file(path)


def batch_server_run(problems_path, output_dir, randomized, ui_root=None):
    global tour_dir, batch, OBSTACLES, MAPPING, UI_ROOT
    tour_dir = output_dir

    OBSTACLES = True
    if ui_root is None:
        UI_ROOT = os.path.join(os.path.dirname(__file__), 'obstacles_batch_ui')
    else:
        UI_ROOT = ui_root
    batch = load_batch(problems_path, load_func=load_obstacles)

    if randomized:
        MAPPING = np.random.permutation(len(batch))
    else:
        MAPPING = np.arange(len(batch))

    if not os.path.isdir(tour_dir):
        os.makedirs(tour_dir)
    with open(os.path.join(tour_dir, 'order.txt'), 'w') as f:
        f.write('{}\n'.format(str(MAPPING)))

    run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a subject on a batch of problems.')
    parser.add_argument('-f', type=str, required=True, help='Path to TSP batch to load/write')
    parser.add_argument('-s', type=str, required=False, help='Path to save tours')
    parser.add_argument('-g', nargs=2, type=int, required=False, help='Generate a new batch of X problems with Y cities each')
    parser.add_argument('-o', nargs='*', type=int, required=False, help='If specified, add N obstacles (lines) of W width')
    parser.add_argument('-r', action='store_true', help='Randomize the order in which problems are presented')
    args = parser.parse_args()

    if args.g is not None:
        if args.o is None:
            save_batch(generate_batch(*args.g), args.f)
        else:
            save_batch(generate_batch_obstacles(*args.g, *args.o), args.f, save_func=save_obstacles)

    tour_dir = args.s if args.s is not None else args.f
    if args.o is None:
        batch = load_batch(args.f)
    else:
        OBSTACLES = True
        UI_ROOT = os.path.join(os.path.dirname(__file__), 'obstacles_batch_ui')
        batch = load_batch(args.f, load_func=load_obstacles)

    if args.r:
        MAPPING = np.random.permutation(len(batch))
    else:
        MAPPING = np.arange(len(batch))

    if not os.path.isdir(tour_dir):
        os.makedirs(tour_dir)
    with open(os.path.join(tour_dir, 'order.txt'), 'w') as f:
        f.write('{}\n'.format(str(MAPPING)))

    run()
