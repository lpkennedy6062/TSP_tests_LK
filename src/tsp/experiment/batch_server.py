"""A user interface (UI) for collecting solutions to problem sets from human subjects.

See `tsp.experiment.run` for a way to automate experiments with multiple experimental conditions.

The interface can run with either `tsp.core.tsp.TSP` or `tsp.extra.obstacles.TSP_O` problems. They
should be saved in a directory with the file naming system described in `tsp.experiment.batch`
(you should just use `tsp.experiment.batch.save_problem_batch` to handle this). In the directory
you pass in as the `output_dir` the UI will save the human-generated tours as well as the time
(in milliseconds) that it took the human to construct each tour segment. The UI is served on
[localhost:8080](http://localhost:8080). It uses the wonderful [Bottle](https://bottlepy.org/)
microframework to glue together the tsp library and a JavaScript/HTML5 canvas frontend.

This script can also be run on the command line with `python3 -m tsp.experiment.batch_server`.
The command line documentation is as follows:

```
usage: batch_server.py [-h] -f F -s S [-r]

Run a subject on a batch of problems.

optional arguments:
  -h, --help  show this help message and exit
  -f F        Path to TSP batch to load
  -s S        Path to save tours
  -r          Randomize the order in which problems are presented
```
"""


import argparse
import json
import os
from threading import Thread
import time
import numpy as np

import bottle
from bottle import route, request, response, abort, WSGIRefServer

from tsp.core.save import save_list
from tsp.experiment.batch import load_problem_batch, save_list_item



tour_dir = None
batch = None

MAPPING = []
SERVER = None


# Utilities

UI_ROOT = os.path.join(os.path.dirname(__file__), 'batch_ui')

def _static_file(path: str):
    return bottle.static_file(path, root=UI_ROOT)

def _run():
    global SERVER
    print('Serving on http://localhost:8080 ...')
    SERVER = WSGIRefServer(host='', port=8080)
    bottle.run(server=SERVER, quiet=True)

def _shutdown():
    print('Stopping server...')
    time.sleep(5)
    SERVER.srv.shutdown()
    SERVER.srv.server_close()


# API

@route('/api/<id_:int>/tour', method='POST')
def _get_tour(id_: int):
    id_ = MAPPING[id_]
    tour_edges, tour_edge_times = json.loads(request.forms.get('data')) # pylint: disable=no-member
    tour_segments = [tour_edges[0][0]] + [edge[1] for edge in tour_edges]
    tour_times = [tour_edge_times[i + 1] - tour_edge_times[i] for i in range(len(tour_edge_times) - 1)]
    assert len(tour_segments) == len(tour_times) + 1
    save_list_item(tour_segments, tour_dir, 'sol', id_ + 1)
    save_list_item(tour_times, tour_dir, 'time', id_ + 1)
    return ['Done']

@route('/api/<id_:int>/cities')
def _send_cities(id_: int):
    if id_ < 0 or id_ >= len(MAPPING):
        if id_ == len(MAPPING):
            Thread(target=_shutdown).start()
        abort(404, 'Problem not found.')
    id_ = MAPPING[id_]
    response.content_type = 'application/json'
    return [json.dumps({
        'cities': batch[id_].cities.tolist(),
        'obstacles': batch[id_].obstacles.tolist() if 'obstacles' in batch[id_].__dict__ else [],
        'height': batch[id_].h,
        'width': batch[id_].w
    })]

@route('/api/<id_:int>/visgraph', method='POST')
def _get_visgraph(id_: int):
    id_ = MAPPING[id_]
    response.content_type = 'application/json'
    vertex = tuple(json.loads(request.forms.get('data'))) # pylint: disable=no-member
    return [json.dumps(batch[id_].to_visgraph()[vertex] if 'obstacles' in batch[id_].__dict__ else batch[id_].cities.tolist())]


# Static

@route('/')
def _serve_main():
    return _static_file('index.html')

@route('/<path:path>')
def _serve_static(path: str):
    return _static_file(path)


def batch_server_run(problems_path: str, output_dir: str, randomized: bool, ui_root: str = None):
    """Run a subject on a batch of problems.

    Args:
        problems_path (str): path to batch of problems
        output_dir (str): path to save solutions
        randomized (bool): whether or not to randomize the order in which problems are presented
        ui_root (str, optional): Path to UI (should only need to be used if creating a standalone executable). Defaults to None.
    """
    global tour_dir, batch, MAPPING, UI_ROOT
    tour_dir = output_dir

    if ui_root is not None:
        UI_ROOT = ui_root
    batch = load_problem_batch(problems_path)

    if randomized:
        MAPPING = np.random.permutation(len(batch))
    else:
        MAPPING = np.arange(len(batch))

    if not os.path.isdir(tour_dir):
        os.makedirs(tour_dir)

    save_list(MAPPING, os.path.join(tour_dir, 'order.txt'))

    _run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a subject on a batch of problems.')
    parser.add_argument('-f', type=str, required=True, help='Path to TSP batch to load')
    parser.add_argument('-s', type=str, required=True, help='Path to save tours')
    parser.add_argument('-r', action='store_true', help='Randomize the order in which problems are presented')
    args = parser.parse_args()

    batch_server_run(args.f, args.s, args.r)
