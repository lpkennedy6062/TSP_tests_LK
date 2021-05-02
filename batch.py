import glob
import os
import time
import pickle
import json
import numpy as np

from tsp.tsp import TSP, N_TSP, TSP_O, convert_tour_segments


def save_stresses(stresses: [float], path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    for i, s in enumerate(stresses, 1):
        with open(os.path.join(path, '{}.stress'.format(str(i).zfill(3))), 'w') as f:
            f.write('{}\n'.format(s))


def load_stresses(path: str):
    result = []
    for path_ in sorted(glob.glob(os.path.join(path, '*.stress'))):
        with open(path_, 'r') as f:
            result.append(float(f.read().strip()))
    return result


def save(tsp: TSP, path: str):
    with open(path, 'w') as f:
        # for x, y in tsp.cities:
        #     f.write('{} {}\n'.format(x, y))
        for c in tsp.cities:
            f.write('{}\n'.format(' '.join(map(str, c))))


def save_obstacles(tsp: TSP_O, path: str):
    with open(path, 'wb') as f:
        pickle.dump(tsp, f)


def load(path: str) -> TSP:
    tsp = None
    with open(path, 'r') as f:
        for line in f:
            c = tuple(map(int, line.strip().split()))
            if tsp is None:
                tsp = TSP() if len(c) == 2 else N_TSP()
            tsp.add_city(*c)
    return tsp


def load_obstacles(path: str) -> TSP_O:
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_batch(tsps: [TSP], path: str, save_func=save, starting_index=1):
    if not os.path.isdir(path):
        os.makedirs(path)
    for i, tsp in enumerate(tsps, starting_index):
        save_func(tsp, os.path.join(path, '{}.tsp'.format(str(i).zfill(3))))


def load_batch(path: str, load_func=load) -> [TSP]:
    result = []
    for path_ in sorted(glob.glob(os.path.join(path, '*.tsp'))):
        result.append(load_func(path_))
    return result


def save_tour(tour: [int], path: str, i: int = None):
    if not os.path.isdir(path):
        os.makedirs(path)
    if i is not None:
        path = os.path.join(path, '{}.sol'.format(str(i).zfill(3)))
    with open(path, 'w') as f:
        f.write('{}\n'.format(','.join(str(v) for v in tour)))


def save_tours(tours: [[int]], path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    for i, tour in enumerate(tours, 1):
        save_tour(tour, path, i)


def save_tour_segments(tour_segments, path: str, i: int = None):
    if not os.path.isdir(path):
        os.makedirs(path)
    if i is not None:
        path = os.path.join(path, '{}.sol'.format(str(i).zfill(3)))
    with open(path, 'w') as f:
        json.dump(tour_segments, f)


def save_tour_times(tour_times, path: str, i: int = None):
    if not os.path.isdir(path):
        os.makedirs(path)
    if i is not None:
        path = os.path.join(path, '{}.time'.format(str(i).zfill(3)))
    with open(path, 'w') as f:
        json.dump(tour_times, f)


def save_all_tour_segments(all_tour_segments, path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    for i, tour_segments in enumerate(all_tour_segments, 1):
        save_tour_segments(tour_segments, path, i)


def load_tour(path: str):
    with open(path, 'r') as f:
        return [int(v) for v in f.read().strip().split(',')]


def load_tours(path: str):
    result = []
    for path_ in sorted(glob.glob(os.path.join(path, '*.sol'))):
        result.append(load_tour(path_))
    return result


def load_tour_segments(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def load_all_tour_segments(path: str):
    result = []
    for path_ in sorted(glob.glob(os.path.join(path, '*.sol'))):
        result.append(load_tour_segments(path_))
    return result


def load_all_tour_segments_to_cities(batch_path: str, load_func=load):
    def _result_func(path: str):
        batch = load_batch(batch_path, load_func)
        tours = load_all_tour_segments(path)
        result = []
        for problem, tour in zip(batch, tours):
            result.append(convert_tour_segments(problem, tour))
        return result
    return _result_func


def generate_batch(size: int, n_cities: int) -> [TSP]:
    result = []
    for i in range(size):
        result.append(TSP.generate_random(n_cities))
    return result


def generate_batch_obstacles(size: int, n_cities: int, n_obstacles: int, obstacle_width: int) -> [TSP_O]:
    result = []
    for i in range(size):
        while True:
            t = TSP_O.generate_random(n_cities)
            t.add_random_obstacles(n_obstacles, obstacle_width)
            try:
                t.to_edge_matrix()
                break
            except Exception:
                print('discarding')
        result.append(t)
    return result
