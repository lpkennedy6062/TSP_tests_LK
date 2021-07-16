import numpy as np

from tsp.tsp import TSP
from tsp.solvers import Solver
from tsp.batch import load_batch, load_tours, save_tours, load, load_all_tour_segments


def run_batch(src: str, solver: Solver, dest: str = None, load_func=load) -> [[int]]:
    batch = load_batch(src, load_func=load_func)
    tours = []
    for p in batch:
        tours.append(solver(p)())
    if dest is not None:
        import os
        save_tours(tours, dest)
    return tours


def score_tours(batch_path: str, tours_path: str, load_func=load) -> [float]:
    batch = load_batch(batch_path, load_func=load_func)
    tours = load_tours(tours_path)
    result = []
    for p, t in zip(batch, tours):
        result.append(p.score(t))
    return result


def score_tours_2(batch_path: str, tours_path: str, base_path: str, load_func=load, segments=False) -> [float]:
    batch = load_batch(batch_path, load_func=load_func)
    tours = load_all_tour_segments(tours_path) if segments else load_tours(tours_path)
    base = load_tours(base_path)
    errors = np.ndarray((len(batch),), dtype=np.float)
    for i, (p, t, b) in enumerate(zip(batch, tours, base)):
        tour_score = p.score_tour_segments(t) if segments else p.score(t)
        base_score = p.score(b)
        errors[i] = (tour_score / base_score) - 1.
    return errors, np.mean(errors), np.std(errors) / np.sqrt(len(errors))


def score_tours_3(batch_path: str, tours_path: str, base_path: str, load_func=load, load_tours_func=load_tours, segments=False) -> [float]:
    batch = load_batch(batch_path, load_func=load_func)
    tours = load_tours_func(tours_path)
    base = load_tours(base_path)
    errors = np.ndarray((len(batch),), dtype=np.float)
    for i, (p, t, b) in enumerate(zip(batch, tours, base)):
        tour_score = p.score_tour_segments(t) if segments else p.score(t)
        base_score = p.score(b)
        errors[i] = (tour_score / base_score) - 1.
    return errors, np.mean(errors), np.std(errors) / np.sqrt(len(errors))
