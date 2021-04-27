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


def score_tours_3(batch_path: str, tours_path: str, base_path: str, load_func=load, load_tours_func=load_tours) -> [float]:
    batch = load_batch(batch_path, load_func=load_func)
    tours = load_tours_func(tours_path)
    base = load_tours(base_path)
    errors = np.ndarray((len(batch),), dtype=np.float)
    for i, (p, t, b) in enumerate(zip(batch, tours, base)):
        tour_score = p.score_tour_segments(t) if segments else p.score(t)
        base_score = p.score(b)
        errors[i] = (tour_score / base_score) - 1.
    return errors, np.mean(errors), np.std(errors) / np.sqrt(len(errors))


def evaluate_pyramid(tsp: TSP) -> ([int], float, float):
    """
    Returns the pyramid solution, distance traveled, and time per city to compute
    """
    s = PyramidSolver(tsp)
    mt = time.perf_counter()
    tour = s()
    mt = (time.perf_counter() - mt) / len(tour)
    score = tsp.score(tour)
    return tour, score, mt


def evaluate_concorde(tsp: TSP) -> ([int], float, float):
    s = ConcordeSolver(tsp)
    mt = time.perf_counter()
    tour = s()
    mt = (time.perf_counter() - mt) / len(tour)
    score = tsp.score(tour)
    return tour, score, mt


def evaluate_batch_pyramid(tsps: [TSP]) -> ([float], [float]):
    """
    Returns the distances traveled and the times per city to compute
    """
    scores = []
    times = []
    for tsp in tsps:
        tour, score, mt = evaluate_pyramid(tsp)
        scores.append(score)
        times.append(mt)
    return scores, times


def evaluate_batch_concorde(tsps: [TSP]) -> ([float], [float]):
    scores = []
    times = []
    for tsp in tsps:
        tour, score, mt = evaluate_concorde(tsp)
        scores.append(score)
        times.append(mt)
    return scores, times


def score_batch(tsps: [TSP]) -> (float, float, float):
    """
    Returns the average error percentage average time per city to compute (pyramid), and average time per city to compute (concorde)
    """
    pyramid_scores, pyramid_times = evaluate_batch_pyramid(tsps)
    concorde_scores, concorde_times = evaluate_batch_concorde(tsps)
    avg_error = 0.
    for i in range(len(tsps)):
        avg_error += max(1., pyramid_scores[i] / concorde_scores[i]) - 1.
    avg_error /= len(tsps)
    return avg_error, np.mean(pyramid_times), np.mean(concorde_times)


def time_batch(tsps: [TSP]) -> float:
    scores, times = evaluate_batch_pyramid(tsps)
    return np.mean(times)
