"""
Problem set file format:
 - List of paths to problem sets (without /problems)
 - Asterisk (*) in front of paths that should be kept in the position that it is in
 - Ampersand (&) in front of paths that should not have their problems presented in random order (after the * if they coincide)

Example:

*./simulations4/data/set_test
./simulations4/data/set_16_128
./simulations4/data/set_16_192
./simulations4/data/set_16_256
./simulations4/data/set_32_128
./simulations4/data/set_32_192
./simulations4/data/set_32_256
./simulations4/data/set_48_128
./simulations4/data/set_48_192
./simulations4/data/set_48_256

"""

from typing import Iterable, Iterator, Tuple
import itertools as it
import argparse
import os
import re
import sys
import numpy as np

from tsp.experiment.batch_server import batch_server_run


def _confirm():
    try:
        input('Press RETURN to begin (otherwise ^C)')
    except KeyboardInterrupt:
        sys.exit(0)


def _parse_problems(paths: Iterable[str]) -> Tuple[str, bool]:
    """
    Format of result: (base_path, randomize problems)
    """
    result = [[]]
    for path in paths:
        path = path.strip()
        if not path:
            continue
        if path[:2] == '*&':
            result.append([(path[2:], False)])
            result.append([])
        elif path[:1] == '*':
            result.append([(path[1:], True)])
            result.append([])
        elif path[:1] == '&':
            result[-1].append((path[1:], False))
        else:
            result[-1].append((path, True))
    return list(it.chain(*map(lambda x: sorted(x, key=lambda _: np.random.rand()), result)))


def _run_problem_set(participant: str, path: str, randomized: bool, ui_root: str = None):
    command = f'python3 -m tsp.experiment.batch_server -f {path}/problems -s {path}/{participant} -o{"-r" if randomized else ""}'
    print()
    print(command)
    _confirm()
    problems_path = f'{path}/problems'
    output_dir = f'{path}/{participant}'
    try:
        batch_server_run(problems_path, output_dir, randomized, ui_root)
    except KeyboardInterrupt:
        return


def _load_save_file(path: str) -> Iterator[str, bool]:
    with open(path) as f:
        for line in f:
            match = re.match(r'\(([R\s])\)\s(.*)\n', line)
            yield match.group(2), match.group(1) == 'R'


def _dump_save_file(save_file_path: str, problem_sets: Iterable[str, bool]):
    with open(save_file_path, 'w') as f:
        for path, randomized in problem_sets:
            f.write('({}) {}\n'.format(('R' if randomized else ' '), path))


def run(participant: str, set_list_path: str, save_file_path: str, ui_root: str = None):
    """Run a subject on a set of experimental conditions.

    Args:
        participant (str): participant identifier
        set_list_path (str): path to set list file (see module documentation for expected format)
        save_file_path (str): path to save the ordering of experimental conditions (mainly useful if randomized)
        ui_root (str, optional): Path to UI (should only need to be used if creating a standalone executable). Defaults to None.
    """
    if save_file_path is not None and os.path.exists(save_file_path):
        problem_sets = list(_load_save_file(save_file_path))
    else:
        with open(set_list_path, 'r') as f:
            problem_sets = _parse_problems(f)
        if save_file_path is not None:
            _dump_save_file(save_file_path, problem_sets)
    print('Running participant "{}".'.format(participant))
    print('Problem sets will be administered in the following order:\n')
    for path, randomized in problem_sets:
        print('({}) {}'.format(('R' if randomized else ' '), path))
    print()
    print('Once the participant has completed a set, continue to the next set with ^C')
    print('Then have the participant refresh the page')
    _confirm()
    for path, randomized in problem_sets:
        if not os.path.exists(os.path.join(path, participant)):
            _run_problem_set(participant, path, randomized, ui_root)
        else:
            print('Found existing directory {}, skipping...'.format(os.path.join(path, participant)))
    print()
    print('All problem sets done! Exiting...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a subject on problem sets.')
    parser.add_argument('id', type=str, help='Participant identifier')
    parser.add_argument('sets', type=str, help='Path to list of problem sets')
    parser.add_argument('-s', type=str, required=False, help='Path to save file')
    args = parser.parse_args()

    run(args.id, args.sets, args.s)
