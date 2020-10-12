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

import numpy as np
import itertools as it
import subprocess
import argparse


def confirm():
    try:
        input('Press RETURN to begin (otherwise ^C)')
    except KeyboardInterrupt:
        exit(0)


def parse_problems(paths):
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
    return list(it.chain(*map(lambda x: sorted(x, key=lambda _: np.random.random()), result)))


def run_problem_set(participant, path, randomized):
    command = 'python3 -m tsp.batch_server -f {}/problems -s {}/{} -o{}'.format(path, path, participant, ' -r' if randomized else '')
    print()
    print(command)
    confirm()
    try:
        subprocess.call(command.split())
    except KeyboardInterrupt:
        return


def run(participant, set_list_path):
    with open(set_list_path, 'r') as f:
        problem_sets = parse_problems(f)
    print('Running participant "{}".'.format(participant))
    print('Problem sets will be administered in the following order:\n')
    for path, randomized in problem_sets:
        print('({}) {}'.format(('R' if randomized else ' '), path))
    print()
    print('Once the participant has completed a set, continue to the next set with ^C')
    print('Then have the participant refresh the page')
    confirm()
    for path, randomized in problem_sets:
        run_problem_set(participant, path, randomized)
    print()
    print('All problem sets done! Exiting...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a subject on problem sets.')
    parser.add_argument('id', type=str, help='Participant identifier')
    parser.add_argument('sets', type=str, help='Path to list of problem sets')
    args = parser.parse_args()
    run(args.id, args.sets)
