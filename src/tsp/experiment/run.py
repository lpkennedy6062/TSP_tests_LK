"""Script for automating experiments with multiple experimental conditions.

This script can be run on the command line with `python3 -m tsp.experiment.run`.
The command line documentation is as follows:

```
usage: run.py [-h] [-s S] id sets

Run a subject on problem sets.

positional arguments:
  id          Participant identifier
  sets        Path to list of problem sets

optional arguments:
  -h, --help  show this help message and exit
  -s S        Path to save file
```

### Problem Set File Format

Problem sets (corresponding to experimental conditions) should be saved under a parent directory
for the data from the experimental condition in a subdirectory named `problems`. The interface will
save the human-generated tours and times in a subdirectory named `<id>` where &lt;id&gt; is the
participant identifier. Then, create a "set list file" with the following format:

 - List of paths to problem sets (without /problems)
 - Asterisk (*) in front of paths that should be kept in the position that it is in
 - Ampersand (&) in front of paths that should not have their problems presented in random order (after the * if they coincide)

Example:

```
*&./simulations4/data/set_test
./simulations4/data/set_16_128
./simulations4/data/set_16_192
./simulations4/data/set_16_256
./simulations4/data/set_32_128
./simulations4/data/set_32_192
./simulations4/data/set_32_256
./simulations4/data/set_48_128
./simulations4/data/set_48_192
./simulations4/data/set_48_256
```

This file would present the first experimental condition (a test set) first and with problems in
unrandomized order. Then it would present all of the rest of the experimental conditions in random
order, with their problems in randomized order.

When provided with a path, the script will generate a "save file" containing the actual order the
experimental conditions are presented in and whether or not each one is randomized. This is useful
both as reference and if the subject needs multiple sessions to complete all conditions. Note that
if an experimental condition has been started on a previous run of the script, it will be skipped
over on the next run, *even if it hasn't been completed*. In this case, you will need to manually
remove the offending directory.

### Packaging

With mildly tech-savvy participants, you can package a simple script which calls
`tsp.experiment.run.run` as a standalone executable using something like
[PyInstaller](https://www.pyinstaller.org/), and have the participants run themselves on the
experimental conditions. In the age of COVID-19, we've had to do this ourselves. The basic
directory structure should look something like:

```
root
 |-- run_this.exe
 |-- problems.txt  # problem set list
 |-- data
      |-- condition1
      |    |-- problems
      |         |-- 001.tsp
      |         |-- ...
      |
      |-- condition2
      |    |-- ...
      |
      |-- ...
 |-- batch_ui  # copy of src/tsp/experiment/batch_ui
      |-- ...
```

Then the Python script to package into `run_this.exe` would be something like:

```python
from tsp.experiment.run import run

run(
    participant='jv',
    set_list_path='problems.txt',
    save_file_path='save.txt',
    ui_root='batch_ui'
)
```

Where "jv" gets replaced with the participant identifier. Theoretically this could also be provided
dynamically if one doesn't want to package individual scripts for each participant. Solutions and
times will then be saved in `data/condition1/jv`, `data/condition2/jv`, etc.

Note that the files for the web interface need to be provided manually in order for the UI to work
correctly as a standalone executable. With a native Python install of the library, the UI stuff is
all handled "under the hood."
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


def _load_save_file(path: str) -> Iterator[Tuple[str, bool]]:
    with open(path) as f:
        for line in f:
            match = re.match(r'\(([R\s])\)\s(.*)\n', line)
            yield match.group(2), match.group(1) == 'R'


def _dump_save_file(save_file_path: str, problem_sets: Iterable[Tuple[str, bool]]):
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
    print('Have the participant refresh the page after starting each condition here.')
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
