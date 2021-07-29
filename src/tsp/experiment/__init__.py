"""This module implements tools for running experiments, including helper functions
for saving whole sets of problems and tours and calculating statistics, as well as a user interface
for collecting human participant tours of problem sets.

`tsp.experiment.batch` contains helper functions for saving and loading sets of problems and tours.

`tsp.experiment.batch_server` contains a user interface (UI) for collecting solutions to problem
sets from human subjects. `tsp.experiment.run` contains some advanced tools for automating
experiments with multiple experimental conditions.

`tsp.experiment.batch_solver` contains helper functions for generating solutions to problem sets
programmatically (e.g., with the Concorde solver), and computing statistics for problem sets.

### Setting Up a Simple Experiment

First, generate a set of problems (in this case, we will create a set of 10 20-city problems).

```python
problems = []
for i in range(10):
    problems.append(TSP.generate_random(20))
```

Then, save the problem set in `test/problems`.

```python
save_problem_batch(problems, 'test/problems')
```

Now, run the UI to collect solutions from the human (the UI can be accessed at
[localhost:8080](http://localhost:8080)). Save them in `test/human`.

```python
batch_server_run('test/problems', 'test/human', randomized=False)
```

Generate Concorde solutions to the problem set, saving them in `test/concorde`.

```python
solve_batch('test/problems', ConcordeSolver, 'test/concorde')
```

Finally, you can generate the errors when comparing the human tours to the optimal tours produced by
Concorde.

```python
errors, mean, ste = score_batch_2('test/problems', 'test/human', 'test/concorde')
```

`errors` will store an array of the errors for the 10 problems, `mean` the mean of the 10 errors,
and `ste` the standard error of the mean.
"""
