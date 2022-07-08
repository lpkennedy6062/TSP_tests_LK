"""A library for cognitive science research on the traveling salesperson problem (TSP).

The library is divided up into three modules, each with a number of submodules. The `tsp.core`
module implements general purpose tools for writing TSP experiments: object-oriented containers for
TSPs of 2 (or more!) dimensions, basic solvers implementing random tour, optimal tour (via
Concorde), and pyramid (hierarchical clustering) tour, serialization for TSPs and tours, and
visualization of TSPs and tours.

The `tsp.experiment` module implements tools for running experiments, including helper functions
for saving whole sets of problems and tours and calculating statistics, as well as a user interface
for collecting human participant tours of problem sets.

The `tsp.extra` module implements extensions for modern TSP research, mainly research on two
different kinds of "not-Euclidean" TSPs, as well as the supporting code for serialization and
visualization.

This code has been used to generate results for a couple of abstracts so far, and implements
some experiments described in others. It serves as a companion to the book [*Problem Solving: Cognitive Mechanisms and Formal Models*](https://www.cambridge.org/core/books/abs/problem-solving/problem-solving/105FC98CEBE3FA277CD43AC34EECBC1B)
by Zygmunt Pizlo. See the [README](https://github.com/jackvandrunen/tsp) for citations.
"""
