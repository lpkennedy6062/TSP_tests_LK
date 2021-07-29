"""General purpose tools for writing TSP experiments.

This module has five component submodules. The `tsp.core.tsp` submodule has object-oriented
containers for both 2-dimensional and n-dimensional TSPs, including procedures for randomly
generating 2-dimensional TSPs.

The `tsp.core.solvers` submodule implements random, optimal, and human-approximate solvers. The
optimal solver uses the [Concorde](https://www.math.uwaterloo.ca/tsp/concorde.html) backend. The
human-approximate solver uses a hierarchical clustering ("pyramid") algorithm implemented in the
`tsp.core.pyramid` submodule.

The `tsp.core.save` submodule implements procedures for serializing TSP objects and tours.

The `tsp.core.viz` submodule implements procedures for visualizing 2D TSPs and tours using either a
Python Imaging Library (PIL) backend or a MatPlotLib backend. Using OpenCV, it can also generate
movies visualizing 3D TSPs using simulated 3D motion.
"""