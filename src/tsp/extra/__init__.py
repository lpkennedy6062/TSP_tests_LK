"""Extensions of `tsp.core` for modern TSP research, mainly research on two
different kinds of "not-Euclidean" TSPs, as well as the supporting code for serialization and
visualization.

The two not-Euclidean TSPs implemented here are "TSP with obstacles" (`tsp.extra.obstacles.TSP_O`)
and "TSP with colors" (`tsp.extra.colors.TSP_Color`). Both of these violate Euclidean axioms - in
the TSP-O the shortest path between two points might not be the straight line between them (the
metric axioms will not be violated), while in the TSP with colors the triangle inequality may also
be violated, making the problem truly non-metric.

Along with these two new TSP types comes some supporting tools. `tsp.extra.save` reimplements the
`tsp.core.save.save_problem` and `load_problem` procedures to support serialization of the new
TSPs. `tsp.extra.viz` likewise provides procedures for visualizing the new TSPs.

The TSP with obstacles implementation also sports some further supporting code, including an
implementation of a visibility graph in `tsp.extra.visgraph`, which is needed for finding tours and
shortest paths. There is also a wrapper around [scikit-learn](https://scikit-learn.org/)'s
multidimensional scaling (MDS) routines for generating Euclidean reconstructions of the new TSPs in
`tsp.extra.mds`.

Finally, `tsp.extra.templates` provides a tool for generating TSPs with obstacles that form more
complex shapes than just straight lines.
"""
