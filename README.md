# tsp

A library for cognitive science research on the traveling salesperson problem (TSP). Documentation
is hosted [here](https://jackv.co/tsp/tsp.html).

**Coming here from the *Problem Solving* textbook? Check out the
[quickstart](examples/quickstart.ipynb) demo, and the
[documentation for the `experiment` module](https://jackv.co/tsp/tsp/experiment.html) to get
started (also available as a [Jupyter notebook](examples/experiment_demo.ipynb)).**

## Installation

Runs anywhere Python (3.8+) and Concorde can be installed.

First, you will have to download (and compile, if necessary)
[Concorde](https://www.math.uwaterloo.ca/tsp/concorde.html) and its supporting libraries.

Then, clone and install this repository:

```
git clone https://github.com/jackvandrunen/tsp.git
cd tsp
python3 -m pip install .
```

And you're ready to go!

## Citations

This library accompanies the forthcoming book *Problem Solving: Cognitive Mechanisms and Formal
Models*. It has been used to generate results for the following abstracts:

> Pizlo, Z. and VanDrunen, J. "The status of mental representations in cognitive functions." *Annual Interdisciplinary Conference*, 2020.
> 
> VanDrunen, J. and Pizlo, Z. "The effectiveness of Multidimensional Scaling in TSPs whose metric is not Euclidean" (poster). *Society for Mathematical Psychology*, 2019.

The pyramid algorithm for approximating human solutions to TSP has been described in a number of papers, none of which precisely describe the algorithm implemented in this library. A paper describing a recent pyramid model:

> Haxhimusa, Y., Kropatsch, W. G., Pizlo, Z., and Ion, A. "Approximative graph pyramid solution of the E-TSP." *Image and Vision Computing* 27 (2009), 887-896.

Other papers describing 3D TSP, not-Euclidean TSP with obstacles, and not-Euclidean TSP with colors:

> Haxhimusa, Y., et al. "2D and 3D traveling salesman problem." *The Journal of Problem Solving* 3 (2011), 167-193.
> 
> Saalweachter, J. and Pizlo, Z. "Non-Euclidean traveling salesman problem." In *Decision Modeling and Behavior in Complex and Uncertain Environments* (Springer, 2008), 339-358.
> 
> Sajedinia, Z., Pizlo, Z., and H&eacute;lie, S. "Investigating the role of the visual system in solving the traveling salesperson problem." *Cognitive Science Society*, 2019.

### Citing This Code

You are welcome to use this code without citation (it is also licensed with a
[copyfree](https://copyfree.org/) license, so you can do *anything* with it in general). However,
if experiments done with the help of this code make their way into a publication, I would greatly
appreciate a citation or acknowledgement in some format which includes my full name (Jacob
VanDrunen) and a link to this repository.

## Acknowledgements

Many thanks to Kevin Nam for tracking down innumerable bugs and usage difficulties, and to Zyg
Pizlo for guidance and encouragement through the years. Also thanks to the Undergraduate
Research Opportunities Program at UC Irvine, the school of social sciences, and Zyg Pizlo for
providing bits of funding here and there to projects ultimately motivating the development of this
software.
