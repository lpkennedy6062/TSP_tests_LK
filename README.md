# tsp

Liam: For the tests I have been working on, I added them in the examples folder.




A library for cognitive science research on the traveling salesperson problem (TSP). Documentation
is hosted [here](https://jackv.co/tsp/tsp.html). Built and maintained by
[Jack VanDrunen](https://jackv.co/),
with contributions from Mark Beers
([github](https://github.com/mabeers-arco),
[linkedin](https://www.linkedin.com/in/mark-beers-3a90a614a/)).

**Coming here from the [*Problem Solving*](https://www.cambridge.org/core/books/abs/problem-solving/problem-solving/105FC98CEBE3FA277CD43AC34EECBC1B)
textbook? Check out the [quickstart](examples/quickstart.ipynb) demo, and the
[setting up a simple experiment](examples/experiment_demo.ipynb) demo.**

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

## License

This work is dual-licensed under the ISC and the
[Copyfree Open Innovation License](https://coil.apotheon.org/about/). You can choose between the
two when you make use of the code. The intent is that you are free to use this code as you please
(ISC), with explicit patent grant if you deem that necessary (COIL). COIL provides a
[more permissive](https://copyfree.org/) alternative to the more popular Apache 2.0 license for
that purpose.

`SPDX-License-Identifier: ISC OR COIL-1.0`

If you use our code for scientific work, we encourage you to cite it, including at least (1) our
full names (Jacob VanDrunen and Mark Beers), (2), the title of the project ("tsp: A library
for cognitive science research on the traveling salesperson problem"), and (3) the GitHub url
([https://github.com/jackvandrunen/tsp](https://github.com/jackvandrunen/tsp)).

## Citations

This library accompanies the forthcoming book
[*Problem Solving: Cognitive Mechanisms and Formal Models*](https://www.cambridge.org/core/books/abs/problem-solving/problem-solving/105FC98CEBE3FA277CD43AC34EECBC1B)
by Zygmunt Pizlo (Cambridge UP, 2022). It has been used to generate results for the following
abstracts:

> VanDrunen, J., Nam, K., Beers, M., and Pizlo, Z. "Traveling salesperson problem with simple
> obstacles: The role of multidimensional scaling and the role of clustering."
> *Computational Brain and Behavior*, 2022.
> 
> Pizlo, Z. and VanDrunen, J. "The status of mental representations in cognitive functions."
> *Annual Interdisciplinary Conference*, 2020.
> 
> VanDrunen, J. and Pizlo, Z. "The effectiveness of Multidimensional Scaling in TSPs whose metric
> is not Euclidean" (poster). *Society for Mathematical Psychology*, 2019.

The pyramid algorithm for approximating human solutions to TSP has been described in a number of
papers, none of which precisely describe the algorithm implemented in this library. A paper
describing the ancestor of our model:

> Haxhimusa, Y., Kropatsch, W. G., Pizlo, Z., and Ion, A. "Approximative graph pyramid solution of
> the E-TSP." *Image and Vision Computing* 27 (2009), 887-896.

Other papers describing 3D TSP, not-Euclidean TSP with obstacles, and not-Euclidean TSP with colors:

> Haxhimusa, Y., et al. "2D and 3D traveling salesman problem." *The Journal of Problem Solving* 3
> (2011), 167-193.
> 
> Saalweachter, J. and Pizlo, Z. "Non-Euclidean traveling salesman problem." In *Decision Modeling
> and Behavior in Complex and Uncertain Environments* (Springer, 2008), 339-358.
> 
> Sajedinia, Z., Pizlo, Z., and H&eacute;lie, S. "Investigating the role of the visual system in
> solving the traveling salesperson problem." *Cognitive Science Society*, 2019.

## Caveats

This code represents the accumulation of 4 years of technical debt, thanks to what I called when I
worked at a startup, "demo-driven development." I have paid off some of it. The code is well-linted
and well-documented. But for those entering into the rabbit-hole of development with it, a few
caveats: (1) it lacks a systematic suite of integration tests to ensure there are no regressions
still lurking around, (2) it very frustratingly uses numpy arrays in some areas and lists in others,
and (3) it is computationally inefficient, despite some algorithms having theoretically-lower
complexity bounds.

Oh, and (4) it still relies on Concorde, which is a notoriously difficult piece of software to
install. I am happy to hear about any alternatives you can suggest.

## Acknowledgements

Many thanks to Kevin Nam for tracking down innumerable bugs and usage difficulties, especially on
Windows, and to Zyg Pizlo for guidance at every step of the development process. Early development
was funded in part by a fellowship to JV from the Division of Undergraduate Education at the
University of California, Irvine. A sizable portion of the development happened at Bad Coffee
Costa Mesa, for whose existence I am also grateful. S.D.G.
