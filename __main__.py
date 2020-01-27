# from tsp.tsp import TSP
# from tsp.solvers import ConcordeSolver, PyramidSolver

# t = TSP.generate_random(50)
# for x, y in t.cities:
#     print(x, y)
# for i in range(10):
#     s = RandomSolver(t)
#     tour = s()
#     print('Tour', i, ':', t.score(tour))
# s = ConcordeSolver(t)
# tour = s()
# print('Optimal:', t.score(tour))
# s2 = PyramidSolver(t)
# tour2 = s2()
# print('Approx:', t.score(tour2))


# from tsp.server import run

# run()




# from tsp.tsp import TSP, TSP_O
# from tsp.solvers import ConcordeSolver, ConcordeMDSSolver, PyramidSolver, PyramidMDSSolver

# t1 = TSP.generate_random(10)
# t2 = TSP_O.from_cities(t1.cities)
# t2.add_random_obstacles(50, 200)

# tour1 = ConcordeSolver(t1)()
# tsp_score = t1.score(tour1)
# tspo_score = t2.score(tour1)
# tour3 = ConcordeMDSSolver(t2)()
# tspo_score_better = t2.score(tour3)

# tour6 = PyramidSolver(t1)()
# tsp_pyramid_score = t1.score(tour6)
# tspo_pyramid_score = t2.score(tour6)
# tour8 = PyramidMDSSolver(t2, dimensions=3)()
# tspo_pyramid_score_better = t2.score(tour8)

# print()
# print('CONCORDE')
# print()
# print('Regular TSP Score (10 cities):', tsp_score)
# print(tour1)
# print()
# print('TSP w/ obstacles score:       ', tspo_score)
# print(list(t2.tour_segments(tour1)))
# print()
# print('TSP w/ obstacles score (MDS): ', tspo_score_better)
# print(tour3)
# print(list(t2.tour_segments(tour3)))

# print()
# print('PYRAMID')
# print()
# print('Regular TSP Score (10 cities):', tsp_pyramid_score)
# print(tour6)
# print()
# print('TSP w/ obstacles score:       ', tspo_pyramid_score)
# print(list(t2.tour_segments(tour6)))
# print()
# print('TSP w/ obstacles score (MDS): ', tspo_pyramid_score_better)
# print(tour8)
# print(list(t2.tour_segments(tour8)))



from tsp.tsp import TSP, TSP_O
from tsp.solvers import ChristofidesSolver, ChristofidesMDSSolver, ConcordeSolver, ConcordeMDSSolver, PyramidSolver, PyramidMDSSolver

t = TSP.generate_random(20)

t_concorde = ConcordeSolver(t)()
t_concorde_score = t.score(t_concorde)

t_christofides = ChristofidesSolver(t)()
t_christofides_score = t.score(t_christofides)

t_pyramid = PyramidSolver(t)()
t_pyramid_score = t.score(t_pyramid)

t_concorde_mds = ConcordeMDSSolver(t)()
t_concorde_mds_score = t.score(t_concorde_mds)

t_christofides_mds = ChristofidesMDSSolver(t)()
t_christofides_mds_score = t.score(t_christofides_mds)

t_pyramid_mds = PyramidMDSSolver(t)()
t_pyramid_mds_score = t.score(t_pyramid_mds)

t = TSP_O.from_cities(t.cities)
t.add_random_obstacles(50, 200)

to_concorde = ConcordeSolver(t)()
to_concorde_score = t.score(to_concorde)

to_concorde_mds = ConcordeMDSSolver(t)()
to_concorde_mds_score = t.score(to_concorde_mds)

to_christofides_mds = ChristofidesMDSSolver(t)()
to_christofides_mds_score = t.score(to_christofides_mds)

to_pyramid_mds = PyramidMDSSolver(t)()
to_pyramid_mds_score = t.score(to_pyramid_mds)

print()
print('Concorde', t_concorde, t_concorde_score)
print('Christofides', t_christofides, t_christofides_score)
print('Pyramid', t_pyramid, t_pyramid_score)
print('Concorde MDS', t_concorde_mds, t_concorde_mds_score)
print('Christofides MDS', t_christofides_mds, t_christofides_mds_score)
print('Pyramid MDS', t_pyramid_mds, t_pyramid_mds_score)
print()
print('Concorde', to_concorde, to_concorde_score)
print('Concorde MDS', to_concorde_mds, to_concorde_mds_score)
print('Christofides MDS', to_christofides_mds, to_christofides_mds_score)
print('Pyramid MDS', to_pyramid_mds, to_pyramid_mds_score)
