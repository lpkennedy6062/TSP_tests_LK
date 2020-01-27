import bottle
from bottle import route, request, response
import os
import json

from tsp.tsp import TSP
from tsp.solvers import PyramidSolver, ConcordeSolver, ChristofidesSolver


# Utilities

UI_ROOT = os.path.join(os.path.dirname(__file__), 'ui')

def static_file(path):
    return bottle.static_file(path, root=UI_ROOT)

def run():
    bottle.run(port=8080)


# API

clients = {}

@route('/api/<id>/push', method='POST')
def receive_city_push(id):
    clients[id] = TSP.from_cities(json.loads(request.forms.get('data')))
    return []

@route('/api/<id>/solve/concorde')
def solve_with_concorde(id):
    s = ConcordeSolver(clients[id])
    t = s()
    response.content_type = 'application/json'
    return [json.dumps(t)]

@route('/api/<id>/solve/christofides')
def solve_with_christofides(id):
    s = ChristofidesSolver(clients[id])
    t = s()
    response.content_type = 'application/json'
    return [json.dumps(t)]

@route('/api/<id>/solve/pyramid')
def solve_with_pyramid(id):
    s = PyramidSolver(clients[id])
    t = s()
    response.content_type = 'application/json'
    return [json.dumps(t)]

@route('/api/<id>/generate/<n>')
def generate_cities(id, n):
    clients[id] = TSP.generate_random(int(n), 500, 500)
    response.content_type = 'application/json'
    return [json.dumps(clients[id].cities)]

@route('/api/<id>/score', method='POST')
def score_tour(id):
    tour = json.loads(request.forms.get('data'))
    score = clients[id].score(tour)
    optimal_tour = ConcordeSolver(clients[id])()
    optimal = clients[id].score(optimal_tour)
    return [json.dumps({'score': score, 'optimal': optimal})]

# @route('/api/<id>/random')
# def random_tour(id):
#     s = RandomSolver(clients[id])
#     response.content_type = 'application/json'
#     return [json.dumps(s())]

# @route('/api/<id>/optimal')
# def optimal_tour(id):
#     s = ConcordeSolver(clients[id])
#     response.content_type = 'application/json'
#     return [json.dumps(list(s()))]


# Static

@route('/')
def serve_main():
    return static_file('index.html')

@route('/<path:path>')
def serve_static(path):
    return static_file(path)
