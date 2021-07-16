from collections import deque
from itertools import permutations
from queue import PriorityQueue
import itertools as it
import numpy as np


class DSNode:
    def __init__(self, value, children=[], height=0):
        self.value = value
        self.children = children
        self.height = height
        self.parent = self

    def __lt__(self, obj):
        return self.height < obj.height

    def __eq__(self, obj):
        return self.height == obj.height

    def find(self):
        if self.parent.parent is not self.parent:
            self.parent = self.parent.find()
        return self.parent

    def split(self, depth=1):
        """
        Returns the depth + 1 clusters produced by descending the dendrogram
        """
        if self.height < 1 or depth < 1:
            return [self]
        result = self.children[:]
        for i in range(depth - 1):
            to_split = max(result)
            result.remove(to_split)
            result += to_split.split()
        return result

    def values(self):
        if not self.children:
            return [self.value]
        else:
            return self.children[0].values() + self.children[1].values()


class DSTree:
    def __init__(self, values=[]):
        self._sets = dict()
        self._unions = 0
        self.root = None
        for value in values:
            self.make_set(value)

    @property
    def sets(self):
        return len(self._sets) - self._unions

    def make_set(self, value):
        self._sets[value] = DSNode(value)

    def find(self, value):
        return self._sets[value].find()

    def union(self, x, y):
        x_parent, y_parent = self.find(x), self.find(y)
        if x_parent is y_parent:
            return x_parent
        self._unions += 1
        new_parent = DSNode(None, [x_parent, y_parent], self._unions)
        x_parent.parent = new_parent
        y_parent.parent = new_parent
        self.root = new_parent
        return new_parent


def calculate_centroid(c, nodes):
    result = np.zeros((nodes.shape[1],), dtype=np.int32)
    total = 0
    for child in c.children:
        if type(child.value) is tuple:
            total += child.value[0]
            result += child.value[1] * child.value[0]
        else:
            total += 1
            result += nodes[child.value]
    return total, result // total


def centroid(c, nodes):
    if not c.children:
        return nodes[c.value]
    else:
        return c.value[1]


def cluster_kruskal(nodes):
    """
    Agglomerate nodes based on their MST, given coordinates in n-dimensions.
    Uses Kruskal's algorithm, which runs in O(v^2 log v)

    Input: np.ndarray((v, n))
    """
    edges = PriorityQueue(len(nodes) * (len(nodes) - 1) // 2)
    tree = DSTree()
    for i in range(len(nodes)):
        tree.make_set(i)
        for j in range(i + 1, len(nodes)):
            value = np.sqrt(np.sum(np.square(nodes[i] - nodes[j])))
            edges.put((value, (i, j)))
    while tree.sets > 1:
        # TODO: Calculate centroids progressively as we build the pyramid
        c = tree.union(*edges.get()[1])
        c.value = calculate_centroid(c, nodes)
    return tree.root


def cluster_boruvka(nodes):
    """
    Agglomerate nodes based on their MST, given coordinates in n-dimensions.
    Uses Boruvka's algorithm: which should produce a more "balanced" tree.

    Input: np.ndarray((v, n))
    """
    v = nodes.shape[0]
    edges = np.zeros((v, v), dtype=np.float)
    tree = DSTree()
    for i in range(v):
        tree.make_set(i)
        edges[i, i] = np.inf
        for j in range(i + 1, v):
            value = np.sqrt(np.sum(np.square(nodes[i] - nodes[j])))
            edges[i, j] = value
    i_lower = np.tril_indices(v, -1)
    edges[i_lower] = edges.T[i_lower]
    while tree.sets > 1:
        clusters = list(set(map(lambda n: tuple(n.find().values()), tree._sets.values())))
        minimum_edges = {i : (np.inf, None) for i in range(len(clusters))}
        for c1, c2 in it.combinations(range(len(clusters)), 2):
            for e in it.product(clusters[c1], clusters[c2]):
                if edges[e] < minimum_edges[c1][0]:
                    minimum_edges[c1] = edges[e], e
                if edges[e] < minimum_edges[c2][0]:
                    minimum_edges[c2] = edges[e], e
        # for _, e in minimum_edges.values():
        for _, e in sorted(minimum_edges.values(), key=lambda t: t[0]):
            c = tree.union(*e)
            c.value = calculate_centroid(c, nodes)
    return tree.root


def mst_boruvka(nodes):
    result = set()
    v = nodes.shape[0]
    edges = np.zeros((v, v), dtype=np.float)
    tree = DSTree()
    for i in range(v):
        tree.make_set(i)
        edges[i, i] = np.inf
        for j in range(i + 1, v):
            value = np.sqrt(np.sum(np.square(nodes[i] - nodes[j])))
            edges[i, j] = value
    i_lower = np.tril_indices(v, -1)
    edges[i_lower] = edges.T[i_lower]
    while tree.sets > 1:
        clusters = list(set(map(lambda n: tuple(n.find().values()), tree._sets.values())))
        minimum_edges = {i : (np.inf, None) for i in range(len(clusters))}
        for c1, c2 in it.combinations(range(len(clusters)), 2):
            for e in it.product(clusters[c1], clusters[c2]):
                if edges[e] < minimum_edges[c1][0]:
                    minimum_edges[c1] = edges[e], e
                if edges[e] < minimum_edges[c2][0]:
                    minimum_edges[c2] = edges[e], e
        for _, e in minimum_edges.values():
            c = tree.union(*e)
            c.value = calculate_centroid(c, nodes)
            result.add(e)
    return result


cluster = cluster_boruvka


def cluster_edges(edges):
    """
    Agglomerate nodes based on their MST, given edge weights.
    Uses Kruskal's algorithm, which runs in O(v^2 log v)

    Input: np.ndarray((v, v))
    """
    edge_pq = PriorityQueue(len(edges) * (len(edges) - 1) // 2)
    tree = DSTree()
    for i in range(len(edges)):
        tree.make_set(i)
        for j in range(i + 1, len(edges)):
            edge_pq.put((edges[i, j], (i, j)))
    while tree.sets > 1:
        c = tree.union(*edge_pq.get()[1])
        # TODO: Can this even work? The entire structure of the problem is changing!
        # c.value = calculate_centroid(c, )


def evaluate_path(nodes, indices):
    current = nodes[0]
    distance = 0.
    for i in indices:
        distance += np.sqrt(np.sum(np.square(nodes[i] - current)))
        current = nodes[i]
    distance += np.sqrt(np.sum(np.square(nodes[-1] - current)))
    return distance


def evaluate_tour(nodes, indices):
    current = nodes[indices[0]]
    first = current
    indices = indices[1:]
    distance = 0.
    for i in indices:
        distance += np.sqrt(np.sum(np.square(nodes[i] - current)))
        current = nodes[i]
    distance += np.sqrt(np.sum(np.square(current - first)))
    return distance


def partial_shortest_tour(nodes, indices, left=None, right=None):
    if left is not None:  # Assume both left and right provided
        nodes = [left] + nodes + [right]
        indices = [i + 1 for i in indices]
    min_score = float('inf')
    min_path = None
    for path in permutations(indices):
        score = evaluate_path(nodes, path) if left is not None else evaluate_tour(nodes, path)
        if score < min_score:
            min_score = score
            min_path = path
    if left is not None:
        return [i - 1 for i in min_path]
    else:
        return min_path


def solve_level(nodes, c, k, left=None, right=None):
    children = c.split(k)
    if len(children) == 1:
        return children
    centroids = [centroid(d, nodes) for d in children]
    tour = partial_shortest_tour(centroids, list(range(len(children))), left, right)
    return [children[i] for i in tour]


def pyramid_solve(nodes, k=6):
    k = k - 1
    c = cluster(nodes)
    result = solve_level(nodes, c, k)
    while len(result) < nodes.shape[0]:
        new_result = []
        for i, c in enumerate(result):
            new_result += solve_level(nodes, c, k, centroid(new_result[-1] if new_result else result[-1], nodes), centroid(result[(i + 1) % len(result)], nodes))
        result = new_result
    result = [n.value for n in result]
    zero = result.index(0)
    return result[zero:] + result[:zero]


def pyramid_debug(nodes, k=6):
    k = k - 1
    c = cluster(nodes)
    result = solve_level(nodes, c, k)
    yield [centroid(d, nodes) for d in result]
    while len(result) < nodes.shape[0]:
        new_result = []
        for i, c in enumerate(result):
            new_result += solve_level(nodes, c, k, centroid(new_result[-1] if new_result else result[-1], nodes), centroid(result[(i + 1) % len(result)], nodes))
            yield [centroid(d, nodes) for d in new_result + result[i + 1:]]
        result = new_result
        yield [centroid(d, nodes) for d in result]
