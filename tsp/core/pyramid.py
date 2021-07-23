from typing import Any, Hashable, Iterable, Iterator, List, Set, Tuple
from itertools import permutations
from queue import PriorityQueue
import itertools as it
from numpy.typing import ArrayLike
import numpy as np


class DSNode:
    """Implements a disjoint-set dendrogram."""

    def __init__(self, value: Any, children: List[DSNode] = None, height: int = 0):
        """
        Args:
            value (Any): value to store at this node (can be literally anything)
            children (List[DSNode], optional): Nodes immediately below this one. Defaults to [].
            height (int, optional): Numerical value expressing the depth of the tree below it. Defaults to 0.
        """
        self.value = value
        self.children = [] if children is None else children
        self.height = height
        self.parent = self

    def __lt__(self, obj: DSNode) -> bool:
        return self.height < obj.height

    def __eq__(self, obj: DSNode) -> bool:
        return self.height == obj.height

    def find(self) -> DSNode:
        """Recursively finds the root of the tree.

        Returns:
            DSNode: root
        """
        if self.parent.parent is not self.parent:
            self.parent = self.parent.find()
        return self.parent

    def split(self, depth: int = 1) -> List[DSNode]:
        """Returns the depth + 1 clusters produced by descending the dendrogram.

        Args:
            depth (int, optional): how many levels down the dendrogram to split. Defaults to 1.

        Returns:
            List[DSNode]: depth + 1 roots of the subtrees
        """
        if self.height < 1 or depth < 1:
            return [self]
        result = self.children[:]
        for _ in range(depth - 1):
            to_split = max(result)
            result.remove(to_split)
            result += to_split.split()
        return result

    def values(self) -> List[Any]:
        """List this node's value and all values below it.

        Returns:
            List[Any]: values
        """
        if not self.children:
            return [self.value]
        return self.children[0].values() + self.children[1].values()


class DSTree:
    """Container for building the disjoint-set dendrogram."""

    def __init__(self, values: List[Hashable] = None):
        """
        Args:
            values (List[Any], optional): Initial leaves to initialize. Defaults to None.
        """
        self.sets = dict()
        self._unions = 0
        self.root = None
        if values is not None:
            for value in values:
                self.make_set(value)

    @property
    def sets(self) -> int:
        """Number of disjoint sets in the tree.

        Returns:
            int: number of sets
        """
        return len(self.sets) - self._unions

    def make_set(self, value: Hashable):
        """Initialize a new set/leaf of tree.

        Args:
            value (Any): value to store at node
        """
        self.sets[value] = DSNode(value)

    def find(self, value: Hashable) -> DSNode:
        """Find root of tree containing node which stores value.

        Args:
            value (Hashable): value of node of interest

        Returns:
            DSNode: root of tree
        """
        return self.sets[value].find()

    def union(self, x: Hashable, y: Hashable) -> DSNode:
        """Create a union of set containing x and set containing y.

        Args:
            x (Hashable): value which picks out first set
            y (Hashable): value which picks out second set

        Returns:
            DSNode: root of new tree which is the union of the sets (or not-new if they're both in the same set)
        """
        x_parent, y_parent = self.find(x), self.find(y)
        if x_parent is y_parent:
            return x_parent
        self._unions += 1
        new_parent = DSNode(None, [x_parent, y_parent], self._unions)
        x_parent.parent = new_parent
        y_parent.parent = new_parent
        self.root = new_parent
        return new_parent


def _calculate_centroid(c: DSNode, nodes: ArrayLike) -> Tuple[float, ArrayLike]:
    """Given a node in a disjoint-set dendrogram, calculate the center of gravity of all its children.

    Args:
        c (DSNode): parent node of relevance
        nodes (ArrayLike): master list of coordinates of points

    Returns:
        Tuple[float, ArrayLike]: (number of children, centroid)
    """
    result = np.zeros((nodes.shape[1],), dtype=np.int32)
    total = 0
    for child in c.children:
        if isinstance(child.value, tuple):
            total += child.value[0]
            result += child.value[1] * child.value[0]
        else:
            total += 1
            result += nodes[child.value]
    return total, result // total


def centroid(c: DSNode, nodes: ArrayLike) -> ArrayLike:
    """Gets pre-computed centroid of node in tree.

    Args:
        c (DSNode): parent node of relevance
        nodes (ArrayLike): master list of coordinates of points

    Returns:
        ArrayLike: centroid
    """
    if not c.children:
        return nodes[c.value]
    return c.value[1]


def cluster_kruskal(nodes: ArrayLike) -> Tuple[Set, DSNode]:
    """Agglomerate nodes based on their MST, given coordinates in n-dimensions.
    Uses Kruskal's algorithm, which runs in O(v^2 log v)

    Args:
        nodes (ArrayLike): ndarray((v, n))

    Returns:
        Tuple[Set, DSNode]: (edges in MST, root of agglomerated tree)
    """
    result = set()
    edges = PriorityQueue(len(nodes) * (len(nodes) - 1) // 2)
    tree = DSTree()
    for i in range(len(nodes)):
        tree.make_set(i)
        for j in range(i + 1, len(nodes)):
            value = np.sqrt(np.sum(np.square(nodes[i] - nodes[j])))
            edges.put((value, (i, j)))
    while tree.sets > 1:
        e = edges.get()[1]
        c = tree.union(*e)
        c.value = _calculate_centroid(c, nodes)
        result.add(e)
    return result, tree.root


def cluster_boruvka(nodes: ArrayLike) -> Tuple[Set, DSNode]:
    """Agglomerate nodes based on their MST, given coordinates in n-dimensions.
    Uses Boruvka's algorithm: which should produce a more "balanced" tree.

    Args:
        nodes (ArrayLike): ndarray((v, n))

    Returns:
        Tuple[Set, DSNode]: (edges in MST, root of agglomerated tree)
    """
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
        clusters = list(set(map(lambda n: tuple(n.find().values()), tree.sets.values())))
        minimum_edges = {i : (np.inf, None) for i in range(len(clusters))}
        for c1, c2 in it.combinations(range(len(clusters)), 2):
            for e in it.product(clusters[c1], clusters[c2]):
                if edges[e] < minimum_edges[c1][0]:
                    minimum_edges[c1] = edges[e], e
                if edges[e] < minimum_edges[c2][0]:
                    minimum_edges[c2] = edges[e], e
        for _, e in minimum_edges.values():
            c = tree.union(*e)
            c.value = _calculate_centroid(c, nodes)
            result.add(e)
    return result, tree.root


mst = lambda nodes: cluster_boruvka(nodes)[0]
cluster = lambda nodes: cluster_boruvka(nodes)[1]


def _evaluate_path(nodes: ArrayLike, indices: Iterable[int]) -> float:
    """Computes length of path.

    Args:
        nodes (ArrayLike): master list of coordinates of points
        indices (Iterable[int]): indices of points making up the path

    Returns:
        float: length
    """
    current = nodes[0]
    distance = 0.
    for i in indices:
        distance += np.sqrt(np.sum(np.square(nodes[i] - current)))
        current = nodes[i]
    distance += np.sqrt(np.sum(np.square(nodes[-1] - current)))
    return distance


def _evaluate_tour(nodes: ArrayLike, indices: Iterable[int]) -> float:
    """Computes length of path with return to first node in the path.

    Args:
        nodes (ArrayLike): master list of coordinates of points
        indices (Iterable[int]): indices of points making up the tour (without repeat of first point)

    Returns:
        float: length
    """
    current = nodes[indices[0]]
    first = current
    indices = indices[1:]
    distance = 0.
    for i in indices:
        distance += np.sqrt(np.sum(np.square(nodes[i] - current)))
        current = nodes[i]
    distance += np.sqrt(np.sum(np.square(current - first)))
    return distance


def _partial_shortest_tour(nodes: ArrayLike, indices: Iterable[int], left: int = None, right: int = None) -> List[int]:
    """Brute force the shortest path (if left and right nodes provided) or tour (if not provided).
    Should only be used on computationally tractable subproblems.

    Args:
        nodes (ArrayLike): master list of coordinates of points
        indices (Iterable[int]): indices of points making up the path/tour
        left (int, optional): Left (starting) node, if computing a path. Defaults to None.
        right (int, optional): Right (ending) node, if computing a path. Defaults to None.

    Returns:
        List[int]: shortest path/tour
    """
    if left is not None:  # Assume both left and right provided
        nodes = [left] + nodes + [right]
        indices = [i + 1 for i in indices]
    min_score = float('inf')
    min_path = None
    for path in permutations(indices):
        score = _evaluate_path(nodes, path) if left is not None else _evaluate_tour(nodes, path)
        if score < min_score:
            min_score = score
            min_path = path
    if left is not None:
        return [i - 1 for i in min_path]
    return min_path


def solve_level(nodes: ArrayLike, c: DSNode, k: int, left: int = None, right: int = None) -> List[int]:
    """Find shortest path/tour through a branch of the tree.

    Args:
        nodes (ArrayLike): master list of coordinates of points
        c (DSNode): parent node of branch
        k (int): cluster size
        left (int, optional): Left (starting) node, if computing a path. Defaults to None.
        right (int, optional): Right (ending) node, if computing a path. Defaults to None.

    Returns:
        List[int]: shortest path/tour
    """
    children = c.split(k)
    if len(children) == 1:
        return children
    centroids = [centroid(d, nodes) for d in children]
    tour = _partial_shortest_tour(centroids, list(range(len(children))), left, right)
    return [children[i] for i in tour]


def pyramid_solve(nodes: ArrayLike, k: int = 6) -> List[int]:
    """Find an approximately-optimal tour using heirarchical clustering algorithm.

    Args:
        nodes (ArrayLike): master list of coordinates of points
        k (int, optional): Cluster size. Defaults to 6.

    Returns:
        List[int]: tour
    """
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


def pyramid_debug(nodes: ArrayLike, k: int = 6) -> Iterator[List[int]]:
    """Starts by yielding the centroids at the top level of the pyramid, then the level below, and so on, in the following pattern:
    (1) tour of centroids at top of pyramid: [a, b, c, d, e, f]
    (2) [cluster below a, b, c, d, e, f]
    (3) [cluster below a, cluster below b, c, d, e, f]
    (4) ...
    (5) [cluster below a, cluster below b, ..., cluster below f]
    Repeating from 1 for the next level of the pyramid (which is the tour produced in 5).

    Args:
        nodes (ArrayLike): master list of coordinates of points
        k (int, optional): Cluster size. Defaults to 6.

    Yields:
        Iterator[List[int]]: [description]
    """
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
