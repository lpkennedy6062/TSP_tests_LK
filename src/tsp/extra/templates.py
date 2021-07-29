from typing import Callable, List, Tuple
import numpy as np


RIGHT_90 = np.pi / 2.
LEFT_90 = -np.pi / 2.


class Point:
    """Container for a 2D point."""

    @classmethod
    def random(cls, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float]):
        """Generates a random point.

        Args:
            x_bounds (Tuple[float, float]): min and max x
            y_bounds (Tuple[float, float]): min and max y
        """
        x = (np.random.rand() * (x_bounds[1] - x_bounds[0])) + x_bounds[0]
        y = (np.random.rand() * (y_bounds[1] - y_bounds[0])) + y_bounds[0]
        return cls(x, y)

    def __init__(self, x: float, y: float):
        self.x, self.y = x, y

    @property
    def coord(self) -> Tuple[float, float]:
        """Cartestian coordinates of the point.

        Returns:
            Tuple[float, float]: (x, y)
        """
        return self.x, self.y


class Direction:
    """Container for a direction (radians)."""

    @classmethod
    def random(cls):
        """Generates a direction of random angle."""
        return cls(2. * np.pi * np.random.rand())

    def __init__(self, a: float):
        self.a = a

    def add(self, p: Point, m: float) -> Point:
        """Find the point m distance from p in direction.

        Args:
            p (Point): origin
            m (float): magnitude

        Returns:
            Point: tip
        """
        x = p.x + (m * np.cos(self.a))
        y = p.y + (m * np.sin(self.a))
        return Point(x, y)

    def __add__(self, obj: object):
        if isinstance(obj, Direction):
            return Direction(self.a + obj.a)
        return Direction(self.a + obj)


class Component:
    """Interface for a template component."""

    def __init__(self):
        pass

    def __call__(self, o: Point, d: Direction, add_obstacle: Callable):
        """
        Args:
            o (Point): origin
            d (Direction): angle
            add_obstacle (Callable): method to add obstacle to problem

        Returns:
            Tuple[Point, Direction]: new origin and angle
        """


class Line(Component):
    """Add line obstacle."""

    def __init__(self, m: float):
        """
        Args:
            m (float): length
        """
        Component.__init__(self)
        self.m = m

    def __call__(self, o: Point, d: Direction, add_obstacle: Callable) -> Tuple[Point, Direction]:
        add_obstacle(d.add(o, -1).coord, d.add(o, self.m + 1).coord)
        return d.add(o, self.m), d


class Turn(Component):
    """Change angle."""

    def __init__(self, a: float):
        """
        Args:
            a (float): amount to rotate
        """
        Component.__init__(self)
        self.a = a

    def __call__(self, o: Point, d: Direction, add_obstacle: Callable) -> Tuple[Point, Direction]:
        return o, d + self.a


class Gap(Component):
    """Skip some amount of space."""

    def __init__(self, m: float):
        """
        Args:
            m (float): distance
        """
        Component.__init__(self)
        self.m = m

    def __call__(self, o: Point, d: Direction, add_obstacle: Callable):
        return d.add(o, self.m), d


class Template:
    """Container for a template (composed of a list of components)."""

    def __init__(self, components: List[Component], o: Point = None, d: Direction = None):
        """
        Args:
            components (List[Component]): components to be sequentially inserted
            o (Point, optional): Origin point, if None then will be randomly generated. Defaults to None.
            d (Direction, optional): Starting direction, if None then will be randomly generated. Defaults to None.
        """
        self.components = components
        self.o, self.d = o, d

    def __call__(self, add_obstacle: Callable, x_bounds: Tuple[float, float] = None, y_bounds: Tuple[float, float] = None):
        """
        Args:
            add_obstacle (Callable): method to add obstacle to problem
            x_bounds (Tuple[float, float], optional): For random origin generation if origin not given in constructor. Defaults to None.
            y_bounds (Tuple[float, float], optional): For random origin generation if origin not given in constructor. Defaults to None.
        """
        o = Point.random(x_bounds, y_bounds) if self.o is None else self.o
        d = Direction.random() if self.d is None else self.d
        for c in self.components:
            o, d = c(o, d, add_obstacle)
