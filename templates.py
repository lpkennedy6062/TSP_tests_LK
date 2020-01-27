import numpy as np


RIGHT_90 = np.pi / 2.
LEFT_90 = -np.pi / 2.


class Point:
    @classmethod
    def random(cls, x_bounds, y_bounds):
        x = (np.random.rand() * (x_bounds[1] - x_bounds[0])) + x_bounds[0]
        y = (np.random.rand() * (y_bounds[1] - y_bounds[0])) + y_bounds[0]
        return cls(x, y)

    def __init__(self, x, y):
        self.x, self.y = x, y

    @property
    def coord(self):
        return self.x, self.y


class Direction:
    @classmethod
    def random(cls):
        return cls(2. * np.pi * np.random.rand())

    def __init__(self, a):
        self.a = a

    def add(self, p: Point, m):
        x = p.x + (m * np.cos(self.a))
        y = p.y + (m * np.sin(self.a))
        return Point(x, y)

    def __add__(self, obj):
        if type(obj) == Direction:
            return Direction(self.a + obj.a)
        return Direction(self.a + obj)


class Component:
    def __init__(self):
        raise NotImplementedError()
    def __call__(self, o: Point, d: Direction, add_obstacle):
        raise NotImplementedError()


class Line(Component):
    def __init__(self, m):
        self.m = m
    def __call__(self, o: Point, d: Direction, add_obstacle):
        add_obstacle(d.add(o, -1).coord, d.add(o, self.m + 1).coord)
        return d.add(o, self.m), d


class Turn(Component):
    def __init__(self, a):
        self.a = a
    def __call__(self, o: Point, d: Direction, add_obstacle):
        return o, d + self.a


class Gap(Component):
    def __init__(self, m):
        self.m = m
    def __call__(self, o: Point, d: Direction, add_obstacle):
        return d.add(o, self.m), d


class Template:
    def __init__(self, components: [Component], o=None, d=None):
        self.components = components
        self.o, self.d = o, d
    def __call__(self, x_bounds, y_bounds, add_obstacle):
        o = Point.random(x_bounds, y_bounds) if self.o is None else self.o
        d = Direction.random() if self.d is None else self.d
        for c in self.components:
            o, d = c(o, d, add_obstacle)
