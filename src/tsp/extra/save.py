from typing import Any, Dict
import json
import os
import numpy as np

from tsp.core.save import LoadError, save_ntsp, load_ntsp, save_tsp, load_tsp, save_list, load_list # pylint: disable=unused-import
from tsp.core.save import save_problem as save_problem_, load_problem as load_problem_
from tsp.extra.obstacles import TSP_O
from tsp.extra.color import TSP_Color


def save_obstacles(obj: TSP_O, path: str):
    """Serialize a TSP_O object.

    Args:
        obj (TSP_O): object
        path (str): path to save
    """
    struct = {
        "type": "TSP_O",
        "cities": obj.cities.tolist(),
        "w": obj.w,
        "h": obj.h,
        "obstacles": obj.obstacles.tolist()
    }
    with open(path, 'w') as f:
        json.dump(struct, f)


def _load_obstacles(struct: Dict) -> TSP_O:
    result = TSP_O.from_cities(struct["cities"], struct["w"], struct["h"])
    result.obstacles = np.array(struct["obstacles"])
    return result


def load_obstacles(path: str) -> TSP_O:
    """Unserialize a TSP_O object.

    Args:
        path (str): path to load

    Returns:
        TSP_O: object
    """
    with open(path, 'r') as f:
        struct = json.load(f)
    return _load_obstacles(struct)


def save_color(obj: TSP_Color, path: str):
    """Serialize a TSP_Color object.

    Args:
        obj (TSP_Color): object
        path (str): path to save
    """
    struct = {
        "type": "TSP_Color",
        "cities": obj.cities.tolist(),
        "w": obj.w,
        "h": obj.h,
        "penalty": obj.penalty,
        "colors": obj.colors.tolist()
    }
    with open(path, 'w') as f:
        json.dump(struct, f)


def _load_color(struct: Dict) -> TSP_Color:
    return TSP_Color.from_cities(
        zip(struct["cities"], struct["colors"]),
        struct["w"],
        struct["h"],
        struct["penalty"]
    )


def load_color(path: str) -> TSP_Color:
    """Unserialize a TSP_Color object.

    Args:
        path (str): path to load

    Returns:
        TSP_Color: object
    """
    with open(path, 'r') as f:
        struct = json.load(f)
    return _load_color(struct)


def save_problem(obj: Any, path: str):
    """Serialize an object (should be descended from N_TSP, supports TSP_O and TSP_Color).

    Args:
        obj (Any): object
        path (str): path to save
    """
    if isinstance(obj, TSP_O):
        save_obstacles(obj, path)
    elif isinstance(obj, TSP_Color):
        save_color(obj, path)
    else:
        save_problem_(obj, path)  # fall through to other save_problem


def load_problem(path: str) -> Any:
    """Unserialize an object (TSP_O, TSP_Color, TSP, or N_TSP).

    Args:
        path (str): path to load

    Raises:
        LoadError: serialized object not supported

    Returns:
        Any: object
    """
    with open(path, 'r') as f:
        struct = json.load(f)
    if struct["type"] == "TSP_O":
        return _load_obstacles(struct)
    if struct["type"] == "TSP_Color":
        return _load_color(struct)
    return load_problem_(path)
