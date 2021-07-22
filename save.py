from typing import Any, Dict, Iterable, List
import os
import json
from tsp.tsp import N_TSP, TSP


class LoadError(Exception):
    """Exception for expected problems with loading files."""


def save_ntsp(obj: N_TSP, path: str):
    """Serialize an N_TSP object.

    Args:
        obj (N_TSP): object
        path (str): path to save
    """
    struct = {
        "type": "N_TSP",
        "cities": obj.cities
    }
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(path, 'w') as f:
        json.dump(struct, f)


def _load_ntsp(struct: Dict) -> N_TSP:
    return N_TSP.from_cities(struct["cities"])


def load_ntsp(path: str) -> N_TSP:
    """Unserialize an N_TSP object.

    Args:
        path (str): path to load

    Returns:
        N_TSP: object
    """
    with open(path, 'r') as f:
        struct = json.load(f)
    return _load_ntsp(struct)


def save_tsp(obj: TSP, path: str):
    """Serialize a TSP object.

    Args:
        obj (TSP): object
        path (str): path to save
    """
    struct = {
        "type": "TSP",
        "cities": obj.cities,
        "w": obj.w,
        "h": obj.h
    }
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(path, 'w') as f:
        json.dump(struct, f)


def _load_tsp(struct: Dict) -> TSP:
    return TSP.from_cities(struct["cities"], struct["w"], struct["h"])


def load_tsp(path: str) -> TSP:
    """Unserialize a TSP object.

    Args:
        path (str): path to load

    Returns:
        TSP: object
    """
    with open(path, 'r') as f:
        struct = json.load(f)
    return _load_tsp(struct)


def save_problem(obj: Any, path: str):
    """Serialize an object (should be descended from N_TSP).

    Args:
        obj (Any): object
        path (str): path to save
    """
    if isinstance(obj, TSP):
        save_tsp(obj, path)
    else:
        save_ntsp(obj, path)  # try to save anything else as if it's a generic N_TSP


def load_problem(path: str) -> Any:
    """Unserialize an object (TSP or N_TSP).

    Args:
        path (str): path to load

    Raises:
        LoadError: serialized object not TSP or N_TSP

    Returns:
        Any: object
    """
    with open(path, 'r') as f:
        struct = json.load(f)
    if struct["type"] == "TSP":
        return _load_tsp(struct)
    if struct["type"] == "N_TSP":
        return _load_ntsp(struct)
    raise LoadError('invalid type')


def save_list(obj: Iterable[Any], path: str):
    """Generic save function for tours of various formats.

    Args:
        obj (Iterable[Any]): list
        path (str): path to save
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    with open(path, 'w') as f:
        json.dump(list(obj), f)


def load_list(path: str) -> List[Any]:
    """Generic load function for tours of various formats.

    Args:
        path (str): path to load

    Returns:
        List[Any]: list
    """
    with open(path, 'r') as f:
        return json.load(f)
