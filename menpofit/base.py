from __future__ import division
from functools import partial
import itertools
import numpy as np
from menpo.visualize import progress_bar_str, print_dynamic


def name_of_callable(c):
    try:
        if isinstance(c, partial):  # partial
            # Recursively call as partial may be wrapping either a callable
            # or a function (or another partial for some reason!)
            return name_of_callable(c.func)
        else:
            return c.__name__  # function
    except AttributeError:
        return c.__class__.__name__  # callable class


def batch(iterable, n):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def build_grid(shape):
    r"""
    """
    shape = np.asarray(shape)
    half_shape = np.floor(shape / 2)
    half_shape = np.require(half_shape, dtype=int)
    start = -half_shape
    end = half_shape + shape % 2
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    return np.rollaxis(sampling_grid, 0, 3)
