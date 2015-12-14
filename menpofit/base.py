from __future__ import division
import itertools
import numpy as np


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
