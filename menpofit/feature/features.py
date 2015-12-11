from __future__ import division
import numpy as np
import warnings
from menpo.feature import ndfeature


# TODO: Document me!
@ndfeature
def centralize(x, axes=(-2, -1)):
    r"""
    """
    mean = np.mean(x, axis=axes, keepdims=True)
    return x - mean


# TODO: Document me!
@ndfeature
def normalize_norm(x, axes=(-2, -1)):
    r"""
    """
    x = centralize(x, axes=axes)
    norm = np.asarray(np.linalg.norm(x, axis=axes))
    positions = np.asarray(axes) + len(x.shape)
    for axis in positions:
        norm = np.expand_dims(norm, axis=axis)
    return handle_div_by_zero(x, norm)


# TODO: document me!
@ndfeature
def normalize_std(x, axes=(-2, -1)):
    r"""
    """
    x = centralize(x, axes=axes)
    std = np.std(x, axis=axes, keepdims=True)
    return handle_div_by_zero(x, std)


# TODO: document me!
@ndfeature
def normalize_var(x, axes=(-2, -1)):
    r"""
    """
    x = centralize(x, axes=axes)
    var = np.var(x, axis=axes, keepdims=True)
    return handle_div_by_zero(x, var)


# TODO: document me!
@ndfeature
def probability_map(x, axes=(-2, -1)):
    r"""
    """
    x = x - np.min(x, axis=axes, keepdims=True)
    total = np.sum(x, axis=axes, keepdims=True)
    nonzero = total > 0
    if np.any(~nonzero):
        warnings.warn("some of x axes have 0 variance - uniform probability "
                      "maps are used them.")
        x[nonzero] /= total[nonzero]
        x[~nonzero] = 1 / np.prod(axes)
    else:
        x /= total
    return x


# TODO: document me!
def handle_div_by_zero(x, normalizer):
    r"""
    """
    nonzero = normalizer > 0
    if np.any(~nonzero):
        warnings.warn("some of the denominators have 0 variance - they cannot "
                      "be normalized.")
        x[nonzero] /= normalizer[nonzero]
    else:
        x /= normalizer
    return x
