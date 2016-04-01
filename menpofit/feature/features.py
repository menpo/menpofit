from __future__ import division
import numpy as np
import warnings

from menpo.feature import ndfeature


@ndfeature
def centralize(x, axes=(-2, -1)):
    r"""
    Normalizes an image so that it has zero mean.

    Parameters
    ----------
    x : `menpo.image.Image` or subclass or ``(C, X, Y, ..., Z)`` `ndarray`
        The input image.
    axes : ``None`` or `int` or `tuple` of `int`, optional
        Axes along which the normalization is performed.

    Returns
    -------
    normalized_image : `menpo.image.Image` or ``(C, X, Y, ..., Z)`` `ndarray`
        The normalized image.
    """
    mean = np.mean(x, axis=axes, keepdims=True)
    return x - mean


@ndfeature
def normalize_norm(x, axes=(-2, -1)):
    r"""
    Normalizes an image so that it has unit norm.

    Parameters
    ----------
    x : `menpo.image.Image` or subclass or `ndarray`
        The input image.
    axes : ``None`` or `int` or `tuple` of `int`, optional
        Axes along which the normalization is performed.

    Returns
    -------
    normalized_image : `menpo.image.Image` or ``(C, X, Y, ..., Z)`` `ndarray`
        The normalized image.
    """
    x = centralize(x, axes=axes)
    norm = np.asarray(np.linalg.norm(x, axis=axes))
    positions = np.asarray(axes) + len(x.shape)
    for axis in positions:
        norm = np.expand_dims(norm, axis=axis)
    return handle_div_by_zero(x, norm)


@ndfeature
def normalize_std(x, axes=(-2, -1)):
    r"""
    Normalizes an image so that it has unit standard deviation.

    Parameters
    ----------
    x : `menpo.image.Image` or subclass or `ndarray`
        The input image.
    axes : ``None`` or `int` or `tuple` of `int`, optional
        Axes along which the normalization is performed.

    Returns
    -------
    normalized_image : `menpo.image.Image` or ``(C, X, Y, ..., Z)`` `ndarray`
        The normalized image.
    """
    x = centralize(x, axes=axes)
    std = np.std(x, axis=axes, keepdims=True)
    return handle_div_by_zero(x, std)


@ndfeature
def normalize_var(x, axes=(-2, -1)):
    r"""
    Normalizes an image with respect to its variance.

    Parameters
    ----------
    x : `menpo.image.Image` or subclass or `ndarray`
        The input image.
    axes : ``None`` or `int` or `tuple` of `int`, optional
        Axes along which the normalization is performed.

    Returns
    -------
    normalized_image : `menpo.image.Image` or ``(C, X, Y, ..., Z)`` `ndarray`
        The normalized image.
    """
    x = centralize(x, axes=axes)
    var = np.var(x, axis=axes, keepdims=True)
    return handle_div_by_zero(x, var)


@ndfeature
def probability_map(x, axes=(-2, -1)):
    r"""
    Generates the probability MAP of the image.

    Parameters
    ----------
    x : `menpo.image.Image` or subclass or `ndarray`
        The input image.
    axes : ``None`` or `int` or `tuple` of `int`, optional
        Axes along which the normalization is performed.

    Returns
    -------
    probability_map : `menpo.image.Image` or ``(C, X, Y, ..., Z)`` `ndarray`
        The probability MAP of the image.
    """
    x -= np.min(x, axis=axes, keepdims=True)
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


def handle_div_by_zero(x, normalizer):
    r"""
    Function that performs division and is able to handle cases of division by
    zero.

    Parameters
    ----------
    x : `ndarray`
        The input array..
    normalizer : `ndarray`
        The normalizer values.

    Returns
    -------
    division_result : `ndarray`
        The result of the division.
    """
    nonzero = normalizer > 0
    if np.any(~nonzero):
        warnings.warn("some of the denominators have 0 variance - they cannot "
                      "be normalized.")
        x[nonzero] /= normalizer[nonzero]
    else:
        x /= normalizer
    return x
