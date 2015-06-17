import numpy as np
from menpofit.base import is_pyramid_on_features


def check_features(features, n_levels):
    r"""
    Checks the feature type per level.

    Parameters
    ----------
    features : callable or list of callables
        The features to apply to the images.
    n_levels : int
        The number of pyramid levels.

    Returns
    -------
    feature_list : list
        A list of feature function.
    """
    # Firstly, make sure we have a list of callables of the right length
    if is_pyramid_on_features(features):
        return features
    else:
        try:
            all_callables = check_list_callables(features, n_levels,
                                                 allow_single=False)
        except ValueError:
            raise ValueError("features must be a callable or a list of "
                             "{} callables".format(n_levels))
        return all_callables


def check_list_callables(callables, n_callables, allow_single=True):
    if not isinstance(callables, list):
        if allow_single:
            # expand to a list of callables for them
            callables = [callables] * n_callables
        else:
            raise ValueError("Expected a list of callables "
                             "(allow_single=False)")
    # must have a list by now
    for c in callables:
        if not callable(c):
            raise ValueError("All items must be callables")
    if len(callables) != n_callables:
        raise ValueError("List of callables must be {} "
                         "long".format(n_callables))
    return callables


def check_diagonal(diagonal):
    r"""
    Checks the diagonal length used to normalize the images' size that
    must be >= 20.
    """
    if diagonal is not None and diagonal < 20:
        raise ValueError("diagonal must be >= 20")


# TODO: document me!
def check_scales(scales):
    if isinstance(scales, (int, float)):
        return [scales], 1
    elif len(scales) == 1 and isinstance(scales[0], (int, float)):
        return list(scales), 1
    elif len(scales) > 1:
        l1, n1 = check_scales(scales[0])
        l2, n2 = check_scales(scales[1:])
        return l1 + l2, n1 + n2
    else:
        raise ValueError("scales must be an int/float or a list/tuple of "
                         "int/float")


# TODO: document me!
def check_patch_shape(patch_shape, n_levels):
    if len(patch_shape) == 2 and isinstance(patch_shape[0], int):
        return [patch_shape] * n_levels
    elif len(patch_shape) == 1:
        return check_patch_shape(patch_shape[0], 1)
    elif len(patch_shape) == n_levels:
        l1 = check_patch_shape(patch_shape[0], 1)
        l2 = check_patch_shape(patch_shape[1:], n_levels-1)
        return l1 + l2
    else:
        raise ValueError("patch_shape must be a list/tuple of int or a "
                         "list/tuple of lit/tuple of int/float with the "
                         "same length as scales")


def check_max_components(max_components, n_levels, var_name):
    r"""
    Checks the maximum number of components per level either of the shape
    or the appearance model. It must be None or int or float or a list of
    those containing 1 or {n_levels} elements.
    """
    str_error = ("{} must be None or an int > 0 or a 0 <= float <= 1 or "
                 "a list of those containing 1 or {} elements").format(
        var_name, n_levels)
    if not isinstance(max_components, list):
        max_components_list = [max_components] * n_levels
    elif len(max_components) == 1:
        max_components_list = [max_components[0]] * n_levels
    elif len(max_components) == n_levels:
        max_components_list = max_components
    else:
        raise ValueError(str_error)
    for comp in max_components_list:
        if comp is not None:
            if not isinstance(comp, int):
                if not isinstance(comp, float):
                    raise ValueError(str_error)
    return max_components_list


# TODO: document me!
def check_max_iters(max_iters, n_levels):
    if type(max_iters) is int:
        max_iters = [np.round(max_iters/n_levels)
                     for _ in range(n_levels)]
    elif len(max_iters) == 1 and n_levels > 1:
        max_iters = [np.round(max_iters[0]/n_levels)
                     for _ in range(n_levels)]
    elif len(max_iters) != n_levels:
        raise ValueError('max_iters can be integer, integer list '
                         'containing 1 or {} elements or '
                         'None'.format(n_levels))
    return np.require(max_iters, dtype=np.int)


# TODO: document me!
def check_sampling(sampling, n_levels):
    if isinstance(sampling, (list, tuple)):
        if len(sampling) == 1:
            sampling = sampling * n_levels
        elif len(sampling) != n_levels:
            raise ValueError('A sampling list can only '
                             'contain 1 element or {} '
                             'elements'.format(n_levels))
    elif isinstance(sampling, np.ndarray):
        sampling = [sampling] * n_levels
    else:
        raise ValueError('sampling can be a ndarray, a ndarray list '
                         'containing 1 or {} elements or '
                         'None'.format(n_levels))
    return sampling


# def check_n_levels(n_levels):
#     r"""
#     Checks the number of pyramid levels - must be int > 0.
#     """
#     if not isinstance(n_levels, int) or n_levels < 1:
#         raise ValueError("n_levels must be int > 0")
#
#
# def check_downscale(downscale):
#     r"""
#     Checks the downscale factor of the pyramid that must be >= 1.
#     """
#     if downscale < 1:
#         raise ValueError("downscale must be >= 1")
#
#
# def check_boundary(boundary):
#     r"""
#     Checks the boundary added around the reference shape that must be
#     int >= 0.
#     """
#     if not isinstance(boundary, int) or boundary < 0:
#         raise ValueError("boundary must be >= 0")
