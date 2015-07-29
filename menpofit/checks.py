import numpy as np
import warnings
from menpo.shape import TriMesh
from menpo.transform import PiecewiseAffine


def check_diagonal(diagonal):
    r"""
    Checks the diagonal length used to normalize the images' size that
    must be >= 20.
    """
    if diagonal is not None and diagonal < 20:
        raise ValueError("diagonal must be >= 20")


def check_trilist(image, transform, group=None):
    trilist = image.landmarks[group].lms

    if not isinstance(trilist, TriMesh) and isinstance(transform,
                                                       PiecewiseAffine):
        warnings.warn('The given images do not have an explicit triangulation '
                      'applied. A Delaunay Triangulation will be computed '
                      'and used for warping. This may be suboptimal and cause '
                      'warping artifacts.')


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


def check_features(features, n_levels):
    r"""
    Checks the feature type per level.

    Parameters
    ----------
    features : callable or list of callables
        The features to apply to the images.
    n_scales : int
        The number of pyramid levels.

    Returns
    -------
    feature_list : list
        A list of feature function.
    """
    if callable(features):
        return [features] * n_levels
    elif len(features) == 1 and np.alltrue([callable(f) for f in features]):
        return list(features) * n_levels
    elif len(features) == n_levels and np.alltrue([callable(f)
                                                   for f in features]):
        return list(features)
    else:
        raise ValueError("features must be a callable or a list/tuple of "
                         "callables with the same length as scales")


# TODO: document me!
def check_scale_features(scale_features, features):
    r"""
    """
    if np.alltrue([f == features[0] for f in features]):
        return scale_features
    elif scale_features:
        # Only raise warning if True was passed.
        warnings.warn('scale_features has been automatically set to False '
                      'because different types of features are used at each '
                      'level.')
        return False
    else:
        return scale_features


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
    those containing 1 or {n_scales} elements.
    """
    str_error = ("{} must be None or an int > 0 or a 0 <= float <= 1 or "
                 "a list of those containing 1 or {} elements").format(
        var_name, n_levels)
    if not isinstance(max_components, (list, tuple)):
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
    if (isinstance(sampling, (list, tuple)) and
        np.alltrue([isinstance(s, (np.ndarray, np.int)) or sampling is None
                    for s in sampling])):
        if len(sampling) == 1:
            return sampling * n_levels
        elif len(sampling) == n_levels:
            return sampling
        else:
            raise ValueError('A sampling list can only '
                             'contain 1 element or {} '
                             'elements'.format(n_levels))
    elif isinstance(sampling, (np.ndarray, np.int)) or sampling is None:
        return [sampling] * n_levels
    else:
        raise ValueError('sampling can be an integer or ndarray, '
                         'a integer or ndarray list '
                         'containing 1 or {} elements or '
                         'None'.format(n_levels))


