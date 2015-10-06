import warnings
import collections
from functools import partial
import numpy as np

from menpo.shape import TriMesh, DirectedGraph, UndirectedGraph, Tree
from menpo.transform import PiecewiseAffine


def check_diagonal(diagonal):
    r"""
    Checks the diagonal length that is used to normalize the images' size. It
    must be >= 20 or ``None``.

    Parameters
    ----------
    diagonal : `int` or `float` or ``None``
        The diagonal value to check.

    Returns
    -------
    diagonal : `int` or `float` or ``None``
        The diagonal value if no errors were raised.

    Raises
    ------
    ValueError
        diagonal must be >= 20 or None.
    """
    if diagonal is not None and diagonal < 20:
        raise ValueError("diagonal must be >= 20 or None.")
    return diagonal


def check_trilist(shape, transform):
    r"""
    Checks if the combination of a shape and a transform is :map:`TriMesh`
    and :map:`PiecewiseAffine`, respectively. If not, it raises a warning.

    Parameters
    ----------
    shape : :map:`TriMesh`
        The shape to check.
    transform : :map:`PiecewiseAffine`
        The transform to check.

    Raises
    ------
    Warning
        The given images do not have an explicit triangulation applied. A
        Delaunay Triangulation will be computed and used for warping. This may
        be suboptimal and cause warping artifacts.
    """
    if not isinstance(shape, TriMesh) and isinstance(transform,
                                                     PiecewiseAffine):
        warnings.warn('The given images do not have an explicit triangulation '
                      'applied. A Delaunay Triangulation will be computed '
                      'and used for warping. This may be suboptimal and cause '
                      'warping artifacts.')


def check_landmark_trilist(image, transform, group=None):
    r"""
    Checks if the combination of the shape of an image and a transform is
    :map:`TriMesh` and :map:`PiecewiseAffine`, respectively. If not,
    it raises a warning.

    Parameters
    ----------
    image : :map:`Image`
        The image from which to check the shape.
    transform : :map:`PiecewiseAffine`
        The transform to check.

    Raises
    ------
    Warning
        The given images do not have an explicit triangulation applied. A
        Delaunay Triangulation will be computed and used for warping. This may
        be suboptimal and cause warping artifacts.
    """
    shape = image.landmarks[group].lms
    check_trilist(shape, transform)


def check_scales(scales):
    r"""
    Checks the provided scales. Scales must be either `int` or `float` or a
    `list`/`tuple` of those.

    Parameters
    ----------
    scales : `int` or `float` or a `list`/`tuple` of those
        The scales to check.

    Returns
    -------
    scales : `int` or `float` or a `list`/`tuple` of those
        The scales value if no errors were raised.

    Raises
    ------
    ValueError
        scales must be an int/float or a list/tuple of int/float.
    """
    if isinstance(scales, (int, float)):
        return [scales]
    elif len(scales) == 1 and isinstance(scales[0], (int, float)):
        return list(scales)
    elif len(scales) > 1:
        return check_scales(scales[0]) + check_scales(scales[1:])
    else:
        raise ValueError("scales must be an int/float or a list/tuple of "
                         "int/float")


def check_multi_scale_param(n_scales, types, param_name, param):
    r"""
    Checks a parameter that is supposed to be defined for different scales.
    The parameter must be among the provided `types` or a `list`/`tuple` of
    those.

    Parameters
    ----------
    n_scales : `int`
        The number of scales.
    types : `tuple`
        A `tuple` of types that are allowed for the param, e.g. `int` or `float`
    param_name : `str`
        The name of the parameter.
    param : `types` or a `list`/`tuple` of those
        The actual parameter value to check.

    Returns
    -------
    param : `list` of `types`
        The parameter value per scale in a `list`.

    Raises
    ------
    ValueError
        {param_name} must be in {types} or a list/tuple of {types} with the
        same length as the number of scales
    """
    error_msg = "{0} must be in {1} or a list/tuple of " \
                "{1} with the same length as the number " \
                "of scales".format(param_name, types)

    # Could be a single value - or we have an error
    if isinstance(param, types):
        return [param] * n_scales
    elif not isinstance(param, collections.Iterable):
        raise ValueError(error_msg)

    # Must be an iterable object
    len_param = len(param)
    isinstance_all_in_param = all(isinstance(p, types) for p in param)

    if len_param == 1 and isinstance_all_in_param:
        return list(param) * n_scales
    elif len_param == n_scales and isinstance_all_in_param:
        return list(param)
    else:
        raise ValueError(error_msg)


def check_features(features, n_scales, param_name):
    r"""
    Checks the feature type per level. Features must be a callable or a
    `list`/`tuple` of `callables`.

    Parameters
    ----------
    features : `callable` or `list` of `callables`
        The features to check.
    n_scales : `int`
        The number of pyramid levels.
    param_name : `str`
        The features parameter name.

    Returns
    -------
    feature_list : `list`
        The `list` of `callables` per pyramidal level.

    Raises
    ------
    ValueError
        {param_name} must be a callable or a list/tuple of callables with the
        same length as the number of scales
    """
    if callable(features):
        return [features] * n_scales
    elif len(features) == 1 and np.alltrue([callable(f) for f in features]):
        return list(features) * n_scales
    elif len(features) == n_scales and np.alltrue([callable(f)
                                                   for f in features]):
        return list(features)
    else:
        raise ValueError("{} must be a callable or a list/tuple of "
                         "callables with the same length as the number "
                         "of scales".format(param_name))


def check_scale_features(scale_features, features):
    r"""
    Checks the combination between features and scale_features.

    Parameters
    ----------
    scale_features : `bool`
        The features to check.
    features : `list` of `callables`
        The `list` of features per pyramidal level.

    Returns
    -------
    scale_features : `bool`
        ``False`` if different types of features are used at each pyramidal
        level.

    Raises
    ------
    Warning
        scale_features has been automatically set to False because different
        types of features are used at each level.
    """
    if all(f == features[0] for f in features):
        return scale_features
    else:
        warnings.warn('scale_features has been automatically set to False '
                      'because different types of features are used at each '
                      'level.')
        return False


def check_patch_shape(patch_shape, n_scales):
    r"""
    Checks the patch shape per level. Patch_shape must be a `list`/`tuple` of
    `int` or a `list`/`tuple` of `lit`/`tuple` of `int`/`float` with the same
    length as the number of scales.

    Parameters
    ----------
    patch_shape : `int/float/tuple` or `list/tuple` of those
        The patch_shape to check.
    n_scales : `int`
        The number of pyramid levels.

    Returns
    -------
    patch_shape : `list` of `tuple`
        The `list` of `tuple` patch shape per pyramidal level.

    Raises
    ------
    ValueError
        patch_shape must be a list/tuple of int or a list/tuple of lit/tuple of
        int/float with the same length as the number of scales
    """
    if len(patch_shape) == 2 and isinstance(patch_shape[0], int):
        return [patch_shape] * n_scales
    elif len(patch_shape) == 1:
        return check_patch_shape(patch_shape[0], 1)
    elif len(patch_shape) == n_scales:
        l1 = check_patch_shape(patch_shape[0], 1)
        l2 = check_patch_shape(patch_shape[1:], n_scales-1)
        return l1 + l2
    else:
        raise ValueError("patch_shape must be a list/tuple of int or a "
                         "list/tuple of lit/tuple of int/float with the "
                         "same length as the number of scales")


def check_max_components(max_components, n_scales, param_name):
    r"""
    Checks the provided maximum number of components per level. It must be
    ``None`` or `int` or `float` or a `list` of those containing ``1`` or
    {``n_scales``} elements.

    Parameters
    ----------
    max_components : `int`/`float`/``None`` or `list` of those
        The max_components to check.
    n_scales : `int`
        The number of pyramid levels.
    param_name : `str`
        The name of the parameter.

    Returns
    -------
    max_components : `list` of `int`/`float`
        The `list` of max_components per level.

    Raises
    ------
    ValueError
        {param_name} must be None or an int > 0 or a 0 <= float <= 1 or a list
        of those containing 1 or {n_scales} elements
    """
    str_error = ("{} must be None or an int > 0 or a 0 <= float <= 1 or "
                 "a list of those containing 1 or {} elements").format(
        param_name, n_scales)
    if not isinstance(max_components, (list, tuple)):
        max_components_list = [max_components] * n_scales
    elif len(max_components) == 1:
        max_components_list = [max_components[0]] * n_scales
    elif len(max_components) == n_scales:
        max_components_list = max_components
    else:
        raise ValueError(str_error)
    for comp in max_components_list:
        if comp is not None:
            if not isinstance(comp, int):
                if not isinstance(comp, float):
                    raise ValueError(str_error)
    return max_components_list


def check_max_iters(max_iters, n_scales):
    r"""
    Checks the maximum number of iterations per level. Max_iters must be a
    `int` or a `list`/`tuple` of `int`.

    Parameters
    ----------
    max_iters : `int` or `list` of `int`
        The max_iters to check.
    n_scales : `int`
        The number of pyramid levels.

    Returns
    -------
    max_iters : `list` of `int`
        The `list` of max_iters per pyramidal level.

    Raises
    ------
    ValueError
        max_iters can be integer, integer list containing 1 or {n_scales}
        elements or None
    """
    if type(max_iters) is int:
        max_iters = [np.round(max_iters/n_scales)
                     for _ in range(n_scales)]
    elif len(max_iters) == 1 and n_scales > 1:
        max_iters = [np.round(max_iters[0]/n_scales)
                     for _ in range(n_scales)]
    elif len(max_iters) != n_scales:
        raise ValueError('max_iters can be integer, integer list '
                         'containing 1 or {} elements or '
                         'None'.format(n_scales))
    return np.require(max_iters, dtype=np.int)


def check_sampling(sampling, n_scales):
    r"""
    Checks the sampling parameter per level. Sampling must be an `int` or
    `ndarray` or a `list` of those or ``None``.

    Parameters
    ----------
    sampling : `int`/`ndarray` or `list` of `int`/`ndarray`
        The sampling to check.
    n_scales : `int`
        The number of pyramid levels.

    Returns
    -------
    sampling : `list` of `int`/`ndarray`
        The `list` of sampling per pyramidal level.

    Raises
    ------
    ValueError
        A sampling list can only contain 1 element or {n_scales} elements
    ValueError
        sampling can be an integer or ndarray, a integer or ndarray list
        containing 1 or {n_scales} elements or None
    """
    if (isinstance(sampling, (list, tuple)) and
        np.alltrue([isinstance(s, (np.ndarray, np.int)) or sampling is None
                    for s in sampling])):
        if len(sampling) == 1:
            return sampling * n_scales
        elif len(sampling) == n_scales:
            return sampling
        else:
            raise ValueError('A sampling list can only '
                             'contain 1 element or {} '
                             'elements'.format(n_scales))
    elif isinstance(sampling, (np.ndarray, np.int)) or sampling is None:
        return [sampling] * n_scales
    else:
        raise ValueError('sampling can be an integer or ndarray, '
                         'a integer or ndarray list '
                         'containing 1 or {} elements or '
                         'None'.format(n_scales))


def set_models_components(models, n_components):
    r"""
    Checks and sets the provided number of components per level to the
    provided models.

    Parameters
    ----------
    models : `list` of :map:`PCAModel` or :map:`PCAInstanceModel`
        The list of PCA models.
    n_components : `int`/`float`/`None` or `list` of those
        The number of components to set as active.

    Raises
    ------
    ValueError
        n_components can be an integer or a float or None or a list containing
        1 or {n_scales} of those
    """
    if n_components is not None:
        n_scales = len(models)
        if type(n_components) is int or type(n_components) is float:
            for am in models:
                am.n_active_components = n_components
        elif len(n_components) == 1 and n_scales > 1:
            for am in models:
                am.n_active_components = n_components[0]
        elif len(n_components) == n_scales:
            for am, n in zip(models, n_components):
                am.n_active_components = n
        else:
            raise ValueError('n_components can be an integer or a float '
                             'or None or a list containing 1 or {} of '
                             'those'.format(n_scales))


def check_algorithm_cls(algorithm_cls, n_scales, base_algorithm_cls):
    r"""
    Checks the algorithm class per level. Algorithm must be a `class` or a
    `list` of `class`.

    Parameters
    ----------
    algorithm_cls : `class` or `list` of `class`
        The selected algorithm class(es) to check.
    n_scales : `int`
        The number of pyramid levels.
    base_algorithm_cls : `class`
        The `class` that must be a `superclass` of `algorithm_cls`.

    Returns
    -------
    algorithm_cls : `list` of `class`
        The `list` of `class` per pyramidal level.

    Raises
    ------
    ValueError
        algorithm_cls must be a subclass of {base_algorithm_cls} or a list/tuple
        of {base_algorithm_cls} subclasses with the same length as the number
        of scales {n_scales}
    """
    if (isinstance(algorithm_cls, partial) and
            base_algorithm_cls in algorithm_cls.func.mro()):
        return [algorithm_cls] * n_scales
    elif (isinstance(algorithm_cls, type) and
          base_algorithm_cls in algorithm_cls.mro()):
        return [algorithm_cls] * n_scales
    elif len(algorithm_cls) == 1:
        return check_algorithm_cls(algorithm_cls[0], n_scales,
                                   base_algorithm_cls)
    elif len(algorithm_cls) == n_scales:
        return [check_algorithm_cls(a, 1, base_algorithm_cls)[0]
                for a in algorithm_cls]
    else:
        raise ValueError("algorithm_cls must be a subclass of {} or a "
                         "list/tuple of {} subclasses with the same length "
                         "as the number of scales {}"
                         .format(base_algorithm_cls, base_algorithm_cls,
                                 n_scales))


def check_precision(precision):
    r"""
    Checks the provided numerical precision. It must be 'single' or 'double'.

    Parameters
    ----------
    precision : {``'single', 'double'``}
        The precision value to check.

    Returns
    -------
    precision : {``'single', 'double'``}
        The precision value if no errors were raised.

    Raises
    ------
    ValueError
        precision can be either ''single or ''double''
    """
    if precision != 'single' and precision != 'double':
        raise ValueError('precision can be either ''single or ''double''')
    return precision


def check_graph(graph, graph_types, param_name, n_scales):
    r"""
    Checks the provided graph per pyramidal level. The graph must be a
    subclass of `graph_types` or a `list` of those.

    Parameters
    ----------
    graph : `graph` or `list` of `graph` types
        The graph argument to check.
    graph_types : `graph` or `tuple` of `graphs`
        The `tuple` of allowed graph types.
    param_name : `str`
        The name of the graph parameter.
    n_scales : `int`
        The number of pyramidal levels.

    Returns
    -------
    graph : `list` of `graph` types
        The graph per scale in a `list`.

    Raises
    ------
    ValueError
        {param_name} must be a list of length equal to the number of scales.
    ValueError
        {param_name} must be a list of {graph_types_str}. {} given instead.
    """
    # check if the provided graph is a list
    if not isinstance(graph, list):
        graphs = [graph] * n_scales
    elif len(graph) == 1:
        graphs = graph * n_scales
    elif len(graph) == n_scales:
        graphs = graph
    else:
        raise ValueError('{} must be a list of length equal to the number of '
                         'scales.'.format(param_name))
    # check if the provided graph_types is a list
    if not isinstance(graph_types, list):
        graph_types = [graph_types]

    # check each member of the graphs list
    for g in graphs:
        if g is not None:
            if type(g) not in graph_types:
                graph_types_str = ' or '.join(gt.__name__ for gt in graph_types)
                raise ValueError('{} must be a list of {}. {} given '
                                 'instead.'.format(param_name, graph_types_str,
                                                   type(g).__name__))
    return graphs
