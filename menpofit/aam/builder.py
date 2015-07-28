from __future__ import division
import numpy as np
from copy import deepcopy
from menpo.model import PCAModel
from menpo.shape import mean_pointcloud
from menpo.feature import no_op
from menpo.visualize import print_dynamic
from menpofit import checks
from menpofit.builder import (
    normalization_wrt_reference_shape, compute_features, scale_images,
    warp_images, extract_patches, build_shape_model, align_shapes,
    build_reference_frame, build_patch_reference_frame, densify_shapes,
    rescale_images_to_reference_shape)
from menpofit.transform import (
    DifferentiablePiecewiseAffine, DifferentiableThinPlateSplines)


# TODO: document me!
class AAMBuilder(object):
    r"""
    Class that builds Active Appearance Models.

    Parameters
    ----------
    features : `callable` or ``[callable]``, optional
        If list of length ``n_levels``, feature extraction is performed at
        each level after downscaling of the image.
        The first element of the list specifies the features to be extracted at
        the lowest pyramidal level and so on.

        If ``callable`` the specified feature will be applied to the original
        image and pyramid generation will be performed on top of the feature
        image. Also see the `pyramid_on_features` property.

        Note that from our experience, this approach of extracting features
        once and then creating a pyramid on top tends to lead to better
        performing AAMs.
    transform : :map:`PureAlignmentTransform`, optional
        The :map:`PureAlignmentTransform` that will be
        used to warp the images.
    trilist : ``(t, 3)`` `ndarray`, optional
        Triangle list that will be used to build the reference frame. If
        ``None``, defaults to performing Delaunay triangulation on the points.
    diagonal : `int` >= ``20``, optional
        During building an AAM, all images are rescaled to ensure that the
        scale of their landmarks matches the scale of the mean shape.

        If `int`, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the diagonal value.

        If ``None``, the mean shape is not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).
    scales : `int` or float` or list of those, optional
    scale_shapes : `boolean`, optional
    scale_features : `boolean`, optional
    max_shape_components : ``None`` or `int` > 0 or ``0`` <= `float` <= ``1`` or list of those, optional
        If list of length ``n_levels``, then a number of shape components is
        defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        shape components will be used for all levels.

        Per level:
            If `int`, it specifies the exact number of components to be
            retained.

            If `float`, it specifies the percentage of variance to be retained.

            If ``None``, all the available components are kept
            (100% of variance).
    max_appearance_components : ``None`` or `int` > 0 or ``0`` <= `float` <= ``1`` or list of those, optional
        If list of length ``n_levels``, then a number of appearance components
        is defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        appearance components will be used for all levels.

        Per level:
            If `int`, it specifies the exact number of components to be
            retained.

            If `float`, it specifies the percentage of variance to be retained.

            If ``None``, all the available components are kept
            (100% of variance).

    Returns
    -------
    aam : :map:`AAMBuilder`
        The AAM Builder object

    Raises
    -------
    ValueError
        ``diagonal`` must be >= ``20``.
    ValueError
        ``scales`` must be `int` or `float` or list of those.
    ValueError
        ``features`` must be a `function` or a list of those
        containing ``1`` or ``len(scales)`` elements
    ValueError
        ``max_shape_components`` must be ``None`` or an `int` > 0 or
        a ``0`` <= `float` <= ``1`` or a list of those containing 1 or
        ``len(scales)`` elements
    ValueError
        ``max_appearance_components`` must be ``None`` or an `int` > ``0`` or a
        ``0`` <= `float` <= ``1`` or a list of those containing 1 or
        ``len(scales)`` elements
    """
    def __init__(self, features=no_op, transform=DifferentiablePiecewiseAffine,
                 trilist=None, diagonal=None, scales=(1, 0.5),
                 scale_shapes=False, scale_features=True,
                 max_shape_components=None, max_appearance_components=None):
        pass


# TODO: document me!
class PatchAAMBuilder(AAMBuilder):
    r"""
    Class that builds Patch based Active Appearance Models.

    Parameters
    ----------
    patch_shape: (`int`, `int`) or list or list of (`int`, `int`)

    features : `callable` or ``[callable]``, optional
        If list of length ``n_levels``, feature extraction is performed at
        each level after downscaling of the image.
        The first element of the list specifies the features to be extracted at
        the lowest pyramidal level and so on.

        If ``callable`` the specified feature will be applied to the original
        image and pyramid generation will be performed on top of the feature
        image. Also see the `pyramid_on_features` property.

        Note that from our experience, this approach of extracting features
        once and then creating a pyramid on top tends to lead to better
        performing AAMs.

    diagonal : `int` >= ``20``, optional
        During building an AAM, all images are rescaled to ensure that the
        scale of their landmarks matches the scale of the mean shape.

        If `int`, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the diagonal value.

        If ``None``, the mean shape is not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).

    scales : `int` or float` or list of those, optional

    scale_shapes : `boolean`, optional

    scale_features : `boolean`, optional

    max_shape_components : ``None`` or `int` > 0 or ``0`` <= `float` <= ``1`` or list of those, optional
        If list of length ``n_levels``, then a number of shape components is
        defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        shape components will be used for all levels.

        Per level:
            If `int`, it specifies the exact number of components to be
            retained.

            If `float`, it specifies the percentage of variance to be retained.

            If ``None``, all the available components are kept
            (100% of variance).

    max_appearance_components : ``None`` or `int` > 0 or ``0`` <= `float` <= ``1`` or list of those, optional
        If list of length ``n_levels``, then a number of appearance components
        is defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        appearance components will be used for all levels.

        Per level:
            If `int`, it specifies the exact number of components to be
            retained.

            If `float`, it specifies the percentage of variance to be retained.

            If ``None``, all the available components are kept
            (100% of variance).

    Returns
    -------
    aam : :map:`AAMBuilder`
        The AAM Builder object

    Raises
    -------
    ValueError
        ``diagonal`` must be >= ``20``.
    ValueError
        ``scales`` must be `int` or `float` or list of those.
    ValueError
        ``patch_shape`` must be (`int`, `int`) or list of (`int`, `int`)
        containing 1 or `len(scales)` elements.
    ValueError
        ``features`` must be a `function` or a list of those
        containing ``1`` or ``len(scales)`` elements
    ValueError
        ``max_shape_components`` must be ``None`` or an `int` > 0 or
        a ``0`` <= `float` <= ``1`` or a list of those containing 1 or
        ``len(scales)`` elements
    ValueError
        ``max_appearance_components`` must be ``None`` or an `int` > ``0`` or a
        ``0`` <= `float` <= ``1`` or a list of those containing 1 or
        ``len(scales)`` elements
    """
    def __init__(self, patch_shape=(17, 17), features=no_op,
                 diagonal=None, scales=(1, .5), scale_shapes=True,
                 scale_features=True, max_shape_components=None,
                 max_appearance_components=None):
        # check parameters
        checks.check_diagonal(diagonal)
        scales, n_levels = checks.check_scales(scales)
        patch_shape = checks.check_patch_shape(patch_shape, n_levels)
        features = checks.check_features(features, n_levels)
        scale_features = checks.check_scale_features(scale_features, features)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_levels, 'max_shape_components')
        max_appearance_components = checks.check_max_components(
            max_appearance_components, n_levels, 'max_appearance_components')
        # set parameters
        self.patch_shape = patch_shape
        self.features = features
        self.transform = DifferentiableThinPlateSplines
        self.diagonal = diagonal
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components

    def _warp_images(self, images, shapes, reference_shape, level, level_str,
                     verbose):
        reference_frame = build_patch_reference_frame(
            reference_shape, patch_shape=self.patch_shape[level])
        return warp_images(images, shapes, reference_frame, self.transform,
                           level_str=level_str, verbose=verbose)

    def _build_aam(self, shape_models, appearance_models, reference_shape):
        return PatchAAM(shape_models, appearance_models, reference_shape,
                        self.patch_shape, self.features, self.scales,
                        self.scale_shapes, self.scale_features)


# TODO: document me!
class LinearAAMBuilder(AAMBuilder):
    r"""
    Class that builds Linear Active Appearance Models.

    Parameters
    ----------
    features : `callable` or ``[callable]``, optional
        If list of length ``n_levels``, feature extraction is performed at
        each level after downscaling of the image.
        The first element of the list specifies the features to be extracted at
        the lowest pyramidal level and so on.

        If ``callable`` the specified feature will be applied to the original
        image and pyramid generation will be performed on top of the feature
        image. Also see the `pyramid_on_features` property.

        Note that from our experience, this approach of extracting features
        once and then creating a pyramid on top tends to lead to better
        performing AAMs.

    transform : :map:`PureAlignmentTransform`, optional
        The :map:`PureAlignmentTransform` that will be
        used to warp the images.

    trilist : ``(t, 3)`` `ndarray`, optional
        Triangle list that will be used to build the reference frame. If
        ``None``, defaults to performing Delaunay triangulation on the points.

    diagonal : `int` >= ``20``, optional
        During building an AAM, all images are rescaled to ensure that the
        scale of their landmarks matches the scale of the mean shape.

        If `int`, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the diagonal value.

        If ``None``, the mean shape is not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).

    scales : `int` or float` or list of those, optional

    scale_shapes : `boolean`, optional

    scale_features : `boolean`, optional

    max_shape_components : ``None`` or `int` > 0 or ``0`` <= `float` <= ``1`` or list of those, optional
        If list of length ``n_levels``, then a number of shape components is
        defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        shape components will be used for all levels.

        Per level:
            If `int`, it specifies the exact number of components to be
            retained.

            If `float`, it specifies the percentage of variance to be retained.

            If ``None``, all the available components are kept
            (100% of variance).

    max_appearance_components : ``None`` or `int` > 0 or ``0`` <= `float` <= ``1`` or list of those, optional
        If list of length ``n_levels``, then a number of appearance components
        is defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        appearance components will be used for all levels.

        Per level:
            If `int`, it specifies the exact number of components to be
            retained.

            If `float`, it specifies the percentage of variance to be retained.

            If ``None``, all the available components are kept
            (100% of variance).

    Returns
    -------
    aam : :map:`AAMBuilder`
        The AAM Builder object

    Raises
    -------
    ValueError
        ``diagonal`` must be >= ``20``.
    ValueError
        ``scales`` must be `int` or `float` or list of those.
    ValueError
        ``features`` must be a `function` or a list of those
        containing ``1`` or ``len(scales)`` elements
    ValueError
        ``max_shape_components`` must be ``None`` or an `int` > 0 or
        a ``0`` <= `float` <= ``1`` or a list of those containing 1 or
        ``len(scales)`` elements
    ValueError
        ``max_appearance_components`` must be ``None`` or an `int` > ``0`` or a
        ``0`` <= `float` <= ``1`` or a list of those containing 1 or
        ``len(scales)`` elements
    """
    def __init__(self, features=no_op, transform=DifferentiablePiecewiseAffine,
                 trilist=None, diagonal=None, scales=(1, .5),
                 scale_shapes=False, scale_features=True,
                 max_shape_components=None, max_appearance_components=None):
        # check parameters
        checks.check_diagonal(diagonal)
        scales, n_levels = checks.check_scales(scales)
        features = checks.check_features(features, n_levels)
        scale_features = checks.check_scale_features(scale_features, features)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_levels, 'max_shape_components')
        max_appearance_components = checks.check_max_components(
            max_appearance_components, n_levels, 'max_appearance_components')
        # set parameters
        self.features = features
        self.transform = transform
        self.trilist = trilist
        self.diagonal = diagonal
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components

    def _build_shape_model(self, shapes, max_components, level):
        mean_aligned_shape = mean_pointcloud(align_shapes(shapes))
        self.n_landmarks = mean_aligned_shape.n_points
        self.reference_frame = build_reference_frame(mean_aligned_shape)
        dense_shapes = densify_shapes(shapes, self.reference_frame,
                                      self.transform)
        # build dense shape model
        shape_model = build_shape_model(
            dense_shapes, max_components=max_components)
        return shape_model

    def _warp_images(self, images, shapes, reference_shape, level, level_str,
                     verbose):
        return warp_images(images, shapes, self.reference_frame,
                           self.transform, level_str=level_str,
                           verbose=verbose)

    def _build_aam(self, shape_models, appearance_models, reference_shape):
        return LinearAAM(shape_models, appearance_models,
                         reference_shape, self.transform,
                         self.features, self.scales,
                         self.scale_shapes, self.scale_features,
                         self.n_landmarks)


# TODO: document me!
class LinearPatchAAMBuilder(AAMBuilder):
    r"""
    Class that builds Linear Patch based Active Appearance Models.

    Parameters
    ----------
    patch_shape: (`int`, `int`) or list or list of (`int`, `int`)

    features : `callable` or ``[callable]``, optional
        If list of length ``n_levels``, feature extraction is performed at
        each level after downscaling of the image.
        The first element of the list specifies the features to be extracted at
        the lowest pyramidal level and so on.

        If ``callable`` the specified feature will be applied to the original
        image and pyramid generation will be performed on top of the feature
        image. Also see the `pyramid_on_features` property.

        Note that from our experience, this approach of extracting features
        once and then creating a pyramid on top tends to lead to better
        performing AAMs.

    diagonal : `int` >= ``20``, optional
        During building an AAM, all images are rescaled to ensure that the
        scale of their landmarks matches the scale of the mean shape.

        If `int`, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the diagonal value.

        If ``None``, the mean shape is not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).

    scales : `int` or float` or list of those, optional

    scale_shapes : `boolean`, optional

    scale_features : `boolean`, optional

    max_shape_components : ``None`` or `int` > 0 or ``0`` <= `float` <= ``1`` or list of those, optional
        If list of length ``n_levels``, then a number of shape components is
        defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        shape components will be used for all levels.

        Per level:
            If `int`, it specifies the exact number of components to be
            retained.

            If `float`, it specifies the percentage of variance to be retained.

            If ``None``, all the available components are kept
            (100% of variance).

    max_appearance_components : ``None`` or `int` > 0 or ``0`` <= `float` <= ``1`` or list of those, optional
        If list of length ``n_levels``, then a number of appearance components
        is defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        appearance components will be used for all levels.

        Per level:
            If `int`, it specifies the exact number of components to be
            retained.

            If `float`, it specifies the percentage of variance to be retained.

            If ``None``, all the available components are kept
            (100% of variance).

    Returns
    -------
    aam : :map:`AAMBuilder`
        The AAM Builder object

    Raises
    -------
    ValueError
        ``diagonal`` must be >= ``20``.
    ValueError
        ``scales`` must be `int` or `float` or list of those.
    ValueError
        ``patch_shape`` must be (`int`, `int`) or list of (`int`, `int`)
        containing 1 or `len(scales)` elements.
    ValueError
        ``features`` must be a `function` or a list of those
        containing ``1`` or ``len(scales)`` elements
    ValueError
        ``max_shape_components`` must be ``None`` or an `int` > 0 or
        a ``0`` <= `float` <= ``1`` or a list of those containing 1 or
        ``len(scales)`` elements
    ValueError
        ``max_appearance_components`` must be ``None`` or an `int` > ``0`` or a
        ``0`` <= `float` <= ``1`` or a list of those containing 1 or
        ``len(scales)`` elements
    """
    def __init__(self, patch_shape=(17, 17), features=no_op,
                 diagonal=None, scales=(1, .5), scale_shapes=False,
                 scale_features=True, max_shape_components=None,
                 max_appearance_components=None):
        # check parameters
        checks.check_diagonal(diagonal)
        scales, n_levels = checks.check_scales(scales)
        patch_shape = checks.check_patch_shape(patch_shape, n_levels)
        features = checks.check_features(features, n_levels)
        scale_features = checks.check_scale_features(scale_features, features)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_levels, 'max_shape_components')
        max_appearance_components = checks.check_max_components(
            max_appearance_components, n_levels, 'max_appearance_components')
        # set parameters
        self.patch_shape = patch_shape
        self.features = features
        self.transform = DifferentiableThinPlateSplines
        self.diagonal = diagonal
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components

    def _build_shape_model(self, shapes, max_components, level):
        mean_aligned_shape = mean_pointcloud(align_shapes(shapes))
        self.n_landmarks = mean_aligned_shape.n_points
        self.reference_frame = build_patch_reference_frame(
            mean_aligned_shape, patch_shape=self.patch_shape[level])
        dense_shapes = densify_shapes(shapes, self.reference_frame,
                                      self.transform)
        # build dense shape model
        shape_model = build_shape_model(dense_shapes,
                                        max_components=max_components)
        return shape_model

    def _warp_images(self, images, shapes, reference_shape, level, level_str,
                     verbose):
        return warp_images(images, shapes, self.reference_frame,
                           self.transform, level_str=level_str,
                           verbose=verbose)

    def _build_aam(self, shape_models, appearance_models, reference_shape):
        return LinearPatchAAM(shape_models, appearance_models,
                              reference_shape, self.patch_shape,
                              self.features, self.scales, self.scale_shapes,
                              self.scale_features, self.n_landmarks)


# TODO: document me!
# TODO: implement offsets support?
class PartsAAMBuilder(AAMBuilder):
    r"""
    Class that builds Parts based Active Appearance Models.

    Parameters
    ----------
    patch_shape: (`int`, `int`) or list or list of (`int`, `int`)

    features : `callable` or ``[callable]``, optional
        If list of length ``n_levels``, feature extraction is performed at
        each level after downscaling of the image.
        The first element of the list specifies the features to be extracted at
        the lowest pyramidal level and so on.

        If ``callable`` the specified feature will be applied to the original
        image and pyramid generation will be performed on top of the feature
        image. Also see the `pyramid_on_features` property.

        Note that from our experience, this approach of extracting features
        once and then creating a pyramid on top tends to lead to better
        performing AAMs.

    normalize_parts : `callable`, optional

    diagonal : `int` >= ``20``, optional
        During building an AAM, all images are rescaled to ensure that the
        scale of their landmarks matches the scale of the mean shape.

        If `int`, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the diagonal value.

        If ``None``, the mean shape is not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).

    scales : `int` or float` or list of those, optional

    scale_shapes : `boolean`, optional

    scale_features : `boolean`, optional

    max_shape_components : ``None`` or `int` > 0 or ``0`` <= `float` <= ``1`` or list of those, optional
        If list of length ``n_levels``, then a number of shape components is
        defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        shape components will be used for all levels.

        Per level:
            If `int`, it specifies the exact number of components to be
            retained.

            If `float`, it specifies the percentage of variance to be retained.

            If ``None``, all the available components are kept
            (100% of variance).

    max_appearance_components : ``None`` or `int` > 0 or ``0`` <= `float` <= ``1`` or list of those, optional
        If list of length ``n_levels``, then a number of appearance components
        is defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        appearance components will be used for all levels.

        Per level:
            If `int`, it specifies the exact number of components to be
            retained.

            If `float`, it specifies the percentage of variance to be retained.

            If ``None``, all the available components are kept
            (100% of variance).

    Returns
    -------
    aam : :map:`AAMBuilder`
        The AAM Builder object

    Raises
    -------
    ValueError
        ``diagonal`` must be >= ``20``.
    ValueError
        ``scales`` must be `int` or `float` or list of those.
    ValueError
        ``patch_shape`` must be (`int`, `int`) or list of (`int`, `int`)
        containing 1 or `len(scales)` elements.
    ValueError
        ``features`` must be a `function` or a list of those
        containing ``1`` or ``len(scales)`` elements
    ValueError
        ``max_shape_components`` must be ``None`` or an `int` > 0 or
        a ``0`` <= `float` <= ``1`` or a list of those containing 1 or
        ``len(scales)`` elements
    ValueError
        ``max_appearance_components`` must be ``None`` or an `int` > ``0`` or a
        ``0`` <= `float` <= ``1`` or a list of those containing 1 or
        ``len(scales)`` elements
    """
    def __init__(self, patch_shape=(17, 17), features=no_op,
                 normalize_parts=no_op, diagonal=None, scales=(1, .5),
                 scale_shapes=False, scale_features=True,
                 max_shape_components=None, max_appearance_components=None):
        # check parameters
        checks.check_diagonal(diagonal)
        scales, n_levels = checks.check_scales(scales)
        patch_shape = checks.check_patch_shape(patch_shape, n_levels)
        features = checks.check_features(features, n_levels)
        scale_features = checks.check_scale_features(scale_features, features)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_levels, 'max_shape_components')
        max_appearance_components = checks.check_max_components(
            max_appearance_components, n_levels, 'max_appearance_components')
        # set parameters
        self.patch_shape = patch_shape
        self.features = features
        self.normalize_parts = normalize_parts
        self.diagonal = diagonal
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components

    def _warp_images(self, images, shapes, reference_shape, level, level_str,
                     verbose):
        return extract_patches(images, shapes, self.patch_shape[level],
                               normalize_function=self.normalize_parts,
                               level_str=level_str, verbose=verbose)

    def _build_aam(self, shape_models, appearance_models, reference_shape):
        return PartsAAM(shape_models, appearance_models, reference_shape,
                        self.patch_shape, self.features,
                        self.normalize_parts, self.scales,
                        self.scale_shapes, self.scale_features)


from .base import AAM, PatchAAM, LinearAAM, LinearPatchAAM, PartsAAM
