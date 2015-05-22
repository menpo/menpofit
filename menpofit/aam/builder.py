from __future__ import division
from copy import deepcopy
from menpo.model import PCAModel
from menpo.shape import mean_pointcloud
from menpo.feature import no_op
from menpo.visualize import print_dynamic
from menpofit import checks
from menpofit.builder import (
    normalization_wrt_reference_shape, compute_features, scale_images,
    warp_images, extract_patches, build_shape_model, align_shapes,
    build_reference_frame, build_patch_reference_frame, densify_shapes)
from menpofit.transform import (
    DifferentiablePiecewiseAffine, DifferentiableThinPlateSplines)


# TODO: fix features checker
# TODO: implement checker for conflict between features and scale_features
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
        # check parameters
        checks.check_diagonal(diagonal)
        scales, n_levels = checks.check_scales(scales)
        features = checks.check_features(features, n_levels)
        max_shape_components = checks.check_max_components(
            max_shape_components, len(scales), 'max_shape_components')
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

    def build(self, images, group=None, label=None, verbose=False):
        r"""
        Builds an Active Appearance Model from a list of landmarked images.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images from which to build the AAM.

        group : `string`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`, optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        verbose : `boolean`, optional
            Flag that controls information and progress printing.

        Returns
        -------
        aam : :map:`AAM`
            The AAM object. Shape and appearance models are stored from
            lowest to highest level
        """
        # normalize images and compute reference shape
        reference_shape, images = normalization_wrt_reference_shape(
            images, group, label, self.diagonal, verbose=verbose)

        # build models at each scale
        if verbose:
            print_dynamic('- Building models\n')
        shape_models = []
        appearance_models = []
        # for each pyramid level (high --> low)
        for j, s in enumerate(self.scales):
            if verbose:
                if len(self.scales) > 1:
                    level_str = '  - Level {}: '.format(j)
                else:
                    level_str = '  - '

            # obtain image representation
            if j == 0:
                # compute features at highest level
                feature_images = compute_features(images, self.features,
                                                  level_str=level_str,
                                                  verbose=verbose)
                level_images = feature_images
            elif self.scale_features:
                # scale features at other levels
                level_images = scale_images(feature_images, s,
                                            level_str=level_str,
                                            verbose=verbose)
            else:
                # scale images and compute features at other levels
                scaled_images = scale_images(images, s, level_str=level_str,
                                             verbose=verbose)
                level_images = compute_features(scaled_images, self.features,
                                                level_str=level_str,
                                                verbose=verbose)

            # extract potentially rescaled shapes
            level_shapes = [i.landmarks[group][label]
                            for i in level_images]

            # obtain shape representation
            if j == 0 or self.scale_shapes:
                # obtain shape model
                if verbose:
                    print_dynamic('{}Building shape model'.format(level_str))
                shape_model = self._build_shape_model(
                    level_shapes, self.max_shape_components[j], j)
                # add shape model to the list
                shape_models.append(shape_model)
            else:
                # copy precious shape model and add it to the list
                shape_models.append(deepcopy(shape_model))

            # obtain warped images
            warped_images = self._warp_images(level_images, level_shapes,
                                              shape_model.mean(), j,
                                              level_str, verbose)

            # obtain appearance model
            if verbose:
                print_dynamic('{}Building appearance model'.format(level_str))
            appearance_model = PCAModel(warped_images)
            # trim appearance model if required
            if self.max_appearance_components is not None:
                appearance_model.trim_components(
                    self.max_appearance_components[j])
            # add appearance model to the list
            appearance_models.append(appearance_model)

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        appearance_models.reverse()
        self.scales.reverse()

        aam = self._build_aam(shape_models, appearance_models, reference_shape)

        return aam

    @classmethod
    def _build_shape_model(cls, shapes, max_components, level):
        return build_shape_model(shapes, max_components=max_components)

    def _warp_images(self, images, shapes, reference_shape, level, level_str,
                     verbose):
        self.reference_frame = build_reference_frame(reference_shape)
        return warp_images(images, shapes, self.reference_frame,
                           self.transform, level_str=level_str,
                           verbose=verbose)

    def _build_aam(self, shape_models, appearance_models, reference_shape):
        return AAM(shape_models, appearance_models, reference_shape,
                   self.transform, self.features, self.scales,
                   self.scale_shapes, self.scale_features)


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
        max_shape_components = checks.check_max_components(
            max_shape_components, len(scales), 'max_shape_components')
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
        self.reference_frame = build_patch_reference_frame(
            reference_shape, patch_shape=self.patch_shape[level])
        return warp_images(images, shapes, self.reference_frame,
                           self.transform, level_str=level_str,
                           verbose=verbose)

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
        max_shape_components = checks.check_max_components(
            max_shape_components, len(scales), 'max_shape_components')
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
        max_shape_components = checks.check_max_components(
            max_shape_components, len(scales), 'max_shape_components')
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
        max_shape_components = checks.check_max_components(
            max_shape_components, len(scales), 'max_shape_components')
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


