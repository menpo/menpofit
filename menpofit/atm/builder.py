from __future__ import division
from copy import deepcopy
from menpo.transform import Scale
from menpofit.transform import (
    DifferentiablePiecewiseAffine, DifferentiableThinPlateSplines)
from menpo.shape import mean_pointcloud
from menpo.image import Image
from menpo.feature import no_op
from menpo.visualize import print_dynamic
from menpofit import checks
from menpofit.aam.builder import (
    align_shapes, densify_shapes,
    build_reference_frame, build_patch_reference_frame)
from menpofit.builder import build_shape_model, compute_reference_shape


# TODO: document me!
class ATMBuilder(object):
    r"""
    Class that builds Active Template Models.

    Parameters
    ----------
    features : `callable` or ``[callable]``, optional
        If list of length ``n_scales``, feature extraction is performed at
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
        If list of length ``n_scales``, then a number of shape components is
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

    Returns
    -------
    atm : :map:`ATMBuilder`
        The ATM Builder object

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
                 max_shape_components=None):
        # check parameters
        checks.check_diagonal(diagonal)
        scales = checks.check_scales(scales)
        features = checks.check_features(features, len(scales))
        scale_features = checks.check_scale_features(scale_features, features)
        max_shape_components = checks.check_max_components(
            max_shape_components, len(scales), 'max_shape_components')
        # set parameters
        self.features = features
        self.transform = transform
        self.trilist = trilist
        self.diagonal = diagonal
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components

    def build(self, shapes, template, group=None, verbose=False):
        r"""
        Builds a Multilevel Active Template Model given a list of shapes and a
        template image.

        Parameters
        ----------
        shapes : list of :map:`PointCloud`
            The set of shapes from which to build the shape model of the ATM.
        template : :map:`Image` or subclass
            The image to be used as template.
        group : `str`, optional
            The key of the landmark set of the template that should be used. If
            ``None``, and if there is only one set of landmarks, this set will
            be used.
        verbose : `bool`, optional
            Flag that controls information and progress printing.

        Returns
        -------
        atm : :map:`ATM`
            The ATM object. Shape and appearance models are stored from lowest
            to highest level.
        """
        # compute reference_shape
        reference_shape = compute_reference_shape(shapes, self.diagonal,
                                                  verbose=verbose)

        # normalize the template size using the reference_shape scaling
        template = template.rescale_to_pointcloud(
            reference_shape, group=group)

        # build models at each scale
        if verbose:
            print_dynamic('- Building models\n')
        shape_models = []
        warped_templates = []
        # for each pyramid level (high --> low)
        for j, s in enumerate(self.scales):
            if verbose:
                if len(self.scales) > 1:
                    level_str = '  - Level {}: '.format(j)
                else:
                    level_str = '  - '

            # obtain shape representation
            if j == 0 or self.scale_shapes:
                if j == 0:
                    level_shapes = shapes
                    level_reference_shape = reference_shape
                else:
                    scale_transform = Scale(scale_factor=s, n_dims=2)
                    level_shapes = [scale_transform.apply(s) for s in shapes]
                    level_reference_shape = scale_transform.apply(
                        reference_shape)
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

            if verbose:
                print_dynamic('{}Building template model'.format(level_str))
            # obtain template representation
            if j == 0:
                # compute features at highest level
                feature_template = self.features[j](template)
                level_template = feature_template
            elif self.scale_features:
                # scale features at other levels
                level_template = feature_template.rescale(s)
            else:
                # scale template and compute features at other levels
                scaled_template = template.rescale(s)
                level_template = self.features[j](scaled_template)

            # extract potentially rescaled template shape
            level_template_shape = level_template.landmarks[group].lms

            # obtain warped template
            warped_template = self._warp_template(level_template,
                                                  level_template_shape,
                                                  level_reference_shape, j)
            # add warped template to the list
            warped_templates.append(warped_template)

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))

        # reverse the list of shape and warped templates so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        warped_templates.reverse()
        self.scales.reverse()

        return self._build_atm(shape_models, warped_templates, reference_shape)

    @classmethod
    def _build_shape_model(cls, shapes, max_components, level):
        return build_shape_model(shapes, max_components=max_components)

    def _warp_template(self, template, template_shape, reference_shape, level):
        # build reference frame
        reference_frame = build_reference_frame(reference_shape)
        # compute transforms
        t = self.transform(reference_frame.landmarks['source'].lms,
                           template_shape)
        # warp template
        warped_template = template.warp_to_mask(reference_frame.mask, t)
        # attach landmarks
        warped_template.landmarks['source'] = reference_frame.landmarks[
            'source']
        return warped_template

    def _build_atm(self, shape_models, warped_templates, reference_shape):
        return ATM(shape_models, warped_templates, reference_shape,
                   self.transform, self.features, self.scales,
                   self.scale_shapes, self.scale_features)


class PatchATMBuilder(ATMBuilder):
    r"""
    Class that builds Multilevel Patch-Based Active Template Models.

    Parameters
    ----------
    patch_shape: (`int`, `int`) or list or list of (`int`, `int`)

    features : `callable` or ``[callable]``, optional
        If list of length ``n_scales``, feature extraction is performed at
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
        If list of length ``n_scales``, then a number of shape components is
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

    Returns
    -------
    atm : :map:`ATMBuilder`
        The ATM Builder object

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
                 scale_features=True, max_shape_components=None):
        # check parameters
        checks.check_diagonal(diagonal)
        scales = checks.check_scales(scales)
        patch_shape = checks.check_patch_shape(patch_shape, len(scales))
        features = checks.check_features(features, len(scales))
        scale_features = checks.check_scale_features(scale_features, features)
        max_shape_components = checks.check_max_components(
            max_shape_components, len(scales), 'max_shape_components')
        # set parameters
        self.patch_shape = patch_shape
        self.features = features
        self.transform = DifferentiableThinPlateSplines
        self.diagonal = diagonal
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components

    def _warp_template(self, template, template_shape, reference_shape, level):
        # build reference frame
        reference_frame = build_patch_reference_frame(
            reference_shape, patch_shape=self.patch_shape[level])
        # compute transforms
        t = self.transform(reference_frame.landmarks['source'].lms,
                           template_shape)
        # warp template
        warped_template = template.warp_to_mask(reference_frame.mask, t)
        # attach landmarks
        warped_template.landmarks['source'] = reference_frame.landmarks[
            'source']
        return warped_template

    def _build_atm(self, shape_models, warped_templates, reference_shape):
        return PatchATM(shape_models, warped_templates, reference_shape,
                        self.patch_shape, self.features, self.scales,
                        self.scale_shapes, self.scale_features)


# TODO: document me!
class LinearATMBuilder(ATMBuilder):
    r"""
    Class that builds Linear Active Template Models.

    Parameters
    ----------
    features : `callable` or ``[callable]``, optional
        If list of length ``n_scales``, feature extraction is performed at
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
        If list of length ``n_scales``, then a number of shape components is
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

    Returns
    -------
    atm : :map:`ATMBuilder`
        The ATM Builder object

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
                 max_shape_components=None):
        # check parameters
        checks.check_diagonal(diagonal)
        scales = checks.check_scales(scales)
        features = checks.check_features(features, len(scales))
        scale_features = checks.check_scale_features(scale_features, features)
        max_shape_components = checks.check_max_components(
            max_shape_components, len(scales), 'max_shape_components')
        # set parameters
        self.features = features
        self.transform = transform
        self.trilist = trilist
        self.diagonal = diagonal
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components

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

    def _warp_template(self, template, template_shape, reference_shape, level):
        # compute transforms
        t = self.transform(self.reference_frame.landmarks['source'].lms,
                           template_shape)
        # warp template
        warped_template = template.warp_to_mask(self.reference_frame.mask, t)
        # attach landmarks
        warped_template.landmarks['source'] = self.reference_frame.landmarks[
            'source']
        return warped_template

    def _build_atm(self, shape_models, warped_templates, reference_shape):
        return LinearATM(shape_models, warped_templates, reference_shape,
                         self.transform, self.features, self.scales,
                         self.scale_shapes, self.scale_features,
                         self.n_landmarks)


# TODO: document me!
class LinearPatchATMBuilder(LinearATMBuilder):
    r"""
    Class that builds Linear Patch based Active Template Models.

    Parameters
    ----------
    patch_shape: (`int`, `int`) or list or list of (`int`, `int`)

    features : `callable` or ``[callable]``, optional
        If list of length ``n_scales``, feature extraction is performed at
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
        If list of length ``n_scales``, then a number of shape components is
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

    Returns
    -------
    atm : :map:`ATMBuilder`
        The ATM Builder object

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
                 scale_features=True, max_shape_components=None):
        # check parameters
        checks.check_diagonal(diagonal)
        scales = checks.check_scales(scales)
        patch_shape = checks.check_patch_shape(patch_shape, len(scales))
        features = checks.check_features(features, len(scales))
        scale_features = checks.check_scale_features(scale_features, features)
        max_shape_components = checks.check_max_components(
            max_shape_components, len(scales), 'max_shape_components')
        # set parameters
        self.patch_shape = patch_shape
        self.features = features
        self.transform = DifferentiableThinPlateSplines
        self.diagonal = diagonal
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components

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

    def _build_atm(self, shape_models, warped_templates, reference_shape):
        return LinearPatchATM(shape_models, warped_templates,
                              reference_shape, self.patch_shape,
                              self.features, self.scales, self.scale_shapes,
                              self.scale_features, self.n_landmarks)


# TODO: document me!
# TODO: implement offsets support?
class PartsATMBuilder(ATMBuilder):
    r"""
    Class that builds Parts based Active Template Models.

    Parameters
    ----------
    patch_shape: (`int`, `int`) or list or list of (`int`, `int`)

    features : `callable` or ``[callable]``, optional
        If list of length ``n_scales``, feature extraction is performed at
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
        If list of length ``n_scales``, then a number of shape components is
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

    Returns
    -------
    atm : :map:`ATMBuilder`
        The ATM Builder object

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
                 max_shape_components=None):
        # check parameters
        checks.check_diagonal(diagonal)
        scales = checks.check_scales(scales)
        patch_shape = checks.check_patch_shape(patch_shape, len(scales))
        features = checks.check_features(features, len(scales))
        scale_features = checks.check_scale_features(scale_features, features)
        max_shape_components = checks.check_max_components(
            max_shape_components, len(scales), 'max_shape_components')
        # set parameters
        self.patch_shape = patch_shape
        self.features = features
        self.normalize_parts = normalize_parts
        self.diagonal = diagonal
        self.scales = list(scales)
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.max_shape_components = max_shape_components

    def _warp_template(self, template, template_shape, reference_shape, level):
        parts = template.extract_patches(template_shape,
                                         patch_size=self.patch_shape[level],
                                         as_single_array=True)
        if self.normalize_parts:
            parts = self.normalize_parts(parts)

        return Image(parts)

    def _build_atm(self, shape_models, warped_templates, reference_shape):
        return PartsATM(shape_models, warped_templates, reference_shape,
                        self.patch_shape, self.features,
                        self.normalize_parts, self.scales,
                        self.scale_shapes, self.scale_features)


from .base import ATM, PatchATM, LinearATM, LinearPatchATM, PartsATM
