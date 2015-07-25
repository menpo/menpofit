from __future__ import division
import numpy as np

from menpo.shape import TriMesh
from menpo.image import MaskedImage
from menpo.transform import Translation
from menpo.feature import igo
from menpo.model import PCAModel
from menpo.visualize import print_dynamic, print_progress

from menpofit import checks
from menpofit.base import create_pyramid
from menpofit.builder import (DeformableModelBuilder, build_shape_model,
                              normalization_wrt_reference_shape)
from menpofit.transform import (DifferentiablePiecewiseAffine,
                                DifferentiableThinPlateSplines)


class AAMBuilder(DeformableModelBuilder):
    r"""
    Class that builds Multilevel Active Appearance Models.

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

    normalization_diagonal : `int` >= ``20``, optional
        During building an AAM, all images are rescaled to ensure that the
        scale of their landmarks matches the scale of the mean shape.

        If `int`, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the normalization_diagonal
        value.

        If ``None``, the mean shape is not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).

    n_levels : `int` > 0, optional
        The number of multi-resolution pyramidal levels to be used.

    downscale : `float` >= ``1``, optional
        The downscale factor that will be used to create the different
        pyramidal levels. The scale factor will be::

            (downscale ** k) for k in range(``n_levels``)

    scaled_shape_models : `boolean`, optional
        If ``True``, the reference frames will be the mean shapes of
        each pyramid level, so the shape models will be scaled.

        If ``False``, the reference frames of all levels will be the mean shape
        of the highest level, so the shape models will not be scaled; they will
        have the same size.

        Note that from our experience, if ``scaled_shape_models`` is ``False``,
        AAMs tend to have slightly better performance.

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

    boundary : `int` >= ``0``, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

    appearance_model_cls: :map:`PCAModel`, optional
        The technique to use to construct an appearance model. Defaults to use
        `PCAModel` class to do so.

    Returns
    -------
    aam : :map:`AAMBuilder`
        The AAM Builder object

    Raises
    -------
    ValueError
        ``n_levels`` must be `int` > ``0``
    ValueError
        ``downscale`` must be >= ``1``
    ValueError
        ``normalization_diagonal`` must be >= ``20``
    ValueError
        ``max_shape_components`` must be ``None`` or an `int` > 0 or
        a ``0`` <= `float` <= ``1`` or a list of those containing 1 or
        ``n_levels`` elements
    ValueError
        ``max_appearance_components`` must be ``None`` or an `int` > ``0`` or a
        ``0`` <= `float` <= ``1`` or a list of those containing 1 or
        ``n_levels`` elements
    ValueError
        ``features`` must be a `function` or a list of those
        containing ``1`` or ``n_levels`` elements
    """
    def __init__(self, features=igo, transform=DifferentiablePiecewiseAffine,
                 trilist=None, normalization_diagonal=None, n_levels=3,
                 downscale=2, scaled_shape_models=True,
                 max_shape_components=None, max_appearance_components=None, appearance_model_cls=PCAModel,
                 boundary=3):
        # check parameters
        checks.check_n_levels(n_levels)
        checks.check_downscale(downscale)
        checks.check_normalization_diagonal(normalization_diagonal)
        checks.check_boundary(boundary)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_levels, 'max_shape_components')
        max_appearance_components = checks.check_max_components(
            max_appearance_components, n_levels, 'max_appearance_components')
        features = checks.check_features(features, n_levels)
        # store parameters
        self.features = features
        self.transform = transform
        self.trilist = trilist
        self.normalization_diagonal = normalization_diagonal
        self.n_levels = n_levels
        self.downscale = downscale
        self.scaled_shape_models = scaled_shape_models
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components
        self.boundary = boundary
        self.appearance_model_cls = appearance_model_cls

    def build(self, images, group=None, label=None, verbose=False):
        r"""
        Builds a Multilevel Active Appearance Model from a list of
        landmarked images.

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
            The AAM object. Shape and appearance models are stored from lowest
            to highest level
        """
        # compute reference_shape and normalize images size
        self.reference_shape, normalized_images = \
            normalization_wrt_reference_shape(images, group, label,
                                              self.normalization_diagonal,
                                              verbose=verbose)

        # create pyramid
        generators = create_pyramid(normalized_images, self.n_levels,
                                    self.downscale, self.features,
                                    verbose=verbose)

        # build the model at each pyramid level
        if verbose:
            if self.n_levels > 1:
                print('- Building model for each of the {} '
                      'pyramid levels'.format(self.n_levels))
            else:
                print('- Building model')

        shape_models = []
        appearance_models = []
        # for each pyramid level (high --> low)
        for j in range(self.n_levels):
            # since models are built from highest to lowest level, the
            # parameters in form of list need to use a reversed index
            rj = self.n_levels - j - 1

            if verbose:
                level_str = '  - '
                if self.n_levels > 1:
                    level_str = '  - Level {}: '.format(j + 1)

            # get feature images of current level
            feature_images = []

            if verbose:
                generators_with_print = print_progress(
                    generators, show_bar=False, show_eta=False,
                    end_with_newline=False,
                    prefix='{}Computing feature space/rescaling'.format(level_str))
            else:
                generators_with_print = generators

            for g in generators_with_print:
                feature_images.append(next(g))

            # extract potentially rescaled shapes
            shapes = [i.landmarks[group][label] for i in feature_images]

            # define shapes that will be used for training
            if j == 0:
                original_shapes = shapes
                train_shapes = shapes
            else:
                if self.scaled_shape_models:
                    train_shapes = shapes
                else:
                    train_shapes = original_shapes

            # train shape model and find reference frame
            if verbose:
                print_dynamic('{}Building shape model'.format(level_str))
            shape_model = build_shape_model(
                train_shapes, self.max_shape_components[rj])
            reference_frame = self._build_reference_frame(shape_model.mean())

            # add shape model to the list
            shape_models.append(shape_model)

            # compute transforms
            if verbose:
                print_dynamic('{}Computing transforms'.format(level_str))


            # Create a dummy initial transform
            s_to_t_transform = self.transform(
                reference_frame.landmarks['source'].lms,
                reference_frame.landmarks['source'].lms)

            if verbose:
                feature_images_with_print = print_progress(
                    feature_images, show_bar=False, show_eta=False,
                    end_with_newline=False,
                    prefix='{}Warping images'.format(level_str))
            else:
                feature_images_with_print = feature_images

            # warp images to reference frame
            warped_images = []
            for i in feature_images_with_print:
                # Setting the target can be significantly faster for transforms
                # such as CachedPiecewiseAffine
                s_to_t_transform.set_target(i.landmarks[group][label])
                warped_images.append(i.warp_to_mask(reference_frame.mask,
                                                    s_to_t_transform))

            # attach reference_frame to images' source shape
            for i in warped_images:
                i.landmarks['source'] = reference_frame.landmarks['source']

            # build appearance model
            if verbose:
                print_dynamic('{}Building appearance model'.format(level_str))
            appearance_model = self.appearance_model_cls(warped_images)
            # trim appearance model if required
            if self.max_appearance_components[rj] is not None:
                appearance_model.trim_components(
                    self.max_appearance_components[rj])

            # add appearance model to the list
            appearance_models.append(appearance_model)

            if verbose:
                print('{}Done'.format(level_str).ljust(80))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        appearance_models.reverse()
        n_training_images = len(images)

        return self._build_aam(shape_models, appearance_models,
                               n_training_images)

    def _build_reference_frame(self, mean_shape):
        r"""
        Generates the reference frame given a mean shape.

        Parameters
        ----------
        mean_shape : :map:`PointCloud`
            The mean shape to use.

        Returns
        -------
        reference_frame : :map:`MaskedImage`
            The reference frame.
        """
        return build_reference_frame(mean_shape, boundary=self.boundary,
                                     trilist=self.trilist)

    def _build_aam(self, shape_models, appearance_models, n_training_images):
        r"""
        Returns an AAM object.

        Parameters
        ----------
        shape_models : :map:`PCAModel`
            The trained multilevel shape models.
            
        appearance_models : :map:`PCAModel`
            The trained multilevel appearance models.
            
        n_training_images : `int`
            The number of training images.

        Returns
        -------
        aam : :map:`AAM`
            The trained AAM object.
        """
        from .base import AAM
        return AAM(shape_models, appearance_models, n_training_images,
                   self.transform, self.features, self.reference_shape,
                   self.downscale, self.scaled_shape_models)


class PatchBasedAAMBuilder(AAMBuilder):
    r"""
    Class that builds Multilevel Patch-Based Active Appearance Models.

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

    patch_shape : tuple of `int`, optional
        The appearance model of the Patch-Based AAM will be obtained by
        sampling appearance patches with the specified shape around each
        landmark.

    normalization_diagonal : `int` >= ``20``, optional
        During building an AAM, all images are rescaled to ensure that the
        scale of their landmarks matches the scale of the mean shape.

        If `int`, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the ``normalization_diagonal``
        value.

        If ``None``, the mean shape is not rescaled.

        .. note::

            Because the reference frame is computed from the mean
            landmarks, this kwarg also specifies the diagonal length of the
            reference frame (provided that features computation does not change
            the image size).

    n_levels : `int` > ``0``, optional
        The number of multi-resolution pyramidal levels to be used.

    downscale : `float` >= 1, optional
        The downscale factor that will be used to create the different
        pyramidal levels. The scale factor will be::

            (downscale ** k) for k in range(``n_levels``)

    scaled_shape_models : `boolean`, optional
        If ``True``, the reference frames will be the mean shapes of each
        pyramid level, so the shape models will be scaled.
        If ``False``, the reference frames of all levels will be the mean shape
        of the highest level, so the shape models will not be scaled; they will
        have the same size.
        Note that from our experience, if scaled_shape_models is ``False``, AAMs
        tend to have slightly better performance.

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

    boundary : `int` >= ``0``, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

    Returns
    -------
    aam : ::map:`PatchBasedAAMBuilder`
        The Patch-Based AAM Builder object

    Raises
    -------
    ValueError
        ``n_levels`` must be `int` > ``0``
    ValueError
        ``downscale`` must be >= ``1``
    ValueError
        ``normalization_diagonal`` must be >= ``20``
    ValueError
        ``max_shape_components must be ``None`` or an `int` > ``0`` or
        a ``0`` <= `float` <= ``1`` or a list of those containing ``1``
        or ``n_levels`` elements
    ValueError
        ``max_appearance_components`` must be ``None`` or an `int` > 0 or a
        ``0`` <= `float` <= ``1`` or a list of those containing ``1``
        or ``n_levels`` elements
    ValueError
        ``features`` must be a `string` or a `function` or a list of those
        containing 1 or ``n_levels`` elements
    ValueError
        ``pyramid_on_features`` is enabled so ``features`` must be a
        `string` or a `function` or a list containing one of those
    """
    def __init__(self, features=igo, patch_shape=(16, 16),
                 normalization_diagonal=None, n_levels=3, downscale=2,
                 scaled_shape_models=True, max_shape_components=None,
                 max_appearance_components=None, boundary=3, appearance_model_cls=PCAModel):
        # check parameters
        checks.check_n_levels(n_levels)
        checks.check_downscale(downscale)
        checks.check_normalization_diagonal(normalization_diagonal)
        checks.check_boundary(boundary)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_levels, 'max_shape_components')
        max_appearance_components = checks.check_max_components(
            max_appearance_components, n_levels, 'max_appearance_components')
        features = checks.check_features(features, n_levels)

        # store parameters
        self.features = features
        self.patch_shape = patch_shape
        self.normalization_diagonal = normalization_diagonal
        self.n_levels = n_levels
        self.downscale = downscale
        self.scaled_shape_models = scaled_shape_models
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components
        self.boundary = boundary
        self.appearance_model_cls = appearance_model_cls

        # patch-based AAMs can only work with TPS transform
        self.transform = DifferentiableThinPlateSplines

    def _build_reference_frame(self, mean_shape):
        r"""
        Generates the reference frame given a mean shape.

        Parameters
        ----------
        mean_shape : :map:`PointCloud`
            The mean shape to use.

        Returns
        -------
        reference_frame : :map:`MaskedImage`
            The patch-based reference frame.
        """
        return build_patch_reference_frame(mean_shape, boundary=self.boundary,
                                           patch_shape=self.patch_shape)

    def _mask_image(self, image):
        r"""
        Creates the patch-based mask of the given image.

        Parameters
        ----------
        image : :map:`MaskedImage`
            The image to be masked.
        """
        image.build_mask_around_landmarks(self.patch_shape, group='source')

    def _build_aam(self, shape_models, appearance_models, n_training_images):
        r"""
        Returns a Patch-Based AAM object.

        Parameters
        ----------
        shape_models : :map:`PCAModel`
            The trained multilevel shape models.

        appearance_models : :map:`PCAModel`
            The trained multilevel appearance models.

        n_training_images : `int`
            The number of training images.

        Returns
        -------
        aam : :map:`PatchBasedAAM`
            The trained Patched-Based AAM object.
        """
        from .base import PatchBasedAAM
        return PatchBasedAAM(shape_models, appearance_models,
                             n_training_images, self.patch_shape,
                             self.transform, self.features,
                             self.reference_shape, self.downscale,
                             self.scaled_shape_models)


def build_reference_frame(landmarks, boundary=3, group='source',
                          trilist=None):
    r"""
    Builds a reference frame from a particular set of landmarks.

    Parameters
    ----------
    landmarks : :map:`PointCloud`
        The landmarks that will be used to build the reference frame.

    boundary : `int`, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

    group : `string`, optional
        Group that will be assigned to the provided set of landmarks on the
        reference frame.

    trilist : ``(t, 3)`` `ndarray`, optional
        Triangle list that will be used to build the reference frame.

        If ``None``, defaults to performing Delaunay triangulation on the
        points.

    Returns
    -------
    reference_frame : :map:`Image`
        The reference frame.
    """
    reference_frame = _build_reference_frame(landmarks, boundary=boundary,
                                             group=group)
    if trilist is not None:
        reference_frame.landmarks[group] = TriMesh(
            reference_frame.landmarks['source'].lms.points, trilist=trilist)

    reference_frame.constrain_mask_to_landmarks(group=group)

    return reference_frame


def build_patch_reference_frame(landmarks, boundary=3, group='source',
                                patch_shape=(16, 16)):
    r"""
    Builds a reference frame from a particular set of landmarks.

    Parameters
    ----------
    landmarks : :map:`PointCloud`
        The landmarks that will be used to build the reference frame.

    boundary : `int`, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

    group : `string`, optional
        Group that will be assigned to the provided set of landmarks on the
        reference frame.

    patch_shape : tuple of ints, optional
        Tuple specifying the shape of the patches.

    Returns
    -------
    patch_based_reference_frame : :map:`Image`
        The patch based reference frame.
    """
    boundary = np.max(patch_shape) + boundary
    reference_frame = _build_reference_frame(landmarks, boundary=boundary,
                                             group=group)

    # mask reference frame
    reference_frame.build_mask_around_landmarks(patch_shape, group=group)

    return reference_frame


def _build_reference_frame(landmarks, boundary=3, group='source'):
    # translate landmarks to the origin
    minimum = landmarks.bounds(boundary=boundary)[0]
    landmarks = Translation(-minimum).apply(landmarks)

    resolution = landmarks.range(boundary=boundary)
    reference_frame = MaskedImage.init_blank(resolution)
    reference_frame.landmarks[group] = landmarks

    return reference_frame
