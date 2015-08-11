from __future__ import division
from copy import deepcopy
import warnings
import numpy as np
from menpo.feature import no_op
from menpo.visualize import print_dynamic
from menpo.model import PCAModel
from menpo.transform import Scale
from menpo.shape import mean_pointcloud
from menpofit import checks
from menpofit.transform import (DifferentiableThinPlateSplines,
                                DifferentiablePiecewiseAffine)
from menpofit.base import name_of_callable, batch
from menpofit.builder import (
    build_reference_frame, build_patch_reference_frame,
    compute_features, scale_images, build_shape_model, warp_images,
    align_shapes, rescale_images_to_reference_shape, densify_shapes,
    extract_patches, MenpoFitBuilderWarning, compute_reference_shape)


# TODO: document me!
class AAM(object):
    r"""
    Active Appearance Model class.

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
    max_appearance_components : ``None`` or `int` > 0 or ``0`` <= `float` <= ``1`` or list of those, optional
        If list of length ``n_scales``, then a number of appearance components
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
    def __init__(self, images, group=None, verbose=False, reference_shape=None,
                 features=no_op, transform=DifferentiablePiecewiseAffine,
                 diagonal=None, scales=(0.5, 1.0), max_shape_components=None,
                 max_appearance_components=None, batch_size=None):

        checks.check_diagonal(diagonal)
        n_scales = len(scales)
        scales = checks.check_scales(scales)
        features = checks.check_features(features, n_scales)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_scales, 'max_shape_components')
        max_appearance_components = checks.check_max_components(
            max_appearance_components, n_scales, 'max_appearance_components')

        self.features = features
        self.transform = transform
        self.diagonal = diagonal
        self.scales = scales
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components
        self.reference_shape = reference_shape
        self.shape_models = []
        self.appearance_models = []

        # Train AAM
        self._train(images, increment=False, group=group, verbose=verbose,
                    batch_size=batch_size)

    def _train(self, images, increment=False, group=None,
               shape_forgetting_factor=1.0, appearance_forgetting_factor=1.0,
               verbose=False, batch_size=None):
        r"""
        """
        # If batch_size is not None, then we may have a generator, else we
        # assume we have a list.
        # If batch_size is not None, then we may have a generator, else we
        # assume we have a list.
        if batch_size is not None:
            # Create a generator of fixed sized batches. Will still work even
            # on an infinite list.
            image_batches = batch(images, batch_size)
        else:
            image_batches = [list(images)]

        for k, image_batch in enumerate(image_batches):
            if k == 0:
                if self.reference_shape is None:
                    # If no reference shape was given, use the mean of the first
                    # batch
                    if batch_size is not None:
                        warnings.warn('No reference shape was provided. The '
                                      'mean of the first batch will be the '
                                      'reference shape. If the batch mean is '
                                      'not representative of the true mean, '
                                      'this may cause issues.',
                                      MenpoFitBuilderWarning)
                    checks.check_landmark_trilist(image_batch[0],
                                                  self.transform, group=group)
                    self.reference_shape = compute_reference_shape(
                        [i.landmarks[group].lms for i in image_batch],
                        self.diagonal, verbose=verbose)

            # After the first batch, we are incrementing the model
            if k > 0:
                increment = True

            if verbose:
                print('Computing batch {}'.format(k))

            # Train each batch
            self._train_batch(
                image_batch, increment=increment, group=group,
                shape_forgetting_factor=shape_forgetting_factor,
                appearance_forgetting_factor=appearance_forgetting_factor,
                verbose=verbose)

    def _train_batch(self, image_batch, increment=False, group=None,
                     verbose=False, shape_forgetting_factor=1.0,
                     appearance_forgetting_factor=1.0):
        r"""
        Builds an Active Appearance Model from a list of landmarked images.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images from which to build the AAM.
        group : `string`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.
        verbose : `boolean`, optional
            Flag that controls information and progress printing.

        Returns
        -------
        aam : :map:`AAM`
            The AAM object. Shape and appearance models are stored from
            lowest to highest scale
        """
        # Rescale to existing reference shape
        image_batch = rescale_images_to_reference_shape(
            image_batch, group, self.reference_shape,
            verbose=verbose)

        # build models at each scale
        if verbose:
            print_dynamic('- Building models\n')

        feature_images = []
        # for each scale (low --> high)
        for j in range(self.n_scales):
            if verbose:
                if len(self.scales) > 1:
                    scale_prefix = '  - Scale {}: '.format(j)
                else:
                    scale_prefix = '  - '
            else:
                scale_prefix = None

            # Handle features
            if j == 0 or self.features[j] is not self.features[j - 1]:
                # Compute features only if this is the first pass through
                # the loop or the features at this scale are different from
                # the features at the previous scale
                feature_images = compute_features(image_batch,
                                                  self.features[j],
                                                  prefix=scale_prefix,
                                                  verbose=verbose)
            # handle scales
            if self.scales[j] != 1:
                # Scale feature images only if scale is different than 1
                scaled_images = scale_images(feature_images, self.scales[j],
                                             prefix=scale_prefix,
                                             verbose=verbose)
            else:
                scaled_images = feature_images

            # Extract potentially rescaled shapes
            scale_shapes = [i.landmarks[group].lms for i in scaled_images]

            # Build the shape model
            if verbose:
                print_dynamic('{}Building shape model'.format(scale_prefix))

            if not increment:
                if j == 0:
                    shape_model = self._build_shape_model(
                        scale_shapes, j)
                    self.shape_models.append(shape_model)
                else:
                    self.shape_models.append(deepcopy(shape_model))
            else:
                self._increment_shape_model(
                    scale_shapes,  self.shape_models[j],
                    forgetting_factor=shape_forgetting_factor)

            # Obtain warped images - we use a scaled version of the
            # reference shape, computed here. This is because the mean
            # moves when we are incrementing, and we need a consistent
            # reference frame.
            scaled_reference_shape = Scale(self.scales[j], n_dims=2).apply(
                self.reference_shape)
            warped_images = self._warp_images(scaled_images, scale_shapes,
                                              scaled_reference_shape,
                                              j, scale_prefix, verbose)

            # obtain appearance model
            if verbose:
                print_dynamic('{}Building appearance model'.format(
                    scale_prefix))

            if not increment:
                appearance_model = PCAModel(warped_images)
                # trim appearance model if required
                if self.max_appearance_components is not None:
                    appearance_model.trim_components(
                        self.max_appearance_components[j])
                # add appearance model to the list
                self.appearance_models.append(appearance_model)
            else:
                # increment appearance model
                self.appearance_models[j].increment(
                    warped_images,
                    forgetting_factor=appearance_forgetting_factor)
                # trim appearance model if required
                if self.max_appearance_components is not None:
                    self.appearance_models[j].trim_components(
                        self.max_appearance_components[j])

            if verbose:
                print_dynamic('{}Done\n'.format(scale_prefix))

        # Because we just copy the shape model, we need to wait to trim
        # it after building each model. This ensures we can have a different
        # number of components per level
        for j, sm in enumerate(self.shape_models):
            max_sc = self.max_shape_components[j]
            if max_sc is not None:
                sm.trim_components(max_sc)

    def increment(self, images, group=None, verbose=False,
                  shape_forgetting_factor=1.0, appearance_forgetting_factor=1.0,
                  batch_size=None):
        # Literally just to fit under 80 characters, but maintain the sensible
        # parameter name
        aff = appearance_forgetting_factor
        return self._train(images, increment=True, group=group,
                           verbose=verbose,
                           shape_forgetting_factor=shape_forgetting_factor,
                           appearance_forgetting_factor=aff,
                           batch_size=batch_size)

    def _build_shape_model(self, shapes, scale_index):
        return build_shape_model(shapes)

    def _increment_shape_model(self, shapes, shape_model,
                               forgetting_factor=1.0):
        # Compute aligned shapes
        aligned_shapes = align_shapes(shapes)
        # Increment shape model
        shape_model.increment(aligned_shapes,
                              forgetting_factor=forgetting_factor)

    def _warp_images(self, images, shapes, reference_shape, scale_index,
                     prefix, verbose):
        reference_frame = build_reference_frame(reference_shape)
        return warp_images(images, shapes, reference_frame, self.transform,
                           prefix=prefix, verbose=verbose)

    @property
    def n_scales(self):
        """
        The number of scales of the AAM.

        :type: `int`
        """
        return len(self.scales)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.
        :type: `string`
        """
        return 'Holistic Active Appearance Model'

    def instance(self, shape_weights=None, appearance_weights=None,
                 scale_index=-1):
        r"""
        Generates a novel AAM instance given a set of shape and appearance
        weights. If no weights are provided, the mean AAM instance is
        returned.

        Parameters
        -----------
        shape_weights : ``(n_weights,)`` `ndarray` or `float` list
            Weights of the shape model that will be used to create
            a novel shape instance. If ``None``, the mean shape
            ``(shape_weights = [0, 0, ..., 0])`` is used.
        appearance_weights : ``(n_weights,)`` `ndarray` or `float` list
            Weights of the appearance model that will be used to create
            a novel appearance instance. If ``None``, the mean appearance
            ``(appearance_weights = [0, 0, ..., 0])`` is used.
        scale_index : `int`, optional
            The scale to be used.

        Returns
        -------
        image : :map:`Image`
            The novel AAM instance.
        """
        sm = self.shape_models[scale_index]
        am = self.appearance_models[scale_index]

        # TODO: this bit of logic should to be transferred down to PCAModel
        if shape_weights is None:
            shape_weights = [0]
        if appearance_weights is None:
            appearance_weights = [0]
        n_shape_weights = len(shape_weights)
        shape_weights *= sm.eigenvalues[:n_shape_weights] ** 0.5
        shape_instance = sm.instance(shape_weights)
        n_appearance_weights = len(appearance_weights)
        appearance_weights *= am.eigenvalues[:n_appearance_weights] ** 0.5
        appearance_instance = am.instance(appearance_weights)

        return self._instance(scale_index, shape_instance, appearance_instance)

    def random_instance(self, scale_index=-1):
        r"""
        Generates a novel random instance of the AAM.

        Parameters
        -----------
        scale_index : `int`, optional
            The scale to be used.

        Returns
        -------
        image : :map:`Image`
            The novel AAM instance.
        """
        sm = self.shape_models[scale_index]
        am = self.appearance_models[scale_index]

        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = (np.random.randn(sm.n_active_components) *
                         sm.eigenvalues[:sm.n_active_components]**0.5)
        shape_instance = sm.instance(shape_weights)
        appearance_weights = (np.random.randn(am.n_active_components) *
                              am.eigenvalues[:am.n_active_components]**0.5)
        appearance_instance = am.instance(appearance_weights)

        return self._instance(scale_index, shape_instance, appearance_instance)

    def _instance(self, scale_index, shape_instance, appearance_instance):
        template = self.appearance_models[scale_index].mean()
        landmarks = template.landmarks['source'].lms

        reference_frame = build_reference_frame(shape_instance)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        return appearance_instance.as_unmasked(copy=False).warp_to_mask(
            reference_frame.mask, transform, warp_landmarks=True)

    def view_shape_models_widget(self, n_parameters=5,
                                 parameters_bounds=(-3.0, 3.0),
                                 mode='multiple', figure_size=(10, 8)):
        r"""
        Visualizes the shape models of the AAM object using the
        `menpo.visualize.widgets.visualize_shape_model` widget.

        Parameters
        -----------
        n_parameters : `int` or `list` of `int` or ``None``, optional
            The number of shape principal components to be used for the
            parameters sliders.
            If `int`, then the number of sliders per level is the minimum
            between `n_parameters` and the number of active components per
            level.
            If `list` of `int`, then a number of sliders is defined per level.
            If ``None``, all the active components per level will have a slider.
        parameters_bounds : (`float`, `float`), optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu.
            If ``'multiple'``, a slider is constructed for each parameter.
        figure_size : (`int`, `int`), optional
            The size of the plotted figures.
        """
        from menpofit.visualize import visualize_shape_model
        visualize_shape_model(self.shape_models, n_parameters=n_parameters,
                              parameters_bounds=parameters_bounds,
                              figure_size=figure_size, mode=mode)

    def view_appearance_models_widget(self, n_parameters=5,
                                      parameters_bounds=(-3.0, 3.0),
                                      mode='multiple', figure_size=(10, 8)):
        r"""
        Visualizes the appearance models of the AAM object using the
        `menpo.visualize.widgets.visualize_appearance_model` widget.
        Parameters
        -----------
        n_parameters : `int` or `list` of `int` or ``None``, optional
            The number of appearance principal components to be used for the
            parameters sliders.
            If `int`, then the number of sliders per scale is the minimum
            between `n_parameters` and the number of active components per
            scale.
            If `list` of `int`, then a number of sliders is defined per scale.
            If ``None``, all the active components per scale will have a slider.
        parameters_bounds : (`float`, `float`), optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu.
            If ``'multiple'``, a slider is constructed for each parameter.
        figure_size : (`int`, `int`), optional
            The size of the plotted figures.
        """
        from menpofit.visualize import visualize_appearance_model
        visualize_appearance_model(self.appearance_models,
                                   n_parameters=n_parameters,
                                   parameters_bounds=parameters_bounds,
                                   figure_size=figure_size, mode=mode)

    # TODO: fix me!
    def view_aam_widget(self, n_shape_parameters=5, n_appearance_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        r"""
        Visualizes both the shape and appearance models of the AAM object using
        the `menpo.visualize.widgets.visualize_aam` widget.
        Parameters
        -----------
        n_shape_parameters : `int` or `list` of `int` or None, optional
            The number of shape principal components to be used for the
            parameters sliders.
            If `int`, then the number of sliders per scale is the minimum
            between `n_parameters` and the number of active components per
            scale.
            If `list` of `int`, then a number of sliders is defined per scale.
            If ``None``, all the active components per scale will have a slider.
        n_appearance_parameters : `int` or `list` of `int` or None, optional
            The number of appearance principal components to be used for the
            parameters sliders.
            If `int`, then the number of sliders per scale is the minimum
            between `n_parameters` and the number of active components per
            scale.
            If `list` of `int`, then a number of sliders is defined per scale.
            If ``None``, all the active components per scale will have a slider.
        parameters_bounds : (`float`, `float`), optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu.
            If ``'multiple'``, a slider is constructed for each parameter.
        figure_size : (`int`, `int`), optional
            The size of the plotted figures.
        """
        from menpofit.visualize import visualize_aam
        visualize_aam(self, n_shape_parameters=n_shape_parameters,
                      n_appearance_parameters=n_appearance_parameters,
                      parameters_bounds=parameters_bounds,
                      figure_size=figure_size, mode=mode)

    def __str__(self):
        return _aam_str(self)


# TODO: document me!
class MaskedAAM(AAM):
    r"""
    Masked Active Appearance Model class.

    Parameters
    -----------
    shape_models : :map:`PCAModel` list
        A list containing the shape models of the AAM.
    appearance_models : :map:`PCAModel` list
        A list containing the appearance models of the AAM.
    reference_shape : :map:`PointCloud`
        The reference shape that was used to resize all training images to a
        consistent object size.
    patch_shape : tuple of `int`
        The shape of the patches used to build the Patch Based AAM.
    features : `callable` or ``[callable]``
        If list of length ``n_scales``, feature extraction is performed at
        each scale after downscaling of the image.
        The first element of the list specifies the features to be extracted at
        the lowest scale and so on.

        If ``callable`` the specified feature will be applied to the original
        image and pyramid generation will be performed on top of the feature
        image. Also see the `pyramid_on_features` property.

        Note that from our experience, this approach of extracting features
        once and then creating a pyramid on top tends to lead to better
        performing AAMs.

    scales : `int` or float` or list of those
    scale_shapes : `boolean`
    """

    def __init__(self, images, group=None, verbose=False, features=no_op,
                 diagonal=None, scales=(0.5, 1.0), patch_shape=(17, 17),
                 max_shape_components=None, max_appearance_components=None,
                 batch_size=None):
        self.patch_shape = checks.check_patch_shape(patch_shape, len(scales))

        super(MaskedAAM, self).__init__(
            images, group=group, verbose=verbose, features=features,
            transform=DifferentiableThinPlateSplines, diagonal=diagonal,
            scales=scales,  max_shape_components=max_shape_components,
            max_appearance_components=max_appearance_components,
            batch_size=batch_size)

    def _warp_images(self, images, shapes, reference_shape, scale_index,
                     prefix, verbose):
        reference_frame = build_patch_reference_frame(
            reference_shape, patch_shape=self.patch_shape[scale_index])
        return warp_images(images, shapes, reference_frame, self.transform,
                           prefix=prefix, verbose=verbose)

    @property
    def _str_title(self):
        return 'Masked Active Appearance Model'

    def _instance(self, scale_index, shape_instance, appearance_instance):
        template = self.appearance_models[scale_index].mean
        landmarks = template.landmarks['source'].lms

        reference_frame = build_patch_reference_frame(
            shape_instance, patch_shape=self.patch_shape)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        return appearance_instance.as_unmasked().warp_to_mask(
            reference_frame.mask, transform, warp_landmarks=True)

    def view_appearance_models_widget(self, n_parameters=5,
                                      parameters_bounds=(-3.0, 3.0),
                                      mode='multiple', figure_size=(10, 8)):
        from menpofit.visualize import visualize_appearance_model
        visualize_appearance_model(self.appearance_models,
                                   n_parameters=n_parameters,
                                   parameters_bounds=parameters_bounds,
                                   figure_size=figure_size, mode=mode)

    def __str__(self):
        return _aam_str(self)


# TODO: document me!
class LinearAAM(AAM):
    r"""
    Linear Active Appearance Model class.

    Parameters
    -----------
    shape_models : :map:`PCAModel` list
        A list containing the shape models of the AAM.
    appearance_models : :map:`PCAModel` list
        A list containing the appearance models of the AAM.
    reference_shape : :map:`PointCloud`
        The reference shape that was used to resize all training images to a
        consistent object size.
    transform : :map:`PureAlignmentTransform`
        The transform used to warp the images from which the AAM was
        constructed.
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

    scales : `int` or float` or list of those
    """

    def __init__(self, images, group=None, verbose=False, features=no_op,
                 transform=DifferentiableThinPlateSplines, diagonal=None,
                 scales=(0.5, 1.0), max_shape_components=None,
                 max_appearance_components=None, batch_size=None):

        super(LinearAAM, self).__init__(
            images, group=group, verbose=verbose, features=features,
            transform=transform, diagonal=diagonal, scales=scales,
            max_shape_components=max_shape_components,
            max_appearance_components=max_appearance_components,
            batch_size=batch_size)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.
        :type: `string`
        """
        return 'Linear Active Appearance Model'

    def _build_shape_model(self, shapes, scale_index):
        mean_aligned_shape = mean_pointcloud(align_shapes(shapes))
        self.n_landmarks = mean_aligned_shape.n_points
        self.reference_frame = build_reference_frame(mean_aligned_shape)
        dense_shapes = densify_shapes(shapes, self.reference_frame,
                                      self.transform)
        # build dense shape model
        shape_model = build_shape_model(dense_shapes)
        return shape_model

    def _increment_shape_model(self, shapes, shape_model,
                               forgetting_factor=1.0):
        aligned_shapes = align_shapes(shapes)
        dense_shapes = densify_shapes(aligned_shapes, self.reference_frame,
                                      self.transform)
        # Increment shape model
        shape_model.increment(dense_shapes,
                              forgetting_factor=forgetting_factor)

    def _warp_images(self, images, shapes, reference_shape, scale_index,
                     prefix, verbose):
        return warp_images(images, shapes, self.reference_frame,
                           self.transform, prefix=prefix,
                           verbose=verbose)

    # TODO: implement me!
    def _instance(self, scale_index, shape_instance, appearance_instance):
        raise NotImplemented

    # TODO: implement me!
    def view_appearance_models_widget(self, n_parameters=5,
                                      parameters_bounds=(-3.0, 3.0),
                                      mode='multiple', figure_size=(10, 8)):
        raise NotImplemented

    # TODO: implement me!
    def view_aam_widget(self, n_shape_parameters=5, n_appearance_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        raise NotImplemented

    def __str__(self):
        return _aam_str(self)


# TODO: document me!
class LinearMaskedAAM(AAM):
    r"""
    Linear Masked Active Appearance Model class.

    Parameters
    -----------
    shape_models : :map:`PCAModel` list
        A list containing the shape models of the AAM.
    appearance_models : :map:`PCAModel` list
        A list containing the appearance models of the AAM.
    reference_shape : :map:`PointCloud`
        The reference shape that was used to resize all training images to a
        consistent object size.
    patch_shape : tuple of `int`
        The shape of the patches used to build the Patch Based AAM.
    features : `callable` or ``[callable]``
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

    scales : `int` or float` or list of those
    """

    def __init__(self, images, group=None, verbose=False, features=no_op,
                 diagonal=None, scales=(0.5, 1.0), patch_shape=(17, 17),
                 max_shape_components=None, max_appearance_components=None,
                 batch_size=None):
        self.patch_shape = checks.check_patch_shape(patch_shape, len(scales))

        super(LinearMaskedAAM, self).__init__(
            images, group=group, verbose=verbose, features=features,
            transform=DifferentiableThinPlateSplines, diagonal=diagonal,
            scales=scales,  max_shape_components=max_shape_components,
            max_appearance_components=max_appearance_components,
            batch_size=batch_size)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.
        :type: `string`
        """
        return 'Linear Masked Active Appearance Model'

    def _build_shape_model(self, shapes, scale_index):
        mean_aligned_shape = mean_pointcloud(align_shapes(shapes))
        self.n_landmarks = mean_aligned_shape.n_points
        self.reference_frame = build_patch_reference_frame(
            mean_aligned_shape, patch_shape=self.patch_shape[scale_index])
        dense_shapes = densify_shapes(shapes, self.reference_frame,
                                      self.transform)
        # build dense shape model
        shape_model = build_shape_model(dense_shapes)
        return shape_model

    def _increment_shape_model(self, shapes, shape_model,
                               forgetting_factor=1.0):
        aligned_shapes = align_shapes(shapes)
        dense_shapes = densify_shapes(aligned_shapes, self.reference_frame,
                                      self.transform)
        # Increment shape model
        shape_model.increment(dense_shapes,
                              forgetting_factor=forgetting_factor)

    def _warp_images(self, images, shapes, reference_shape, scale_index,
                     prefix, verbose):
        return warp_images(images, shapes, self.reference_frame,
                           self.transform, prefix=prefix,
                           verbose=verbose)

    # TODO: implement me!
    def _instance(self, scale_index, shape_instance, appearance_instance):
        raise NotImplemented

    # TODO: implement me!
    def view_appearance_models_widget(self, n_parameters=5,
                                      parameters_bounds=(-3.0, 3.0),
                                      mode='multiple', figure_size=(10, 8)):
        raise NotImplemented

    # TODO: implement me!
    def view_aam_widget(self, n_shape_parameters=5, n_appearance_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        raise NotImplemented

    def __str__(self):
        return _aam_str(self)


# TODO: document me!
# TODO: implement offsets support?
class PatchAAM(AAM):
    r"""
    Patch-based Active Appearance Model class.

    Parameters
    -----------
    shape_models : :map:`PCAModel` list
        A list containing the shape models of the AAM.
    appearance_models : :map:`PCAModel` list
        A list containing the appearance models of the AAM.
    reference_shape : :map:`PointCloud`
        The reference shape that was used to resize all training images to a
        consistent object size.
    patch_shape : tuple of `int`
        The shape of the patches used to build the Patch Based AAM.
    features : `callable` or ``[callable]``
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

    normalize_parts: `callable`
    scales : `int` or float` or list of those
    """

    def __init__(self, images, group=None, verbose=False, features=no_op,
                 normalize_parts=no_op, diagonal=None, scales=(0.5, 1.0),
                 patch_shape=(17, 17), max_shape_components=None,
                 max_appearance_components=None, batch_size=None):
        self.patch_shape = checks.check_patch_shape(patch_shape, len(scales))
        self.normalize_parts = normalize_parts

        super(PatchAAM, self).__init__(
            images, group=group, verbose=verbose, features=features,
            transform=None, diagonal=diagonal,
            scales=scales,  max_shape_components=max_shape_components,
            max_appearance_components=max_appearance_components,
            batch_size=batch_size)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.
        :type: `string`
        """
        return 'Patch-based Active Appearance Model'

    def _warp_images(self, images, shapes, reference_shape, scale_index,
                     prefix, verbose):
        return extract_patches(images, shapes, self.patch_shape[scale_index],
                               normalize_function=self.normalize_parts,
                               prefix=prefix, verbose=verbose)

    # TODO: implement me!
    def _instance(self, scale_index, shape_instance, appearance_instance):
        raise NotImplemented

    # TODO: implement me!
    def view_appearance_models_widget(self, n_parameters=5,
                                      parameters_bounds=(-3.0, 3.0),
                                      mode='multiple', figure_size=(10, 8)):
        raise NotImplemented

    # TODO: implement me!
    def view_aam_widget(self, n_shape_parameters=5, n_appearance_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        raise NotImplemented

    def __str__(self):
        return _aam_str(self)


def _aam_str(aam):
    if aam.diagonal is not None:
        diagonal = aam.diagonal
    else:
        y, x = aam.reference_shape.range()
        diagonal = np.sqrt(x ** 2 + y ** 2)

    # Compute scale info strings
    scales_info = []
    lvl_str_tmplt = r"""  - Scale {}
   - Holistic feature: {}
   - {} appearance components
   - {} shape components"""
    for k, s in enumerate(aam.scales):
        scales_info.append(lvl_str_tmplt.format(
            s, name_of_callable(aam.features[k]),
            aam.appearance_models[k].n_components,
            aam.shape_models[k].n_components))
    # Patch based AAM
    if hasattr(aam, 'patch_shape'):
        for k in range(len(scales_info)):
            scales_info[k] += '\n   - Patch shape: {}'.format(
                aam.patch_shape[k])
    scales_info = '\n'.join(scales_info)

    if aam.transform is not None:
        transform_str = 'Images warped with {transform} transform'
    else:
        transform_str = 'No image warping performed'

    cls_str = r"""{class_title}
 - Images scaled to diagonal: {diagonal:.2f}
 - {transform}
 - Scales: {scales}
{scales_info}
""".format(class_title=aam._str_title,
           transform=transform_str,
           diagonal=diagonal,
           scales=aam.scales,
           scales_info=scales_info)
    return cls_str

HolisticAAM = AAM
