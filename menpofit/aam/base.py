from __future__ import division
from copy import deepcopy
import numpy as np
from menpo.shape import TriMesh
from menpo.feature import no_op
from menpo.visualize import print_dynamic
from menpo.model import PCAModel
from menpo.transform import Scale
from menpofit import checks
from menpofit.transform import DifferentiableThinPlateSplines, \
    DifferentiablePiecewiseAffine
from menpofit.base import name_of_callable, batch
from menpofit.builder import (
    build_reference_frame, build_patch_reference_frame,
    normalization_wrt_reference_shape, compute_features, scale_images,
    build_shape_model, warp_images, align_shapes,
    rescale_images_to_reference_shape)


# TODO: document me!
class AAM(object):
    r"""
    Active Appearance Models.

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
    def __init__(self, images, group=None, verbose=False,
                 features=no_op, transform=DifferentiablePiecewiseAffine,
                 diagonal=None, scales=(0.5, 1.0), scale_features=True,
                 max_shape_components=None, forgetting_factor=1.0,
                 max_appearance_components=None, batch_size=None):
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
        self.scale_features = scale_features
        self.diagonal = diagonal
        self.scales = scales
        self.forgetting_factor = forgetting_factor
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components
        self.reference_shape = None
        self.shape_models = []
        self.appearance_models = []

        # Train AAM
        self._train(images, group=group, verbose=verbose, increment=False,
                    batch_size=batch_size)

    def _train(self, images, group=None, verbose=False, increment=False,
               batch_size=None):
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
            lowest to highest level
        """
        # If batch_size is not None, then we may have a generator, else we
        # assume we have a list.
        if batch_size is not None:
            # Create a generator of fixed sized batches. Will still work even
            # on an infinite list.
            image_batches = batch(images, batch_size)
        else:
            image_batches = [list(images)]

        for k, image_batch in enumerate(image_batches):
            # After the first batch, we are incrementing the model
            if k > 0:
                increment = True

            if verbose:
                print('Computing batch {}'.format(k))

            if not increment:
                checks.check_trilist(image_batch[0], self.transform,
                                     group=group)
                # Normalize images and compute reference shape
                self.reference_shape, image_batch = normalization_wrt_reference_shape(
                    image_batch, group, self.diagonal, verbose=verbose)
            else:
                # We are incrementing, so rescale to existing reference shape
                image_batch = rescale_images_to_reference_shape(
                    image_batch, group, self.reference_shape,
                    verbose=verbose)

            # build models at each scale
            if verbose:
                print_dynamic('- Building models\n')

            feature_images = []
            # for each pyramid level (low --> high)
            for j in range(self.n_levels):
                if verbose:
                    if len(self.scales) > 1:
                        level_str = '  - Level {}: '.format(j)
                    else:
                        level_str = '  - '
                else:
                    level_str = None

                # obtain image representation
                if self.scale_features:
                    if j == 0:
                        # Compute features at highest level
                        feature_images = compute_features(image_batch,
                                                          self.features[0],
                                                          level_str=level_str,
                                                          verbose=verbose)
                    # Scale features at other levels
                    level_images = scale_images(feature_images,
                                                self.scales[j],
                                                level_str=level_str,
                                                verbose=verbose)
                else:
                    # scale images and compute features at other levels
                    scaled_images = scale_images(image_batch, self.scales[j],
                                                 level_str=level_str,
                                                 verbose=verbose)
                    level_images = compute_features(scaled_images,
                                                    self.features[j],
                                                    level_str=level_str,
                                                    verbose=verbose)

                # Extract potentially rescaled shapes
                level_shapes = [i.landmarks[group].lms for i in level_images]

                # Build the shape model
                if not increment:
                    if j == 0:
                        if verbose:
                            print_dynamic('{}Building shape model'.format(level_str))
                        shape_model = self._build_shape_model(
                            level_shapes, self.max_shape_components[j], j)
                        # Store shape model
                        self.shape_models.append(shape_model)
                    else:
                        # Copy shape model
                        self.shape_models.append(deepcopy(shape_model))
                else:
                    # Compute aligned shapes
                    aligned_shapes = align_shapes(level_shapes)
                    # Increment shape model
                    self.shape_models[j].increment(
                        aligned_shapes,
                        forgetting_factor=self.forgetting_factor)
                    if self.max_shape_components is not None:
                        self.shape_models[j].trim_components(
                            self.max_appearance_components[j])

                # Obtain warped images - we use a scaled version of the
                # reference shape, computed here. This is because the mean
                # moves when we are incrementing, and we need a consistent
                # reference frame.
                scaled_reference_shape = Scale(self.scales[j], n_dims=2).apply(
                    self.reference_shape)
                warped_images = self._warp_images(level_images, level_shapes,
                                                  scaled_reference_shape,
                                                  j, level_str, verbose)

                # obtain appearance model
                if verbose:
                    print_dynamic('{}Building appearance model'.format(level_str))

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
                    self.appearance_models[j].increment(warped_images)
                    # trim appearance model if required
                    if self.max_appearance_components is not None:
                        self.appearance_models[j].trim_components(
                            self.max_appearance_components[j])

                if verbose:
                    print_dynamic('{}Done\n'.format(level_str))

    def _build_shape_model(self, shapes, max_components, level):
        return build_shape_model(shapes, max_components=max_components)

    def _warp_images(self, images, shapes, reference_shape, level, level_str,
                     verbose):
        reference_frame = build_reference_frame(reference_shape)
        return warp_images(images, shapes, reference_frame, self.transform,
                           level_str=level_str, verbose=verbose)

    @property
    def n_levels(self):
        """
        The number of scale levels of the AAM.

        :type: `int`
        """
        return len(self.scales)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.
        :type: `string`
        """
        return 'Active Appearance Model'

    def instance(self, shape_weights=None, appearance_weights=None, level=-1):
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
        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        image : :map:`Image`
            The novel AAM instance.
        """
        sm = self.shape_models[level]
        am = self.appearance_models[level]

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

        return self._instance(level, shape_instance, appearance_instance)

    def random_instance(self, level=-1):
        r"""
        Generates a novel random instance of the AAM.

        Parameters
        -----------
        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        image : :map:`Image`
            The novel AAM instance.
        """
        sm = self.shape_models[level]
        am = self.appearance_models[level]

        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = (np.random.randn(sm.n_active_components) *
                         sm.eigenvalues[:sm.n_active_components]**0.5)
        shape_instance = sm.instance(shape_weights)
        appearance_weights = (np.random.randn(am.n_active_components) *
                              am.eigenvalues[:am.n_active_components]**0.5)
        appearance_instance = am.instance(appearance_weights)

        return self._instance(level, shape_instance, appearance_instance)

    def _instance(self, level, shape_instance, appearance_instance):
        template = self.appearance_models[level].mean()
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
            If `int`, then the number of sliders per level is the minimum
            between `n_parameters` and the number of active components per
            level.
            If `list` of `int`, then a number of sliders is defined per level.
            If ``None``, all the active components per level will have a slider.
        n_appearance_parameters : `int` or `list` of `int` or None, optional
            The number of appearance principal components to be used for the
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
        from menpofit.visualize import visualize_aam
        visualize_aam(self, n_shape_parameters=n_shape_parameters,
                      n_appearance_parameters=n_appearance_parameters,
                      parameters_bounds=parameters_bounds,
                      figure_size=figure_size, mode=mode)

    # TODO: fix me!
    def __str__(self):
        out = "{}\n - {} training images.\n".format(self._str_title,
                                                    self.n_training_images)
        # small strings about number of channels, channels string and downscale
        n_channels = []
        down_str = []
        for j in range(self.n_levels):
            n_channels.append(
                self.appearance_models[j].template_instance.n_channels)
            if j == self.n_levels - 1:
                down_str.append('(no downscale)')
            else:
                down_str.append('(downscale by {})'.format(
                    self.downscale**(self.n_levels - j - 1)))
        # string about features and channels
        if self.pyramid_on_features:
            feat_str = "- Feature is {} with ".format(
                name_of_callable(self.features))
            if n_channels[0] == 1:
                ch_str = ["channel"]
            else:
                ch_str = ["channels"]
        else:
            feat_str = []
            ch_str = []
            for j in range(self.n_levels):
                feat_str.append("- Feature is {} with ".format(
                    name_of_callable(self.features[j])))
                if n_channels[j] == 1:
                    ch_str.append("channel")
                else:
                    ch_str.append("channels")
        out = "{} - {} Warp.\n".format(out, name_of_callable(self.transform))
        if self.n_levels > 1:
            if self.scaled_shape_models:
                out = "{} - Gaussian pyramid with {} levels and downscale " \
                      "factor of {}.\n   - Each level has a scaled shape " \
                      "model (reference frame).\n".format(out, self.n_levels,
                                                          self.downscale)

            else:
                out = "{} - Gaussian pyramid with {} levels and downscale " \
                      "factor of {}:\n   - Shape models (reference frames) " \
                      "are not scaled.\n".format(out, self.n_levels,
                                                 self.downscale)
            if self.pyramid_on_features:
                out = "{}   - Pyramid was applied on feature space.\n   " \
                      "{}{} {} per image.\n".format(out, feat_str,
                                                    n_channels[0], ch_str[0])
                if not self.scaled_shape_models:
                    out = "{}   - Reference frames of length {} " \
                          "({} x {}C, {} x {}C)\n".format(
                        out,
                        self.appearance_models[0].n_features,
                        self.appearance_models[0].template_instance.n_true_pixels(),
                        n_channels[0],
                        self.appearance_models[0].template_instance._str_shape,
                        n_channels[0])
            else:
                out = "{}   - Features were extracted at each pyramid " \
                      "level.\n".format(out)
            for i in range(self.n_levels - 1, -1, -1):
                out = "{}   - Level {} {}: \n".format(out, self.n_levels - i,
                                                      down_str[i])
                if not self.pyramid_on_features:
                    out = "{}     {}{} {} per image.\n".format(
                        out, feat_str[i], n_channels[i], ch_str[i])
                if (self.scaled_shape_models or
                        (not self.pyramid_on_features)):
                    out = "{}     - Reference frame of length {} " \
                          "({} x {}C, {} x {}C)\n".format(
                        out, self.appearance_models[i].n_features,
                        self.appearance_models[i].template_instance.n_true_pixels(),
                        n_channels[i],
                        self.appearance_models[i].template_instance._str_shape,
                        n_channels[i])
                out = "{0}     - {1} shape components ({2:.2f}% of " \
                      "variance)\n     - {3} appearance components " \
                      "({4:.2f}% of variance)\n".format(
                    out, self.shape_models[i].n_components,
                    self.shape_models[i].variance_ratio() * 100,
                    self.appearance_models[i].n_components,
                    self.appearance_models[i].variance_ratio() * 100)
        else:
            if self.pyramid_on_features:
                feat_str = [feat_str]
            out = "{0} - No pyramid used:\n   {1}{2} {3} per image.\n" \
                  "   - Reference frame of length {4} ({5} x {6}C, " \
                  "{7} x {8}C)\n   - {9} shape components ({10:.2f}% of " \
                  "variance)\n   - {11} appearance components ({12:.2f}% of " \
                  "variance)\n".format(
                out, feat_str[0], n_channels[0], ch_str[0],
                self.appearance_models[0].n_features,
                self.appearance_models[0].template_instance.n_true_pixels(),
                n_channels[0],
                self.appearance_models[0].template_instance._str_shape,
                n_channels[0], self.shape_models[0].n_components,
                self.shape_models[0].variance_ratio() * 100,
                self.appearance_models[0].n_components,
                self.appearance_models[0].variance_ratio() * 100)
        return out


# TODO: document me!
class PatchAAM(AAM):
    r"""
    Patch based Based Active Appearance Model class.

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

    scales : `int` or float` or list of those
    scale_shapes : `boolean`
    scale_features : `boolean`
    """

    def __init__(self, shape_models, appearance_models, reference_shape,
                 patch_shape, features, scales, scale_shapes, scale_features,
                 transform):
        super(PatchAAM, self).__init__(shape_models, appearance_models,
                                       reference_shape, transform, features,
                                       scales, scale_shapes, scale_features)
        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.transform = DifferentiableThinPlateSplines
        self.patch_shape = patch_shape
        self.features = features
        self.reference_shape = reference_shape
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features

    @property
    def _str_title(self):
        return 'Patch-Based Active Appearance Model'

    def _instance(self, level, shape_instance, appearance_instance):
        template = self.appearance_models[level].mean
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

    # TODO: fix me!
    def __str__(self):
        out = super(PatchAAM, self).__str__()
        out_splitted = out.splitlines()
        out_splitted[0] = self._str_title
        out_splitted.insert(5, "   - Patch size is {}W x {}H.".format(
            self.patch_shape[1], self.patch_shape[0]))
        return '\n'.join(out_splitted)


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

    scales : `int` or float` or list of those
    scale_shapes : `boolean`
    scale_features : `boolean`
    """

    def __init__(self, shape_models, appearance_models, reference_shape,
                 transform, features, scales, scale_shapes, scale_features,
                 n_landmarks):
        super(LinearAAM, self).__init__(shape_models, appearance_models,
                                        reference_shape, transform, features,
                                        scales, scale_shapes, scale_features)
        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.transform = transform
        self.features = features
        self.reference_shape = reference_shape
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.n_landmarks = n_landmarks

    # TODO: implement me!
    def _instance(self, level, shape_instance, appearance_instance):
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

    # TODO: implement me!
    def __str__(self):
        raise NotImplemented


# TODO: document me!
class LinearPatchAAM(AAM):
    r"""
    Linear Patch based Active Appearance Model class.

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

    scales : `int` or float` or list of those
    scale_shapes : `boolean`
    scale_features : `boolean`
    n_landmarks: `int`
    """

    def __init__(self, shape_models, appearance_models, reference_shape,
                 patch_shape, features, scales, scale_shapes, scale_features,
                 n_landmarks, transform):
        super(LinearPatchAAM, self).__init__(shape_models, appearance_models,
                                             reference_shape, transform,
                                             features, scales, scale_shapes,
                                             scale_features)
        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.transform = DifferentiableThinPlateSplines
        self.patch_shape = patch_shape
        self.features = features
        self.reference_shape = reference_shape
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.n_landmarks = n_landmarks

    # TODO: implement me!
    def _instance(self, level, shape_instance, appearance_instance):
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

    # TODO: implement me!
    def __str__(self):
        raise NotImplemented


# TODO: document me!
class PartsAAM(AAM):
    r"""
    Parts based Active Appearance Model class.

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

    normalize_parts: `callable`
    scales : `int` or float` or list of those
    scale_shapes : `boolean`
    scale_features : `boolean`
    """

    def __init__(self, shape_models, appearance_models, reference_shape,
                 patch_shape, features, normalize_parts, scales, scale_shapes,
                 scale_features, transform):
        super(PartsAAM, self).__init__(shape_models, appearance_models,
                                       reference_shape, transform, features,
                                       scales, scale_shapes, scale_features)
        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.patch_shape = patch_shape
        self.features = features
        self.normalize_parts = normalize_parts
        self.reference_shape = reference_shape
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features

    # TODO: implement me!
    def _instance(self, level, shape_instance, appearance_instance):
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

    # TODO: implement me!
    def __str__(self):
        raise NotImplemented
