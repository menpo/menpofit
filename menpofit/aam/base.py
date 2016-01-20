from __future__ import division
import warnings
import numpy as np

from menpo.feature import no_op
from menpo.visualize import print_dynamic
from menpo.model import PCAModel
from menpo.transform import Scale
from menpo.shape import mean_pointcloud
from menpo.base import name_of_callable

from menpofit import checks
from menpofit.aam.algorithm.lk import (LucasKanadeStandardInterface,
                                       LucasKanadePatchInterface,
                                       LucasKanadeLinearInterface)
from menpofit.modelinstance import OrthoPDM
from menpofit.transform import (DifferentiableThinPlateSplines,
                                DifferentiablePiecewiseAffine, OrthoMDTransform,
                                LinearOrthoMDTransform)
from menpofit.base import batch
from menpofit.builder import (
    build_reference_frame, build_patch_reference_frame,
    compute_features, scale_images, warp_images,
    align_shapes, rescale_images_to_reference_shape, densify_shapes,
    extract_patches, MenpoFitBuilderWarning, compute_reference_shape)


class AAM(object):
    r"""
    Class for training a multi-level holistic Active Appearance Model.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The `list` of training images.
    group : `str` or ``None``, optional
        The landmark group that will be used to train the AAM. Note that all
        the training images need to have the specified landmark group.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for building the AAM. If
        ``None``, then the mean shape will be used.
    holistic_features : `closure`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before warping the images to the
        reference shape. Please refer to `menpo.feature` for a list of
        potential features.
    diagonal : `int` or ``None``, optional
        This parameter is used to normalize the scale of the training images
        so that the extracted features are in correspondence. The
        normalization is performed by rescaling all the training images so
        that the diagonal of their groundtruth shapes' bounding boxes
        equals to the provided value. The reference scale gets rescaled as
        well. If ``None``, then the images are rescaled with respect to the
        reference shape's diagonal.
    scales : `tuple` of `float`, optional
        The scale value of each level. They must provided in ascending order,
        i.e. from lowest to highest scale.
    transform : `menpofit.transform.DifferentiablePiecewiseAffine`, optional
        A differential warp transform object.
    shape_model_cls : `menpofit.modelinstance.OrthoPDM` or subclass, optional
        The class to be used for building the shape model.
    max_shape_components : `int`, `float`, `list` of those or ``None``, optional
        The number of shape components to keep. If `int`, then it sets the exact
        number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should
        define a value per level. If a single number, then this will be
        applied to all scales. If ``None``, then all the components are kept.
    max_appearance_components : `int`, `float`, `list` of those or ``None``, optional
        The number of appearance components to keep. If `int`, then it sets the
        exact number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should define a value
        per level. If a single number, then this will be applied to all
        scales. If ``None``, then all the components are kept.
    verbose : `bool`, optional
        If ``True``, then the progress of building the AAM will be printed.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.
    """
    def __init__(self, images, group=None, reference_shape=None,
                 holistic_features=no_op, diagonal=None, scales=(0.5, 1.0),
                 transform=DifferentiablePiecewiseAffine,
                 shape_model_cls=OrthoPDM, max_shape_components=None,
                 max_appearance_components=None, verbose=False,
                 batch_size=None):
        # Check parameters
        checks.check_diagonal(diagonal)
        scales = checks.check_scales(scales)
        n_scales = len(scales)
        holistic_features = checks.check_callable(holistic_features, n_scales)
        shape_model_cls = checks.check_callable(shape_model_cls, n_scales)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_scales, 'max_shape_components')
        max_appearance_components = checks.check_max_components(
            max_appearance_components, n_scales, 'max_appearance_components')
        # Assign attributes
        self.holistic_features = holistic_features
        self.transform = transform
        self.diagonal = diagonal
        self.scales = scales
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components
        self.reference_shape = reference_shape
        self._shape_model_cls = shape_model_cls
        self.shape_models = []
        self.appearance_models = []
        # Train AAM
        self._train(images, increment=False, group=group, verbose=verbose,
                    batch_size=batch_size)

    def _train(self, images, increment=False, group=None,
               shape_forgetting_factor=1.0, appearance_forgetting_factor=1.0,
               verbose=False, batch_size=None):
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

            # Handle holistic features
            if j == 0 and self.holistic_features[j] == no_op:
                # Saves a lot of memory
                feature_images = image_batch
            elif j == 0 or self.holistic_features[j] is not self.holistic_features[j - 1]:
                # Compute features only if this is the first pass through
                # the loop or the features at this scale are different from
                # the features at the previous scale
                feature_images = compute_features(image_batch,
                                                  self.holistic_features[j],
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
                shape_model = self._build_shape_model(scale_shapes, j)
                self.shape_models.append(shape_model)
            else:
                self._increment_shape_model(
                    scale_shapes, j, forgetting_factor=shape_forgetting_factor)

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

    def increment(self, images, group=None, shape_forgetting_factor=1.0,
                  appearance_forgetting_factor=1.0, verbose=False,
                  batch_size=None):
        r"""
        Method to increment the trained AAM with a new set of training images.

        Parameters
        ----------
        images : `list` of `menpo.image.Image`
            The `list` of training images.
        group : `str` or ``None``, optional
            The landmark group that will be used to train the AAM. Note that all
            the training images need to have the specified landmark group.
        shape_forgetting_factor : ``[0.0, 1.0]`` `float`, optional
            Forgetting factor that weights the relative contribution of new
            samples vs old samples for the shape model. If ``1.0``, all samples
            are weighted equally and, hence, the result is the exact same as
            performing batch PCA on the concatenated list of old and new
            simples. If ``<1.0``, more emphasis is put on the new samples.
        appearance_forgetting_factor : ``[0.0, 1.0]`` `float`, optional
            Forgetting factor that weights the relative contribution of new
            samples vs old samples for the appearance model. If ``1.0``,
            all samples are weighted equally and, hence, the result is the
            exact same as performing batch PCA on the concatenated list of
            old and new simples. If ``<1.0``, more emphasis is put on the new
            samples.
        verbose : `bool`, optional
            If ``True``, then the progress of building the AAM will be printed.
        batch_size : `int` or ``None``, optional
            If an `int` is provided, then the training is performed in an
            incremental fashion on image batches of size equal to the provided
            value. If ``None``, then the training is performed directly on the
            all the images.
        """
        return self._train(
                images, increment=True, group=group, verbose=verbose,
                shape_forgetting_factor=shape_forgetting_factor,
                appearance_forgetting_factor=appearance_forgetting_factor,
                batch_size=batch_size)

    def _build_shape_model(self, shapes, scale_index):
        return self._shape_model_cls[scale_index](
            shapes, max_n_components=self.max_shape_components[scale_index])

    def _increment_shape_model(self, shapes, scale_index,
                               forgetting_factor=None):
        self.shape_models[scale_index].increment(
            shapes, forgetting_factor=forgetting_factor,
            max_n_components=self.max_shape_components[scale_index])

    def _warp_images(self, images, shapes, reference_shape, scale_index,
                     prefix, verbose):
        reference_frame = build_reference_frame(reference_shape)
        return warp_images(images, shapes, reference_frame, self.transform,
                           prefix=prefix, verbose=verbose)

    @property
    def n_scales(self):
        """
        Returns the number of scales.

        :type: `int`
        """
        return len(self.scales)

    @property
    def _str_title(self):
        return 'Holistic Active Appearance Model'

    def instance(self, shape_weights=None, appearance_weights=None,
                 scale_index=-1):
        r"""
        Generates a novel AAM instance given a set of shape and appearance
        weights. If no weights are provided, the mean AAM instance is
        returned.

        Parameters
        ----------
        shape_weights : ``(n_weights,)`` `ndarray` or `list` or ``None``, optional
            The weights of the shape model that will be used to create a novel
            shape instance. If ``None``, the weights are assumed to be zero,
            thus the mean shape is used.
        appearance_weights : ``(n_weights,)`` `ndarray` or `list` or ``None``, optional
            The weights of the appearance model that will be used to create a
            novel appearance instance. If ``None``, the weights are assumed
            to be zero, thus the mean appearance is used.
        scale_index : `int`, optional
            The scale to be used.

        Returns
        -------
        image : `menpo.image.Image`
            The novel AAM instance.
        """
        if shape_weights is None:
            shape_weights = [0]
        if appearance_weights is None:
            appearance_weights = [0]
        sm = self.shape_models[scale_index].model
        am = self.appearance_models[scale_index]

        shape_instance = sm.instance(shape_weights, normalized_weights=True)
        appearance_instance = am.instance(appearance_weights,
                                          normalized_weights=True)

        return self._instance(scale_index, shape_instance, appearance_instance)

    def random_instance(self, scale_index=-1):
        r"""
        Generates a novel random instance of the AAM.

        Parameters
        ----------
        scale_index : `int`, optional
            The scale to be used.

        Returns
        -------
        image : `menpo.image.Image`
            The novel AAM instance.
        """
        sm = self.shape_models[scale_index].model
        am = self.appearance_models[scale_index]

        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = np.random.randn(sm.n_active_components)
        shape_instance = sm.instance(shape_weights, normalized_weights=True)
        appearance_weights = np.random.randn(sm.n_active_components)
        appearance_instance = am.instance(appearance_weights,
                                          normalized_weights=True)

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
        Visualizes the shape models of the AAM object using an interactive
        widget.

        Parameters
        ----------
        n_parameters : `int` or `list` of `int` or ``None``, optional
            The number of shape principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            level is the minimum between `n_parameters` and the number of
            active components per level. If `list` of `int`, then a number of
            sliders is defined per level. If ``None``, all the active
            components per level will have a slider.
        parameters_bounds : ``(float, float)``, optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu. If ``'multiple'``, a slider is constructed for
            each parameter.
        figure_size : ``(int, int)``, optional
            The size of the rendered figure.
        """
        try:
            from menpowidgets import visualize_shape_model
            visualize_shape_model(
                [sm.model for sm in self.shape_models],
                n_parameters=n_parameters, parameters_bounds=parameters_bounds,
                figure_size=figure_size, mode=mode)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def view_appearance_models_widget(self, n_parameters=5,
                                      parameters_bounds=(-3.0, 3.0),
                                      mode='multiple', figure_size=(10, 8)):
        r"""
        Visualizes the appearance models of the AAM object using an
        interactive widget.

        Parameters
        ----------
        n_parameters : `int` or `list` of `int` or ``None``, optional
            The number of appearance principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            level is the minimum between `n_parameters` and the number of
            active components per level. If `list` of `int`, then a number of
            sliders is defined per level. If ``None``, all the active
            components per level will have a slider.
        parameters_bounds : ``(float, float)``, optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu. If ``'multiple'``, a slider is constructed for
            each parameter.
        figure_size : ``(int, int)``, optional
            The size of the rendered figure.
        """
        try:
            from menpowidgets import visualize_appearance_model
            visualize_appearance_model(self.appearance_models,
                                       n_parameters=n_parameters,
                                       parameters_bounds=parameters_bounds,
                                       figure_size=figure_size, mode=mode)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def view_aam_widget(self, n_shape_parameters=5, n_appearance_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        r"""
        Visualizes the AAM using an interactive widget.

        Parameters
        ----------
        n_shape_parameters : `int` or `list` of `int` or ``None``, optional
            The number of shape principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            level is the minimum between `n_parameters` and the number of
            active components per level. If `list` of `int`, then a number of
            sliders is defined per level. If ``None``, all the active
            components per level will have a slider.
        n_appearance_parameters : `int` or `list` of `int` or ``None``, optional
            The number of appearance principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            level is the minimum between `n_parameters` and the number of
            active components per level. If `list` of `int`, then a number of
            sliders is defined per level. If ``None``, all the active
            components per level will have a slider.
        parameters_bounds : ``(float, float)``, optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu. If ``'multiple'``, a slider is constructed for
            each parameter.
        figure_size : ``(int, int)``, optional
            The size of the rendered figure.
        """
        try:
            from menpowidgets import visualize_aam
            visualize_aam(self, n_shape_parameters=n_shape_parameters,
                          n_appearance_parameters=n_appearance_parameters,
                          parameters_bounds=parameters_bounds,
                          figure_size=figure_size, mode=mode)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def build_fitter_interfaces(self, sampling):
        r"""
        Method that builds the correct Lucas-Kanade fitting interface. It
        only applies in case you wish to fit the AAM with a Lucas-Kanade
        algorithm

        Parameters
        ----------
        sampling : `int` or ``None``, optional
            The sub-sampling step of the sampling mask. If ``None``, then no
            sampling is applied on the template.
        """
        interfaces = []
        for am, sm, s in zip(self.appearance_models, self.shape_models,
                             sampling):
            template = am.mean()
            md_transform = OrthoMDTransform(
                sm, self.transform,
                source=template.landmarks['source'].lms)
            interface = LucasKanadeStandardInterface(
                am, md_transform, template, sampling=s)
            interfaces.append(interface)
        return interfaces

    def __str__(self):
        return _aam_str(self)


class MaskedAAM(AAM):
    r"""
    Class for training a multi-level patch-based Masked Active Appearance Model.
    The appearance of this model is formulated by simply masking an image
    with a patch-based mask.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The `list` of training images.
    group : `str` or ``None``, optional
        The landmark group that will be used to train the AAM. Note that all
        the training images need to have the specified landmark group.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for building the AAM. If
        ``None``, then the mean shape will be used.
    holistic_features : `closure`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before warping the images to the
        reference shape. Please refer to `menpo.feature` for a list of
        potential features.
    diagonal : `int` or ``None``, optional
        This parameter is used to normalize the scale of the training images
        so that the extracted features are in correspondence. The
        normalization is performed by rescaling all the training images so
        that the diagonal of their groundtruth shapes' bounding boxes
        equals to the provided value. The reference scale gets rescaled as
        well. If ``None``, then the images are rescaled with respect to the
        reference shape's diagonal.
    scales : `tuple` of `float`, optional
        The scale value of each level. They must provided in ascending order,
        i.e. from lowest to highest scale.
    patch_shape : ``(int, int)``, optional
        The size of the patches of the mask that is used to sample the
        appearance vectors.
    shape_model_cls : `menpofit.modelinstance.OrthoPDM` or subclass, optional
        The class to be used for building the shape model.
    max_shape_components : `int`, `float`, `list` of those or ``None``, optional
        The number of shape components to keep. If `int`, then it sets the exact
        number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should
        define a value per level. If a single number, then this will be
        applied to all scales. If ``None``, then all the components are kept.
    max_appearance_components : `int`, `float`, `list` of those or ``None``, optional
        The number of appearance components to keep. If `int`, then it sets the
        exact number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should define a value
        per level. If a single number, then this will be applied to all
        scales. If ``None``, then all the components are kept.
    verbose : `bool`, optional
        If ``True``, then the progress of building the AAM will be printed.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.
    """
    def __init__(self, images, group=None, reference_shape=None,
                 holistic_features=no_op, diagonal=None, scales=(0.5, 1.0),
                 patch_shape=(17, 17), shape_model_cls=OrthoPDM,
                 max_shape_components=None, max_appearance_components=None,
                 verbose=False, batch_size=None):
        # Check arguments
        n_scales = len(checks.check_scales(scales))
        self.patch_shape = checks.check_patch_shape(patch_shape, n_scales)
        # Call superclass
        super(MaskedAAM, self).__init__(
            images, group=group, verbose=verbose,
            reference_shape=reference_shape,
            holistic_features=holistic_features,
            transform=DifferentiableThinPlateSplines, diagonal=diagonal,
            scales=scales,  max_shape_components=max_shape_components,
            max_appearance_components=max_appearance_components,
            shape_model_cls=shape_model_cls, batch_size=batch_size)

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
        template = self.appearance_models[scale_index].mean()
        landmarks = template.landmarks['source'].lms

        reference_frame = build_patch_reference_frame(
            shape_instance, patch_shape=self.patch_shape[scale_index])

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        return appearance_instance.as_unmasked(copy=False).warp_to_mask(
            reference_frame.mask, transform, warp_landmarks=True)

    def __str__(self):
        return _aam_str(self)


class LinearAAM(AAM):
    r"""
    Class for training a multi-level Linear Active Appearance Model.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The `list` of training images.
    group : `str` or ``None``, optional
        The landmark group that will be used to train the AAM. Note that all
        the training images need to have the specified landmark group.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for building the AAM. If
        ``None``, then the mean shape will be used.
    holistic_features : `closure`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before warping the images to the
        reference shape. Please refer to `menpo.feature` for a list of
        potential features.
    diagonal : `int` or ``None``, optional
        This parameter is used to normalize the scale of the training images
        so that the extracted features are in correspondence. The
        normalization is performed by rescaling all the training images so
        that the diagonal of their groundtruth shapes' bounding boxes
        equals to the provided value. The reference scale gets rescaled as
        well. If ``None``, then the images are rescaled with respect to the
        reference shape's diagonal.
    scales : `tuple` of `float`, optional
        The scale value of each level. They must provided in ascending order,
        i.e. from lowest to highest scale.
    transform : `menpofit.transform.DifferentiableThinPlateSplines`, optional
        A differential warp transform object.
    shape_model_cls : `menpofit.modelinstance.OrthoPDM` or subclass, optional
        The class to be used for building the shape model.
    max_shape_components : `int`, `float`, `list` of those or ``None``, optional
        The number of shape components to keep. If `int`, then it sets the exact
        number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should
        define a value per level. If a single number, then this will be
        applied to all scales. If ``None``, then all the components are kept.
    max_appearance_components : `int`, `float`, `list` of those or ``None``, optional
        The number of appearance components to keep. If `int`, then it sets the
        exact number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should define a value
        per level. If a single number, then this will be applied to all
        scales. If ``None``, then all the components are kept.
    verbose : `bool`, optional
        If ``True``, then the progress of building the AAM will be printed.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.
    """
    def __init__(self, images, group=None, reference_shape=None,
                 holistic_features=no_op, diagonal=None, scales=(0.5, 1.0),
                 transform=DifferentiableThinPlateSplines,
                 shape_model_cls=OrthoPDM,  max_shape_components=None,
                 max_appearance_components=None, verbose=False,
                 batch_size=None):
        super(LinearAAM, self).__init__(
            images, group=group, verbose=verbose,
            reference_shape=reference_shape,
            holistic_features=holistic_features, transform=transform,
            diagonal=diagonal, scales=scales,
            max_shape_components=max_shape_components,
            max_appearance_components=max_appearance_components,
            shape_model_cls=shape_model_cls, batch_size=batch_size)

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

        # Build dense shape model
        max_sc = self.max_shape_components[scale_index]
        return self._shape_model_cls[scale_index](dense_shapes,
                                                  max_n_components=max_sc)

    def _increment_shape_model(self, shapes, scale_index,
                               forgetting_factor=1.0):
        aligned_shapes = align_shapes(shapes)
        dense_shapes = densify_shapes(aligned_shapes, self.reference_frame,
                                      self.transform)
        # Increment shape model
        self.shape_models[scale_index].increment(
            dense_shapes, forgetting_factor=forgetting_factor,
            max_n_components=self.max_shape_components[scale_index])

    def _warp_images(self, images, shapes, reference_shape, scale_index,
                     prefix, verbose):
        return warp_images(images, shapes, self.reference_frame,
                           self.transform, prefix=prefix,
                           verbose=verbose)

    # TODO: implement me!
    def _instance(self, scale_index, shape_instance, appearance_instance):
        raise NotImplementedError()

    # TODO: implement me!
    def view_appearance_models_widget(self, n_parameters=5,
                                      parameters_bounds=(-3.0, 3.0),
                                      mode='multiple', figure_size=(10, 8)):
        raise NotImplementedError()

    # TODO: implement me!
    def view_aam_widget(self, n_shape_parameters=5, n_appearance_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        raise NotImplementedError()

    def build_fitter_interfaces(self, sampling):
        r"""
        Method that builds the correct Lucas-Kanade fitting interface. It
        only applies in case you wish to fit the AAM with a Lucas-Kanade
        algorithm

        Parameters
        ----------
        sampling : `int` or ``None``, optional
            The sub-sampling step of the sampling mask. If ``None``, then no
            sampling is applied on the template.
        """
        interfaces = []
        for am, sm, s in zip(self.appearance_models, self.shape_models,
                             sampling):
            template = am.mean()
            # This is pretty hacky as we just steal the OrthoPDM's PCAModel
            md_transform = LinearOrthoMDTransform(
                sm.model, self.reference_shape)
            interface = LucasKanadeLinearInterface(am, md_transform,
                                                   template, sampling=s)
            interfaces.append(interface)
        return interfaces

    def __str__(self):
        return _aam_str(self)


class LinearMaskedAAM(AAM):
    r"""
    Class for training a multi-level Linear Masked Active Appearance Model.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The `list` of training images.
    group : `str` or ``None``, optional
        The landmark group that will be used to train the AAM. Note that all
        the training images need to have the specified landmark group.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for building the AAM. If
        ``None``, then the mean shape will be used.
    holistic_features : `closure`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before warping the images to the
        reference shape. Please refer to `menpo.feature` for a list of
        potential features.
    diagonal : `int` or ``None``, optional
        This parameter is used to normalize the scale of the training images
        so that the extracted features are in correspondence. The
        normalization is performed by rescaling all the training images so
        that the diagonal of their groundtruth shapes' bounding boxes
        equals to the provided value. The reference scale gets rescaled as
        well. If ``None``, then the images are rescaled with respect to the
        reference shape's diagonal.
    scales : `tuple` of `float`, optional
        The scale value of each level. They must provided in ascending order,
        i.e. from lowest to highest scale.
    patch_shape : ``(int, int)``, optional
        The size of the patches of the mask that is used to sample the
        appearance vectors.
    shape_model_cls : `menpofit.modelinstance.OrthoPDM` or subclass, optional
        The class to be used for building the shape model.
    max_shape_components : `int`, `float`, `list` of those or ``None``, optional
        The number of shape components to keep. If `int`, then it sets the exact
        number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should
        define a value per level. If a single number, then this will be
        applied to all scales. If ``None``, then all the components are kept.
    max_appearance_components : `int`, `float`, `list` of those or ``None``, optional
        The number of appearance components to keep. If `int`, then it sets the
        exact number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should define a value
        per level. If a single number, then this will be applied to all
        scales. If ``None``, then all the components are kept.
    verbose : `bool`, optional
        If ``True``, then the progress of building the AAM will be printed.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.
    """
    def __init__(self, images, group=None, reference_shape=None,
                 holistic_features=no_op, diagonal=None, scales=(0.5, 1.0),
                 patch_shape=(17, 17), shape_model_cls=OrthoPDM,
                 max_shape_components=None, max_appearance_components=None,
                 verbose=False, batch_size=None):
        # Check arguments
        n_scales = len(checks.check_scales(scales))
        self.patch_shape = checks.check_patch_shape(patch_shape, n_scales)
        # Call superclass
        super(LinearMaskedAAM, self).__init__(
            images, group=group, verbose=verbose,
            reference_shape=reference_shape,
            holistic_features=holistic_features,
            transform=DifferentiableThinPlateSplines, diagonal=diagonal,
            scales=scales,  max_shape_components=max_shape_components,
            max_appearance_components=max_appearance_components,
            shape_model_cls=shape_model_cls, batch_size=batch_size)

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
        # Build dense shape model
        max_sc = self.max_shape_components[scale_index]
        return self._shape_model_cls[scale_index](dense_shapes,
                                                  max_n_components=max_sc)

    def _increment_shape_model(self, shapes, scale_index,
                               forgetting_factor=1.0):
        aligned_shapes = align_shapes(shapes)
        dense_shapes = densify_shapes(aligned_shapes, self.reference_frame,
                                      self.transform)
        # Increment shape model
        self.shape_models[scale_index].increment(
            dense_shapes, forgetting_factor=forgetting_factor,
            max_n_components=self.max_shape_components[scale_index])

    def _warp_images(self, images, shapes, reference_shape, scale_index,
                     prefix, verbose):
        return warp_images(images, shapes, self.reference_frame,
                           self.transform, prefix=prefix,
                           verbose=verbose)

    # TODO: implement me!
    def _instance(self, scale_index, shape_instance, appearance_instance):
        raise NotImplementedError()

    # TODO: implement me!
    def view_appearance_models_widget(self, n_parameters=5,
                                      parameters_bounds=(-3.0, 3.0),
                                      mode='multiple', figure_size=(10, 8)):
        raise NotImplementedError()

    # TODO: implement me!
    def view_aam_widget(self, n_shape_parameters=5, n_appearance_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        raise NotImplementedError()

    def build_fitter_interfaces(self, sampling):
        interfaces = []
        for am, sm, s in zip(self.appearance_models, self.shape_models,
                             sampling):
            template = am.mean()
            # This is pretty hacky as we just steal the OrthoPDM's PCAModel
            md_transform = LinearOrthoMDTransform(
                sm.model, self.reference_shape)
            interface = LucasKanadeLinearInterface(am, md_transform,
                                                   template, sampling=s)
            interfaces.append(interface)
        return interfaces

    def __str__(self):
        return _aam_str(self)


# TODO: implement offsets support?
class PatchAAM(AAM):
    r"""
    Class for training a multi-level Patch-Based Active Appearance Model.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The `list` of training images.
    group : `str` or ``None``, optional
        The landmark group that will be used to train the AAM. Note that all
        the training images need to have the specified landmark group.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for building the AAM. If
        ``None``, then the mean shape will be used.
    holistic_features : `closure`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before extracting the patches. Please
        refer to `menpo.feature` for a list of potential features.
    diagonal : `int` or ``None``, optional
        This parameter is used to normalize the scale of the training images
        so that the extracted features are in correspondence. The
        normalization is performed by rescaling all the training images so
        that the diagonal of their groundtruth shapes' bounding boxes
        equals to the provided value. The reference scale gets rescaled as
        well. If ``None``, then the images are rescaled with respect to the
        reference shape's diagonal.
    scales : `tuple` of `float`, optional
        The scale value of each level. They must provided in ascending order,
        i.e. from lowest to highest scale.
    patch_shape : ``(int, int)`` or `list` of ``(int, int)``, optional
        The shape of the patches to be extracted. If a list is provided,
        then it defines a patch shape per level.
    shape_model_cls : `menpofit.modelinstance.OrthoPDM` or subclass, optional
        The class to be used for building the shape model.
    max_shape_components : `int`, `float`, `list` of those or ``None``, optional
        The number of shape components to keep. If `int`, then it sets the exact
        number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should
        define a value per level. If a single number, then this will be
        applied to all scales. If ``None``, then all the components are kept.
    max_appearance_components : `int`, `float`, `list` of those or ``None``, optional
        The number of appearance components to keep. If `int`, then it sets the
        exact number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should define a value
        per level. If a single number, then this will be applied to all
        scales. If ``None``, then all the components are kept.
    verbose : `bool`, optional
        If ``True``, then the progress of building the AAM will be printed.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.
    """
    def __init__(self, images, group=None, reference_shape=None,
                 holistic_features=no_op, diagonal=None, scales=(0.5, 1.0),
                 patch_shape=(17, 17), patch_normalisation=no_op,
                 shape_model_cls=OrthoPDM, max_shape_components=None,
                 max_appearance_components=None, verbose=False,
                 batch_size=None):
        n_scales = len(checks.check_scales(scales))
        self.patch_shape = checks.check_patch_shape(patch_shape, n_scales)
        self.patch_normalisation = patch_normalisation

        super(PatchAAM, self).__init__(
            images, group=group, verbose=verbose,
            reference_shape=reference_shape,
            holistic_features=holistic_features, transform=None,
            diagonal=diagonal, scales=scales,
            max_shape_components=max_shape_components,
            max_appearance_components=max_appearance_components,
            shape_model_cls=shape_model_cls, batch_size=batch_size)

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
                               normalise_function=self.patch_normalisation,
                               prefix=prefix, verbose=verbose)

    def _instance(self, scale_index, shape_instance, appearance_instance):
        return shape_instance, appearance_instance

    def view_appearance_models_widget(self, n_parameters=5,
                                      parameters_bounds=(-3.0, 3.0),
                                      mode='multiple', figure_size=(10, 8)):
        r"""
        Visualizes the appearance models of the AAM object using an
        interactive widget.

        Parameters
        ----------
        n_parameters : `int` or `list` of `int` or ``None``, optional
            The number of appearance principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            level is the minimum between `n_parameters` and the number of
            active components per level. If `list` of `int`, then a number of
            sliders is defined per level. If ``None``, all the active
            components per level will have a slider.
        parameters_bounds : ``(float, float)``, optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu. If ``'multiple'``, a slider is constructed for
            each parameter.
        figure_size : ``(int, int)``, optional
            The size of the rendered figure.
        """
        try:
            from menpowidgets import visualize_patch_appearance_model
            centers = [sp.model.mean() for sp in self.shape_models]
            visualize_patch_appearance_model(self.appearance_models, centers,
                                             n_parameters=n_parameters,
                                             parameters_bounds=parameters_bounds,
                                             figure_size=figure_size, mode=mode)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def view_aam_widget(self, n_shape_parameters=5, n_appearance_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        r"""
        Visualizes the AAM using an interactive widget.

        Parameters
        ----------
        n_shape_parameters : `int` or `list` of `int` or ``None``, optional
            The number of shape principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            level is the minimum between `n_parameters` and the number of
            active components per level. If `list` of `int`, then a number of
            sliders is defined per level. If ``None``, all the active
            components per level will have a slider.
        n_appearance_parameters : `int` or `list` of `int` or ``None``, optional
            The number of appearance principal components to be used for the
            parameters sliders. If `int`, then the number of sliders per
            level is the minimum between `n_parameters` and the number of
            active components per level. If `list` of `int`, then a number of
            sliders is defined per level. If ``None``, all the active
            components per level will have a slider.
        parameters_bounds : ``(float, float)``, optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu. If ``'multiple'``, a slider is constructed for
            each parameter.
        figure_size : ``(int, int)``, optional
            The size of the rendered figure.
        """
        try:
            from menpowidgets import visualize_patch_aam
            visualize_patch_aam(self, n_shape_parameters=n_shape_parameters,
                                n_appearance_parameters=n_appearance_parameters,
                                parameters_bounds=parameters_bounds,
                                figure_size=figure_size, mode=mode)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def build_fitter_interfaces(self, sampling):
        r"""
        Method that builds the correct Lucas-Kanade fitting interface. It
        only applies in case you wish to fit the AAM with a Lucas-Kanade
        algorithm

        Parameters
        ----------
        sampling : `int` or ``None``, optional
            The sub-sampling step of the sampling mask. If ``None``, then no
            sampling is applied on the template.
        """
        interfaces = []
        for j, (am, sm, s) in enumerate(zip(self.appearance_models,
                                            self.shape_models,
                                            sampling)):
            template = am.mean()
            interface = LucasKanadePatchInterface(
                am, sm, template, sampling=s,
                patch_shape=self.patch_shape[j],
                patch_normalisation=self.patch_normalisation)
            interfaces.append(interface)
        return interfaces

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
   - Appearance model class: {}
   - {} appearance components
   - Shape model class: {}
   - {} shape components"""
    for k, s in enumerate(aam.scales):
        scales_info.append(lvl_str_tmplt.format(
            s, name_of_callable(aam.holistic_features[k]),
            name_of_callable(aam.appearance_models[k]),
            aam.appearance_models[k].n_components,
            name_of_callable(aam.shape_models[k]),
            aam.shape_models[k].model.n_components))
    # Patch based AAM
    if hasattr(aam, 'patch_shape'):
        for k in range(len(scales_info)):
            scales_info[k] += '\n   - Patch shape: {}'.format(
                aam.patch_shape[k])
    scales_info = '\n'.join(scales_info)

    if aam.transform is not None:
        transform_str = 'Images warped with {} transform'.format(
            name_of_callable(aam.transform))
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
