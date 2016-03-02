from __future__ import division
import warnings
import numpy as np
from menpo.feature import no_op
from menpo.visualize import print_dynamic
from menpo.transform import Scale
from menpo.shape import mean_pointcloud
from menpo.base import name_of_callable
from menpofit import checks
from menpofit.atm.algorithm import (ATMLKStandardInterface,
                                    ATMLKLinearInterface, ATMLKPatchInterface)
from menpofit.modelinstance import OrthoPDM
from menpofit.transform import (DifferentiableThinPlateSplines,
                                DifferentiablePiecewiseAffine, OrthoMDTransform,
                                LinearOrthoMDTransform)
from menpofit.base import batch
from menpofit.builder import (
    build_reference_frame, build_patch_reference_frame,
    compute_features, scale_images, warp_images,
    align_shapes, densify_shapes,
    extract_patches, MenpoFitBuilderWarning, compute_reference_shape)


# TODO: document me!
class ATM(object):
    r"""
    Active Template Model class.
    """
    def __init__(self, template, shapes, group=None, verbose=False,
                 reference_shape=None, holistic_features=no_op,
                 shape_model_cls=OrthoPDM,
                 transform=DifferentiablePiecewiseAffine, diagonal=None,
                 scales=(0.5, 1.0), max_shape_components=None,
                 batch_size=None):

        checks.check_diagonal(diagonal)
        n_scales = len(scales)
        scales = checks.check_scales(scales)
        holistic_features = checks.check_callable(holistic_features, n_scales)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_scales, 'max_shape_components')
        shape_model_cls = checks.check_callable(shape_model_cls, n_scales)

        self.holistic_features = holistic_features
        self.transform = transform
        self.diagonal = diagonal
        self.scales = scales
        self.max_shape_components = max_shape_components
        self.reference_shape = reference_shape
        self.shape_models = []
        self.warped_templates = []
        self._shape_model_cls = shape_model_cls

        # Train ATM
        self._train(template, shapes, increment=False, group=group,
                    verbose=verbose, batch_size=batch_size)

    def _train(self, template, shapes, increment=False, group=None,
               shape_forgetting_factor=1.0, verbose=False, batch_size=None):
        r"""
        """
        # If batch_size is not None, then we may have a generator, else we
        # assume we have a list.
        # If batch_size is not None, then we may have a generator, else we
        # assume we have a list.
        if batch_size is not None:
            # Create a generator of fixed sized batches. Will still work even
            # on an infinite list.
            shape_batches = batch(shapes, batch_size)
        else:
            shape_batches = [list(shapes)]

        for k, shape_batch in enumerate(shape_batches):
            if k == 0:
                # Rescale the template the reference shape
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
                    checks.check_trilist(shape_batch[0], self.transform)
                    self.reference_shape = compute_reference_shape(
                        shape_batch, self.diagonal, verbose=verbose)

                # Rescale the template the reference shape
                template = template.rescale_to_pointcloud(
                    self.reference_shape, group=group)

            # After the first batch, we are incrementing the model
            if k > 0:
                increment = True

            if verbose:
                print('Computing batch {}'.format(k))

            # Train each batch
            self._train_batch(template, shape_batch, increment=increment,
                              group=group,
                              shape_forgetting_factor=shape_forgetting_factor,
                              verbose=verbose)

    def _train_batch(self, template, shape_batch, increment=False, group=None,
                     shape_forgetting_factor=1.0, verbose=False):
        r"""
        Builds an Active Template Model from a list of landmarked images.
        """
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
            if j == 0 or self.holistic_features[j] is not self.holistic_features[j - 1]:
                # Compute features only if this is the first pass through
                # the loop or the features at this scale are different from
                # the features at the previous scale
                feature_images = compute_features([template],
                                                  self.holistic_features[j],
                                                  prefix=scale_prefix,
                                                  verbose=verbose)
            # handle scales
            if self.scales[j] != 1:
                # Scale feature images only if scale is different than 1
                scaled_images = scale_images(feature_images, self.scales[j],
                                             prefix=scale_prefix,
                                             verbose=verbose)
                # Extract potentially rescaled shapes
                scale_transform = Scale(scale_factor=self.scales[j],
                                        n_dims=2)
                scale_shapes = [scale_transform.apply(s)
                                for s in shape_batch]
            else:
                scaled_images = feature_images
                scale_shapes = shape_batch

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
            warped_template = self._warp_template(scaled_images[0], group,
                                                  scaled_reference_shape,
                                                  j, scale_prefix, verbose)
            self.warped_templates.append(warped_template[0])

            if verbose:
                print_dynamic('{}Done\n'.format(scale_prefix))

    def increment(self, template, shapes, group=None, verbose=False,
                  shape_forgetting_factor=1.0, batch_size=None):
        return self._train(template, shapes, group=group,
                           verbose=verbose,
                           shape_forgetting_factor=shape_forgetting_factor,
                           increment=True, batch_size=batch_size)

    def _build_shape_model(self, shapes, scale_index):
        return self._shape_model_cls[scale_index](
            shapes, max_n_components=self.max_shape_components[scale_index])

    def _increment_shape_model(self, shapes, scale_index,
                               forgetting_factor=None):
        self.shape_models[scale_index].increment(
            shapes, forgetting_factor=forgetting_factor,
            max_n_components=self.max_shape_components[scale_index])

    def _warp_template(self, template, group, reference_shape, scale_index,
                       prefix, verbose):
        reference_frame = build_reference_frame(reference_shape)
        shape = template.landmarks[group].lms
        return warp_images([template], [shape], reference_frame, self.transform,
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
        return 'Holistic Active Template Model'

    def instance(self, shape_weights=None, scale_index=-1):
        r"""
        Returns
        -------
        image : :map:`Image`
            The novel AAM instance.
        """
        if shape_weights is None:
            shape_weights = [0]

        sm = self.shape_models[scale_index].model
        template = self.warped_templates[scale_index]

        shape_instance = sm.instance(shape_weights, normalized_weights=True)
        return self._instance(shape_instance, template)

    def random_instance(self, scale_index=-1):
        r"""
        Generates a novel random instance of the ATM.

        Parameters
        -----------
        scale_index : `int`, optional
            The scale to be used.

        Returns
        -------
        image : :map:`Image`
            The novel AAM instance.
        """
        sm = self.shape_models[scale_index].model
        template = self.warped_templates[scale_index]

        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = np.random.randn(sm.n_active_components)
        shape_instance = sm.instance(shape_weights, normalized_weights=True)

        return self._instance(shape_instance, template)

    def _instance(self, shape_instance, template):
        landmarks = template.landmarks['source'].lms

        reference_frame = build_reference_frame(shape_instance)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        return template.as_unmasked(copy=False).warp_to_mask(
            reference_frame.mask, transform, warp_landmarks=True)

    def view_shape_models_widget(self, n_parameters=5,
                                 parameters_bounds=(-3.0, 3.0),
                                 mode='multiple', figure_size=(10, 8)):
        r"""
        Visualizes the shape models of the AAM object using an interactive
        widget.

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
        try:
            from menpowidgets import visualize_shape_model
            visualize_shape_model(
                [sm.model for sm in self.shape_models],
                n_parameters=n_parameters, parameters_bounds=parameters_bounds,
                figure_size=figure_size, mode=mode)
        except:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def view_atm_widget(self, n_shape_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        r"""
        Visualizes both the shape and appearance models of the AAM object using
        an interactive widget.
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
        try:
            from menpowidgets import visualize_atm
            visualize_atm(self, n_shape_parameters=n_shape_parameters,
                          parameters_bounds=parameters_bounds,
                          figure_size=figure_size, mode=mode)
        except:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def build_fitter_interfaces(self, sampling):
        interfaces = []
        for wt, sm, s in zip(self.warped_templates, self.shape_models,
                             sampling):
            md_transform = OrthoMDTransform(
                sm, self.transform,
                source=wt.landmarks['source'].lms)
            interface = ATMLKStandardInterface(
                md_transform, wt, sampling=s)
            interfaces.append(interface)
        return interfaces

    def __str__(self):
        return _atm_str(self)


# TODO: document me!
class MaskedATM(ATM):
    r"""
    Masked Based Active Appearance Model class.
    """

    def __init__(self, template, shapes, group=None, verbose=False,
                 holistic_features=no_op, diagonal=None, scales=(0.5, 1.0),
                 patch_shape=(17, 17), max_shape_components=None,
                 batch_size=None):
        self.patch_shape = checks.check_patch_shape(patch_shape, len(scales))

        super(MaskedATM, self).__init__(
            template, shapes, group=group, verbose=verbose,
            holistic_features=holistic_features,
            transform=DifferentiableThinPlateSplines, diagonal=diagonal,
            scales=scales,  max_shape_components=max_shape_components,
            batch_size=batch_size)

    def _warp_template(self, template, group, reference_shape, scale_index,
                       prefix, verbose):
        reference_frame = build_patch_reference_frame(
            reference_shape, patch_shape=self.patch_shape[scale_index])
        shape = template.landmarks[group].lms
        return warp_images([template], [shape], reference_frame, self.transform,
                           prefix=prefix, verbose=verbose)

    @property
    def _str_title(self):
        return 'Masked Active Template Model'

    def _instance(self, shape_instance, template):
        landmarks = template.landmarks['source'].lms

        reference_frame = build_patch_reference_frame(
            shape_instance, patch_shape=self.patch_shape)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        return template.as_unmasked().warp_to_mask(
            reference_frame.mask, transform, warp_landmarks=True)

    def __str__(self):
        return _atm_str(self)


# TODO: document me!
class LinearATM(ATM):
    r"""
    Linear Active Template Model class.
    """

    def __init__(self, template, shapes, group=None, verbose=False,
                 holistic_features=no_op,
                 transform=DifferentiableThinPlateSplines, diagonal=None,
                 scales=(0.5, 1.0), max_shape_components=None, batch_size=None):

        super(LinearATM, self).__init__(
            template, shapes, group=group, verbose=verbose,
            holistic_features=holistic_features, transform=transform,
            diagonal=diagonal, scales=scales,
            max_shape_components=max_shape_components, batch_size=batch_size)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.
        :type: `string`
        """
        return 'Linear Active Template Model'

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

    def _warp_template(self, template, group, reference_shape, scale_index,
                       prefix, verbose):
        shape = template.landmarks[group].lms
        return warp_images([template], [shape], self.reference_frame,
                           self.transform, prefix=prefix,
                           verbose=verbose)

    # TODO: implement me!
    def _instance(self, shape_instance, template):
        raise NotImplementedError()

    # TODO: implement me!
    def view_atm_widget(self, n_shape_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        raise NotImplementedError()

    def build_fitter_interfaces(self, sampling):
        interfaces = []
        for wt, sm, s in zip(self.warped_templates, self.shape_models,
                             sampling):
            # This is pretty hacky as we just steal the OrthoPDM's PCAModel
            md_transform = LinearOrthoMDTransform(
                sm.model, self.reference_shape)
            interface = ATMLKLinearInterface(md_transform, wt,
                                             sampling=s)
            interfaces.append(interface)
        return interfaces

    def __str__(self):
        return _atm_str(self)


# TODO: document me!
class LinearMaskedATM(ATM):
    r"""
    Linear Masked Active Template Model class.
    """

    def __init__(self, template, shapes, group=None, verbose=False,
                 holistic_features=no_op, diagonal=None, scales=(0.5, 1.0),
                 patch_shape=(17, 17), max_shape_components=None,
                 batch_size=None):
        self.patch_shape = checks.check_patch_shape(patch_shape, len(scales))

        super(LinearMaskedATM, self).__init__(
            template, shapes, group=group, verbose=verbose,
            holistic_features=holistic_features,
            transform=DifferentiableThinPlateSplines, diagonal=diagonal,
            scales=scales,  max_shape_components=max_shape_components,
            batch_size=batch_size)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.
        :type: `string`
        """
        return 'Linear Masked Active Template Model'

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

    def _warp_template(self, template, group, reference_shape, scale_index,
                       prefix, verbose):
        shape = template.landmarks[group].lms
        return warp_images([template], [shape], self.reference_frame,
                           self.transform, prefix=prefix,
                           verbose=verbose)

    # TODO: implement me!
    def _instance(self, shape_instance, template):
        raise NotImplementedError()

    # TODO: implement me!
    def view_atm_widget(self, n_shape_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        raise NotImplementedError()

    def build_fitter_interfaces(self, sampling):
        interfaces = []
        for wt, sm, s in zip(self.warped_templates, self.shape_models,
                             sampling):
            # This is pretty hacky as we just steal the OrthoPDM's PCAModel
            md_transform = LinearOrthoMDTransform(
                sm.model, self.reference_shape)
            interface = ATMLKLinearInterface(md_transform, wt,
                                             sampling=s)
            interfaces.append(interface)
        return interfaces

    def __str__(self):
        return _atm_str(self)


# TODO: document me!
# TODO: implement offsets support?
class PatchATM(ATM):
    r"""
    Patch-based Active Template Model class.
    """

    def __init__(self, template, shapes, group=None, verbose=False,
                 holistic_features=no_op, patch_normalisation=no_op,
                 diagonal=None, scales=(0.5, 1.0), patch_shape=(17, 17),
                 max_shape_components=None, batch_size=None):
        self.patch_shape = checks.check_patch_shape(patch_shape, len(scales))
        self.patch_normalisation = patch_normalisation

        super(PatchATM, self).__init__(
            template, shapes, group=group, verbose=verbose,
            holistic_features=holistic_features,
            transform=DifferentiableThinPlateSplines, diagonal=diagonal,
            scales=scales,  max_shape_components=max_shape_components,
            batch_size=batch_size)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.
        :type: `string`
        """
        return 'Patch-based Active Template Model'

    def _warp_template(self, template, group, reference_shape, scale_index,
                       prefix, verbose):
        shape = template.landmarks[group].lms
        return extract_patches([template], [shape],
                               self.patch_shape[scale_index],
                               normalise_function=self.patch_normalisation,
                               prefix=prefix, verbose=verbose)

    def _instance(self, shape_instance, template):
        return shape_instance, template

    def view_atm_widget(self, n_shape_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        try:
            from menpowidgets import visualize_patch_atm
            visualize_patch_atm(self, n_shape_parameters=n_shape_parameters,
                                parameters_bounds=parameters_bounds,
                                figure_size=figure_size, mode=mode)
        except:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def build_fitter_interfaces(self, sampling):
        interfaces = []
        for j, (wt, sm, s) in enumerate(zip(self.warped_templates,
                                            self.shape_models,
                                            sampling)):
            interface = ATMLKPatchInterface(
                sm, wt, sampling=s,
                patch_shape=self.patch_shape[j],
                patch_normalisation=self.patch_normalisation)
            interfaces.append(interface)
        return interfaces

    def __str__(self):
        return _atm_str(self)


def _atm_str(atm):
    if atm.diagonal is not None:
        diagonal = atm.diagonal
    else:
        y, x = atm.reference_shape.range()
        diagonal = np.sqrt(x ** 2 + y ** 2)

    # Compute scale info strings
    scales_info = []
    lvl_str_tmplt = r"""  - Scale {}
   - Holistic feature: {}
   - Template shape: {}
   - Shape model class: {}
   - {} shape components"""
    for k, s in enumerate(atm.scales):
        scales_info.append(lvl_str_tmplt.format(
            s, name_of_callable(atm.holistic_features[k]),
            atm.warped_templates[k].shape,
            name_of_callable(atm.shape_models[k]),
            atm.shape_models[k].model.n_components))
    # Patch based ATM
    if hasattr(atm, 'patch_shape'):
        for k in range(len(scales_info)):
            scales_info[k] += '\n   - Patch shape: {}'.format(
                atm.patch_shape[k])
    scales_info = '\n'.join(scales_info)

    cls_str = r"""{class_title}
 - Images warped with {transform} transform
 - Images scaled to diagonal: {diagonal:.2f}
 - Scales: {scales}
{scales_info}
""".format(class_title=atm._str_title,
           transform=name_of_callable(atm.transform),
           diagonal=diagonal,
           scales=atm.scales,
           scales_info=scales_info)
    return cls_str

HolisticATM = ATM
