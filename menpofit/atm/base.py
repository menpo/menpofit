from __future__ import division
import numpy as np
from menpo.shape import TriMesh
from menpofit.transform import DifferentiableThinPlateSplines
from menpofit.base import name_of_callable
from menpofit.aam.builder import (
    build_patch_reference_frame, build_reference_frame)


class ATM(object):
    r"""
    Active Template Model class.

    Parameters
    -----------
    shape_models : :map:`PCAModel` list
        A list containing the shape models of the ATM.

    warped_templates : :map:`MaskedImage` list
        A list containing the warped templates models of the ATM.

    reference_shape : :map:`PointCloud`
        The reference shape that was used to resize all training images to a
        consistent object size.

    transform : :map:`PureAlignmentTransform`
        The transform used to warp the images from which the AAM was
        constructed.

    features : `callable` or ``[callable]``,
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

    scales : `int` or float` or list of those, optional

    scale_shapes : `boolean`

    scale_features : `boolean`

    """
    def __init__(self, shape_models, warped_templates, reference_shape,
                 transform, features, scales, scale_shapes, scale_features):
        self.shape_models = shape_models
        self.warped_templates = warped_templates
        self.transform = transform
        self.features = features
        self.reference_shape = reference_shape
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features

    @property
    def n_levels(self):
        """
        The number of scale level of the ATM.

        :type: `int`
        """
        return len(self.scales)

    # TODO: Could we directly use class names instead of this?
    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.

        :type: `string`
        """
        return 'Active Template Model'

    def instance(self, shape_weights=None, level=-1):
        r"""
        Generates a novel ATM instance given a set of shape weights. If no
        weights are provided, the mean shape instance is returned.

        Parameters
        -----------
        shape_weights : ``(n_weights,)`` `ndarray` or `float` list
            Weights of the shape model that will be used to create
            a novel shape instance. If ``None``, the mean shape
            ``(shape_weights = [0, 0, ..., 0])`` is used.

        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        image : :map:`Image`
            The novel ATM instance.
        """
        sm = self.shape_models[level]

        # TODO: this bit of logic should to be transferred down to PCAModel
        if shape_weights is None:
            shape_weights = [0]
        n_shape_weights = len(shape_weights)
        shape_weights *= sm.eigenvalues[:n_shape_weights] ** 0.5
        shape_instance = sm.instance(shape_weights)

        return self._instance(level, shape_instance)

    def random_instance(self, level=-1):
        r"""
        Generates a novel random instance of the ATM.

        Parameters
        -----------
        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        image : :map:`Image`
            The novel ATM instance.
        """
        sm = self.shape_models[level]

        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = (np.random.randn(sm.n_active_components) *
                         sm.eigenvalues[:sm.n_active_components]**0.5)
        shape_instance = sm.instance(shape_weights)

        return self._instance(level, shape_instance)

    def _instance(self, level, shape_instance):
        template = self.warped_templates[level]
        landmarks = template.landmarks['source'].lms

        if type(landmarks) == TriMesh:
            trilist = landmarks.trilist
        else:
            trilist = None
        reference_frame = build_reference_frame(shape_instance,
                                                trilist=trilist)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        return template.as_unmasked().warp_to_mask(
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

    # TODO: fix me!
    def view_atm_widget(self, n_shape_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        r"""
        Visualizes the ATM object using the
        menpo.visualize.widgets.visualize_atm widget.

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
        parameters_bounds : (`float`, `float`), optional
            The minimum and maximum bounds, in std units, for the sliders.
        mode : {``single``, ``multiple``}, optional
            If ``'single'``, only a single slider is constructed along with a
            drop down menu.
            If ``'multiple'``, a slider is constructed for each pp window.
        figure_size : (`int`, `int`), optional
            The size of the plotted figures.
        """
        from menpofit.visualize import visualize_atm
        visualize_atm(self, n_shape_parameters=n_shape_parameters,
                      parameters_bounds=parameters_bounds,
                      figure_size=figure_size, mode=mode)

    # TODO: fix me!
    def __str__(self):
        out = "{}\n - {} training shapes.\n".format(self._str_title,
                                                    self.n_training_shapes)
        # small strings about number of channels, channels string and downscale
        n_channels = []
        down_str = []
        for j in range(self.n_levels):
            n_channels.append(
                self.warped_templates[j].n_channels)
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
                          self.warped_templates[0].n_true_pixels() *
                          n_channels[0],
                          self.warped_templates[0].n_true_pixels(),
                          n_channels[0],
                          self.warped_templates[0]._str_shape,
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
                          out,
                          self.warped_templates[i].n_true_pixels() *
                                                                  n_channels[i],
                          self.warped_templates[i].n_true_pixels(),
                          n_channels[i],
                          self.warped_templates[i]._str_shape,
                          n_channels[i])
                out = "{0}     - {1} shape components ({2:.2f}% of " \
                      "variance)\n".format(
                      out, self.shape_models[i].n_components,
                      self.shape_models[i].variance_ratio() * 100)
        else:
            if self.pyramid_on_features:
                feat_str = [feat_str]
            out = "{0} - No pyramid used:\n   {1}{2} {3} per image.\n" \
                  "   - Reference frame of length {4} ({5} x {6}C, " \
                  "{7} x {8}C)\n   - {9} shape components ({10:.2f}% of " \
                  "variance)\n".format(
                  out, feat_str[0], n_channels[0], ch_str[0],
                  self.warped_templates[0].n_true_pixels() * n_channels[0],
                  self.warped_templates[0].n_true_pixels(),
                  n_channels[0],
                  self.warped_templates[0]._str_shape,
                  n_channels[0], self.shape_models[0].n_components,
                  self.shape_models[0].variance_ratio() * 100)
        return out


class PatchATM(ATM):
    r"""
    Patch Based Active Template Model class.

    Parameters
    -----------
    shape_models : :map:`PCAModel` list
        A list containing the shape models of the ATM.

    warped_templates : :map:`MaskedImage` list
        A list containing the warped templates models of the ATM.

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

    scale_shapes : `boolean`

    scale_features : `boolean`

    """
    def __init__(self, shape_models, warped_templates, reference_shape,
                 patch_shape, features, scales, scale_shapes, scale_features):
        self.shape_models = shape_models
        self.warped_templates = warped_templates
        self.transform = DifferentiableThinPlateSplines
        self.patch_shape = patch_shape
        self.features = features
        self.reference_shape = reference_shape
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features

    @property
    def _str_title(self):
        return 'Patch-Based Active Template Model'

    def _instance(self, level, shape_instance):
        template = self.warped_templates[level]
        landmarks = template.landmarks['source'].lms

        reference_frame = build_patch_reference_frame(
            shape_instance, patch_shape=self.patch_shape)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        return template.as_unmasked().warp_to_mask(
            reference_frame.mask, transform, warp_landmarks=True)

    # TODO: fix me!
    def __str__(self):
        out = super(PatchBasedATM, self).__str__()
        out_splitted = out.splitlines()
        out_splitted[0] = self._str_title
        out_splitted.insert(5, "   - Patch size is {}W x {}H.".format(
            self.patch_shape[1], self.patch_shape[0]))
        return '\n'.join(out_splitted)


# TODO: document me!
class LinearATM(ATM):
    r"""
    Linear Active Template Model class.

    Parameters
    -----------
    shape_models : :map:`PCAModel` list
        A list containing the shape models of the AAM.

    warped_templates : :map:`MaskedImage` list
        A list containing the warped templates models of the ATM.

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

    scale_shapes : `boolean`

    scale_features : `boolean`

    """
    def __init__(self, shape_models, warped_templates, reference_shape,
                 transform, features, scales, scale_shapes, scale_features,
                 n_landmarks):
        self.shape_models = shape_models
        self.warped_templates = warped_templates
        self.transform = transform
        self.features = features
        self.reference_shape = reference_shape
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.n_landmarks = n_landmarks

    # TODO: implement me!
    def _instance(self, level, shape_instance):
        raise NotImplemented

    # TODO: implement me!
    def view_atm_widget(self, n_shape_parameters=5, n_appearance_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        raise NotImplemented

    # TODO: implement me!
    def __str__(self):
        raise NotImplemented


# TODO: document me!
class LinearPatchATM(ATM):
    r"""
    Linear Patch based Active Template Model class.

    Parameters
    -----------
    shape_models : :map:`PCAModel` list
        A list containing the shape models of the AAM.

    warped_templates : :map:`MaskedImage` list
        A list containing the warped templates models of the ATM.

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

    scale_shapes : `boolean`

    scale_features : `boolean`

    n_landmarks: `int`

    """
    def __init__(self, shape_models, warped_templates, reference_shape,
                 patch_shape, features, scales, scale_shapes,
                 scale_features, n_landmarks):
        self.shape_models = shape_models
        self.warped_templates = warped_templates
        self.transform = DifferentiableThinPlateSplines
        self.patch_shape = patch_shape
        self.features = features
        self.reference_shape = reference_shape
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features
        self.n_landmarks = n_landmarks

    # TODO: implement me!
    def _instance(self, level, shape_instance):
        raise NotImplemented

    # TODO: implement me!
    def view_atm_widget(self, n_shape_parameters=5, n_appearance_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        raise NotImplemented

    # TODO: implement me!
    def __str__(self):
        raise NotImplemented


# TODO: document me!
class PartsATM(ATM):
    r"""
    Parts based Active Template Model class.

    Parameters
    -----------
    shape_models : :map:`PCAModel` list
        A list containing the shape models of the AAM.

    warped_templates : :map:`MaskedImage` list
        A list containing the warped templates models of the ATM.

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

    scale_shapes : `boolean`

    scale_features : `boolean`

    """
    def __init__(self, shape_models, warped_templates, reference_shape,
                 patch_shape, features, normalize_parts, scales,
                 scale_shapes, scale_features):
        self.shape_models = shape_models
        self.warped_templates = warped_templates
        self.patch_shape = patch_shape
        self.features = features
        self.normalize_parts = normalize_parts
        self.reference_shape = reference_shape
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features

    # TODO: implement me!
    def _instance(self, level, shape_instance):
        raise NotImplemented

    # TODO: implement me!
    def view_atm_widget(self, n_shape_parameters=5, n_appearance_parameters=5,
                        parameters_bounds=(-3.0, 3.0), mode='multiple',
                        figure_size=(10, 8)):
        raise NotImplemented

    # TODO: implement me!
    def __str__(self):
        raise NotImplemented
