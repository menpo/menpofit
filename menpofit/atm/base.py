from __future__ import division

import numpy as np
from menpo.shape import TriMesh

from menpofit.base import DeformableModel, name_of_callable
from menpofit.aam.builder import (build_patch_reference_frame,
                                  build_reference_frame)


class ATM(DeformableModel):
    r"""
    Active Template Model class.

    Parameters
    -----------
    shape_models : :map:`PCAModel` list
        A list containing the shape models of the ATM.

    warped_templates : :map:`MaskedImage` list
        A list containing the warped templates models of the ATM.

    n_training_shapes: `int`
        The number of training shapes used to build the ATM.

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

    reference_shape : :map:`PointCloud`
        The reference shape that was used to resize all training images to a
        consistent object size.

    downscale : `float`
        The downscale factor that was used to create the different pyramidal
        levels.

    scaled_shape_models : `boolean`, optional
        If ``True``, the reference frames are the mean shapes of each pyramid
        level, so the shape models are scaled.

        If ``False``, the reference frames of all levels are the mean shape of
        the highest level, so the shape models are not scaled; they have the
        same size.

        Note that from our experience, if scaled_shape_models is ``False``, AAMs
        tend to have slightly better performance.

    """
    def __init__(self, shape_models, warped_templates, n_training_shapes,
                 transform, features, reference_shape, downscale,
                 scaled_shape_models):
        DeformableModel.__init__(self, features)
        self.n_training_shapes = n_training_shapes
        self.shape_models = shape_models
        self.warped_templates = warped_templates
        self.transform = transform
        self.reference_shape = reference_shape
        self.downscale = downscale
        self.scaled_shape_models = scaled_shape_models

    @property
    def n_levels(self):
        """
        The number of multi-resolution pyramidal levels of the ATM.

        :type: `int`
        """
        return len(self.warped_templates)

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

        reference_frame = self._build_reference_frame(
            shape_instance, landmarks)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        return template.as_unmasked(copy=False).warp_to_mask(
            reference_frame.mask, transform, warp_landmarks=True)

    def _build_reference_frame(self, reference_shape, landmarks):
        if type(landmarks) == TriMesh:
            trilist = landmarks.trilist
        else:
            trilist = None
        return build_reference_frame(
            reference_shape, trilist=trilist)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.

        :type: `string`
        """
        return 'Active Template Model'

    def view_shape_models_widget(self, n_parameters=5, mode='multiple',
                                 parameters_bounds=(-3.0, 3.0),
                                 figure_size=(10, 8), style='coloured'):
        r"""
        Visualizes the shape models of the ATM object using the
        `menpo.visualize.widgets.visualize_shape_model` widget.

        Parameters
        -----------
        n_parameters : `int` or `list` of `int` or ``None``, optional
            The number of principal components to be used for the parameters
            sliders. If `int`, then the number of sliders per level is the
            minimum between `n_parameters` and the number of active components
            per level. If `list` of `int`, then a number of sliders is defined
            per level. If ``None``, all the active components per level will
            have a slider.
        mode : {``'single'``, ``'multiple'``}, optional
            If ``'single'``, then only a single slider is constructed along with
            a drop down menu. If ``'multiple'``, then a slider is constructed
            for each parameter.
        parameters_bounds : (`float`, `float`), optional
            The minimum and maximum bounds, in std units, for the sliders.
        figure_size : (`int`, `int`), optional
            The size of the plotted figures.
        style : {``'coloured'``, ``'minimal'``}, optional
            If ``'coloured'``, then the style of the widget will be coloured. If
            ``minimal``, then the style is simple using black and white colours.
        """
        from menpofit.visualize import visualize_shape_model
        visualize_shape_model(
            self.shape_models, n_parameters=n_parameters,
            parameters_bounds=parameters_bounds, figure_size=figure_size,
            mode=mode, style=style)

    def view_atm_widget(self, n_shape_parameters=5, mode='multiple',
                        parameters_bounds=(-3.0, 3.0), figure_size=(10, 8),
                        style='coloured'):
        r"""
        Visualizes the ATM object using the
        menpo.visualize.widgets.visualize_atm widget.

        Parameters
        -----------
        n_shape_parameters : `int` or `list` of `int` or ``None``, optional
            The number of principal components to be used for the shape
            parameters sliders. If `int`, then the number of sliders per level
            is the minimum between `n_parameters` and the number of active
            components per level. If `list` of `int`, then a number of sliders
            is defined per level. If ``None``, all the active components per
            level will have a slider.
        mode : {``'single'``, ``'multiple'``}, optional
            If ``'single'``, then only a single slider is constructed along with
            a drop down menu. If ``'multiple'``, then a slider is constructed
            for each parameter.
        parameters_bounds : (`float`, `float`), optional
            The minimum and maximum bounds, in std units, for the sliders.
        figure_size : (`int`, `int`), optional
            The size of the plotted figures.
        style : {``'coloured'``, ``'minimal'``}, optional
            If ``'coloured'``, then the style of the widget will be coloured. If
            ``minimal``, then the style is simple using black and white colours.
        """
        from menpofit.visualize import visualize_atm
        visualize_atm(self, n_shape_parameters=n_shape_parameters,
                      parameters_bounds=parameters_bounds,
                      figure_size=figure_size, mode=mode, style=style)

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


class PatchBasedATM(ATM):
    r"""
    Patch Based Active Template Model class.

    Parameters
    -----------
    shape_models : :map:`PCAModel` list
        A list containing the shape models of the ATM.

    warped_templates : :map:`MaskedImage` list
        A list containing the warped templates models of the ATM.

    n_training_shapes: `int`
        The number of training shapes used to build the ATM.

    patch_shape : tuple of `int`
        The shape of the patches used to build the Patch Based ATM.

    transform : :map:`PureAlignmentTransform`
        The transform used to warp the images from which the ATM was
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

    reference_shape : :map:`PointCloud`
        The reference shape that was used to resize all training images to a
        consistent object size.

    downscale : `float`
        The downscale factor that was used to create the different pyramidal
        levels.

    scaled_shape_models : `boolean`, optional
        If ``True``, the reference frames are the mean shapes of each pyramid
        level, so the shape models are scaled.

        If ``False``, the reference frames of all levels are the mean shape of
        the highest level, so the shape models are not scaled; they have the
        same size.

        Note that from our experience, if ``scaled_shape_models`` is ``False``,
        AAMs tend to have slightly better performance.

    """
    def __init__(self, shape_models, warped_templates, n_training_shapes,
                 patch_shape, transform, features, reference_shape,
                 downscale, scaled_shape_models):
        super(PatchBasedATM, self).__init__(
            shape_models, warped_templates, n_training_shapes, transform,
            features, reference_shape, downscale, scaled_shape_models)
        self.patch_shape = patch_shape

    def _build_reference_frame(self, reference_shape, landmarks):
        return build_patch_reference_frame(
            reference_shape, patch_shape=self.patch_shape)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.

        :type: `string`
        """
        return 'Patch-Based Active Template Model'

    def __str__(self):
        out = super(PatchBasedATM, self).__str__()
        out_splitted = out.splitlines()
        out_splitted[0] = self._str_title
        out_splitted.insert(5, "   - Patch size is {}W x {}H.".format(
            self.patch_shape[1], self.patch_shape[0]))
        return '\n'.join(out_splitted)
