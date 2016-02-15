from __future__ import division
import numpy as np
from copy import deepcopy

from menpo.base import name_of_callable
from menpo.transform import AlignmentUniformScale
from menpo.image import BooleanImage

from menpofit.fitter import ModelFitter, noisy_shape_from_bounding_box
from menpofit.sdm import SupervisedDescentFitter
import menpofit.checks as checks
from menpofit.result import MultiScaleParametricIterativeResult

from .algorithm.lk import WibergInverseCompositional
from .algorithm.sd import ProjectOutNewton
from .result import AAMResult


class AAMFitter(ModelFitter):
    r"""
    Abstract class for defining an AAM fitter.
    """
    @property
    def aam(self):
        r"""
        The trained AAM model.

        :type: :map:`AAM` or `subclass`
        """
        return self._model

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return AAMResult(results=algorithm_results, scales=self.aam.scales,
                         affine_correction=affine_correction, image=image,
                         gt_shape=gt_shape)


class LucasKanadeAAMFitter(AAMFitter):
    r"""
    Class for defining an AAM fitter using the Lucas-Kanade optimization.

    Parameters
    ----------
    aam : :map:`AAM` or `subclass`
        The trained AAM model.
    lk_algorithm_cls : `class`, optional
        The Lukas-Kanade optimisation algorithm that will get applied. The
        possible algorithms are:

        ============================================== =====================
        Class                                          Method
        ============================================== =====================
        :map:`AlternatingForwardCompositional`         Alternating
        :map:`AlternatingInverseCompositional`
        :map:`ModifiedAlternatingForwardCompositional` Modified Alternating
        :map:`ModifiedAlternatingInverseCompositional`
        :map:`ProjectOutForwardCompositional`          Project-Out
        :map:`ProjectOutInverseCompositional`
        :map:`SimultaneousForwardCompositional`        Simultaneous
        :map:`SimultaneousInverseCompositional`
        :map:`WibergForwardCompositional`              Wiberg
        :map:`WibergInverseCompositional`
        ============================================== =====================

    n_shape : `int` or `list` or ``None``, optional
        The number of shape components that will be used. If `int`, then the
        provided value will be applied on all scales. If `list`, then it
        defines a value per scale. If ``None``, then all the components will
        be used.
    n_appearance : `int` or `list` or ``None``, optional
        The number of appearance components that will be used. If `int`,
        then the provided value will be applied on all scales. If `list`, then
        it defines a value per scale. If ``None``, then all the components will
        be used.
    sampling : `list` of `int` or `ndarray` or ``None``
        It defines a sampling mask per scale. If `int`, then it defines the
        sub-sampling step of the sampling mask. If `ndarray`, then it
        explicitly defines the sampling mask. If ``None``, then no
        sub-sampling is applied.
    """
    def __init__(self, aam, lk_algorithm_cls=WibergInverseCompositional,
                 n_shape=None, n_appearance=None, sampling=None):
        self._model = aam
        # Check parameters
        checks.set_models_components(aam.shape_models, n_shape)
        checks.set_models_components(aam.appearance_models, n_appearance)
        self._sampling = checks.check_sampling(sampling, aam.n_scales)
        # Set up algorithm
        self._set_up(lk_algorithm_cls)

    def _set_up(self, lk_algorithm_cls):
        interfaces = self.aam.build_fitter_interfaces(self._sampling)
        self.algorithms = [lk_algorithm_cls(interface)
                           for interface in interfaces]

    def appearance_reconstructions(self, appearance_parameters,
                                   n_iters_per_scale):
        r"""
        Method that generates the appearance reconstructions given a set of
        appearance parameters. This is to be combined with a :map:`AAMResult`
        object, in order to generate the appearance reconstructions of a
        fitting procedure.

        Parameters
        ----------
        appearance_parameters : `list` of ``(n_params,)`` `ndarray`
            A set of appearance parameters per fitting iteration. It can be
            retrieved as a property of an :map:`AAMResult` object.
        n_iters_per_scale : `list` of `int`
            The number of iterations per scale. This is necessary in order to
            figure out which appearance parameters correspond to the model of
            each scale. It can be retrieved as a property of a :map:`AAMResult`
            object.

        Returns
        -------
        appearance_reconstructions : `list` of `menpo.image.Image`
            `List` of the appearance reconstructions that correspond to the
            provided parameters.
        """
        return self.aam.appearance_reconstructions(
                appearance_parameters=appearance_parameters,
                n_iters_per_scale=n_iters_per_scale)

    def warped_images(self, image, shapes):
        r"""
        Given an input test image and a list of shapes, it warps the image
        into the shapes. This is useful for generating the warped images of a
        fitting procedure stored within an :map:`AAMResult`.

        Parameters
        ----------
        image : `menpo.image.Image` or `subclass`
            The input image to be warped.
        shapes : `list` of `menpo.shape.PointCloud`
            The list of shapes in which the image will be warped. The shapes
            are obtained during the iterations of a fitting procedure.

        Returns
        -------
        warped_images : `list` of `menpo.image.MaskedImage` or `ndarray`
            The warped images.
        """
        return self.algorithms[-1].interface.warped_images(image=image,
                                                           shapes=shapes)

    def __str__(self):
        # Compute scale info strings
        scales_info = []
        lvl_str_tmplt = r"""   - Scale {}
     - {} active shape components
     - {} similarity transform components
     - {} active appearance components"""
        for k, s in enumerate(self.scales):
            scales_info.append(lvl_str_tmplt.format(
                    s,
                    self.aam.shape_models[k].n_active_components,
                    self.aam.shape_models[k].n_global_parameters,
                    self.aam.appearance_models[k].n_active_components))
        scales_info = '\n'.join(scales_info)

        cls_str = r"""{class_title}
 - Scales: {scales}
{scales_info}
    """.format(class_title=self.algorithms[0].__str__(),
               scales=self.scales,
               scales_info=scales_info)
        return self.aam.__str__() + cls_str


class SupervisedDescentAAMFitter(SupervisedDescentFitter):
    r"""
    Class for training a multi-scale cascaded-regression Supervised Descent AAM
    fitter.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The `list` of training images.
    aam : :map:`AAM` or `subclass`
        The trained AAM model.
    group : `str` or ``None``, optional
        The landmark group that will be used to train the fitter. If ``None`` and
        the images only have a single landmark group, then that is the one
        that will be used. Note that all the training images need to have the
        specified landmark group.
    bounding_box_group_glob : `glob` or ``None``, optional
        Glob that defines the bounding boxes to be used for training. If
        ``None``, then the bounding boxes of the ground truth shapes are used.
    n_shape : `int` or `list` or ``None``, optional
        The number of shape components that will be used. If `int`, then the
        provided value will be applied on all scales. If `list`, then it
        defines a value per scale. If ``None``, then all the components will
        be used.
    n_appearance : `int` or `list` or ``None``, optional
        The number of appearance components that will be used. If `int`,
        then the provided value will be applied on all scales. If `list`, then
        it defines a value per scale. If ``None``, then all the components will
        be used.
    sampling : `list` of `int` or `ndarray` or ``None``
        It defines a sampling mask per scale. If `int`, then it defines the
        sub-sampling step of the sampling mask. If `ndarray`, then it explicitly
        defines the sampling mask. If ``None``, then no sub-sampling is applied.
    sd_algorithm_cls : `class`, optional
        The Supervised Descent algorithm to be used. The possible algorithms
        are:

        =================================== ============= =====================
        Class                               Features      Regression
        =================================== ============= =====================
        :map:`MeanTemplateNewton`           Mean Template :map:`IRLRegression`
        :map:`MeanTemplateGaussNewton`                    :map:`IIRLRegression`
        :map:`ProjectOutNewton`             Project-Out   :map:`IRLRegression`
        :map:`ProjectOutGaussNewton`                      :map:`IIRLRegression`
        :map:`AppearanceWeightsNewton`      App. Weights  :map:`IRLRegression`
        :map:`AppearanceWeightsGaussNewton`               :map:`IIRLRegression`
        =================================== ============= =====================
    n_iterations : `int` or `list` of `int`, optional
        The number of iterations (cascades) of each level. If `list`, it must
        specify a value per scale. If `int`, then it defines the total number of
        iterations (cascades) over all scales.
    n_perturbations : `int` or ``None``, optional
        The number of perturbations to be generated from the provided bounding
        boxes.
    perturb_from_gt_bounding_box : `callable`, optional
        The function that will be used to generate the perturbations.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.
    verbose : `bool`, optional
        If ``True``, then the progress of training will be printed.
    """
    def __init__(self, images, aam, group=None, bounding_box_group_glob=None,
                 n_shape=None, n_appearance=None, sampling=None,
                 sd_algorithm_cls=ProjectOutNewton,
                 n_iterations=6, n_perturbations=30,
                 perturb_from_gt_bounding_box=noisy_shape_from_bounding_box,
                 batch_size=None, verbose=False):
        self.aam = aam
        # Check parameters
        checks.set_models_components(aam.shape_models, n_shape)
        checks.set_models_components(aam.appearance_models, n_appearance)
        self._sampling = checks.check_sampling(sampling, aam.n_scales)

        # patch_feature and patch_shape are not actually
        # used because they are fully defined by the AAM already. Therefore,
        # we just leave them as their 'defaults' because they won't be used.
        super(SupervisedDescentAAMFitter, self).__init__(
            images, group=group, bounding_box_group_glob=bounding_box_group_glob,
            reference_shape=self.aam.reference_shape,
            sd_algorithm_cls=sd_algorithm_cls,
            holistic_features=self.aam.holistic_features,
            diagonal=self.aam.diagonal,
            scales=self.aam.scales, n_iterations=n_iterations,
            n_perturbations=n_perturbations,
            perturb_from_gt_bounding_box=perturb_from_gt_bounding_box,
            batch_size=batch_size, verbose=verbose)

    def _setup_algorithms(self):
        interfaces = self.aam.build_fitter_interfaces(self._sampling)
        self.algorithms = [self._sd_algorithm_cls[j](
                               interface, n_iterations=self.n_iterations[j])
                           for j, interface in enumerate(interfaces)]

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return MultiScaleParametricIterativeResult(
                results=algorithm_results, scales=self.aam.scales,
                affine_correction=affine_correction, image=image,
                gt_shape=gt_shape)

    def warped_images(self, image, shapes):
        r"""
        Given an input test image and a list of shapes, it warps the image
        into the shapes. This is useful for generating the warped images of a
        fitting procedure stored within an :map:`AAMResult`.

        Parameters
        ----------
        image : `menpo.image.Image` or `subclass`
            The input image to be warped.
        shapes : `list` of `menpo.shape.PointCloud`
            The list of shapes in which the image will be warped. The shapes
            are obtained during the iterations of a fitting procedure.

        Returns
        -------
        warped_images : `list` of `menpo.image.MaskedImage` or `ndarray`
            The warped images.
        """
        return self.algorithms[-1].interface.warped_images(image=image,
                                                           shapes=shapes)

    def __str__(self):
        is_custom_perturb_func = (self._perturb_from_gt_bounding_box !=
                                  noisy_shape_from_bounding_box)
        if is_custom_perturb_func:
            is_custom_perturb_func = name_of_callable(
                    self._perturb_from_gt_bounding_box)
        regressor_cls = self.algorithms[0]._regressor_cls

        # Compute scale info strings
        scales_info = []
        lvl_str_tmplt = r"""   - Scale {}
     - {} iterations"""
        for k, s in enumerate(self.scales):
            scales_info.append(lvl_str_tmplt.format(s, self.n_iterations[k]))
        scales_info = '\n'.join(scales_info)

        cls_str = r"""Supervised Descent Method
 - Regression performed using the {reg_alg} algorithm
   - Regression class: {reg_cls}
 - Perturbations generated per shape: {n_perturbations}
 - Custom perturbation scheme used: {is_custom_perturb_func}
 - Scales: {scales}
{scales_info}
""".format(
                reg_alg=name_of_callable(self._sd_algorithm_cls[0]),
                reg_cls=name_of_callable(regressor_cls),
                n_perturbations=self.n_perturbations,
                is_custom_perturb_func=is_custom_perturb_func,
                scales=self.scales,
                scales_info=scales_info)

        return self.aam.__str__() + cls_str


def holistic_sampling_from_scale(aam, scale=0.35):
    r"""
    Function that generates a sampling reference mask given a scale value.

    Parameters
    ----------
    aam : :map:`AAM` or subclass
        The trained AAM.
    scale : `float`, optional
        The scale value.

    Returns
    -------
    true_positions : `ndarray` of `bool`
        The array that has ``True`` for the points of the reference shape that
        belong to the new mask.
    boolean_image : `menpo.image.BooleanImage`
        The boolean image of the mask.
    """
    reference = aam.appearance_models[0].mean()
    scaled_reference = reference.rescale(scale)

    t = AlignmentUniformScale(scaled_reference.landmarks['source'].lms,
                              reference.landmarks['source'].lms)
    new_indices = np.require(np.round(t.apply(
        scaled_reference.mask.true_indices())), dtype=np.int)

    modified_mask = deepcopy(reference.mask.pixels)
    modified_mask[:] = False
    modified_mask[:, new_indices[:, 0], new_indices[:, 1]] = True

    true_positions = np.nonzero(
        modified_mask[:, reference.mask.mask].ravel())[0]

    return true_positions, BooleanImage(modified_mask[0])


def holistic_sampling_from_step(aam, step=8):
    r"""
    Function that generates a sampling reference mask given a sampling step.

    Parameters
    ----------
    aam : :map:`AAM` or subclass
        The trained AAM.
    step : `int`, optional
        The sampling step.

    Returns
    -------
    true_positions : `ndarray` of `bool`
        The array that has ``True`` for the points of the reference shape that
        belong to the new mask.
    boolean_image : `menpo.image.BooleanImage`
        The boolean image of the mask.
    """
    reference = aam.appearance_models[0].mean()

    n_true_pixels = reference.n_true_pixels()
    true_positions = np.zeros(n_true_pixels, dtype=np.bool)
    sampling = xrange(0, n_true_pixels, step)
    true_positions[sampling] = True

    modified_mask = reference.mask.copy()
    new_indices = modified_mask.true_indices()[sampling, :]
    modified_mask.mask[:] = False
    modified_mask.mask[new_indices[:, 0], new_indices[:, 1]] = True

    return true_positions, modified_mask
