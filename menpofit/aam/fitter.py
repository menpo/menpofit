from __future__ import division
import numpy as np
from copy import deepcopy

from menpo.transform import AlignmentUniformScale
from menpo.image import BooleanImage

from menpofit.fitter import ModelFitter, noisy_shape_from_bounding_box
from menpofit.sdm import SupervisedDescentFitter
import menpofit.checks as checks

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

        :type: `menpofit.aam.AAM` or subclass
        """
        return self._model

    def _check_n_appearance(self, n_appearance):
        checks.set_models_components(self.aam.appearance_models, n_appearance)

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
    aam : `menpofit.aam.base.AAM` or subclass
        The trained AAM model.
    lk_algorithm_cls : `menpofit.aam.algorithm.lk.LucasKanade` or subclass, optional
        The Lukas-Kanade optimization algorithm that will get applied. All
        possible algorithms are stored in `menpofit.aam.algorithm.lk`.
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
    sampling : `int` or ``None``, optional
        The sub-sampling step of the sampling mask. If ``None``, then no
        sampling is applied on the template.
    """
    def __init__(self, aam, lk_algorithm_cls=WibergInverseCompositional,
                 n_shape=None, n_appearance=None, sampling=None):
        self._model = aam
        self._check_n_shape(n_shape)
        self._check_n_appearance(n_appearance)
        self._sampling = checks.check_sampling(sampling, aam.n_scales)
        self._set_up(lk_algorithm_cls)

    def _set_up(self, lk_algorithm_cls):
        interfaces = self.aam.build_fitter_interfaces(self._sampling)
        self.algorithms = [lk_algorithm_cls(interface)
                           for interface in interfaces]

    def appearance_reconstructions(self, appearance_parameters,
                                   n_iters_per_scale):
        r"""
        Method that generates the appearance reconstructions given a set of
        appearance parameters. This is to be combined with a
        `menpofit.aam.result.AAMResult` object, in order to generate the
        appearance reconstructions of a fitting procedure.

        Parameters
        ----------
        appearance_parameters : `list` of `ndarray`
            A set of appearance parameters per fitting iteration. It can be
            retrieved as a property of a `menpofit.aam.result.AAMResult` object.
        n_iters_per_scale : `list` of `int`
            The number of iterations per scale. This is necessary in order to
            figure out which appearance parameters correspond to the model of
            each scale. It can be retrieved as a property of a
            `menpofit.aam.result.AAMResult` object.

        Returns
        -------
        appearance_reconstructions : `list` of `menpo.image.Image`
            List of the appearance reconstructions that correspond to the
            provided parameters.
        """
        return self.aam.appearance_reconstructions(
                appearance_parameters=appearance_parameters,
                n_iters_per_scale=n_iters_per_scale)

    def warped_images(self, image, shapes):
        r"""
        Given an input test image and a list of shapes, it warps the image
        into the shapes. This is useful for generating the warped images of a
        fitting procedure.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
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


# # TODO: document me!
class SupervisedDescentAAMFitter(SupervisedDescentFitter):
    r"""
    """
    def __init__(self, images, aam, group=None, bounding_box_group_glob=None,
                 n_shape=None, n_appearance=None, sampling=None,
                 sd_algorithm_cls=ProjectOutNewton,
                 n_iterations=6, n_perturbations=30,
                 perturb_from_gt_bounding_box=noisy_shape_from_bounding_box,
                 batch_size=None, verbose=False):
        self.aam = aam
        checks.set_models_components(aam.appearance_models, n_appearance)
        checks.set_models_components(aam.shape_models, n_shape)
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


# TODO: Document me!
def holistic_sampling_from_scale(aam, scale=0.35):
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


# TODO: Document me!
def holistic_sampling_from_step(aam, step=8):
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
