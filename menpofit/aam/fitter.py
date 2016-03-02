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
from .result import AAMFitterResult


# TODO: document me!
class AAMFitter(ModelFitter):
    r"""
    """
    @property
    def aam(self):
        return self._model

    def _check_n_appearance(self, n_appearance):
        checks.set_models_components(self.aam.appearance_models, n_appearance)

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return AAMFitterResult(image, self, algorithm_results,
                               affine_correction, gt_shape=gt_shape)


# TODO: document me!
class LucasKanadeAAMFitter(AAMFitter):
    r"""
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
