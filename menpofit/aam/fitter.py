from __future__ import division
from menpo.transform import Scale
from menpofit.builder import (
    rescale_images_to_reference_shape, compute_features, scale_images)
from menpofit.fitter import ModelFitter
from menpofit.modelinstance import OrthoPDM
from menpofit.transform import OrthoMDTransform, LinearOrthoMDTransform
import menpofit.checks as checks
from .base import AAM, PatchAAM, LinearAAM, LinearPatchAAM, PartsAAM
from .algorithm.lk import (
    LKAAMInterface, LinearLKAAMInterface, PartsLKAAMInterface, AIC)
from .algorithm.cr import (
    CRAAMInterface, CRLinearAAMInterface, CRPartsAAMInterface, PAJ)
from .result import AAMFitterResult


# TODO: document me!
class AAMFitter(ModelFitter):
    r"""
    """
    def __init__(self, aam, n_shape=None, n_appearance=None):
        super(AAMFitter, self).__init__(aam)
        self._algorithms = []
        self._check_n_shape(n_shape)
        self._check_n_appearance(n_appearance)

    @property
    def aam(self):
        return self._model

    @property
    def algorithms(self):
        return self._algorithms

    def _check_n_appearance(self, n_appearance):
        if n_appearance is not None:
            if type(n_appearance) is int or type(n_appearance) is float:
                for am in self.aam.appearance_models:
                    am.n_active_components = n_appearance
            elif len(n_appearance) == 1 and self.aam.n_levels > 1:
                for am in self.aam.appearance_models:
                    am.n_active_components = n_appearance[0]
            elif len(n_appearance) == self.aam.n_levels:
                for am, n in zip(self.aam.appearance_models, n_appearance):
                    am.n_active_components = n
            else:
                raise ValueError('n_appearance can be an integer or a float '
                                 'or None or a list containing 1 or {} of '
                                 'those'.format(self.aam.n_levels))

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return AAMFitterResult(image, self, algorithm_results,
                               affine_correction, gt_shape=gt_shape)


# TODO: document me!
class LKAAMFitter(AAMFitter):
    r"""
    """
    def __init__(self, aam,  n_shape=None, n_appearance=None,
                 lk_algorithm_cls=AIC, sampling=None, **kwargs):
        super(LKAAMFitter, self).__init__(
            aam, n_shape=n_shape, n_appearance=n_appearance)
        self._set_up(lk_algorithm_cls, sampling, **kwargs)

    def _set_up(self, lk_algorithm_cls, sampling, **kwargs):
        for j, (am, sm) in enumerate(zip(self.aam.appearance_models,
                                         self.aam.shape_models)):

            if type(self.aam) is AAM or type(self.aam) is PatchAAM:
                # build orthonormal model driven transform
                md_transform = OrthoMDTransform(
                    sm, self.aam.transform,
                    source=am.mean().landmarks['source'].lms)
                # set up algorithm using standard aam interface
                algorithm = lk_algorithm_cls(
                    LKAAMInterface, am, md_transform, sampling=sampling,
                    **kwargs)

            elif (type(self.aam) is LinearAAM or
                  type(self.aam) is LinearPatchAAM):
                # build linear version of orthogonal model driven transform
                md_transform = LinearOrthoMDTransform(
                    sm, self.aam.n_landmarks)
                # set up algorithm using linear aam interface
                algorithm = lk_algorithm_cls(
                    LinearLKAAMInterface, am, md_transform, sampling=sampling,
                    **kwargs)

            elif type(self.aam) is PartsAAM:
                # build orthogonal point distribution model
                pdm = OrthoPDM(sm)
                # set up algorithm using parts aam interface
                algorithm = lk_algorithm_cls(
                    PartsLKAAMInterface, am, pdm,
                    sampling=sampling, patch_shape=self.aam.patch_shape[j],
                    normalize_parts=self.aam.normalize_parts, **kwargs)

            else:
                raise ValueError("AAM object must be of one of the "
                                 "following classes: {}, {}, {}, {}, "
                                 "{}".format(AAM, PatchAAM, LinearAAM,
                                             LinearPatchAAM, PartsAAM))

            # append algorithms to list
            self._algorithms.append(algorithm)


# TODO: document me!
class CRAAMFitter(AAMFitter):
    r"""
    """
    def __init__(self, aam, cr_algorithm_cls=PAJ, n_shape=None,
                 n_appearance=None, sampling=None, n_perturbations=10,
                 max_iters=6, **kwargs):
        super(CRAAMFitter, self).__init__(
            aam, n_shape=n_shape, n_appearance=n_appearance)
        self.n_perturbations = n_perturbations
        self.max_iters = checks.check_max_iters(max_iters, self.n_levels)
        self._set_up(cr_algorithm_cls, sampling, **kwargs)

    def _set_up(self, cr_algorithm_cls, sampling, **kwargs):
        for j, (am, sm) in enumerate(zip(self.aam.appearance_models,
                                         self.aam.shape_models)):

            if type(self.aam) is AAM or type(self.aam) is PatchAAM:
                # build orthonormal model driven transform
                md_transform = OrthoMDTransform(
                    sm, self.aam.transform,
                    source=am.mean().landmarks['source'].lms)
                # set up algorithm using standard aam interface
                algorithm = cr_algorithm_cls(
                    CRAAMInterface, am, md_transform, sampling=sampling,
                    max_iters=self.max_iters[j], **kwargs)

            elif (type(self.aam) is LinearAAM or
                  type(self.aam) is LinearPatchAAM):
                # build linear version of orthogonal model driven transform
                md_transform = LinearOrthoMDTransform(
                    sm, self.aam.n_landmarks)
                # set up algorithm using linear aam interface
                algorithm = cr_algorithm_cls(
                    CRLinearAAMInterface, am, md_transform,
                    sampling=sampling, max_iters=self.max_iters[j], **kwargs)

            elif type(self.aam) is PartsAAM:
                # build orthogonal point distribution model
                pdm = OrthoPDM(sm)
                # set up algorithm using parts aam interface
                algorithm = cr_algorithm_cls(
                    CRPartsAAMInterface, am, pdm,
                    sampling=sampling, max_iters=self.max_iters[j],
                    patch_shape=self.aam.patch_shape[j],
                    normalize_parts=self.aam.normalize_parts, **kwargs)

            else:
                raise ValueError("AAM object must be of one of the "
                                 "following classes: {}, {}, {}, {}, "
                                 "{}".format(AAM, PatchAAM, LinearAAM,
                                             LinearPatchAAM, PartsAAM))

            # append algorithms to list
            self._algorithms.append(algorithm)

    def train(self, images, group=None, label=None, verbose=False, **kwargs):
        # normalize images with respect to reference shape of aam
        images = rescale_images_to_reference_shape(
            images, group, label, self.reference_shape, verbose=verbose)

        # for each pyramid level (low --> high)
        for j, s in enumerate(self.scales):
            if verbose:
                if len(self.scales) > 1:
                    level_str = '  - Level {}: '.format(j)
                else:
                    level_str = '  - '

            # obtain image representation
            if s == self.scales[-1]:
                # compute features at highest level
                feature_images = compute_features(images, self.features,
                                                  level_str=level_str,
                                                  verbose=verbose)
                level_images = feature_images
            elif self.scale_features:
                # compute features at highest level
                feature_images = compute_features(images, self.features,
                                                  level_str=level_str,
                                                  verbose=verbose)
                # scale features at other levels
                level_images = scale_images(feature_images, s,
                                            level_str=level_str,
                                            verbose=verbose)
            else:
                # scale images and compute features at other levels
                scaled_images = scale_images(images, s, level_str=level_str,
                                             verbose=verbose)
                level_images = compute_features(scaled_images, self.features,
                                                level_str=level_str,
                                                verbose=verbose)

            # extract ground truth shapes for current level
            level_gt_shapes = [i.landmarks[group][label] for i in level_images]

            if j == 0:
                # generate perturbed shapes
                current_shapes = []
                for gt_s in level_gt_shapes:
                    perturbed_shapes = []
                    for _ in range(self.n_perturbations):
                        perturbed_shapes.append(self.perturb_shape(gt_s))
                    current_shapes.append(perturbed_shapes)

            # train cascaded regression algorithm
            current_shapes = self.algorithms[j].train(
                level_images, level_gt_shapes, current_shapes,
                verbose=verbose, **kwargs)

            # scale current shapes to next level resolution
            if s != self.scales[-1]:
                transform = Scale(self.scales[j+1]/s, n_dims=2)
                for image_shapes in current_shapes:
                    for shape in image_shapes:
                        transform.apply_inplace(shape)


