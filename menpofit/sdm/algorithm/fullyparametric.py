import numpy as np
from functools import partial
from menpo.feature import no_op
from menpofit.result import euclidean_bb_normalised_error

from .base import (BaseSupervisedDescentAlgorithm,
                   compute_parametric_delta_x, features_per_patch,
                   update_parametric_estimates, print_parametric_info,
                   build_appearance_model, fit_parametric_shape)
from menpo.model import PCAVectorModel
from menpofit.math import IIRLRegression, IRLRegression, OPPRegression
from menpofit.modelinstance import OrthoPDM
from menpofit.visualize import print_progress


class FullyParametricSDAlgorithm(BaseSupervisedDescentAlgorithm):
    r"""
    """

    def __init__(self, shape_model_cls=OrthoPDM,
                 appearance_model_cls=PCAVectorModel):
        super(FullyParametricSDAlgorithm, self).__init__()
        self.regressors = []
        self.shape_model_cls = shape_model_cls
        self.appearance_model_cls = appearance_model_cls
        self.appearance_model = None
        self.shape_model = None

    def _compute_delta_x(self, gt_shapes, current_shapes):
        # This is called first - so train shape model here
        if self.shape_model is None:
            self.shape_model = self.shape_model_cls(gt_shapes)

        return compute_parametric_delta_x(gt_shapes, current_shapes,
                                          self.shape_model)

    def _update_estimates(self, estimated_delta_x, delta_x, gt_x,
                          current_shapes):
        update_parametric_estimates(estimated_delta_x, delta_x, gt_x,
                                    current_shapes, self.shape_model)

    def _compute_training_features(self, images, gt_shapes, current_shapes,
                                   prefix='', verbose=False):
        if self.appearance_model is None:
            self.appearance_model = build_appearance_model(
                images, gt_shapes, self.patch_shape, self.patch_features,
                self.appearance_model_cls, verbose=verbose, prefix=prefix)

        wrap = partial(print_progress,
                       prefix='{}Extracting patches'.format(prefix),
                       end_with_newline=not prefix, verbose=verbose)

        features = []
        for im, shapes in wrap(zip(images, current_shapes)):
            for s in shapes:
                param_feature = self._compute_test_features(im, s)
                features.append(param_feature)

        return np.vstack(features)

    def _compute_parametric_features(self, patch):
        raise NotImplementedError()

    def _compute_test_features(self, image, current_shape):
        patch_feature = features_per_patch(
            image, current_shape, self.patch_shape, self.patch_features)
        return self._compute_parametric_features(patch_feature)

    def _print_regression_info(self, _, gt_shapes, n_perturbations,
                               delta_x, estimated_delta_x, level_index,
                               prefix=''):
        print_parametric_info(self.shape_model, gt_shapes, n_perturbations,
                              delta_x, estimated_delta_x, level_index,
                              self._compute_error, prefix=prefix)

    def run(self, image, initial_shape, gt_shape=None, **kwargs):
        return fit_parametric_shape(image, initial_shape, self,
                                    gt_shape=gt_shape)


class ParametricAppearanceProjectOut(FullyParametricSDAlgorithm):

    def _compute_parametric_features(self, patch):
        return self.appearance_model.project_out(patch.ravel())


class ParametricAppearanceWeights(FullyParametricSDAlgorithm):

    def _compute_parametric_features(self, patch):
        return self.appearance_model.project(patch.ravel())


class ParametricAppearanceMeanTemplate(FullyParametricSDAlgorithm):

    def _compute_parametric_features(self, patch):
        return patch.ravel() - self.appearance_model.mean().ravel()


class FullyParametricWeightsNewton(ParametricAppearanceWeights):
    r"""
    """

    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3, shape_model_cls=OrthoPDM,
                 appearance_model_cls=PCAVectorModel,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10 ** -5, alpha=0, bias=True):
        super(FullyParametricWeightsNewton, self).__init__(
            shape_model_cls=shape_model_cls,
            appearance_model_cls=appearance_model_cls)

        self._regressor_cls = partial(IRLRegression, alpha=alpha, bias=bias)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error
        self.eps = eps


class FullyParametricMeanTemplateNewton(ParametricAppearanceMeanTemplate):
    r"""
    """

    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3, shape_model_cls=OrthoPDM,
                 appearance_model_cls=PCAVectorModel,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10 ** -5, alpha=0, bias=True):
        super(FullyParametricMeanTemplateNewton, self).__init__(
            shape_model_cls=shape_model_cls,
            appearance_model_cls=appearance_model_cls)

        self._regressor_cls = partial(IRLRegression, alpha=alpha, bias=bias)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error
        self.eps = eps


class FullyParametricProjectOutNewton(ParametricAppearanceProjectOut):
    r"""
    """

    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3, shape_model_cls=OrthoPDM,
                 appearance_model_cls=PCAVectorModel,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10 ** -5, alpha=0, bias=True):
        super(FullyParametricProjectOutNewton, self).__init__(
            shape_model_cls=shape_model_cls,
            appearance_model_cls=appearance_model_cls)

        self._regressor_cls = partial(IRLRegression, alpha=alpha, bias=bias)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error
        self.eps = eps


# TODO: document me!
class FullyParametricProjectOutGaussNewton(ParametricAppearanceProjectOut):
    r"""
    """

    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3, shape_model_cls=OrthoPDM,
                 appearance_model_cls=PCAVectorModel,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10 ** -5, alpha=0, bias=True, alpha2=0):
        super(FullyParametricProjectOutGaussNewton, self).__init__(
            shape_model_cls=shape_model_cls,
            appearance_model_cls=appearance_model_cls)

        self._regressor_cls = partial(IIRLRegression, alpha=alpha, bias=bias,
                                      alpha2=alpha2)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error
        self.eps = eps


class FullyParametricProjectOutOPP(ParametricAppearanceProjectOut):
    r"""
    """

    def __init__(self, patch_features=no_op, patch_shape=(17, 17),
                 n_iterations=3, shape_model_cls=OrthoPDM,
                 appearance_model_cls=PCAVectorModel,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10 ** -5, bias=True):
        super(FullyParametricProjectOutOPP, self).__init__(
            shape_model_cls=shape_model_cls,
            appearance_model_cls=appearance_model_cls)

        self._regressor_cls = partial(OPPRegression, bias=bias)
        self.patch_shape = patch_shape
        self.patch_features = patch_features
        self.n_iterations = n_iterations
        self._compute_error = compute_error
        self.eps = eps
