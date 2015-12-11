from __future__ import division
from functools import partial
import numpy as np
from menpofit.math import IRLRegression, IIRLRegression
from menpofit.result import euclidean_bb_normalised_error
from menpofit.sdm.algorithm.base import (
    BaseSupervisedDescentAlgorithm, compute_parametric_delta_x,
    update_parametric_estimates, print_parametric_info)
from menpofit.visualize import print_progress


class ParametricSupervisedDescentAlgorithm(BaseSupervisedDescentAlgorithm):
    r"""
    """

    def __init__(self, aam_interface, n_iterations=10,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10**-5):
        super(ParametricSupervisedDescentAlgorithm, self).__init__()

        self.interface = aam_interface
        self.n_iterations = n_iterations
        self.eps = eps

        self._compute_error = compute_error
        self._precompute()

    @property
    def appearance_model(self):
        return self.interface.appearance_model

    @property
    def transform(self):
        return self.interface.transform

    def _precompute(self):
        # Grab appearance model mean
        a_bar = self.appearance_model.mean()
        # Vectorise it and mask it
        self.a_bar_m = a_bar.as_vector()[self.interface.i_mask]

    def _compute_delta_x(self, gt_shapes, current_shapes):
        # This is called first - so train shape model here
        return compute_parametric_delta_x(gt_shapes, current_shapes,
                                          self.transform)

    def _update_estimates(self, estimated_delta_x, delta_x, gt_x,
                          current_shapes):
        update_parametric_estimates(estimated_delta_x, delta_x, gt_x,
                                    current_shapes, self.transform)

    def _compute_training_features(self, images, gt_shapes, current_shapes,
                                   prefix='', verbose=False):
        wrap = partial(print_progress,
                       prefix='{}Extracting patches'.format(prefix),
                       end_with_newline=not prefix, verbose=verbose)

        features = []
        for im, shapes in wrap(zip(images, current_shapes)):
            for s in shapes:
                param_feature = self._compute_test_features(im, s)
                features.append(param_feature)

        return np.vstack(features)

    def _compute_test_features(self, image, current_shape):
        # Make sure you call: self.transform.set_target(current_shape)
        # before calculating the warp
        raise NotImplementedError()

    def _print_regression_info(self, _, gt_shapes, n_perturbations,
                               delta_x, estimated_delta_x, level_index,
                               prefix=''):
        print_parametric_info(self.transform, gt_shapes, n_perturbations,
                              delta_x, estimated_delta_x, level_index,
                              self._compute_error, prefix=prefix)

    def run(self, image, initial_shape, gt_shape=None, **kwargs):
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # Cascaded Regression loop
        for r in self.regressors:
            # Assumes that the transform is correctly set
            features = self._compute_test_features(image,
                                                   self.transform.target)

            # solve for increments on the shape parameters
            dx = r.predict(features)

            # We need to update the transform to set the state for the warping
            # of the image above.
            new_x = p_list[-1] + dx
            self.transform._from_vector_inplace(new_x)
            p_list.append(new_x)

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, gt_shape=gt_shape)


# TODO: document me!
class MeanTemplate(ParametricSupervisedDescentAlgorithm):
    r"""
    """
    def _compute_test_features(self, image, current_shape):
        self.transform.set_target(current_shape)
        i = self.interface.warp(image)
        i_m = i.as_vector()[self.interface.i_mask]
        return i_m - self.a_bar_m


# TODO: document me!
class MeanTemplateNewton(MeanTemplate):
    r"""
    """
    def __init__(self, aam_interface, n_iterations=3,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10**-5, alpha=0, bias=True):
        super(MeanTemplateNewton, self).__init__(
            aam_interface, n_iterations=n_iterations,
            compute_error=compute_error, eps=eps)

        self._regressor_cls = partial(IRLRegression, alpha=alpha, bias=bias)


# TODO: document me!
class MeanTemplateGaussNewton(MeanTemplate):
    r"""
    """
    def __init__(self, aam_interface, n_iterations=3,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10**-5, alpha=0, alpha2=0, bias=True):
        super(MeanTemplateGaussNewton, self).__init__(
            aam_interface, n_iterations=n_iterations,
            compute_error=compute_error, eps=eps)

        self._regressor_cls = partial(IIRLRegression, alpha=alpha,
                                      alpha2=alpha2, bias=bias)


# TODO: document me!
class ProjectOut(ParametricSupervisedDescentAlgorithm):
    r"""
    """
    def _precompute(self):
        super(ProjectOut, self)._precompute()
        A = self.appearance_model.components
        self.A_m = A.T[self.interface.i_mask, :]

        self.pinv_A_m = np.linalg.pinv(self.A_m)

    def project_out(self, J):
        # Project-out appearance bases from a particular vector or matrix
        return J - self.A_m.dot(self.pinv_A_m.dot(J))

    def _compute_test_features(self, image, current_shape):
        self.transform.set_target(current_shape)
        i = self.interface.warp(image)
        i_m = i.as_vector()[self.interface.i_mask]
        # TODO: This project out could actually be cached at test time -
        # but we need to think about the best way to implement this and still
        # allow incrementing
        e_m = i_m - self.a_bar_m
        return self.project_out(e_m)


# TODO: document me!
class ProjectOutNewton(ProjectOut):
    r"""
    """
    def __init__(self, aam_interface, n_iterations=3,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10**-5, alpha=0, bias=True):
        super(ProjectOutNewton, self).__init__(
            aam_interface, n_iterations=n_iterations,
            compute_error=compute_error, eps=eps)

        self._regressor_cls = partial(IRLRegression, alpha=alpha, bias=bias)


# TODO: document me!
class ProjectOutGaussNewton(ProjectOut):
    r"""
    """
    def __init__(self, aam_interface, n_iterations=3,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10**-5, alpha=0, alpha2=0, bias=True):
        super(ProjectOutGaussNewton, self).__init__(
            aam_interface, n_iterations=n_iterations,
            compute_error=compute_error, eps=eps)

        self._regressor_cls = partial(IIRLRegression, alpha=alpha,
                                      alpha2=alpha2, bias=bias)

# TODO: document me!
class AppearanceWeights(ParametricSupervisedDescentAlgorithm):
    r"""
    """
    def _precompute(self):
        super(AppearanceWeights, self)._precompute()
        A = self.appearance_model.components
        A_m = A.T[self.interface.i_mask, :]

        self.pinv_A_m = np.linalg.pinv(A_m)

    def project(self, J):
        # Project a particular vector or matrix onto the appearance bases
        return self.pinv_A_m.dot(J - self.a_bar_m)

    def _compute_test_features(self, image, current_shape):
        self.transform.set_target(current_shape)
        i = self.interface.warp(image)
        i_m = i.as_vector()[self.interface.i_mask]
        # Project image onto the appearance model
        return self.project(i_m)


# TODO: document me!
class AppearanceWeightsNewton(AppearanceWeights):
    r"""
    """
    def __init__(self, aam_interface, n_iterations=3,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10**-5, alpha=0, bias=True):
        super(AppearanceWeightsNewton, self).__init__(
            aam_interface, n_iterations=n_iterations,
            compute_error=compute_error, eps=eps)

        self._regressor_cls = partial(IRLRegression, alpha=alpha,
                                      bias=bias)


# TODO: document me!
class AppearanceWeightsGaussNewton(AppearanceWeights):
    r"""
    """
    def __init__(self, aam_interface, n_iterations=3,
                 compute_error=euclidean_bb_normalised_error,
                 eps=10**-5, alpha=0, alpha2=0, bias=True):
        super(AppearanceWeightsGaussNewton, self).__init__(
            aam_interface, n_iterations=n_iterations,
            compute_error=compute_error, eps=eps)

        self._regressor_cls = partial(IIRLRegression, alpha=alpha,
                                      alpha2=alpha2, bias=bias)
