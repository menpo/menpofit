from __future__ import division
from functools import partial
import numpy as np
from menpo.image import Image
from menpo.feature import no_op
from menpo.visualize import print_dynamic
from menpofit.math import IRLRegression, IIRLRegression
from menpofit.result import compute_normalise_point_to_point_error
from menpofit.sdm.algorithm import SupervisedDescentAlgorithm
from menpofit.visualize import print_progress
from ..result import AAMAlgorithmResult, LinearAAMAlgorithmResult


# TODO document me!
class SupervisedDescentStandardInterface(object):
    r"""
    """
    def __init__(self, appearance_model, transform, template, sampling=None):
        self.appearance_model = appearance_model
        self.transform = transform
        self.template = template

        self._build_sampling_mask(sampling)

    def _build_sampling_mask(self, sampling):
        n_true_pixels = self.template.n_true_pixels()
        n_channels = self.template.n_channels
        sampling_mask = np.zeros(n_true_pixels, dtype=np.bool)

        if sampling is None:
            sampling = 1
        sampling_pattern = xrange(0, n_true_pixels, sampling)
        sampling_mask[sampling_pattern] = 1

        self.i_mask = np.nonzero(np.tile(
            sampling_mask[None, ...], (n_channels, 1)).flatten())[0]

    @property
    def shape_model(self):
        return self.transform.pdm.model

    @property
    def n(self):
        return self.transform.n_parameters

    @property
    def m(self):
        return self.appearance_model.n_active_components

    def warp(self, image):
        return image.warp_to_mask(self.template.mask,
                                  self.transform)

    def algorithm_result(self, image, shape_parameters,
                         appearance_parameters=None, gt_shape=None):
        return AAMAlgorithmResult(
            image, self, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# TODO document me!
class SupervisedDescentLinearInterface(SupervisedDescentStandardInterface):
    r"""
    """
    @property
    def shape_model(self):
        return self.transform.model

    def algorithm_result(self, image, shape_parameters,
                         appearance_parameters=None, gt_shape=None):
        return LinearAAMAlgorithmResult(
            image, self, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# TODO document me!
class SupervisedDescentPatchInterface(SupervisedDescentStandardInterface):
    r"""
    """
    def __init__(self, appearance_model, transform, template, sampling=None,
                 patch_size=(17, 17), patch_normalisation=no_op):
        self.patch_size = patch_size
        self.patch_normalisation = patch_normalisation

        super(SupervisedDescentPatchInterface, self).__init__(
            appearance_model, transform, template, sampling=sampling)

    def _build_sampling_mask(self, sampling):
        if sampling is None:
            sampling = np.ones(self.patch_size, dtype=np.bool)

        image_shape = self.template.pixels.shape
        image_mask = np.tile(sampling[None, None, None, ...],
                             image_shape[:3] + (1, 1))
        self.i_mask = np.nonzero(image_mask.flatten())[0]

    @property
    def shape_model(self):
        return self.transform.model

    def warp(self, image):
        parts = image.extract_patches(self.transform.target,
                                      patch_size=self.patch_size,
                                      as_single_array=True)
        parts = self.patch_normalisation(parts)
        return Image(parts, copy=False)


def _weights_for_target(transform, target):
    transform.set_target(target)
    return transform.as_vector()


# TODO document me!
def obtain_parametric_delta_x(gt_shapes, current_shapes, transform):
    # initialize current and delta parameters arrays
    n_samples = len(gt_shapes) * len(current_shapes[0])
    gt_params = np.empty((n_samples, transform.n_parameters))
    delta_params = np.empty_like(gt_params)

    k = 0
    for gt_s, c_s in zip(gt_shapes, current_shapes):
        # Compute and cache ground truth parameters
        c_gt_params = _weights_for_target(transform, gt_s)
        for s in c_s:
            gt_params[k] = c_gt_params

            current_params = _weights_for_target(transform, s)
            delta_params[k] = c_gt_params - current_params

            k += 1

    return delta_params, gt_params


class ParametricSupervisedDescentAlgorithm(SupervisedDescentAlgorithm):
    r"""
    """
    def __init__(self, aam_interface, n_iterations=3,
                 compute_error=compute_normalise_point_to_point_error,
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

    def _train(self, images, gt_shapes, current_shapes, increment=False,
               prefix='', verbose=False):

        if not increment:
            # Reset the regressors
            self.regressors = []

        n_perturbations = len(current_shapes[0])
        template_shape = gt_shapes[0]

        # obtain delta_x and gt_x (parameters rather than shapes)
        delta_x, gt_x = obtain_parametric_delta_x(gt_shapes, current_shapes,
                                                  self.transform)

        # Cascaded Regression loop
        for k in range(self.n_iterations):
            # generate regression data
            features = self._generate_features(
                images, current_shapes,
                prefix='{}(Iteration {}) - '.format(prefix, k),
                verbose=verbose)

            if verbose:
                print_dynamic('{}(Iteration {}) - Performing regression'.format(
                    prefix, k))

            if not increment:
                r = self._regressor_cls()
                r.train(features, delta_x)
                self.regressors.append(r)
            else:
                self.regressors[k].increment(features, delta_x)

            # Estimate delta_points
            estimated_delta_x = self.regressors[k].predict(features)
            if verbose:
                self._print_regression_info(template_shape, gt_shapes,
                                            n_perturbations, delta_x,
                                            estimated_delta_x, k,
                                            prefix=prefix)

            j = 0
            for shapes in current_shapes:
                for s in shapes:
                    # Estimate parameters
                    edx = estimated_delta_x[j]
                    # Current parameters
                    cx = _weights_for_target(self.transform, s) + edx

                    # Uses less memory to find updated target shape
                    self.transform.from_vector_inplace(cx)
                    # Update current shape inplace
                    s.from_vector_inplace(self.transform.target.as_vector())

                    delta_x[j] = gt_x[j] - cx
                    j += 1

        return current_shapes

    def _generate_features(self, images, current_shapes, prefix='',
                           verbose=False):
        # Initialize features array - since current_shapes is a list of lists
        # we need to know the total size
        n_samples = len(images) * len(current_shapes[0])
        features = np.empty((n_samples,) + self.a_bar_m.shape)

        wrap = partial(print_progress,
                       prefix='{}Computing features'.format(prefix),
                       end_with_newline=not prefix, verbose=verbose)

        # initialize sample counter
        k = 0
        for img, img_shapes in wrap(zip(images, current_shapes)):
            for s in img_shapes:
                self.transform.set_target(s)
                # Assumes that the transform is correctly set
                features[k] = self._compute_features(img)

                k += 1

        return features

    def run(self, image, initial_shape, gt_shape=None, **kwargs):
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # Cascaded Regression loop
        for r in self.regressors:
            # Assumes that the transform is correctly set
            features = self._compute_features(image)

            # solve for increments on the shape parameters
            dx = r.predict(features)

            # We need to update the transform to set the state for the warping
            # of the image above.
            new_x = p_list[-1] + dx
            self.transform.from_vector_inplace(new_x)
            p_list.append(new_x)

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, gt_shape=gt_shape)

    def _print_regression_info(self, template_shape, gt_shapes, n_perturbations,
                               delta_x, estimated_delta_x, level_index,
                               prefix=''):
        print_dynamic('{}(Iteration {}) - Calculating errors'.format(
            prefix, level_index))
        errors = []
        for j, (dx, edx) in enumerate(zip(delta_x, estimated_delta_x)):
            self.transform.from_vector_inplace(dx)
            s1 = self.transform.target
            self.transform.from_vector_inplace(edx)
            s2 = self.transform.target

            gt_s = gt_shapes[np.floor_divide(j, n_perturbations)]
            errors.append(self._compute_error(s1, s2, gt_s))
        mean = np.mean(errors)
        std = np.std(errors)
        median = np.median(errors)
        print_dynamic('{}(Iteration {}) - Training error -> '
                      'mean: {:.4f}, std: {:.4f}, median: {:.4f}.\n'.
                      format(prefix, level_index, mean, std, median))


# TODO: document me!
class MeanTemplate(ParametricSupervisedDescentAlgorithm):
    r"""
    """
    def _compute_features(self, image):
        i = self.interface.warp(image)
        i_m = i.as_vector()[self.interface.i_mask]
        return i_m - self.a_bar_m


# TODO: document me!
class MeanTemplateNewton(MeanTemplate):
    r"""
    """
    def __init__(self, aam_interface, n_iterations=3,
                 compute_error=compute_normalise_point_to_point_error,
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
                 compute_error=compute_normalise_point_to_point_error,
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

    def _compute_features(self, image):
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
                 compute_error=compute_normalise_point_to_point_error,
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
                 compute_error=compute_normalise_point_to_point_error,
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

    def _compute_features(self, image):
        i = self.interface.warp(image)
        i_m = i.as_vector()[self.interface.i_mask]
        # Project image onto the appearance model
        return self.project(i_m)


# TODO: document me!
class AppearanceWeightsNewton(AppearanceWeights):
    r"""
    """
    def __init__(self, aam_interface, n_iterations=3,
                 compute_error=compute_normalise_point_to_point_error,
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
                 compute_error=compute_normalise_point_to_point_error,
                 eps=10**-5, alpha=0, alpha2=0, bias=True):
        super(AppearanceWeightsGaussNewton, self).__init__(
            aam_interface, n_iterations=n_iterations,
            compute_error=compute_error, eps=eps)

        self._regressor_cls = partial(IIRLRegression, alpha=alpha,
                                      alpha2=alpha2, bias=bias)
