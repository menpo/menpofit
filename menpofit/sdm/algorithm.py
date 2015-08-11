from __future__ import division
from functools import partial
import numpy as np
from menpo.feature import no_op
from menpo.visualize import print_dynamic
from menpofit.visualize import print_progress
from menpofit.result import (
    NonParametricAlgorithmResult, compute_normalise_point_to_point_error)
from menpofit.math import IRLRegression, IIRLRegression


# TODO: document me!
class SupervisedDescentAlgorithm(object):
    r"""
    """

    def __init__(self):
        self.regressors = []

    def train(self, images, gt_shapes, current_shapes, prefix='',
              verbose=False):
        return self._train(images, gt_shapes, current_shapes, increment=False,
                           prefix=prefix, verbose=verbose)

    def increment(self, images, gt_shapes, current_shapes, prefix='',
                  verbose=False):
        return self._train(images, gt_shapes, current_shapes, increment=True,
                           prefix=prefix, verbose=verbose)

    def _train(self, images, gt_shapes, current_shapes, increment=False,
               prefix='', verbose=False):

        if not increment:
            # Reset the regressors
            self.regressors = []

        n_perturbations = len(current_shapes[0])
        template_shape = gt_shapes[0]

        # obtain delta_x and gt_x
        delta_x, gt_x = obtain_delta_x(gt_shapes, current_shapes)

        # Cascaded Regression loop
        for k in range(self.n_iterations):
            # generate regression data
            features = features_per_image(
                images, current_shapes, self.patch_size, self.features,
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
                    # update current x
                    current_x = s.as_vector() + estimated_delta_x[j]
                    # update current shape inplace
                    s.from_vector_inplace(current_x)
                    # update delta_x
                    delta_x[j] = gt_x[j] - current_x
                    # increase index
                    j += 1

        return current_shapes

    def run(self, image, initial_shape, gt_shape=None, **kwargs):
        # set current shape and initialize list of shapes
        current_shape = initial_shape
        shapes = [initial_shape]

        # Cascaded Regression loop
        for r in self.regressors:
            # compute regression features
            features = features_per_patch(image, current_shape,
                                          self.patch_size, self.features)

            # solve for increments on the shape vector
            dx = r.predict(features)

            # update current shape
            current_shape = current_shape.from_vector(
                current_shape.as_vector() + dx)
            shapes.append(current_shape)

        # return algorithm result
        return NonParametricAlgorithmResult(image, shapes,
                                            gt_shape=gt_shape)

    def _print_regression_info(self, template_shape, gt_shapes, n_perturbations,
                               delta_x, estimated_delta_x, level_index,
                               prefix=''):
        print_dynamic('{}(Iteration {}) - Calculating errors'.format(
            prefix, level_index))
        errors = []
        for j, (dx, edx) in enumerate(zip(delta_x, estimated_delta_x)):
            s1 = template_shape.from_vector(dx)
            s2 = template_shape.from_vector(edx)
            gt_s = gt_shapes[np.floor_divide(j, n_perturbations)]
            errors.append(self._compute_error(s1, s2, gt_s))
        mean = np.mean(errors)
        std = np.std(errors)
        median = np.median(errors)
        print_dynamic('{}(Iteration {}) - Training error -> '
                      'mean: {:.4f}, std: {:.4f}, median: {:.4f}.\n'.
                      format(prefix, level_index, mean, std, median))


# TODO: document me!
class Newton(SupervisedDescentAlgorithm):
    r"""
    """
    def __init__(self, features=no_op, patch_size=(17, 17), n_iterations=3,
                 compute_error=compute_normalise_point_to_point_error,
                 eps=10**-5, alpha=0, bias=True):
        super(Newton, self).__init__()

        self._regressor_cls = partial(IRLRegression, alpha=alpha, bias=bias)
        self.patch_size = patch_size
        self.features = features
        self.n_iterations = n_iterations
        self._compute_error = compute_error
        self.eps = eps


# TODO: document me!
class GaussNewton(SupervisedDescentAlgorithm):
    r"""
    """
    def __init__(self, features=no_op, patch_size=(17, 17), n_iterations=3,
                 compute_error=compute_normalise_point_to_point_error,
                 eps=10**-5, alpha=0, bias=True, alpha2=0):
        super(GaussNewton, self).__init__()

        self._regressor_cls = partial(IIRLRegression, alpha=alpha, bias=bias,
                                      alpha2=alpha2)
        self.patch_size = patch_size
        self.features = features
        self.n_iterations = n_iterations
        self._compute_error = compute_error
        self.eps = eps


# TODO: document me!
def features_per_patch(image, shape, patch_size, features_callable):
    """r
    """
    patches = image.extract_patches(shape, patch_size=patch_size,
                                    as_single_array=True)

    patch_features = [features_callable(p[0]).ravel() for p in patches]
    return np.asarray(patch_features).ravel()


# TODO: document me!
def features_per_shape(image, shapes, patch_size, features_callable):
    """r
    """
    patch_features = [features_per_patch(image, s, patch_size,
                                         features_callable)
                      for s in shapes]

    return np.asarray(patch_features)


# TODO: document me!
def features_per_image(images, shapes, patch_size, features_callable,
                       prefix='', verbose=False):
    """r
    """
    wrap = partial(print_progress,
                   prefix='{}Extracting patches'.format(prefix),
                   end_with_newline=not prefix, verbose=verbose)

    patch_features = [features_per_shape(i, shapes[j], patch_size,
                                         features_callable)
                      for j, i in enumerate(wrap(images))]
    patch_features = np.asarray(patch_features)
    return patch_features.reshape((-1, patch_features.shape[-1]))


def compute_delta_x(gt_shape, current_shapes):
    r"""
    """
    n_x = gt_shape.n_parameters
    n_current_shapes = len(current_shapes)

    # initialize ground truth and delta shape vectors
    gt_x = np.empty((n_current_shapes, n_x))
    delta_x = np.empty((n_current_shapes, n_x))

    for j, s in enumerate(current_shapes):
        # compute ground truth shape vector
        gt_x[j] = gt_shape.as_vector()
        # compute delta shape vector
        delta_x[j] = gt_x[j] - s.as_vector()

    return delta_x, gt_x


def obtain_delta_x(gt_shapes, current_shapes):
    r"""
    """
    n_x = gt_shapes[0].n_parameters
    n_gt_shapes = len(gt_shapes)
    n_current_shapes = len(current_shapes[0])

    # initialize current, ground truth and delta parameters
    gt_x = np.empty((n_gt_shapes, n_current_shapes, n_x))
    delta_x = np.empty((n_gt_shapes, n_current_shapes, n_x))

    # obtain ground truth points and compute delta points
    for j, (gt_s, shapes) in enumerate(zip(gt_shapes, current_shapes)):
        # compute ground truth par
        delta_x[j], gt_x[j] = compute_delta_x(gt_s, shapes)

    return delta_x.reshape((-1, n_x)), gt_x.reshape((-1, n_x))


def compute_features_info(image, shape, features_callable,
                          patch_size=(17, 17)):
    # TODO: include offsets support?
    patches = image.extract_patches(shape, patch_size=patch_size,
                                    as_single_array=True)

    # TODO: include offsets support?
    features_patch_size = features_callable(patches[0, 0]).shape
    features_patch_length = np.prod(features_patch_size)
    features_shape = patches.shape[:1] + features_patch_size
    features_length = np.prod(features_shape)

    return (features_patch_size, features_patch_length,
            features_shape, features_length)
