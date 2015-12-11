from __future__ import division
from functools import partial
from menpofit.result import NonParametricAlgorithmResult
import numpy as np
from menpo.visualize import print_dynamic
from menpofit.visualize import print_progress


# TODO: document me!
class BaseSupervisedDescentAlgorithm(object):
    r"""
    """

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
        elif increment and not (hasattr(self, 'regressors') and self.regressors):
            raise ValueError('Algorithm must be trained before it can be '
                             'incremented.')

        n_perturbations = len(current_shapes[0])
        template_shape = gt_shapes[0]

        # obtain delta_x and gt_x
        delta_x, gt_x = self._compute_delta_x(gt_shapes, current_shapes)

        # Cascaded Regression loop
        for k in range(self.n_iterations):
            # generate regression data
            features_prefix = '{}(Iteration {}) - '.format(prefix, k)
            features = self._compute_training_features(images, gt_shapes,
                                                       current_shapes,
                                                       prefix=features_prefix,
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

            self._update_estimates(estimated_delta_x, delta_x, gt_x,
                                   current_shapes)

        return current_shapes

    def _compute_delta_x(self, gt_shapes, current_shapes):
        raise NotImplementedError()

    def _update_estimates(self, estimated_delta_x, delta_x, gt_x,
                          current_shapes):
        raise NotImplementedError()

    def _compute_training_features(self, images, gt_shapes, current_shapes,
                                   prefix='', verbose=False):
        raise NotImplementedError()

    def _compute_test_features(self, image, current_shape):
        raise NotImplementedError()

    def run(self, image, initial_shape, gt_shape=None, **kwargs):
        raise NotImplementedError()

    def _print_regression_info(self, template_shape, gt_shapes, n_perturbations,
                               delta_x, estimated_delta_x, level_index,
                               prefix=''):
        raise NotImplementedError()



# TODO: document me!
def features_per_patch(image, shape, patch_shape, features_callable):
    """r
    """
    patches = image.extract_patches(shape, patch_shape=patch_shape,
                                    as_single_array=True)

    patch_features = [features_callable(p[0]).ravel() for p in patches]
    return np.hstack(patch_features)


# TODO: document me!
def features_per_shapes(image, shapes, patch_shape, features_callable):
    """r
    """
    patch_features = [features_per_patch(image, s, patch_shape,
                                         features_callable)
                      for s in shapes]

    return np.vstack(patch_features)


# TODO: document me!
def features_per_image(images, shapes, patch_shape, features_callable,
                       prefix='', verbose=False):
    """r
    """
    wrap = partial(print_progress,
                   prefix='{}Extracting patches'.format(prefix),
                   end_with_newline=not prefix, verbose=verbose)

    patch_features = [features_per_shapes(i, shapes[j], patch_shape,
                                          features_callable)
                      for j, i in enumerate(wrap(images))]
    return np.vstack(patch_features)


def compute_non_parametric_delta_x(gt_shapes, current_shapes):
    r"""
    """
    n_x = gt_shapes[0].n_parameters
    n_gt_shapes = len(gt_shapes)
    n_current_shapes = len(current_shapes[0])

    # initialize current, ground truth and delta parameters
    gt_x = np.empty((n_gt_shapes * n_current_shapes, n_x))
    delta_x = np.empty((n_gt_shapes * n_current_shapes, n_x))

    # obtain ground truth points and compute delta points
    k = 0
    for gt_s, shapes in zip(gt_shapes, current_shapes):
        c_gt_s = gt_s.as_vector()
        for s in shapes:
            # compute ground truth shape vector
            gt_x[k] = c_gt_s
            # compute delta shape vector
            delta_x[k] = c_gt_s - s.as_vector()
            k += 1

    return delta_x, gt_x


def update_non_parametric_estimates(estimated_delta_x, delta_x, gt_x,
                                    current_shapes):
    j = 0
    for shapes in current_shapes:
        for s in shapes:
            # update current x
            current_x = s.as_vector() + estimated_delta_x[j]
            # update current shape inplace
            s._from_vector_inplace(current_x)
            # update delta_x
            delta_x[j] = gt_x[j] - current_x
            # increase index
            j += 1


def print_non_parametric_info(template_shape, gt_shapes, n_perturbations,
                              delta_x, estimated_delta_x, level_index,
                              compute_error_f, prefix=''):
    print_dynamic('{}(Iteration {}) - Calculating errors'.format(
        prefix, level_index))
    errors = []
    for j, (dx, edx) in enumerate(zip(delta_x, estimated_delta_x)):
        s1 = template_shape.from_vector(dx)
        s2 = template_shape.from_vector(edx)
        gt_s = gt_shapes[np.floor_divide(j, n_perturbations)]
        errors.append(compute_error_f(s1, s2, gt_s))
    mean = np.mean(errors)
    std = np.std(errors)
    median = np.median(errors)
    print_dynamic('{}(Iteration {}) - Training error -> '
                  'mean: {:.4f}, std: {:.4f}, median: {:.4f}.\n'.
                  format(prefix, level_index, mean, std, median))


def print_parametric_info(model, gt_shapes, n_perturbations,
                          delta_x, estimated_delta_x, level_index,
                          compute_error_f, prefix=''):
    print_dynamic('{}(Iteration {}) - Calculating errors'.format(
        prefix, level_index))
    errors = []
    for j, (dx, edx) in enumerate(zip(delta_x, estimated_delta_x)):
        model._from_vector_inplace(dx)
        s1 = model.target
        model._from_vector_inplace(edx)
        s2 = model.target

        gt_s = gt_shapes[np.floor_divide(j, n_perturbations)]
        errors.append(compute_error_f(s1, s2, gt_s))
    mean = np.mean(errors)
    std = np.std(errors)
    median = np.median(errors)
    print_dynamic('{}(Iteration {}) - Training error -> '
                  'mean: {:.4f}, std: {:.4f}, median: {:.4f}.\n'.
                  format(prefix, level_index, mean, std, median))


def compute_parametric_delta_x(gt_shapes, current_shapes, model):
    # initialize current and delta parameters arrays
    n_samples = len(gt_shapes) * len(current_shapes[0])
    gt_params = np.empty((n_samples, model.n_parameters))
    delta_params = np.empty_like(gt_params)

    k = 0
    for gt_s, c_s in zip(gt_shapes, current_shapes):
        # Compute and cache ground truth parameters
        model.set_target(gt_s)
        c_gt_params = model.as_vector()
        for s in c_s:
            gt_params[k] = c_gt_params

            model.set_target(s)
            current_params = model.as_vector()
            delta_params[k] = c_gt_params - current_params

            k += 1

    return delta_params, gt_params


def update_parametric_estimates(estimated_delta_x, delta_x, gt_x,
                                current_shapes, model):
    j = 0
    for shapes in current_shapes:
        for s in shapes:
            # Estimate parameters
            edx = estimated_delta_x[j]
            # Current parameters
            model.set_target(s)
            cx = model.as_vector() + edx
            model._from_vector_inplace(cx)

            # Update current shape inplace
            s._from_vector_inplace(model.target.as_vector().copy())

            delta_x[j] = gt_x[j] - cx
            j += 1


def build_appearance_model(images, gt_shapes, patch_shape, patch_features,
                           appearance_model_cls, verbose=False, prefix=''):
    wrap = partial(print_progress,
                   prefix='{}Extracting ground truth patches'.format(prefix),
                   end_with_newline=not prefix, verbose=verbose)
    n_images = len(images)
    # Extract patches from ground truth
    gt_patches = [features_per_patch(im, gt_s, patch_shape,
                                     patch_features)
                  for gt_s, im in wrap(zip(gt_shapes, images))]
    # Calculate appearance model from extracted gt patches
    gt_patches = np.array(gt_patches).reshape([n_images, -1])
    if verbose:
        print_dynamic('{}Building Appearance Model'.format(prefix))
    return appearance_model_cls(gt_patches)


def fit_parametric_shape(image, initial_shape, parametric_algo, gt_shape=None):
    # set current shape and initialize list of shapes
    parametric_algo.shape_model.set_target(initial_shape)
    current_shape = initial_shape.from_vector(
        parametric_algo.shape_model.target.as_vector().copy())
    shapes = [current_shape]

    # Cascaded Regression loop
    for r in parametric_algo.regressors:
        # compute regression features
        features = parametric_algo._compute_test_features(image, current_shape)

        # solve for increments on the shape vector
        dx = r.predict(features).ravel()

        # update current shape
        p = parametric_algo.shape_model.as_vector() + dx
        parametric_algo.shape_model._from_vector_inplace(p)
        current_shape = current_shape.from_vector(
            parametric_algo.shape_model.target.as_vector().copy())
        shapes.append(current_shape)

    # return algorithm result
    return NonParametricAlgorithmResult(image, shapes,
                                        gt_shape=gt_shape)


def fit_non_parametric_shape(image, initial_shape, non_parametric_algo,
                             gt_shape=None):
    # set current shape and initialize list of shapes
    current_shape = initial_shape
    shapes = [initial_shape]

    # Cascaded Regression loop
    for r in non_parametric_algo.regressors:
        # compute regression features
        features = non_parametric_algo._compute_test_features(image,
                                                              current_shape)

        # solve for increments on the shape vector
        dx = r.predict(features)

        # update current shape
        current_shape = current_shape.from_vector(
            current_shape.as_vector() + dx)
        shapes.append(current_shape)

    # return algorithm result
    return NonParametricAlgorithmResult(image, shapes,
                                        gt_shape=gt_shape)
