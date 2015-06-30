from __future__ import division
import numpy as np
from menpo.image import Image
from menpo.feature import no_op
from menpo.visualize import print_dynamic, progress_bar_str
from ..result import AAMAlgorithmResult, LinearAAMAlgorithmResult


# TODO: implement more clever sampling for the standard interface?
# TODO document me!
class SupervisedDescentStandardInterface(object):
    r"""
    """
    def __init__(self, cr_aam_algorithm, sampling=None):
        self.algorithm = cr_aam_algorithm

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
    def appearance_model(self):
        return self.algorithm.appearance_model

    @property
    def template(self):
        return self.algorithm.template

    @property
    def transform(self):
        return self.algorithm.transform

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
            image, self.algorithm, shape_parameters,
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
            image, self.algorithm, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# TODO document me!
class SupervisedDescentPartsInterface(SupervisedDescentStandardInterface):
    r"""
    """
    def __init__(self, cr_aam_algorithm, sampling=None, patch_shape=(17, 17),
                 normalize_parts=no_op):
        self.algorithm = cr_aam_algorithm
        self.patch_shape = patch_shape
        self.normalize_parts = normalize_parts

        if sampling is None:
            sampling = np.ones(self.patch_shape, dtype=np.bool)

        image_shape = self.algorithm.template.pixels.shape
        image_mask = np.tile(sampling[None, None, None, ...],
                             image_shape[:3] + (1, 1))
        self.i_mask = np.nonzero(image_mask.flatten())[0]

    @property
    def shape_model(self):
        return self.transform.model

    def warp(self, image):
        parts = image.extract_patches(self.transform.target,
                                      patch_size=self.patch_shape,
                                      as_single_array=True)
        parts = self.normalize_parts(parts)
        return Image(parts)


# TODO document me!
class SupervisedDescent(object):
    r"""
    """
    def __init__(self, aam_interface, appearance_model, transform, max_iters=3,
                 eps=10**-5, **kwargs):
        # set common state for all AAM algorithms
        self.appearance_model = appearance_model
        self.template = appearance_model.mean()
        self.transform = transform
        self.max_iters = max_iters
        # TODO: Make use of eps in self.train?
        self.eps = eps
        # set interface
        self.interface = aam_interface(self, **kwargs)
        # perform pre-computations
        self._precompute()

    def _precompute(self):
        # grab appearance model mean
        a_bar = self.appearance_model.mean()
        # vectorize it and mask it
        self.a_bar_m = a_bar.as_vector()[self.interface.i_mask]

    def train(self, images, gt_shapes, current_shapes, verbose=False,
              **kwargs):
        n_images = len(images)
        n_samples_image = len(current_shapes[0])

        # set number of iterations and initialize list of regressors
        self.regressors = []

        # compute current and delta parameters from current and ground truth
        # shapes
        delta_params, current_params, gt_params = self._generate_params(
            gt_shapes, current_shapes)
        # initialize iteration counter
        k = 0

        # Cascaded Regression loop
        while k < self.max_iters:
            # generate regression data
            features = self._generate_features(images, current_params,
                                               verbose=verbose)

            # perform regression
            if verbose:
                print_dynamic('- Performing regression...')
            regressor = self._perform_regression(features, delta_params,
                                                 **kwargs)
            # add regressor to list
            self.regressors.append(regressor)

            # compute regression rmse
            estimated_delta_params = regressor(features)
            # TODO: Should print a more informative error here?
            rmse = _compute_rmse(delta_params, estimated_delta_params)
            if verbose:
                print_dynamic('- Regression RMSE is {0:.5f}.\n'.format(rmse))

            current_params += estimated_delta_params

            delta_params = gt_params - current_params
            # increase iteration counter
            k += 1

        # obtain current shapes from current parameters
        current_shapes = []
        for p in current_params:
            current_shapes.append(self.transform.from_vector(p).target)

        # convert current shapes into a list of list and return
        final_shapes = []
        for j in range(n_images):
            k = j * n_samples_image
            l = k + n_samples_image
            final_shapes.append(current_shapes[k:l])
        return final_shapes

    def _generate_params(self, gt_shapes, current_shapes):
        # initialize current and delta parameters arrays
        n_samples = len(gt_shapes) * len(current_shapes[0])
        current_params = np.empty((n_samples, self.transform.n_parameters))
        gt_params = np.empty((n_samples, self.transform.n_parameters))
        delta_params = np.empty((n_samples, self.transform.n_parameters))
        # initialize sample counter
        k = 0
        # compute ground truth and current shape parameters
        for gt_s, c_s in zip(gt_shapes, current_shapes):
            for s in c_s:
                # compute current parameters
                current_params[k] = self._compute_params(s)
                # compute ground truth parameters
                gt_params[k] = self._compute_params(gt_s)
                # compute delta parameters
                delta_params[k] = gt_params[k] - current_params[k]
                # increment counter
                k += 1

        return delta_params, current_params, gt_params

    def _compute_params(self, shape):
        self.transform.set_target(shape)
        return self.transform.as_vector()

    def _generate_features(self, images, current_params, verbose=False):
        # initialize features array
        n_images = len(images)
        n_samples = len(current_params)
        n_samples_image = int(n_samples / n_images)
        features = np.zeros((n_samples,) + self.a_bar_m.shape)

        # initialize sample counter
        k = 0
        for i in images:
            for _ in range(n_samples_image):
                if verbose:
                    print_dynamic('- Generating regression features - {'
                                  '}'.format(
                        progress_bar_str((k + 1.) / n_samples,
                                         show_bar=False)))
                # set transform
                self.transform.from_vector_inplace(current_params[k])
                # compute regression features
                f = self._compute_train_features(i)
                # add to features array
                features[k] = f
                # increment counter
                k += 1

        return features

    def run(self, image, initial_shape, gt_shape=None, **kwargs):
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter
        k = 0

        # Cascaded Regression loop
        while k < self.max_iters:
            # compute regression features
            features = self._compute_test_features(image)

            # solve for increments on the shape parameters
            dp = self.regressors[k](features)

            # update warp
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            p_list.append(self.transform.as_vector())

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, gt_shape=gt_shape)


# TODO: document me!
class SumOfSquares(SupervisedDescent):
    r"""
    """
    def _compute_train_features(self, image):
        # warp image
        i = self.interface.warp(image)
        # vectorize it and mask it
        i_m = i.as_vector()[self.interface.i_mask]
        # compute masked error
        return i_m - self.a_bar_m

    def _compute_test_features(self, image):
        # warp image
        i = self.interface.warp(image)
        # vectorize it and mask it
        i_m = i.as_vector()[self.interface.i_mask]
        # compute masked error
        return i_m - self.a_bar_m


# TODO: document me!
class SumOfSquaresNewton(SumOfSquares):
    r"""
    """
    def _perform_regression(self, features, deltas, gamma=None,
                            dtype=np.float64):
        return _supervised_newton(features, deltas, gamma=gamma, dtype=dtype)


# TODO: document me!
class SumOfSquaresGaussNewton(SumOfSquares):
    r"""
    """
    def _perform_regression(self, features, deltas, gamma=None, psi=None,
                            dtype=np.float64):
        return _supervised_gauss_newton(features, deltas, gamma=gamma,
                                        psi=psi, dtype=dtype)


# TODO: document me!
class ProjectOut(SupervisedDescent):
    r"""
    """
    def _precompute(self):
        # call super method
        super(ProjectOut, self)._precompute()
        # grab appearance model components
        A = self.appearance_model.components
        # mask them
        self.A_m = A.T[self.interface.i_mask, :]
        # compute their pseudoinverse
        self.pinv_A_m = np.linalg.pinv(self.A_m)

    def project_out(self, J):
        # project-out appearance bases from a particular vector or matrix
        return J - self.A_m.dot(self.pinv_A_m.dot(J))

    def _compute_train_features(self, image):
        # warp image
        i = self.interface.warp(image)
        # vectorize it and mask it
        i_m = i.as_vector()[self.interface.i_mask]
        # compute masked error
        e_m = i_m - self.a_bar_m
        return self.project_out(e_m)

    def _compute_test_features(self, image):
        # warp image
        i = self.interface.warp(image)
        # vectorize it and mask it
        i_m = i.as_vector()[self.interface.i_mask]
        # compute masked error
        return i_m - self.a_bar_m


# TODO: document me!
class ProjectOutNewton(ProjectOut):
    r"""
    """
    def _perform_regression(self, features, deltas, gamma=None,
                            dtype=np.float64):
        regressor = _supervised_newton(features, deltas, gamma=gamma,
                                       dtype=dtype)
        regressor.R = self.project_out(regressor.R)
        return regressor


# TODO: document me!
class ProjectOutGaussNewton(ProjectOut):
    r"""
    """
    def _perform_regression(self, features, deltas, gamma=None, psi=None,
                            dtype=np.float64):
        return _supervised_gauss_newton(features, deltas, gamma=gamma,
                                        psi=psi, dtype=dtype)


# TODO: document me!
class AppearanceWeights(SupervisedDescent):
    r"""
    """
    def _precompute(self):
        # call super method
        super(AppearanceWeights, self)._precompute()
        # grab appearance model components
        A = self.appearance_model.components
        # mask them
        A_m = A.T[self.interface.i_mask, :]
        # compute their pseudoinverse
        self.pinv_A_m = np.linalg.pinv(A_m)

    def project(self, J):
        # project a particular vector or matrix onto the appearance bases
        return self.pinv_A_m.dot(J - self.a_bar_m)

    def _compute_train_features(self, image):
        # warp image
        i = self.interface.warp(image)
        # vectorize it and mask it
        i_m = i.as_vector()[self.interface.i_mask]
        # project it onto the appearance model
        return self.project(i_m)

    def _compute_test_features(self, image):
        # warp image
        i = self.interface.warp(image)
        # vectorize it and mask it
        i_m = i.as_vector()[self.interface.i_mask]
        # project it onto the appearance model
        return self.project(i_m)


# TODO: document me!
class AppearanceWeightsNewton(AppearanceWeights):
    r"""
    """
    def _perform_regression(self, features, deltas, gamma=None,
                            dtype=np.float64):
        return _supervised_newton(features, deltas, gamma=gamma, dtype=dtype)


# TODO: document me!
class AppearanceWeightsGaussNewton(AppearanceWeights):
    r"""
    """
    def _perform_regression(self, features, deltas, gamma=None, psi=None,
                            dtype=np.float64):
        return _supervised_gauss_newton(features, deltas, gamma=gamma,
                                        psi=psi, dtype=dtype)


# TODO: document me!
class _supervised_newton(object):
    r"""
    """
    def __init__(self, features, deltas, gamma=None, dtype=np.float64):
        features = features.astype(dtype)
        deltas = deltas.astype(dtype)
        XX = features.T.dot(features)
        XT = features.T.dot(deltas)
        if gamma:
            np.fill_diagonal(XX, gamma + np.diag(XX))
        # descent direction
        self.R = np.linalg.solve(XX, XT)

    def __call__(self, features):
        return np.dot(features, self.R)


# TODO: document me!
class _supervised_gauss_newton(object):
    r"""
    """
    def __init__(self, features, deltas, gamma=None, psi=None,
                 dtype=np.float64):
        features = features.astype(dtype)
        # ridge regression
        deltas = deltas.astype(dtype)
        XX = deltas.T.dot(deltas)
        XT = deltas.T.dot(features)
        if gamma:
            np.fill_diagonal(XX, gamma + np.diag(XX))
        # average Jacobian
        self.J = np.linalg.solve(XX, XT)
        # average Hessian
        self.H = self.J.dot(self.J.T)
        if psi:
            np.fill_diagonal(self.H, psi + np.diag(self.H))
        # descent direction
        self.R = np.linalg.solve(self.H, self.J).T

    def __call__(self, features):
        return np.dot(features, self.R)


# TODO: document me!
def _compute_rmse(x1, x2):
    return np.sqrt(np.mean(np.sum((x1 - x2) ** 2, axis=1)))
