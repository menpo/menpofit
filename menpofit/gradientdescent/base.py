from __future__ import division
import numpy as np
from menpofit.base import build_sampling_grid

multivariate_normal = None  # expensive, from scipy.stats

from menpofit.fitter import Fitter
from menpofit.fittingresult import SemiParametricFittingResult


# TODO: incorporate different residuals
# TODO: generalize transform prior, and map the changes to LK methods
class GradientDescent(Fitter):
    r"""
    Abstract Interface for defining Gradient Descent based fitting algorithms
    for Constrained Local Models [1]_.

    Parameters
    ----------
    classifiers : `list` of ``classifier_callable``
        The list containing the classifier that will produce the response
        maps for each landmark point.
    patch_shape : `tuple` of `int`
        The shape of the patches used to train the classifiers.
    transform : :map:`GlobalPDM` or subclass
        The global point distribution model to be used.

        .. note::

            Only :map:`GlobalPDM` and its subclasses are supported.
            :map:`PDM` is not supported at the moment.
    eps : `float`, optional
        The convergence value. When calculating the level of convergence, if
        the norm of the delta parameter updates is less than ``eps``, the
        algorithm is considered to have converged.

    References
    ----------
    .. [1] J. Saragih, S. Lucey and J. Cohn, ''Deformable Model Fitting by
    Regularized Landmark Mean-Shifts", International Journal of Computer
    Vision (IJCV), 2010.
    """
    def __init__(self, classifiers, patch_shape, pdm, eps=10**-10):
        self.classifiers = classifiers
        self.patch_shape = patch_shape
        self.transform = pdm
        self.eps = eps
        # pre-computations
        self._set_up()

    def _create_fitting_result(self, image, parameters, gt_shape=None):
        return SemiParametricFittingResult(
            image, self, parameters=[parameters], gt_shape=gt_shape)

    def fit(self, image, initial_parameters, gt_shape=None, **kwargs):
        self.transform.from_vector_inplace(initial_parameters)
        return Fitter.fit(self, image, initial_parameters, gt_shape=gt_shape,
                          **kwargs)

    def get_parameters(self, shape):
        self.transform.set_target(shape)
        return self.transform.as_vector()


class RLMS(GradientDescent):
    r"""
    Implementation of the Regularized Landmark Mean-Shifts algorithm for
    fitting Constrained Local Models described in [1]_.

    Parameters
    ----------
    classifiers : `list` of ``classifier_callable``
        The list containing the classifier that will produce the response
        maps for each landmark point.
    patch_shape : `tuple` of `int`
        The shape of the patches used to train the classifiers.
    transform : :map:`GlobalPDM` or subclass
        The global point distribution model to be used.

        .. note::

            Only :map:`GlobalPDM` and its subclasses are supported.
            :map:`PDM` is not supported at the moment.
    eps : `float`, optional
        The convergence value. When calculating the level of convergence, if
        the norm of the delta parameter updates is less than ``eps``, the
        algorithm is considered to have converged.
    scale: `float`, optional
        Constant value that will be multiplied to the `noise_variance` of
        the pdm in order to compute the covariance of the KDE
        approximation.

    References
    ----------
    .. [1] J. Saragih, S. Lucey and J. Cohn, ''Deformable Model Fitting by
    Regularized Landmark Mean-Shifts", International Journal of Computer
    Vision (IJCV), 2010.
    """
    def __init__(self, classifiers, patch_shape, pdm, eps=10**-10, scale=10):
        self.scale = scale
        super(RLMS, self).__init__(
            classifiers, patch_shape, pdm, eps=eps)

    @property
    def algorithm(self):
        return 'RLMS'

    def _set_up(self):
        global multivariate_normal
        if multivariate_normal is None:
            from scipy.stats import multivariate_normal  # expensive
        # Build the sampling grid associated to the patch shape
        self._sampling_grid = build_sampling_grid(self.patch_shape)
        # Define the 2-dimensional gaussian distribution
        mean = np.zeros(self.transform.n_dims)
        covariance = self.scale * self.transform.model.noise_variance()
        mvn = multivariate_normal(mean=mean, cov=covariance)
        # Compute Gaussian-KDE grid
        self._kernel_grid = mvn.pdf(self._sampling_grid)

        # Jacobian
        self._J = self.transform.d_dp([])

        # Prior
        sim_prior = np.zeros((4,))
        pdm_prior = 1 / self.transform.model.eigenvalues
        self._J_prior = np.hstack((sim_prior, pdm_prior))

        # Inverse Hessian
        H = np.einsum('ijk, ilk -> jl', self._J, self._J)
        self._inv_H = np.linalg.inv(np.diag(self._J_prior) + H)

    def _fit(self, fitting_result, max_iters=20):
        # Initial error > eps
        error = self.eps + 1
        image = fitting_result.image
        target = self.transform.target
        n_iters = 0

        max_h = image.shape[-2] - 1
        max_w = image.shape[-1] - 1

        image_pixels = np.reshape(image.pixels, (image.n_channels, -1)).T
        response_image = np.zeros((target.n_points, image.shape[-2],
                                   image.shape[-1]))

        # Compute response maps
        for j, clf in enumerate(self.classifiers):
            response_image[j, :, :] = np.reshape(clf(image_pixels),
                                                 image.shape)

        while n_iters < max_iters and error > self.eps:

            mean_shift_target = np.zeros_like(target.points)

            # Compute mean-shift vectors
            for j, point in enumerate(target.points):

                patch_grid = (self._sampling_grid +
                              np.round(point[None, None, ...]).astype(int))

                x = patch_grid[:, :, 0]
                y = patch_grid[:, :, 1]

                # deal with boundaries
                x[x > max_h] = max_h
                y[y > max_w] = max_w
                x[x < 0] = 0
                y[y < 0] = 0

                kernel_response = response_image[j, x, y] * self._kernel_grid
                normalizer = np.sum(kernel_response)
                normalized_kernel_response = kernel_response / normalizer

                mean_shift_target[j, :] = np.sum(
                    normalized_kernel_response * (x, y), axis=(1, 2))

            # Compute (shape) error term
            error = mean_shift_target - target.points

            # Compute steepest descent parameter updates
            sd_delta_p = np.einsum('ijk, ik -> j', self._J, error)

            # TODO: a similar approach could be implemented in LK
            # Deal with prior
            prior = self._J_prior * self.transform.as_vector()

            # Compute parameter updates
            delta_p = -np.dot(self._inv_H, prior - sd_delta_p)

            # Update transform weights
            parameters = self.transform.as_vector() + delta_p
            fitting_result.parameters.append(parameters)
            self.transform.from_vector_inplace(parameters)
            target = self.transform.target

            # Test convergence
            error = np.abs(np.linalg.norm(delta_p))
            n_iters += 1

        return fitting_result
