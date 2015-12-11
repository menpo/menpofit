from __future__ import division
import numpy as np
from menpofit.base import build_grid
from menpofit.clm.result import CLMAlgorithmResult

multivariate_normal = None  # expensive, from scipy.stats


# TODO: document me!
class GradientDescentCLMAlgorithm(object):
    r"""
    """


# TODO: Document me!
class ActiveShapeModel(GradientDescentCLMAlgorithm):
    r"""
    Active Shape Model (ASM) algorithm
    """
    def __init__(self, expert_ensemble, shape_model, gaussian_covariance=10,
                 eps=10**-5):
        # Set parameters
        self.expert_ensemble, = expert_ensemble,
        self.transform = shape_model
        self.gaussian_covariance = gaussian_covariance
        self.eps = eps
        # Perform pre-computations
        self._precompute()

    def _precompute(self):
        r"""
        """
        # Import multivariate normal distribution from scipy
        global multivariate_normal
        if multivariate_normal is None:
            from scipy.stats import multivariate_normal  # expensive

        # Build grid associated to size of the search space
        search_size = self.expert_ensemble.search_size
        self.half_search_size = np.round(
            np.asarray(self.expert_ensemble.search_size) / 2)
        self.search_grid = build_grid(search_size)[None, None]

        # set rho2
        self.rho2 = self.transform.model.noise_variance()

        # Compute Gaussian-KDE grid
        self.mvn = multivariate_normal(mean=np.zeros(2),
                                       cov=self.gaussian_covariance)

        # Compute shape model prior
        sim_prior = np.zeros((4,))
        pdm_prior = self.rho2 / self.transform.model.eigenvalues
        self.rho2_inv_L = np.hstack((sim_prior, pdm_prior))

        # Compute Jacobian
        J = np.rollaxis(self.transform.d_dp(None), -1, 1)
        self.J = J.reshape((-1, J.shape[-1]))
        # Compute inverse Hessian
        self.JJ = self.J.T.dot(self.J)
        # Compute Jacobian pseudo-inverse
        self.pinv_J = np.linalg.solve(self.JJ, self.J.T)
        self.inv_JJ_prior = np.linalg.inv(self.JJ + np.diag(self.rho2_inv_L))

    def run(self, image, initial_shape, max_iters=20, gt_shape=None,
            map_inference=False):
        r"""
        """
        # Initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # Initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Expectation-Maximisation loop
        while k < max_iters and eps > self.eps:

            target = self.transform.target
            # Obtain all landmark positions l_i = (x_i, y_i) being considered
            # ie all pixel positions in each landmark's search space
            candidate_landmarks = (target.points[:, None, None, None, :] +
                                   self.search_grid)

            # Compute responses
            responses = self.expert_ensemble.predict_probability(image, target)

            # Approximate responses using isotropic Gaussian
            max_indices = np.argmax(
                responses.reshape(responses.shape[:2] + (-1,)), axis=-1)
            max_indices = np.unravel_index(max_indices, responses.shape)[-2:]
            max_indices = np.hstack((max_indices[0], max_indices[1]))
            max_indices = max_indices[:, None, None, None, ...]
            max_indices -= self.half_search_size
            gaussian_responses = self.mvn.pdf(max_indices + self.search_grid)
            # Normalise smoothed responses
            gaussian_responses /= np.sum(gaussian_responses,
                                         axis=(-2, -1))[..., None, None]

            # Compute new target
            new_target = np.sum(gaussian_responses[:, None, ..., None] *
                                candidate_landmarks, axis=(-3, -2))

            # Compute shape error term
            error = target.as_vector() - new_target.ravel()

            # Solve for increments on the shape parameters
            if map_inference:
                Je = (self.rho2_inv_L * self.transform.as_vector() -
                      self.J.T.dot(error))
                dp = -self.inv_JJ_prior.dot(Je)
            else:
                dp = self.pinv_J.dot(error)

            # Update pdm
            s_k = self.transform.target.points
            self.transform._from_vector_inplace(self.transform.as_vector() + dp)
            p_list.append(self.transform.as_vector())

            # Test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # Increase iteration counter
            k += 1

        # Return algorithm result
        return CLMAlgorithmResult(image, self.transform, p_list,
                                  gt_shape=gt_shape)


# TODO: Document me!
class RegularisedLandmarkMeanShift(GradientDescentCLMAlgorithm):
    r"""
    Regularized Landmark Mean-Shift (RLMS) algorithm
    """
    def __init__(self, expert_ensemble, shape_model, kernel_covariance=10,
                 eps=10**-5):
        # Set parameters
        self.expert_ensemble, = expert_ensemble,
        self.transform = shape_model
        self.kernel_covariance = kernel_covariance
        self.eps = eps
        # Perform pre-computations
        self._precompute()

    def _precompute(self):
        r"""
        """
        # Import multivariate normal distribution from scipy
        global multivariate_normal
        if multivariate_normal is None:
            from scipy.stats import multivariate_normal  # expensive

        # Build grid associated to size of the search space
        search_size = self.expert_ensemble.search_size
        self.search_grid = build_grid(search_size)

        # set rho2
        self.rho2 = self.transform.model.noise_variance()

        # Compute Gaussian-KDE grid
        mvn = multivariate_normal(mean=np.zeros(2), cov=self.kernel_covariance)
        self.kernel_grid = mvn.pdf(self.search_grid)[None, None]

        # Compute shape model prior
        sim_prior = np.zeros((4,))
        pdm_prior = self.rho2 / self.transform.model.eigenvalues
        self.rho2_inv_L = np.hstack((sim_prior, pdm_prior))

        # Compute Jacobian
        J = np.rollaxis(self.transform.d_dp(None), -1, 1)
        self.J = J.reshape((-1, J.shape[-1]))
        # Compute inverse Hessian
        self.JJ = self.J.T.dot(self.J)
        # Compute Jacobian pseudo-inverse
        self.pinv_J = np.linalg.solve(self.JJ, self.J.T)
        self.inv_JJ_prior = np.linalg.inv(self.JJ + np.diag(self.rho2_inv_L))

    def run(self, image, initial_shape, max_iters=20, gt_shape=None,
            map_inference=False):
        r"""
        """
        # Initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # Initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Expectation-Maximisation loop
        while k < max_iters and eps > self.eps:

            target = self.transform.target
            # Obtain all landmark positions l_i = (x_i, y_i) being considered
            # ie all pixel positions in each landmark's search space
            candidate_landmarks = (target.points[:, None, None, None, :] +
                                   self.search_grid)

            # Compute patch responses
            patch_responses = self.expert_ensemble.predict_probability(image,
                                                                       target)

            # Smooth responses using the Gaussian-KDE grid
            patch_kernels = patch_responses * self.kernel_grid
            # Normalise smoothed responses
            patch_kernels /= np.sum(patch_kernels,
                                    axis=(-2, -1))[..., None, None]

            # Compute mean shift target
            mean_shift_target = np.sum(patch_kernels[..., None] *
                                       candidate_landmarks, axis=(-3, -2))

            # Compute shape error term
            error = mean_shift_target.ravel() - target.as_vector()

            # Solve for increments on the shape parameters
            if map_inference:
                Je = (self.rho2_inv_L * self.transform.as_vector() -
                      self.J.T.dot(error))
                dp = -self.inv_JJ_prior.dot(Je)
            else:
                dp = self.pinv_J.dot(error)

            # Update pdm
            s_k = self.transform.target.points
            self.transform._from_vector_inplace(self.transform.as_vector() + dp)
            p_list.append(self.transform.as_vector())

            # Test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # Increase iteration counter
            k += 1

        # Return algorithm result
        return CLMAlgorithmResult(image, self, p_list, gt_shape=gt_shape)
