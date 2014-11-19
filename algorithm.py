from __future__ import division
import abc

import numpy as np

from menpofast.utils import build_parts_image

from menpofit.base import build_sampling_grid

from .result import UnifiedAlgorithmResult

from alabortcvpr2015.aam.algorithm import PartsAAMInterface


multivariate_normal = None  # expensive, from scipy.stats


# Abstract Interface for AAM Algorithms ---------------------------------------

class UnifiedAlgorithm(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, aam_interface, appearance_model, transform,
                 multiple_clf, parts_shape, normalize_parts, covariance, pdm,
                 eps=10**-5, **kwargs):

        # AAM part ------------------------------------------------------------

        # set state
        self.appearance_model = appearance_model
        self.template = appearance_model.mean()
        self.transform = transform
        # set interface
        self.interface = aam_interface(self, **kwargs)
        # mask appearance model
        self._U = self.appearance_model.components.T
        self._pinv_U = np.linalg.pinv(
            self._U[self.interface.image_vec_mask, :]).T

        # CLM part ------------------------------------------------------------

        # set state
        self.multiple_clf = multiple_clf
        self.parts_shape = parts_shape
        self.normalize_parts = normalize_parts
        self.covariance = covariance
        self.pdm = pdm

        # Unified part --------------------------------------------------------

        self.eps = eps
        # pre-compute
        self._precompute()

    @abc.abstractmethod
    def _precompute(self, **kwargs):
        pass

    @abc.abstractmethod
    def run(self, image, initial_shape, max_iters=20, gt_shape=None, **kwargs):
        pass


# Concrete Implementations of AAM Algorithm -----------------------------------

class PICRLMS(UnifiedAlgorithm):
    r"""
    Project-Out Inverse Compositional + Regularized Landmark Mean Shift
    """

    def _precompute(self):

        # AAM part ------------------------------------------------------------

        # sample appearance model
        self._U = self._U[self.interface.image_vec_mask, :]

        # compute model's gradient
        nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        dw_dp = self.interface.dw_dp()

        # set inverse sigma2
        self._inv_sigma2 = self.appearance_model.inverse_noise_variance()

        # compute AAM jacobian
        j = self.interface.steepest_descent_images(nabla_t, dw_dp)
        j_po = j - self._U.dot(self._pinv_U.T.dot(j))
        self._j_aam = self._inv_sigma2 * j_po

        # compute inverse hessian
        self._h_aam = self._j_aam.T.dot(j_po)

        # CLM part ------------------------------------------------------------

        global multivariate_normal
        if multivariate_normal is None:
            from scipy.stats import multivariate_normal  # expensive

        # set inverse rho2
        self._inv_rho2 = self.pdm.model.inverse_noise_variance()

        # compute Gaussian-KDE grid
        self._sampling_grid = build_sampling_grid(self.parts_shape)
        mean = np.zeros(self.transform.n_dims)
        covariance = self.covariance * self._inv_rho2
        mvn = multivariate_normal(mean=mean, cov=covariance)
        self._kernel_grid = mvn.pdf(self._sampling_grid)

        # compute CLM jacobian
        j_clm = np.rollaxis(self.pdm.d_dp(None), -1, 1)
        j_clm = j_clm.reshape((-1, j_clm.shape[-1]))
        self._j_clm = self._inv_rho2 * j_clm

        # compute CLM hessian
        self._h_clm = self._j_clm.T.dot(j_clm)

        # Unified part --------------------------------------------------------

        # set Prior
        sim_prior = np.zeros((4,))
        transform_prior = 1 / self.pdm.model.eigenvalues
        self._j_prior = np.hstack((sim_prior, transform_prior))

        # compute Unified hessian inverse and jacobian pseudo-inverse
        h = self._h_aam + self._h_clm
        self._pinv_j_aam = np.linalg.solve(h, self._j_aam.T)
        self._pinv_j_clm = np.linalg.solve(h, self._j_clm.T)
        self._inv_h_prior = np.linalg.inv(h + np.diag(self._j_prior))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, a=0.5):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # AAM part --------------------------------------------------------

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e_aam = masked_m - masked_i

            # CLM part --------------------------------------------------------

            target = self.transform.target
            # get all (x, y) pairs being considered
            xys = (target.points[:, None, None, ...] +
                   self._sampling_grid)

            # build parts image
            if not isinstance(self.interface, PartsAAMInterface):
                i = build_parts_image(
                    image, target, parts_shape=self.parts_shape,
                    normalize_parts=self.normalize_parts)

            # compute parts response
            parts_response = self.multiple_clf(i)
            parts_response[np.logical_not(np.isfinite(parts_response))] = .5

            # compute parts kernel
            parts_kernel = parts_response * self._kernel_grid
            parts_kernel /= np.sum(
                parts_kernel, axis=(-2, -1))[..., None, None]

            # compute mean shift target
            mean_shift_target = np.sum(parts_kernel[..., None] * xys,
                                       axis=(-3, -2))

            # compute (shape) error term
            e_clm = mean_shift_target.ravel() - target.as_vector()

            # Unified ---------------------------------------------------------

            # compute gauss-newton parameter updates
            if prior:
                b = (self._j_prior * self.transform.as_vector() -
                     a * self._j_aam.T.dot(e_aam) -
                     (1 - a) * self._j_clm.T.dot(e_clm))
                dp = -self._inv_h_prior.dot(b)
            else:
                dp = self._pinv_j_aam.dot(e_aam) + self._pinv_j_clm.dot(e_clm)

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return dm algorithm result
        return UnifiedAlgorithmResult(image, self, shape_parameters,
                                      gt_shape=gt_shape)


class AICRLMS(UnifiedAlgorithm):
    r"""
    Alternating Inverse Compositional + Regularized Landmark Mean Shift
    """

    def _precompute(self):

        # AAM part ------------------------------------------------------------

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

        # set inverse sigma2
        self._inv_sigma2 = self.appearance_model.inverse_noise_variance()

        # CLM part ------------------------------------------------------------

        global multivariate_normal
        if multivariate_normal is None:
            from scipy.stats import multivariate_normal  # expensive

        # set inverse rho2
        self._inv_rho2 = self.pdm.model.inverse_noise_variance()

        # compute Gaussian-KDE grid
        self._sampling_grid = build_sampling_grid(self.parts_shape)
        mean = np.zeros(self.transform.n_dims)
        covariance = self.covariance * self._inv_rho2
        mvn = multivariate_normal(mean=mean, cov=covariance)
        self._kernel_grid = mvn.pdf(self._sampling_grid)

        # compute CLM jacobian
        j_clm = np.rollaxis(self.pdm.d_dp(None), -1, 1)
        j_clm = j_clm.reshape((-1, j_clm.shape[-1]))
        self._j_clm = self._inv_rho2 * j_clm

        # compute CLM hessian
        self._h_clm = self._j_clm.T.dot(j_clm)

        # Unified part --------------------------------------------------------

        # set Prior
        sim_prior = np.zeros((4,))
        transform_prior = 1 / self.pdm.model.eigenvalues
        self._j_prior = np.hstack((sim_prior, transform_prior))
        self._h_prior = np.diag(self._j_prior)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False, a=0.5):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # initial appearance parameters
        appearance_parameters = [0]
        # model mean
        m = self.appearance_model.mean().as_vector()
        # masked model mean
        masked_m = m[self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # AAM part --------------------------------------------------------

            # warp image
            i = self.interface.warp(image)
            # mask image
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # reconstruct appearance
            c = self._pinv_U.T.dot(masked_i - masked_m)
            t = self._U.dot(c) + m
            self.template.from_vector_inplace(t)
            appearance_parameters.append(c)

            # compute (image) error
            e_aam = (self.template.as_vector()[self.interface.image_vec_mask] -
                     masked_i)

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)

            # compute AAM jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            j_aam = self._inv_sigma2 * j

            # compute AAM hessian
            h_aam = j_aam.T.dot(j)

            # CLM part --------------------------------------------------------

            # compute all position (y, x) pairs being considered
            target = self.transform.target
            yxs = (target.points[:, None, None, ...] +
                   self._sampling_grid)

            # build parts image
            if not isinstance(self.interface, PartsAAMInterface):
                i = build_parts_image(
                    image, target, parts_shape=self.parts_shape,
                    normalize_parts=self.normalize_parts)

            # compute parts response
            parts_response = self.multiple_clf(i)
            parts_response[np.logical_not(np.isfinite(parts_response))] = .5

            # compute parts kernel
            parts_kernel = parts_response * self._kernel_grid
            parts_kernel /= np.sum(
                parts_kernel, axis=(-2, -1))[..., None, None]

            # compute mean shift target
            mean_shift_target = np.sum(parts_kernel[..., None] * yxs,
                                       axis=(-3, -2))

            # compute (shape) error
            e_clm = mean_shift_target.ravel() - target.as_vector()

            # Unified part ----------------------------------------------------

            # compute Gauss-Newton parameter updates
            if prior:
                h = a * h_aam + (1 - a) * self._h_clm + self._h_prior
                b = (self._j_prior * self.transform.as_vector() -
                     a * j_aam.T.dot(e_aam) -
                     (1 - a) * self._j_clm.T.dot(e_clm))
                dp = -np.linalg.solve(h, b)
            else:
                dp = np.linalg.solve(a * h_aam + (1 - a) * self._h_clm,
                                     a * j_aam.T.dot(e_aam) +
                                     (1 - a) * self._j_clm.T.dot(e_clm))

            # update transform
            target = self.transform.target
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            shape_parameters.append(self.transform.as_vector())

            # test convergence
            error = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))
            if error < self.eps:
                break

        # return Unified algorithm result
        return UnifiedAlgorithmResult(
            image, self, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)
