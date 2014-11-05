from __future__ import division
import abc

import numpy as np

from menpofast.utils import build_parts_image

from menpofit.base import build_sampling_grid

from .result import UnifiedAlgorithmResult

multivariate_normal = None  # expensive, from scipy.stats


# Abstract Interface for AAM Algorithms ---------------------------------------

class UnifiedAlgorithm(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, aam_interface, appearance_model, transform,
                 multiple_clf, parts_shape, normalize_parts, pdm,
                 eps=10**-5, scale=10, **kwargs):

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
        self.pdm = pdm
        self._scale = scale

        # Unified -------------------------------------------------------------

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

class ProbPIC(UnifiedAlgorithm):
    r"""
    Project-Out Inverse Compositional Algorithm
    """

    def _precompute(self):

        # sample appearance model
        self._U = self._U[self.interface.image_vec_mask, :]

        # compute model's gradient
        nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        dw_dp = self.interface.dw_dp()

        # compute steepest descent images
        j = self.interface.steepest_descent_images(nabla_t, dw_dp)

        # project out appearance model from J
        self._j_po = j - self._U.dot(self._pinv_U.T.dot(j))

        # compute inverse hessian
        self._h = self._j_po.T.dot(j)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

        # initialize transform
        self.transform.set_target(initial_shape)
        shape_parameters = [self.transform.as_vector()]
        # masked model mean
        masked_m = self.appearance_model.mean().as_vector()[
            self.interface.image_vec_mask]

        for _ in xrange(max_iters):

            # compute warped image with current weights
            i = self.interface.warp(image)

            # reconstruct appearance
            masked_i = i.as_vector()[self.interface.image_vec_mask]

            # compute error image
            e = masked_m - masked_i

            # compute gauss-newton parameter updates
            dp = self.interface.solve(self._h, self._j_po, e, prior)

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


class ProbAIC(UnifiedAlgorithm):
    r"""
    Alternating Inverse Compositional Algorithm
    """

    def _precompute(self):

        # AAM part ------------------------------------------------------------

        # compute warp jacobian
        self._dw_dp = self.interface.dw_dp()

        self._sigma2 = self.appearance_model.noise_variance()

        # CLM part ------------------------------------------------------------

        global multivariate_normal
        if multivariate_normal is None:
            from scipy.stats import multivariate_normal  # expensive

        # build sampling grid associated to patch shape
        self._sampling_grid = build_sampling_grid(self.parts_shape)

        self._rho2 = self.pdm.model.noise_variance()

        # compute Gaussian-KDE grid
        mean = np.zeros(self.pdm.n_dims)
        covariance = self._scale * self._rho2
        mvn = multivariate_normal(mean=mean, cov=covariance)
        self._kernel_grid = mvn.pdf(self._sampling_grid)

        # compute Jacobian
        j_clm = np.rollaxis(self.pdm.d_dp(None), -1, 1)
        self._j_clm = j_clm.reshape((-1, j_clm.shape[-1]))

        # compute Hessian inverse
        self._h_clm = self._j_clm.T.dot(self._j_clm)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            prior=False):

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

            # compute error image
            e_aam = (self.template.as_vector()[self.interface.image_vec_mask] -
                     masked_i)

            # compute model gradient
            nabla_t = self.interface.gradient(self.template)

            # compute model jacobian
            j_aam = self.interface.steepest_descent_images(nabla_t,
                                                           self._dw_dp)

            # compute hessian
            h_aam = j_aam.T.dot(j_aam)

            # CLM part --------------------------------------------------------

            target = self.transform.target
            rounded_target = target.copy()
            rounded_target.points = np.round(target.points)
            # get all (x, y) pairs being considered
            xys = (rounded_target.points[:, None, None, ...] +
                   self._sampling_grid)

            # build parts image
            parts_image = build_parts_image(
                image, target, parts_shape=self.parts_shape,
                normalize_parts=self.normalize_parts)

            # compute parts response
            parts_response = self.multiple_clf(parts_image)

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

            dp = np.linalg.solve(
                self._rho2 * h_aam + self._sigma2 * self._h_clm,
                self._rho2 * j_aam.T.dot(e_aam) +
                self._sigma2 * self._j_clm.T.dot(e_clm))

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
