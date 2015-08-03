from __future__ import division
import numpy as np
from menpo.image import Image
from menpo.feature import no_op
from menpo.feature import gradient as fast_gradient
from .result import ATMAlgorithmResult, LinearATMAlgorithmResult


# TODO document me!
class LucasKanadeStandardInterface(object):
    r"""
    """
    def __init__(self, lk_algorithm, sampling=None):
        self.algorithm = lk_algorithm

        n_true_pixels = self.template.n_true_pixels()
        n_channels = self.template.n_channels
        n_parameters = self.transform.n_parameters
        sampling_mask = np.zeros(n_true_pixels, dtype=np.bool)

        if sampling is None:
            sampling = 1
        sampling_pattern = xrange(0, n_true_pixels, sampling)
        sampling_mask[sampling_pattern] = 1

        self.i_mask = np.nonzero(np.tile(
            sampling_mask[None, ...], (n_channels, 1)).flatten())[0]
        self.dW_dp_mask = np.nonzero(np.tile(
            sampling_mask[None, ..., None], (2, 1, n_parameters)))
        self.nabla_mask = np.nonzero(np.tile(
            sampling_mask[None, None, ...], (2, n_channels, 1)))
        self.nabla2_mask = np.nonzero(np.tile(
            sampling_mask[None, None, None, ...], (2, 2, n_channels, 1)))

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
    def true_indices(self):
        return self.template.mask.true_indices()

    @property
    def shape_model(self):
        return self.transform.pdm.model

    def warp_jacobian(self):
        dW_dp = np.rollaxis(self.transform.d_dp(self.true_indices), -1)
        return dW_dp[self.dW_dp_mask].reshape((dW_dp.shape[0], -1,
                                               dW_dp.shape[2]))

    def warp(self, image):
        return image.warp_to_mask(self.template.mask,
                                  self.transform)

    def gradient(self, img):
        nabla = fast_gradient(img)
        nabla.set_boundary_pixels()
        return nabla.as_vector().reshape((2, img.n_channels, -1))

    def steepest_descent_images(self, nabla, dW_dp):
        # reshape gradient
        # nabla: n_dims x n_channels x n_pixels
        nabla = nabla[self.nabla_mask].reshape(nabla.shape[:2] + (-1,))
        # compute steepest descent images
        # nabla: n_dims x n_channels x n_pixels
        # warp_jacobian: n_dims x            x n_pixels x n_params
        # sdi:            n_channels x n_pixels x n_params
        sdi = 0
        a = nabla[..., None] * dW_dp[:, None, ...]
        for d in a:
            sdi += d
        # reshape steepest descent images
        # sdi: (n_channels x n_pixels) x n_params
        return sdi.reshape((-1, sdi.shape[2]))

    @classmethod
    def solve_shape_map(cls, H, J, e, J_prior, p):
        if p.shape[0] is not H.shape[0]:
            # Bidirectional Compositional case
            J_prior = np.hstack((J_prior, J_prior))
            p = np.hstack((p, p))
        # compute and return MAP solution
        H += np.diag(J_prior)
        Je = J_prior * p + J.T.dot(e)
        return - np.linalg.solve(H, Je)

    @classmethod
    def solve_shape_ml(cls, H, J, e):
        # compute and return ML solution
        return -np.linalg.solve(H, J.T.dot(e))

    def algorithm_result(self, image, shape_parameters, cost_functions=None,
                         gt_shape=None):
        return ATMAlgorithmResult(
            image, self.algorithm, shape_parameters,
            cost_functions=cost_functions, gt_shape=gt_shape)


# TODO document me!
class LucasKanadeLinearInterface(LucasKanadeStandardInterface):
    r"""
    """
    @property
    def shape_model(self):
        return self.transform.model

    def algorithm_result(self, image, shape_parameters, cost_functions=None,
                         gt_shape=None):
        return LinearATMAlgorithmResult(
            image, self.algorithm, shape_parameters,
            cost_functions=cost_functions, gt_shape=gt_shape)


# TODO document me!
class LucasKanadePartsInterface(LucasKanadeStandardInterface):
    r"""
    """
    def __init__(self, lk_algorithm, patch_shape=(17, 17),
                 normalize_parts=no_op, sampling=None):
        self.algorithm = lk_algorithm
        self.patch_shape = patch_shape
        self.normalize_parts = normalize_parts

        if sampling is None:
            sampling = np.ones(self.patch_shape, dtype=np.bool)

        image_shape = self.algorithm.template.pixels.shape
        image_mask = np.tile(sampling[None, None, None, ...],
                             image_shape[:3] + (1, 1))
        self.i_mask = np.nonzero(image_mask.flatten())[0]
        self.nabla_mask = np.nonzero(np.tile(
            image_mask[None, ...], (2, 1, 1, 1, 1, 1)))
        self.nabla2_mask = np.nonzero(np.tile(
            image_mask[None, None, ...], (2, 2, 1, 1, 1, 1, 1)))

    @property
    def shape_model(self):
        return self.transform.model

    def warp_jacobian(self):
        return np.rollaxis(self.transform.d_dp(None), -1)

    def warp(self, image):
        parts = image.extract_patches(self.transform.target,
                                      patch_size=self.patch_shape,
                                      as_single_array=True)
        parts = self.normalize_parts(parts)
        return Image(parts)

    def gradient(self, image):
        pixels = image.pixels
        g = fast_gradient(pixels.reshape((-1,) + self.patch_shape))
        # remove 1st dimension gradient which corresponds to the gradient
        # between parts
        return g.reshape((2,) + pixels.shape)

    def steepest_descent_images(self, nabla, dw_dp):
        # reshape nabla
        # nabla: dims x parts x off x ch x (h x w)
        nabla = nabla[self.nabla_mask].reshape(
            nabla.shape[:-2] + (-1,))
        # compute steepest descent images
        # nabla: dims x parts x off x ch x (h x w)
        # ds_dp:    dims x parts x                             x params
        # sdi:             parts x off x ch x (h x w) x params
        sdi = 0
        a = nabla[..., None] * dw_dp[..., None, None, None, :]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (parts x offsets x ch x w x h) x params
        return sdi.reshape((-1, sdi.shape[-1]))


# TODO document me!
class LucasKanade(object):

    def __init__(self, lk_atm_interface_cls, template, transform,
                 eps=10**-5, **kwargs):
        # set common state for all ATM algorithms
        self.template = template
        self.transform = transform
        self.eps = eps
        # set interface
        self.interface = lk_atm_interface_cls(self, **kwargs)
        # perform pre-computations
        self._precompute()

    def _precompute(self, **kwargs):
        # grab number of shape and appearance parameters
        self.n = self.transform.n_parameters

        # vectorize template and mask it
        self.t_m = self.template.as_vector()[self.interface.i_mask]

        # compute warp jacobian
        self.dW_dp = self.interface.warp_jacobian()

        # compute shape model prior
        s2 = 1 / self.interface.shape_model.noise_variance()
        L = self.interface.shape_model.eigenvalues
        self.s2_inv_L = np.hstack((np.ones((4,)), s2 / L))


# TODO document me!
class Compositional(LucasKanade):
    r"""
    Abstract Interface for Compositional ATM algorithms
    """
    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False):
        # define cost closure
        def cost_closure(x):
            return lambda: x.T.dot(x)

        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop -------------------------------------

        # warp image
        self.i = self.interface.warp(image)
        # vectorize it and mask it
        i_m = self.i.as_vector()[self.interface.i_mask]

        # compute masked error
        self.e_m = i_m - self.t_m

        # update cost
        cost_functions = [cost_closure(self.e_m)]

        while k < max_iters and eps > self.eps:
            # solve for increments on the shape parameters
            self.dp = self._solve(map_inference)

            # update warp
            s_k = self.transform.target.points
            self._update_warp()
            p_list.append(self.transform.as_vector())

            # warp image
            self.i = self.interface.warp(image)
            # vectorize it and mask it
            i_m = self.i.as_vector()[self.interface.i_mask]

            # compute masked error
            self.e_m = i_m - self.t_m

            # update cost
            cost_functions.append(cost_closure(self.e_m))

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, cost_functions=cost_functions, gt_shape=gt_shape)


# TODO document me!
class ForwardCompositional(Compositional):
    r"""
    Forward Compositional (FC) Gauss-Newton algorithm
    """
    def _solve(self, map_inference):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # compute masked forward Jacobian
        J_m = self.interface.steepest_descent_images(nabla_i, self.dW_dp)
        # compute masked forward Hessian
        JJ_m = J_m.T.dot(J_m)
        # solve for increments on the shape parameters
        if map_inference:
            return self.interface.solve_shape_map(
                JJ_m, J_m, self.e_m,  self.s2_inv_L,
                self.transform.as_vector())
        else:
            return self.interface.solve_shape_ml(JJ_m, J_m, self.e_m)

    def _update_warp(self):
        # update warp based on forward composition
        self.transform.from_vector_inplace(
            self.transform.as_vector() + self.dp)


# TODO document me!
class InverseCompositional(Compositional):
    r"""
    Inverse Compositional (IC) Gauss-Newton algorithm
    """
    def _precompute(self):
        # call super method
        super(InverseCompositional, self)._precompute()
        # compute appearance model mean gradient
        nabla_t = self.interface.gradient(self.template)
        # compute masked inverse Jacobian
        self.J_m = self.interface.steepest_descent_images(-nabla_t, self.dW_dp)
        # compute masked inverse Hessian
        self.JJ_m = self.J_m.T.dot(self.J_m)
        # compute masked Jacobian pseudo-inverse
        self.pinv_J_m = np.linalg.solve(self.JJ_m, self.J_m.T)

    def _solve(self, map_inference):
        # solve for increments on the shape parameters
        if map_inference:
            return self.interface.solve_shape_map(
                self.JJ_m, self.J_m, self.e_m, self.s2_inv_L,
                self.transform.as_vector())
        else:
            return -self.pinv_J_m.dot(self.e_m)

    def _update_warp(self):
        # update warp based on inverse composition
        self.transform.from_vector_inplace(
            self.transform.as_vector() - self.dp)
