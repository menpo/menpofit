from __future__ import division
import numpy as np
from menpo.image import Image
from menpo.feature import gradient as fast_gradient, no_op
from ..result import AAMAlgorithmResult, LinearAAMAlgorithmResult


def _solve_all_map(H, J, e, Ja_prior, c, Js_prior, p, m, n):
    if n is not H.shape[0] - m:
        # Bidirectional Compositional case
        Js_prior = np.hstack((Js_prior, Js_prior))
        p = np.hstack((p, p))
        # compute and return MAP solution
    J_prior = np.hstack((Ja_prior, Js_prior))
    H += np.diag(J_prior)
    Je = J_prior * np.hstack((c, p)) + J.T.dot(e)
    dq = - np.linalg.solve(H, Je)
    return dq[:m], dq[m:]


def _solve_all_ml(H, J, e, m):
    # compute ML solution
    dq = - np.linalg.solve(H, J.T.dot(e))
    return dq[:m], dq[m:]


# TODO document me!
class LucasKanadeBaseInterface(object):
    r"""
    """
    def __init__(self, transform, template, sampling=None):
        self.transform = transform
        self.template = template

        self._build_sampling_mask(sampling)

    def _build_sampling_mask(self, sampling):
        n_true_pixels = self.template.n_true_pixels()
        n_channels = self.template.n_channels
        n_parameters = self.transform.n_parameters

        sampling_mask = np.zeros(n_true_pixels, dtype=np.bool)

        if sampling is None:
            sampling = range(0, n_true_pixels, 1)
        elif isinstance(sampling, np.int):
            sampling = range(0, n_true_pixels, sampling)

        sampling_mask[sampling] = 1

        self.i_mask = np.nonzero(np.tile(
            sampling_mask[None, ...], (n_channels, 1)).flatten())[0]
        self.dW_dp_mask = np.nonzero(np.tile(
            sampling_mask[None, ..., None], (2, 1, n_parameters)))
        self.nabla_mask = np.nonzero(np.tile(
            sampling_mask[None, None, ...], (2, n_channels, 1)))
        self.nabla2_mask = np.nonzero(np.tile(
            sampling_mask[None, None, None, ...], (2, 2, n_channels, 1)))

    @property
    def shape_model(self):
        return self.transform.pdm.model

    @property
    def n(self):
        return self.transform.n_parameters

    @property
    def true_indices(self):
        return self.template.mask.true_indices()

    def warp_jacobian(self):
        dW_dp = np.rollaxis(self.transform.d_dp(self.true_indices), -1)
        return dW_dp[self.dW_dp_mask].reshape((dW_dp.shape[0], -1,
                                               dW_dp.shape[2]))

    def warp(self, image):
        return image.warp_to_mask(self.template.mask, self.transform,
                                  warp_landmarks=False)

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


class LucasKanadeStandardInterface(LucasKanadeBaseInterface):

    def __init__(self, appearance_model, transform, template, sampling=None):
        super(LucasKanadeStandardInterface, self).__init__(transform, template,
                                                           sampling=sampling)
        self.appearance_model = appearance_model

    @property
    def m(self):
        return self.appearance_model.n_active_components

    def solve_all_map(self, H, J, e, Ja_prior, c, Js_prior, p):
        return _solve_all_map(H, J, e, Ja_prior, c, Js_prior, p,
                              self.m, self.n)

    def solve_all_ml(self, H, J, e):
        return _solve_all_ml(H, J, e, self.m)

    def algorithm_result(self, image, shape_parameters, cost_functions=None,
                         appearance_parameters=None, gt_shape=None):
        return AAMAlgorithmResult(
            image, self, shape_parameters,
            cost_functions=cost_functions,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# TODO document me!
class LucasKanadeLinearInterface(LucasKanadeStandardInterface):
    r"""
    """
    @property
    def shape_model(self):
        return self.transform.model

    def algorithm_result(self, image, shape_parameters, cost_functions=None,
                         appearance_parameters=None, gt_shape=None):
        return LinearAAMAlgorithmResult(
            image, self, shape_parameters,
            cost_functions=cost_functions,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# TODO document me!
class LucasKanadePatchBaseInterface(LucasKanadeBaseInterface):
    r"""
    """
    def __init__(self, transform, template, sampling=None,
                 patch_shape=(17, 17), patch_normalisation=no_op):
        self.patch_shape = patch_shape
        self.patch_normalisation = patch_normalisation

        super(LucasKanadePatchBaseInterface, self).__init__(
            transform, template, sampling=sampling)

    def _build_sampling_mask(self, sampling):
        if sampling is None:
            sampling = np.ones(self.patch_shape, dtype=np.bool)

        image_shape = self.template.pixels.shape
        image_mask = np.tile(sampling[None, None, None, ...],
                             image_shape[:3] + (1, 1))
        self.i_mask = np.nonzero(image_mask.flatten())[0]
        self.gradient_mask = np.nonzero(np.tile(
            image_mask[None, ...], (2, 1, 1, 1, 1, 1)))
        self.gradient2_mask = np.nonzero(np.tile(
            image_mask[None, None, ...], (2, 2, 1, 1, 1, 1, 1)))

    @property
    def shape_model(self):
        return self.transform.model

    def warp_jacobian(self):
        return np.rollaxis(self.transform.d_dp(None), -1)

    def warp(self, image):
        parts = image.extract_patches(self.transform.target,
                                      patch_shape=self.patch_shape,
                                      as_single_array=True)
        parts = self.patch_normalisation(parts)
        return Image(parts, copy=False)

    def gradient(self, image):
        pixels = image.pixels
        nabla = fast_gradient(pixels.reshape((-1,) + self.patch_shape))
        # remove 1st dimension gradient which corresponds to the gradient
        # between parts
        return nabla.reshape((2,) + pixels.shape)

    def steepest_descent_images(self, nabla, dw_dp):
        # reshape nabla
        # nabla: dims x parts x off x ch x (h x w)
        nabla = nabla[self.gradient_mask].reshape(
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
class LucasKanadePatchInterface(LucasKanadePatchBaseInterface):
    r"""
    """
    def __init__(self, appearance_model, transform, template, sampling=None,
                 patch_shape=(17, 17), patch_normalisation=no_op):
        self.appearance_model = appearance_model

        super(LucasKanadePatchInterface, self).__init__(
            transform, template, patch_shape=patch_shape,
            patch_normalisation=patch_normalisation, sampling=sampling)

    @property
    def m(self):
        return self.appearance_model.n_active_components

    def solve_all_map(self, H, J, e, Ja_prior, c, Js_prior, p):
        return _solve_all_map(H, J, e, Ja_prior, c, Js_prior, p,
                              self.m, self.n)

    def solve_all_ml(self, H, J, e):
        return _solve_all_ml(H, J, e, self.m)

    def algorithm_result(self, image, shape_parameters, cost_functions=None,
                         appearance_parameters=None, gt_shape=None):
        return AAMAlgorithmResult(
            image, self, shape_parameters,
            cost_functions=cost_functions,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


# TODO document me!
class LucasKanade(object):
    r"""
    """
    def __init__(self, aam_interface, eps=10**-5):
        self.eps = eps
        self.interface = aam_interface
        self._precompute()

    @property
    def appearance_model(self):
        return self.interface.appearance_model

    @property
    def transform(self):
        return self.interface.transform

    @property
    def template(self):
        return self.interface.template

    def _precompute(self):
        # grab number of shape and appearance parameters
        self.n = self.transform.n_parameters
        self.m = self.appearance_model.n_active_components

        # grab appearance model components
        self.A = self.appearance_model.components
        # mask them
        self.A_m = self.A.T[self.interface.i_mask, :]
        # compute their pseudoinverse
        self.pinv_A_m = np.linalg.pinv(self.A_m)

        # grab appearance model mean
        self.a_bar = self.appearance_model.mean()
        # vectorize it and mask it
        self.a_bar_m = self.a_bar.as_vector()[self.interface.i_mask]

        # compute warp jacobian
        self.dW_dp = self.interface.warp_jacobian()

        # compute shape model prior
        # TODO: Is this correct? It's like modelling no noise at all
        sm_noise_variance = self.interface.shape_model.noise_variance() or 1
        s2 = self.appearance_model.noise_variance() / sm_noise_variance
        L = self.interface.shape_model.eigenvalues
        self.s2_inv_L = np.hstack((np.ones((4,)), s2 / L))
        # compute appearance model prior
        S = self.appearance_model.eigenvalues
        self.s2_inv_S = s2 / S


# TODO: Document me!
class ProjectOut(LucasKanade):
    r"""
    Abstract Interface for Project-out AAM algorithms
    """
    def project_out(self, J):
        # project-out appearance bases from a particular vector or matrix
        return J - self.A_m.dot(self.pinv_A_m.dot(J))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False):
        # define cost closure
        def cost_closure(x, f):
            return lambda: x.T.dot(f(x))

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
        self.e_m = i_m - self.a_bar_m

        # update cost_functions
        cost_functions = [cost_closure(self.e_m, self.project_out)]

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
            self.e_m = i_m - self.a_bar_m

            # update cost
            cost_functions.append(cost_closure(self.e_m, self.project_out))

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, cost_functions=cost_functions, gt_shape=gt_shape)


# TODO: Document me!
class ProjectOutForwardCompositional(ProjectOut):
    r"""
    Project-out Forward Compositional (PFC) Gauss-Newton algorithm
    """
    def _solve(self, map_inference):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # compute masked forward Jacobian
        J_m = self.interface.steepest_descent_images(nabla_i, self.dW_dp)
        # project out appearance model from it
        QJ_m = self.project_out(J_m)
        # compute masked forward Hessian
        JQJ_m = QJ_m.T.dot(J_m)
        # solve for increments on the shape parameters
        if map_inference:
            return self.interface.solve_shape_map(
                JQJ_m, QJ_m, self.e_m,  self.s2_inv_L,
                self.transform.as_vector())
        else:
            return self.interface.solve_shape_ml(JQJ_m, QJ_m, self.e_m)

    def _update_warp(self):
        # update warp based on forward composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() + self.dp)


# TODO: Document me!
class ProjectOutInverseCompositional(ProjectOut):
    r"""
    Project-out Inverse Compositional (PIC) Gauss-Newton algorithm
    """
    def _precompute(self):
        # call super method
        super(ProjectOutInverseCompositional, self)._precompute()
        # compute appearance model mean gradient
        nabla_a = self.interface.gradient(self.a_bar)
        # compute masked inverse Jacobian
        J_m = self.interface.steepest_descent_images(-nabla_a, self.dW_dp)
        # project out appearance model from it
        self.QJ_m = self.project_out(J_m)
        # compute masked inverse Hessian
        self.JQJ_m = self.QJ_m.T.dot(J_m)
        # compute masked Jacobian pseudo-inverse
        self.pinv_QJ_m = np.linalg.solve(self.JQJ_m, self.QJ_m.T)

    def _solve(self, map_inference):
        # solve for increments on the shape parameters
        if map_inference:
            return self.interface.solve_shape_map(
                self.JQJ_m, self.QJ_m, self.e_m, self.s2_inv_L,
                self.transform.as_vector())
        else:
            return -self.pinv_QJ_m.dot(self.e_m)

    def _update_warp(self):
        # update warp based on inverse composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() - self.dp)


# TODO: Document me!
class Simultaneous(LucasKanade):
    r"""
    Abstract Interface for Simultaneous AAM algorithms
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
        # mask warped image
        i_m = self.i.as_vector()[self.interface.i_mask]

        # initialize appearance parameters by projecting masked image
        # onto masked appearance model
        self.c = self.pinv_A_m.dot(i_m - self.a_bar_m)
        self.a = self.appearance_model.instance(self.c)
        a_m = self.a.as_vector()[self.interface.i_mask]
        c_list = [self.c]

        # compute masked error
        self.e_m = i_m - a_m

        # update cost
        cost_functions = [cost_closure(self.e_m)]

        while k < max_iters and eps > self.eps:
            # solve for increments on the appearance and shape parameters
            # simultaneously
            dc, self.dp = self._solve(map_inference)

            # update appearance parameters
            self.c = self.c + dc
            self.a = self.appearance_model.instance(self.c)
            a_m = self.a.as_vector()[self.interface.i_mask]
            c_list.append(self.c)

            # update warp
            s_k = self.transform.target.points
            self._update_warp()
            p_list.append(self.transform.as_vector())

            # warp image
            self.i = self.interface.warp(image)
            # mask warped image
            i_m = self.i.as_vector()[self.interface.i_mask]

            # compute masked error
            self.e_m = i_m - a_m

            # update cost
            cost_functions.append(cost_closure(self.e_m))

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, cost_functions=cost_functions,
            appearance_parameters=c_list, gt_shape=gt_shape)

    def _solve(self, map_inference):
        # compute masked Jacobian
        J_m = self._compute_jacobian()
        # assemble masked simultaneous Jacobian
        J_sim_m = np.hstack((-self.A_m, J_m))
        # compute masked Hessian
        H_sim_m = J_sim_m.T.dot(J_sim_m)
        # solve for increments on the appearance and shape parameters
        # simultaneously
        if map_inference:
            return self.interface.solve_all_map(
                H_sim_m, J_sim_m, self.e_m, self.s2_inv_S, self.c,
                self.s2_inv_L, self.transform.as_vector())
        else:
            return self.interface.solve_all_ml(H_sim_m, J_sim_m, self.e_m)


# TODO: Document me!
class SimultaneousForwardCompositional(Simultaneous):
    r"""
    Simultaneous Forward Compositional (SFC) Gauss-Newton algorithm
    """
    def _compute_jacobian(self):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # return forward Jacobian
        return self.interface.steepest_descent_images(nabla_i, self.dW_dp)

    def _update_warp(self):
        # update warp based on forward composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() + self.dp)


# TODO: Document me!
class SimultaneousInverseCompositional(Simultaneous):
    r"""
    Simultaneous Inverse Compositional (SIC) Gauss-Newton algorithm
    """
    def _compute_jacobian(self):
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def _update_warp(self):
        # update warp based on inverse composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() - self.dp)


# TODO: Document me!
class Alternating(LucasKanade):
    r"""
    Abstract Interface for Alternating AAM algorithms
    """
    def _precompute(self, **kwargs):
        # call super method
        super(Alternating, self)._precompute()
        # compute MAP appearance Hessian
        self.AA_m_map = self.A_m.T.dot(self.A_m) + np.diag(self.s2_inv_S)

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
        # mask warped image
        i_m = self.i.as_vector()[self.interface.i_mask]

        # initialize appearance parameters by projecting masked image
        # onto masked appearance model
        c = self.pinv_A_m.dot(i_m - self.a_bar_m)
        self.a = self.appearance_model.instance(c)
        a_m = self.a.as_vector()[self.interface.i_mask]
        c_list = [c]
        Jdp = 0

        # compute masked error
        e_m = i_m - a_m

        # update cost
        cost_functions = [cost_closure(e_m)]

        while k < max_iters and eps > self.eps:
            # solve for increment on the appearance parameters
            if map_inference:
                Ae_m_map = - self.s2_inv_S * c + self.A_m.dot(e_m + Jdp)
                dc = np.linalg.solve(self.AA_m_map, Ae_m_map)
            else:
                dc = self.pinv_A_m.dot(e_m + Jdp)

            # compute masked Jacobian
            J_m = self._compute_jacobian()
            # compute masked Hessian
            H_m = J_m.T.dot(J_m)
            # solve for increments on the shape parameters
            if map_inference:
                self.dp = self.interface.solve_shape_map(
                    H_m, J_m, e_m - self.A_m.T.dot(dc), self.s2_inv_L,
                    self.transform.as_vector())
            else:
                self.dp = self.interface.solve_shape_ml(H_m, J_m,
                                                        e_m - self.A_m.dot(dc))

            # update appearance parameters
            c = c + dc
            self.a = self.appearance_model.instance(c)
            a_m = self.a.as_vector()[self.interface.i_mask]
            c_list.append(c)

            # update warp
            s_k = self.transform.target.points
            self._update_warp()
            p_list.append(self.transform.as_vector())

            # warp image
            self.i = self.interface.warp(image)
            # mask warped image
            i_m = self.i.as_vector()[self.interface.i_mask]

            # compute Jdp
            Jdp = J_m.dot(self.dp)

            # compute masked error
            e_m = i_m - a_m

            # update cost
            cost_functions.append(cost_closure(e_m))

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, cost_functions=cost_functions,
            appearance_parameters=c_list, gt_shape=gt_shape)


# TODO: Document me!
class AlternatingForwardCompositional(Alternating):
    r"""
    Alternating Forward Compositional (AFC) Gauss-Newton algorithm
    """
    def _compute_jacobian(self):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # return forward Jacobian
        return self.interface.steepest_descent_images(nabla_i, self.dW_dp)

    def _update_warp(self):
        # update warp based on forward composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() + self.dp)


# TODO: Document me!
class AlternatingInverseCompositional(Alternating):
    r"""
    Alternating Inverse Compositional (AIC) Gauss-Newton algorithm
    """
    def _compute_jacobian(self):
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def _update_warp(self):
        # update warp based on inverse composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() - self.dp)


# TODO: Document me!
class ModifiedAlternating(Alternating):
    r"""
    Abstract Interface for Modified Alternating AAM algorithms
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
        a_m = self.a_bar_m
        c_list = []
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop -------------------------------------

        # warp image
        self.i = self.interface.warp(image)
        # mask warped image
        i_m = self.i.as_vector()[self.interface.i_mask]

        # initialize appearance parameters by projecting masked image
        # onto masked appearance model
        c = self.pinv_A_m.dot(i_m - a_m)
        self.a = self.appearance_model.instance(c)
        a_m = self.a.as_vector()[self.interface.i_mask]
        c_list.append(c)

        # compute masked error
        e_m = i_m - a_m

        # update cost
        cost_functions = [cost_closure(e_m)]

        while k < max_iters and eps > self.eps:
            # compute masked Jacobian
            J_m = self._compute_jacobian()
            # compute masked Hessian
            H_m = J_m.T.dot(J_m)
            # solve for increments on the shape parameters
            if map_inference:
                self.dp = self.interface.solve_shape_map(
                    H_m, J_m, e_m, self.s2_inv_L, self.transform.as_vector())
            else:
                self.dp = self.interface.solve_shape_ml(H_m, J_m, e_m)

            # update warp
            s_k = self.transform.target.points
            self._update_warp()
            p_list.append(self.transform.as_vector())

            # warp image
            self.i = self.interface.warp(image)
            # mask warped image
            i_m = self.i.as_vector()[self.interface.i_mask]

            # update appearance parameters
            c = self.pinv_A_m.dot(i_m - self.a_bar_m)
            self.a = self.appearance_model.instance(c)
            a_m = self.a.as_vector()[self.interface.i_mask]
            c_list.append(c)

            # compute masked error
            e_m = i_m - a_m

            # update cost
            cost_functions.append(cost_closure(e_m))

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, cost_functions=cost_functions,
            appearance_parameters=c_list, gt_shape=gt_shape)


# TODO: Document me!
class ModifiedAlternatingForwardCompositional(ModifiedAlternating):
    r"""
    Modified Alternating Forward Compositional (MAFC) Gauss-Newton algorithm
    """
    def _compute_jacobian(self):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # return forward Jacobian
        return self.interface.steepest_descent_images(nabla_i, self.dW_dp)

    def _update_warp(self):
        # update warp based on forward composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() + self.dp)


# TODO: Document me!
class ModifiedAlternatingInverseCompositional(ModifiedAlternating):
    r"""
    Modified Alternating Inverse Compositional (MAIC) Gauss-Newton algorithm
    """
    def _compute_jacobian(self):
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def _update_warp(self):
        # update warp based on inverse composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() - self.dp)


# TODO: Document me!
class Wiberg(LucasKanade):
    r"""
    Abstract Interface for Wiberg AAM algorithms
    """
    def project_out(self, J):
        # project-out appearance bases from a particular vector or matrix
        return J - self.A_m.dot(self.pinv_A_m.dot(J))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False):
        # define cost closure
        def cost_closure(x, f):
            return lambda: x.T.dot(f(x))

        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop -------------------------------------

        # warp image
        self.i = self.interface.warp(image)
        # mask warped image
        i_m = self.i.as_vector()[self.interface.i_mask]

        # initialize appearance parameters by projecting masked image
        # onto masked appearance model
        c = self.pinv_A_m.dot(i_m - self.a_bar_m)
        self.a = self.appearance_model.instance(c)
        a_m = self.a.as_vector()[self.interface.i_mask]
        c_list = [c]

        # compute masked error
        e_m = i_m - self.a_bar_m

        # update cost
        cost_functions = [cost_closure(e_m, self.project_out)]

        while k < max_iters and eps > self.eps:
            # compute masked Jacobian
            J_m = self._compute_jacobian()
            # project out appearance models
            QJ_m = self.project_out(J_m)
            # compute masked Hessian
            JQJ_m = QJ_m.T.dot(J_m)
            # solve for increments on the shape parameters
            if map_inference:
                self.dp = self.interface.solve_shape_map(
                    JQJ_m, QJ_m, e_m, self.s2_inv_L,
                    self.transform.as_vector())
            else:
                self.dp = self.interface.solve_shape_ml(JQJ_m, QJ_m, e_m)

            # update warp
            s_k = self.transform.target.points
            self._update_warp()
            p_list.append(self.transform.as_vector())

            # warp image
            self.i = self.interface.warp(image)
            # mask warped image
            i_m = self.i.as_vector()[self.interface.i_mask]

            # update appearance parameters
            dc = self.pinv_A_m.dot(i_m - a_m + J_m.dot(self.dp))
            c = c + dc
            self.a = self.appearance_model.instance(c)
            a_m = self.a.as_vector()[self.interface.i_mask]
            c_list.append(c)

            # compute masked error
            e_m = i_m - self.a_bar_m

            # update cost
            cost_functions.append(cost_closure(e_m, self.project_out))

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, cost_functions=cost_functions,
            appearance_parameters=c_list, gt_shape=gt_shape)


# TODO: Document me!
class WibergForwardCompositional(Wiberg):
    r"""
    Wiberg Forward Compositional (WFC) Gauss-Newton algorithm
    """
    def _compute_jacobian(self):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # return forward Jacobian
        return self.interface.steepest_descent_images(nabla_i, self.dW_dp)

    def _update_warp(self):
        # update warp based on forward composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() + self.dp)


# TODO: Document me!
class WibergInverseCompositional(Wiberg):
    r"""
    Wiberg Inverse Compositional (WIC) Gauss-Newton algorithm
    """
    def _compute_jacobian(self):
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def _update_warp(self):
        # update warp based on inverse composition
        self.transform._from_vector_inplace(
            self.transform.as_vector() - self.dp)
