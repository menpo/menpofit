from __future__ import division
import abc
import numpy as np
from menpo.image import Image
from menpo.feature import gradient as fast_gradient
from .result import AAMAlgorithmResult, LinearAAMAlgorithmResult


class AAMInterface(object):

    def __init__(self, aam_algorithm, sampling=None):
        self.algorithm = aam_algorithm

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

    @property
    def true_indices(self):
        return self.template.mask.true_indices()

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

    def solve_all_map(self, H, J, e, Ja_prior, c, Js_prior, p):
        if self.n is not H.shape[0] - self.m:
            # Bidirectional Compositional case
            Js_prior = np.hstack((Js_prior, Js_prior))
            p = np.hstack((p, p))
        # compute and return MAP solution
        J_prior = np.hstack((Ja_prior, Js_prior))
        H += np.diag(J_prior)
        Je = J_prior * np.hstack((c, p)) + J.T.dot(e)
        dq = - np.linalg.solve(H, Je)
        return dq[:self.m], dq[self.m:]

    def solve_all_ml(self, H, J, e):
        # compute ML solution
        dq = - np.linalg.solve(H, J.T.dot(e))
        return dq[:self.m], dq[self.m:]

    def algorithm_result(self, image, shape_parameters,
                         appearance_parameters=None, gt_shape=None):
        return AAMAlgorithmResult(
            image, self.algorithm, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class LinearAAMInterface(AAMInterface):

    @property
    def shape_model(self):
        return self.transform.model

    def algorithm_result(self, image, shape_parameters,
                         appearance_parameters=None, gt_shape=None):
        return LinearAAMAlgorithmResult(
            image, self.algorithm, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class PartsAAMInterface(AAMInterface):

    def __init__(self, aam_algorithm, sampling=None):
        self.algorithm = aam_algorithm

        if sampling is None:
            sampling = np.ones(self.patch_shape, dtype=np.bool)

        image_shape = self.algorithm.template.pixels.shape
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

    @property
    def patch_shape(self):
        return self.appearance_model.patch_shape

    def warp_jacobian(self):
        return np.rollaxis(self.transform.d_dp(None), -1)

    def warp(self, image):
        return Image(image.extract_patches(
            self.transform.target, patch_size=self.patch_shape,
            as_single_array=True))

    def gradient(self, image):
        pixels = image.pixels
        patch_shape = self.algorithm.appearance_model.patch_shape
        g = fast_gradient(pixels.reshape((-1,) + patch_shape))
        # remove 1st dimension gradient which corresponds to the gradient
        # between parts
        return g.reshape((2,) + pixels.shape)

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

    def algorithm_result(self, image, shape_parameters,
                         appearance_parameters=None, gt_shape=None):
        return AAMAlgorithmResult(
            image, self.algorithm, shape_parameters,
            appearance_parameters=appearance_parameters, gt_shape=gt_shape)


class AAMAlgorithm(object):

    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # set common state for all AAM algorithms
        self.appearance_model = appearance_model
        self.template = appearance_model.mean()
        self.transform = transform
        self.eps = eps
        # set interface
        self.interface = aam_interface(self, **kwargs)
        # perform pre-computations
        self.precompute()

    def precompute(self, **kwargs):
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
        s2 = (self.appearance_model.noise_variance() /
              self.interface.shape_model.noise_variance())
        L = self.interface.shape_model.eigenvalues
        self.s2_inv_L = np.hstack((np.ones((4,)), s2 / L))
        # compute appearance model prior
        S = self.appearance_model.eigenvalues
        self.s2_inv_S = s2 / S

    @abc.abstractmethod
    def run(self, image, initial_shape, max_iters=20, gt_shape=None,
            map_inference=False):
        pass


class ProjectOut(AAMAlgorithm):
    r"""
    Abstract Interface for Project-out AAM algorithms
    """
    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(ProjectOut, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

    def project_out(self, J):
        # project-out appearance bases from a particular vector or matrix
        return J - self.A_m.dot(self.pinv_A_m.dot(J))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False):
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop
        while k < max_iters and eps > self.eps:
            # warp image
            self.i = self.interface.warp(image)
            # vectorize it and mask it
            i_m = self.i.as_vector()[self.interface.i_mask]

            # compute masked error
            self.e_m = i_m - self.a_bar_m

            # solve for increments on the shape parameters
            self.dp = self.solve(map_inference)

            # update warp
            s_k = self.transform.target.points
            self.update_warp()
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, gt_shape=gt_shape)

    @abc.abstractmethod
    def solve(self, map_inference):
        pass

    @abc.abstractmethod
    def update_warp(self):
        pass


class PFC(ProjectOut):
    r"""
    Project-out Forward Compositional (PFC) Gauss-Newton algorithm
    """
    def solve(self, map_inference):
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

    def update_warp(self):
        # update warp based on forward composition
        self.transform.from_vector_inplace(
            self.transform.as_vector() + self.dp)


class PIC(ProjectOut):
    r"""
    Project-out Inverse Compositional (PIC) Gauss-Newton algorithm
    """
    def precompute(self):
        # call super method
        super(PIC, self).precompute()
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

    def solve(self, map_inference):
        # solve for increments on the shape parameters
        if map_inference:
            return self.interface.solve_shape_map(
                self.JQJ_m, self.QJ_m, self.e_m, self.s2_inv_L,
                self.transform.as_vector())
        else:
            return -self.pinv_QJ_m.dot(self.e_m)

    def update_warp(self):
        # update warp based on inverse composition
        self.transform.from_vector_inplace(
            self.transform.as_vector() - self.dp)


class Simultaneous(AAMAlgorithm):
    r"""
    Abstract Interface for Simultaneous AAM algorithms
    """
    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(Simultaneous, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False):
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop
        while k < max_iters and eps > self.eps:
            # warp image
            self.i = self.interface.warp(image)
            # mask warped image
            i_m = self.i.as_vector()[self.interface.i_mask]

            if k == 0:
                # initialize appearance parameters by projecting masked image
                # onto masked appearance model
                self.c = self.pinv_A_m.dot(i_m - self.a_bar_m)
                self.a = self.appearance_model.instance(self.c)
                a_m = self.a.as_vector()[self.interface.i_mask]
                c_list = [self.c]

            # compute masked error
            self.e_m = i_m - a_m

            # solve for increments on the appearance and shape parameters
            # simultaneously
            dc, self.dp = self.solve(map_inference)

            # update appearance parameters
            self.c += dc
            self.a = self.appearance_model.instance(self.c)
            a_m = self.a.as_vector()[self.interface.i_mask]
            c_list.append(self.c)

            # update warp
            s_k = self.transform.target.points
            self.update_warp()
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, appearance_parameters=c_list, gt_shape=gt_shape)

    @abc.abstractmethod
    def compute_jacobian(self):
        pass

    def solve(self, map_inference):
        # compute masked Jacobian
        J_m = self.compute_jacobian()
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

    @abc.abstractmethod
    def update_warp(self):
        pass


class SFC(Simultaneous):
    r"""
    Simultaneous Forward Compositional (SFC) Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # return forward Jacobian
        return self.interface.steepest_descent_images(nabla_i, self.dW_dp)

    def update_warp(self):
        # update warp based on forward composition
        self.transform.from_vector_inplace(
            self.transform.as_vector() + self.dp)


class SIC(Simultaneous):
    r"""
    Simultaneous Inverse Compositional (SIC) Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def update_warp(self):
        # update warp based on inverse composition
        self.transform.from_vector_inplace(
            self.transform.as_vector() - self.dp)


class Alternating(AAMAlgorithm):
    r"""
    Abstract Interface for Alternating AAM algorithms
    """
    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(Alternating, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

    def precompute(self, **kwargs):
        # call super method
        super(Alternating, self).precompute()
        # compute MAP appearance Hessian
        self.AA_m_map = self.A_m.T.dot(self.A_m) + np.diag(self.s2_inv_S)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False):
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop
        while k < max_iters and eps > self.eps:
            # warp image
            self.i = self.interface.warp(image)
            # mask warped image
            i_m = self.i.as_vector()[self.interface.i_mask]

            if k == 0:
                # initialize appearance parameters by projecting masked image
                # onto masked appearance model
                c = self.pinv_A_m.dot(i_m - self.a_bar_m)
                self.a = self.appearance_model.instance(c)
                a_m = self.a.as_vector()[self.interface.i_mask]
                c_list = [c]
                Jdp = 0
            else:
                Jdp = J_m.dot(self.dp)

            # compute masked error
            e_m = i_m - a_m

            # solve for increment on the appearance parameters
            if map_inference:
                Ae_m_map = - self.s2_inv_S * c + self.A_m.dot(e_m + Jdp)
                dc = np.linalg.solve(self.AA_m_map, Ae_m_map)
            else:
                dc = self.pinv_A_m.dot(e_m + Jdp)

            # compute masked Jacobian
            J_m = self.compute_jacobian()
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
            c += dc
            self.a = self.appearance_model.instance(c)
            a_m = self.a.as_vector()[self.interface.i_mask]
            c_list.append(c)

            # update warp
            s_k = self.transform.target.points
            self.update_warp()
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, appearance_parameters=c_list, gt_shape=gt_shape)

    @abc.abstractmethod
    def compute_jacobian(self):
        pass

    @abc.abstractmethod
    def update_warp(self):
        pass


class AFC(Alternating):
    r"""
    Alternating Forward Compositional (AFC) Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # return forward Jacobian
        return self.interface.steepest_descent_images(nabla_i, self.dW_dp)

    def update_warp(self):
        # update warp based on forward composition
        self.transform.from_vector_inplace(
            self.transform.as_vector() + self.dp)


class AIC(Alternating):
    r"""
    Alternating Inverse Compositional (AIC) Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def update_warp(self):
        # update warp based on inverse composition
        self.transform.from_vector_inplace(
            self.transform.as_vector() - self.dp)


class ModifiedAlternating(Alternating):
    r"""
    Abstract Interface for Modified Alternating AAM algorithms
    """
    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(ModifiedAlternating, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False):
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        a_m = self.a_bar_m
        c_list = []
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop
        while k < max_iters and eps > self.eps:
            # warp image
            self.i = self.interface.warp(image)
            # mask warped image
            i_m = self.i.as_vector()[self.interface.i_mask]

            c = self.pinv_A_m.dot(i_m - a_m)
            self.a = self.appearance_model.instance(c)
            a_m = self.a.as_vector()[self.interface.i_mask]
            c_list.append(c)

            # compute masked error
            e_m = i_m - a_m

            # compute masked Jacobian
            J_m = self.compute_jacobian()
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
            self.update_warp()
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, appearance_parameters=c_list, gt_shape=gt_shape)


class MAFC(ModifiedAlternating):
    r"""
    Modified Alternating Forward Compositional (MAFC) Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # return forward Jacobian
        return self.interface.steepest_descent_images(nabla_i, self.dW_dp)

    def update_warp(self):
        # update warp based on forward composition
        self.transform.from_vector_inplace(
            self.transform.as_vector() + self.dp)


class MAIC(ModifiedAlternating):
    r"""
    Modified Alternating Inverse Compositional (MAIC) Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def update_warp(self):
        # update warp based on inverse composition
        self.transform.from_vector_inplace(
            self.transform.as_vector() - self.dp)


class Wiberg(AAMAlgorithm):
    r"""
    Abstract Interface for Wiberg AAM algorithms
    """
    def __init__(self, aam_interface, appearance_model, transform,
                 eps=10**-5, **kwargs):
        # call super constructor
        super(Wiberg, self).__init__(
            aam_interface, appearance_model, transform, eps, **kwargs)

    def project_out(self, J):
        # project-out appearance bases from a particular vector or matrix
        return J - self.A_m.dot(self.pinv_A_m.dot(J))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False):
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Compositional Gauss-Newton loop
        while k < max_iters and eps > self.eps:
            # warp image
            self.i = self.interface.warp(image)
            # mask warped image
            i_m = self.i.as_vector()[self.interface.i_mask]

            if k == 0:
                # initialize appearance parameters by projecting masked image
                # onto masked appearance model
                c = self.pinv_A_m.dot(i_m - self.a_bar_m)
                self.a = self.appearance_model.instance(c)
                a_m = self.a.as_vector()[self.interface.i_mask]
                c_list = [c]
            else:
                c = self.pinv_A_m.dot(i_m - a_m + J_m.dot(self.dp))
                self.a = self.appearance_model.instance(c)
                a_m = self.a.as_vector()[self.interface.i_mask]
                c_list.append(c)

            # compute masked error
            e_m = i_m - self.a_bar_m

            # compute masked Jacobian
            J_m = self.compute_jacobian()
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
            self.update_warp()
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return self.interface.algorithm_result(
            image, p_list, appearance_parameters=c_list, gt_shape=gt_shape)


class WFC(Wiberg):
    r"""
    Wiberg Forward Compositional (WFC) Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        # compute warped image gradient
        nabla_i = self.interface.gradient(self.i)
        # return forward Jacobian
        return self.interface.steepest_descent_images(nabla_i, self.dW_dp)

    def update_warp(self):
        # update warp based on forward composition
        self.transform.from_vector_inplace(
            self.transform.as_vector() + self.dp)


class WIC(Wiberg):
    r"""
    Wiberg Inverse Compositional (WIC) Gauss-Newton algorithm
    """
    def compute_jacobian(self):
        # compute warped appearance model gradient
        nabla_a = self.interface.gradient(self.a)
        # return inverse Jacobian
        return self.interface.steepest_descent_images(-nabla_a, self.dW_dp)

    def update_warp(self):
        # update warp based on inverse composition
        self.transform.from_vector_inplace(
            self.transform.as_vector() - self.dp)

