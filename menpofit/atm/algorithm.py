from __future__ import division
import numpy as np
from menpofit.aam.algorithm.lk import (LucasKanadeBaseInterface,
                                       LucasKanadePatchBaseInterface)
from .result import ATMAlgorithmResult, LinearATMAlgorithmResult


# TODO document me!
class ATMLKStandardInterface(LucasKanadeBaseInterface):
    r"""
    """
    def __init__(self, transform, template, sampling=None):
        super(ATMLKStandardInterface, self).__init__(transform, template,
                                                     sampling=sampling)

    def algorithm_result(self, image, shape_parameters, cost_functions=None,
                         gt_shape=None):
        return ATMAlgorithmResult(
            image, self, shape_parameters,
            cost_functions=cost_functions, gt_shape=gt_shape)


# TODO document me!
class ATMLKLinearInterface(ATMLKStandardInterface):
    r"""
    """
    @property
    def shape_model(self):
        return self.transform.model

    def algorithm_result(self, image, shape_parameters, cost_functions=None,
                         gt_shape=None):
        return LinearATMAlgorithmResult(
            image, self, shape_parameters,
            cost_functions=cost_functions, gt_shape=gt_shape)


# TODO document me!
class ATMLKPatchInterface(LucasKanadePatchBaseInterface):
    r"""
    """
    def algorithm_result(self, image, shape_parameters, cost_functions=None,
                         gt_shape=None):
        return ATMAlgorithmResult(
            image, self, shape_parameters,
            cost_functions=cost_functions, gt_shape=gt_shape)


# TODO document me!
class LucasKanade(object):

    def __init__(self, atm_interface, eps=10**-5):
        self.eps = eps
        self.interface = atm_interface
        self._precompute()

    @property
    def transform(self):
        return self.interface.transform

    @property
    def template(self):
        return self.interface.template

    def _precompute(self):
        # grab number of shape and appearance parameters
        self.n = self.transform.n_parameters

        # vectorize template and mask it
        self.t_m = self.template.as_vector()[self.interface.i_mask]

        # compute warp jacobian
        self.dW_dp = self.interface.warp_jacobian()

        # compute shape model prior
        # TODO: Is this correct? It's like modelling no noise at all
        noise_variance = self.interface.shape_model.noise_variance() or 1
        s2 = 1.0 / noise_variance
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
        self.transform._from_vector_inplace(
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
        self.transform._from_vector_inplace(
            self.transform.as_vector() - self.dp)
