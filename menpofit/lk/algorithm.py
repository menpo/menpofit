from scipy.linalg import norm
import abc
import numpy as np
from .result import LKAlgorithmResult


# TODO: implement Inverse Additive Algorithm?
# TODO: implement Linear, Parts interfaces? Will they play nice with residuals?
# TODO: implement sampling?
# TODO: handle costs for all LKAlgorithms
# TODO: document me!
class LKAlgorithm(object):
    r"""
    """
    def __init__(self, template, transform, residual, eps=10**-10):
        self.template = template
        self.transform = transform
        self.residual = residual
        self.eps = eps

    @abc.abstractmethod
    def run(self, image, initial_shape, max_iters=20, gt_shape=None):
        pass


class FA(LKAlgorithm):
    r"""
    Forward Additive Lucas-Kanade algorithm
    """
    def run(self, image, initial_shape, max_iters=20, gt_shape=None):
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Forward Compositional Algorithm
        while k < max_iters and eps > self.eps:
            # warp image
            IWxp = image.warp_to_mask(self.template.mask, self.transform)

            # compute warp jacobian
            dW_dp = np.rollaxis(
                self.transform.d_dp(self.template.indices()), -1)

            # compute steepest descent images
            filtered_J, J = self.residual.steepest_descent_images(
                image, dW_dp, forward=(self.template, self.transform))

            # compute hessian
            H = self.residual.hessian(filtered_J, sdi2=J)

            # compute steepest descent parameter updates.
            sd_dp = self.residual.steepest_descent_update(
                filtered_J, IWxp, self.template)

            # compute gradient descent parameter updates
            dp = np.real(np.linalg.solve(H, sd_dp))

            # Update warp weights
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(norm(dp))

            # increase iteration counter
            k += 1

        return LKAlgorithmResult(image, self, p_list, gt_shape=None)


class FC(LKAlgorithm):
    r"""
    Forward Compositional Lucas-Kanade algorithm
    """
    def __init__(self, template, transform, residual, eps=10**-10):
        super(FC, self).__init__(template, transform, residual, eps=eps)
        self.precompute()

    def precompute(self):
        # compute warp jacobian
        self.dW_dp = np.rollaxis(
            self.transform.d_dp(self.template.indices()), -1)

    def run(self, image, initial_shape, max_iters=20, gt_shape=None):
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Forward Compositional Algorithm
        while k < max_iters and eps > self.eps:
            # warp image
            IWxp = image.warp_to_mask(self.template.mask, self.transform)

            # compute steepest descent images
            filtered_J, J = self.residual.steepest_descent_images(
                IWxp, self.dW_dp)

            # compute hessian
            H = self.residual.hessian(filtered_J, sdi2=J)

            # compute steepest descent parameter updates.
            sd_dp = self.residual.steepest_descent_update(
                filtered_J, IWxp, self.template)

            # compute gradient descent parameter updates
            dp = np.real(np.linalg.solve(H, sd_dp))

            # Update warp weights
            self.transform.compose_after_from_vector_inplace(dp)
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(norm(dp))

            # increase iteration counter
            k += 1

        return LKAlgorithmResult(image, self, p_list, gt_shape=None)


class IC(LKAlgorithm):
    r"""
    Inverse Compositional Lucas-Kanade algorithm
    """
    def __init__(self, template, transform, residual, eps=10**-10):
        super(IC, self).__init__(template, transform, residual, eps=eps)
        self.precompute()

    def precompute(self):
        # compute warp jacobian
        dW_dp = np.rollaxis(self.transform.d_dp(self.template.indices()), -1)
        dW_dp = dW_dp.reshape(dW_dp.shape[:1] + self.template.shape +
                              dW_dp.shape[-1:])
        # compute steepest descent images
        self.filtered_J, J = self.residual.steepest_descent_images(
            self.template, dW_dp)
        # compute hessian
        self.H = self.residual.hessian(self.filtered_J, sdi2=J)

    def run(self, image, initial_shape, max_iters=20, gt_shape=None):
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Baker-Matthews, Inverse Compositional Algorithm
        while k < max_iters and eps > self.eps:
            # warp image
            IWxp = image.warp_to_mask(self.template.mask, self.transform)

            # compute steepest descent parameter updates.
            sd_dp = self.residual.steepest_descent_update(
                self.filtered_J, IWxp, self.template)

            # compute gradient descent parameter updates
            dp = np.real(np.linalg.solve(self.H, sd_dp))

            # update warp
            inv_dp = self.transform.pseudoinverse_vector(dp)
            self.transform.compose_after_from_vector_inplace(inv_dp)
            p_list.append(self.transform.as_vector())

            # test convergence
            eps = np.abs(norm(dp))

            # increase iteration counter
            k += 1

        return LKAlgorithmResult(image, self, p_list, gt_shape=None)
