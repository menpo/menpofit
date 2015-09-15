from scipy.linalg import norm
import numpy as np
from .result import LucasKanadeAlgorithmResult


# TODO: implement Inverse Additive Algorithm?
# TODO: implement sampling?
# TODO: document me!
class LucasKanade(object):
    r"""
    """
    def __init__(self, template, transform, residual, eps=10**-10):
        self.template = template
        self.transform = transform
        self.residual = residual
        self.eps = eps


# TODO: document me!
class ForwardAdditive(LucasKanade):
    r"""
    Forward Additive Lucas-Kanade algorithm
    """
    def run(self, image, initial_shape, max_iters=20, gt_shape=None):
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        cost_functions = []

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Forward Compositional Algorithm
        while k < max_iters and eps > self.eps:
            # warp image
            IWxp = image.warp_to_mask(self.template.mask, self.transform,
                                      warp_landmarks=False)

            # compute warp jacobian
            dW_dp = np.rollaxis(
                self.transform.d_dp(self.template.indices()), -1)
            dW_dp = dW_dp.reshape(dW_dp.shape[:1] + self.template.shape +
                                  dW_dp.shape[-1:])

            # compute steepest descent images
            filtered_J, J = self.residual.steepest_descent_images(
                image, dW_dp, forward=(self.template, self.transform))

            # compute hessian
            H = self.residual.hessian(filtered_J, sdi2=J)

            # compute steepest descent parameter updates.
            sd_dp = self.residual.steepest_descent_update(
                filtered_J, IWxp, self.template)

            # compute gradient descent parameter updates
            dp = -np.real(np.linalg.solve(H, sd_dp))

            # Update warp weights
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            p_list.append(self.transform.as_vector())

            # update cost
            cost_functions.append(self.residual.cost_closure())

            # test convergence
            eps = np.abs(norm(dp))

            # increase iteration counter
            k += 1

        return LucasKanadeAlgorithmResult(image, self, p_list,
                                          cost_functions=cost_functions,
                                          gt_shape=gt_shape)


# TODO: document me!
class ForwardCompositional(LucasKanade):
    r"""
    Forward Compositional Lucas-Kanade algorithm
    """
    def __init__(self, template, transform, residual, eps=10**-10):
        super(ForwardCompositional, self).__init__(
            template, transform, residual, eps=eps)
        self._precompute()

    def _precompute(self):
        # compute warp jacobian
        dW_dp = np.rollaxis(
            self.transform.d_dp(self.template.indices()), -1)
        self.dW_dp = dW_dp.reshape(dW_dp.shape[:1] + self.template.shape +
                                   dW_dp.shape[-1:])

    def run(self, image, initial_shape, max_iters=20, gt_shape=None):
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        cost_functions = []

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Forward Compositional Algorithm
        while k < max_iters and eps > self.eps:
            # warp image
            IWxp = image.warp_to_mask(self.template.mask, self.transform,
                                      warp_landmarks=False)

            # compute steepest descent images
            filtered_J, J = self.residual.steepest_descent_images(
                IWxp, self.dW_dp)

            # compute hessian
            H = self.residual.hessian(filtered_J, sdi2=J)

            # compute steepest descent parameter updates.
            sd_dp = self.residual.steepest_descent_update(
                filtered_J, IWxp, self.template)

            # compute gradient descent parameter updates
            dp = -np.real(np.linalg.solve(H, sd_dp))

            # Update warp weights
            self.transform.compose_after_from_vector_inplace(dp)
            p_list.append(self.transform.as_vector())

            # update cost
            cost_functions.append(self.residual.cost_closure())

            # test convergence
            eps = np.abs(norm(dp))

            # increase iteration counter
            k += 1

        return LucasKanadeAlgorithmResult(image, self, p_list,
                                          cost_functions=cost_functions,
                                          gt_shape=gt_shape)


# TODO: document me!
class InverseCompositional(LucasKanade):
    r"""
    Inverse Compositional Lucas-Kanade algorithm
    """
    def __init__(self, template, transform, residual, eps=10**-10):
        super(InverseCompositional, self).__init__(
            template, transform, residual, eps=eps)
        self._precompute()

    def _precompute(self):
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

        cost_functions = []

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Baker-Matthews, Inverse Compositional Algorithm
        while k < max_iters and eps > self.eps:
            # warp image
            IWxp = image.warp_to_mask(self.template.mask, self.transform,
                                      warp_landmarks=False)

            # compute steepest descent parameter updates.
            sd_dp = self.residual.steepest_descent_update(
                self.filtered_J, IWxp, self.template)

            # compute gradient descent parameter updates
            dp = np.real(np.linalg.solve(self.H, sd_dp))

            # update warp
            inv_dp = self.transform.pseudoinverse_vector(dp)
            self.transform.compose_after_from_vector_inplace(inv_dp)
            p_list.append(self.transform.as_vector())

            # update cost
            cost_functions.append(self.residual.cost_closure())

            # test convergence
            eps = np.abs(norm(dp))

            # increase iteration counter
            k += 1

        return LucasKanadeAlgorithmResult(image, self, p_list,
                                          cost_functions=cost_functions,
                                          gt_shape=gt_shape)
