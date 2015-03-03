from scipy.linalg import norm
import numpy as np

from .base import AppearanceLucasKanade


class SFA(AppearanceLucasKanade):
    r"""
    Simultaneous Forward Additive algorithm
    """
    @property
    def algorithm(self):
        return 'Simultaneous-FA'

    def _fit(self, fitting_result, max_iters=20, project=True):
        # Initial error > eps
        error = self.eps + 1
        image = fitting_result.image
        fitting_result.weights = []
        n_iters = 0

        # Forward Additive Algorithm
        while n_iters < max_iters and error > self.eps:
            # Compute warped image with current weights
            IWxp = image.warp_to_mask(self.template.mask, self.transform,
                                      warp_landmarks=False)

            # Compute warp Jacobian
            dW_dp = np.rollaxis(
                self.transform.d_dp(self.template.indices()), -1)

            # Compute steepest descent images, VI_dW_dp
            J_aux = self.residual.steepest_descent_images(
                image, dW_dp, forward=(self.template, self.transform))

            # Project out appearance model from VT_dW_dp
            self._J = self.appearance_model.project_out_vectors(J_aux.T).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp weights
            parameters = self.transform.as_vector() + delta_p
            self.transform.from_vector_inplace(parameters)
            fitting_result.parameters.append(parameters)

            # Test convergence
            error = np.abs(norm(delta_p))
            n_iters += 1

        return fitting_result


class SFC(AppearanceLucasKanade):
    r"""
    Simultaneous Forward Compositional algorithm
    """
    @property
    def algorithm(self):
        return 'Simultaneous-FC'

    def _set_up(self):
        # Compute warp Jacobian
        self._dW_dp = np.rollaxis(
            self.transform.d_dp(self.template.indices()), -1)

    def _fit(self, fitting_result, max_iters=20, project=True):
        # Initial error > eps
        error = self.eps + 1
        image = fitting_result.image
        fitting_result.weights = []
        n_iters = 0

        # Forward Additive Algorithm
        while n_iters < max_iters and error > self.eps:
            # Compute warped image with current weights
            IWxp = image.warp_to_mask(self.template.mask, self.transform,
                                      warp_landmarks=False)

            # Compute steepest descent images, VI_dW_dp
            J_aux = self.residual.steepest_descent_images(IWxp, self._dW_dp)

            # Project out appearance model from VT_dW_dp
            self._J = self.appearance_model.project_out_vectors(J_aux.T).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp weights
            self.transform.compose_after_from_vector_inplace(delta_p)
            fitting_result.parameters.append(self.transform.as_vector())

            # Test convergence
            error = np.abs(norm(delta_p))
            n_iters += 1

        return fitting_result


class SIC(AppearanceLucasKanade):
    r"""
    Simultaneous Inverse Compositional algorithm
    """
    @property
    def algorithm(self):
        return 'Simultaneous-IC'

    def _set_up(self):
        # Compute warp Jacobian
        self._dW_dp = np.rollaxis(
            self.transform.d_dp(self.template.indices()), -1)

    def _fit(self, fitting_result, max_iters=20, project=True):
        # Initial error > eps
        error = self.eps + 1
        image = fitting_result.image
        fitting_result.weights = []
        n_iters = 0

        mean = self.appearance_model.mean()

        while n_iters < max_iters and error > self.eps:
            # Compute warped image with current weights
            IWxp = image.warp_to_mask(self.template.mask, self.transform,
                                      warp_landmarks=False)

            if n_iters == 0:
                # Project image onto the model bases
                weights = self.appearance_model.project(IWxp)
            else:
                # Compute Gauss-Newton appearance parameters updates
                diff = (self.template.as_vector() - mean.as_vector())
                self.template.from_vector_inplace(IWxp.as_vector() - diff -
                                                  np.dot(J_aux, delta_p))
                delta_weights = self.appearance_model.project(self.template)
                weights += delta_weights

            # Reconstruct appearance
            self.template = self.appearance_model.instance(weights)
            fitting_result.weights.append(weights)

            # Compute steepest descent images, VT_dW_dp
            J_aux = self.residual.steepest_descent_images(self.template,
                                                          self._dW_dp)

            # Project out appearance model from VT_dW_dp
            self._J = self.appearance_model.project_out_vectors(J_aux.T).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, IWxp, mean)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Request the pesudoinverse vector from the transform
            inv_delta_p = self.transform.pseudoinverse_vector(delta_p)

            # Update warp weights
            self.transform.compose_after_from_vector_inplace(inv_delta_p)
            fitting_result.parameters.append(self.transform.as_vector())

            # Test convergence
            error = np.abs(norm(delta_p))
            n_iters += 1

        return fitting_result
