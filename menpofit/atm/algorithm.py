from __future__ import division
import numpy as np

from menpofit.aam.algorithm.lk import (LucasKanadeBaseInterface,
                                       LucasKanadePatchBaseInterface)

from .result import ATMAlgorithmResult


# ----------- INTERFACES -----------
class ATMLucasKanadeStandardInterface(LucasKanadeBaseInterface):
    r"""
    Interface for Lucas-Kanade optimization of standard ATM. Suitable for
    `menpofit.atm.HolisticATM`.

    Parameters
    ----------
    transform : `menpofit.transform.ModelDrivenTransform` or subclass
        The model driven transform object of the ATM.
    template : `menpo.image.Image` or subclass
        The image template.
    sampling : `int` or ``None``, optional
        The sub-sampling step of the sampling mask. If ``None``, then no
        sampling is applied on the template.
    """
    def __init__(self, transform, template, sampling=None):
        super(ATMLucasKanadeStandardInterface, self).__init__(
                transform, template, sampling=sampling)

    def algorithm_result(self, image, shapes, shape_parameters,
                         cost_functions=None, gt_shape=None):
        r"""
        Returns an ATM iterative optimization result object.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image on which the optimization is applied.
        shapes : `list` of `menpo.shape.PointCloud`
            The `list` of shapes per iteration.
        shape_parameters : `list` of `ndarray`
            The `list` of shape parameters per iteration.
        cost_functions : `list` of `closures` or ``None``, optional
            The `list` of functions that compute the cost per iteration. If
            ``None``, then it is assumed that the cost computation for that
            particular algorithm is not well defined.
        gt_shape : `menpo.shape.PointCloud` or ``None``, optional
            The ground truth shape that corresponds to the test image.

        Returns
        -------
        result : `menpofit.atm.result.ATMAlgorithmResult`
            The optimization result object.
        """
        return ATMAlgorithmResult(
                shapes=shapes, shape_parameters=shape_parameters,
                cost_functions=cost_functions, image=image, gt_shape=gt_shape)


class ATMLucasKanadeLinearInterface(ATMLucasKanadeStandardInterface):
    r"""
    Interface for Lucas-Kanade optimization of linear ATM. Suitable for
    `menpofit.atm.LinearATM` and `menpofit.atm.LinearMaskedATM`.
    """
    @property
    def shape_model(self):
        r"""
        Returns the shape model of the ATM.

        :type: `menpo.model.PCAModel`
        """
        return self.transform.model

    def algorithm_result(self, image, shapes, shape_parameters,
                         cost_functions=None, gt_shape=None):
        r"""
        Returns an ATM iterative optimization result object.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image on which the optimization is applied.
        shapes : `list` of `menpo.shape.PointCloud`
            The `list` of sparse shapes per iteration.
        shape_parameters : `list` of `ndarray`
            The `list` of shape parameters per iteration.
        cost_functions : `list` of `closures` or ``None``, optional
            The `list` of functions that compute the cost per iteration. If
            ``None``, then it is assumed that the cost computation for that
            particular algorithm is not well defined.
        gt_shape : `menpo.shape.PointCloud` or ``None``, optional
            The ground truth shape that corresponds to the test image.

        Returns
        -------
        result : `menpofit.atm.result.ATMAlgorithmResult`
            The optimization result object.
        """
        # TODO: Separate result for linear ATM that stores both the sparse
        #       and dense shapes per iteration (@patricksnape will fix this)
        # This means that the linear ATM will only store the sparse shapes
        shapes = [self.transform.from_vector(p).sparse_target
                  for p in shape_parameters]
        return ATMAlgorithmResult(
                shapes=shapes, shape_parameters=shape_parameters,
                cost_functions=cost_functions, image=image, gt_shape=gt_shape)


class ATMLucasKanadePatchInterface(LucasKanadePatchBaseInterface):
    r"""
    Interface for Lucas-Kanade optimization of patch-based ATM. Suitable for
    `menpofit.atm.PatchATM`.
    """
    def algorithm_result(self, image, shapes, shape_parameters,
                         cost_functions=None, gt_shape=None):
        r"""
        Returns an ATM iterative optimization result object.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image on which the optimization is applied.
        shapes : `list` of `menpo.shape.PointCloud`
            The `list` of shapes per iteration.
        shape_parameters : `list` of `ndarray`
            The `list` of shape parameters per iteration.
        cost_functions : `list` of `closures` or ``None``, optional
            The `list` of functions that compute the cost per iteration. If
            ``None``, then it is assumed that the cost computation for that
            particular algorithm is not well defined.
        gt_shape : `menpo.shape.PointCloud` or ``None``, optional
            The ground truth shape that corresponds to the test image.

        Returns
        -------
        result : `menpofit.atm.result.ATMAlgorithmResult`
            The optimization result object.
        """
        return ATMAlgorithmResult(
                shapes=shapes, shape_parameters=shape_parameters,
                cost_functions=cost_functions, image=image, gt_shape=gt_shape)


# ----------- ALGORITHMS -----------
class LucasKanade(object):
    r"""
    Abstract class for a Lucas-Kanade optimization algorithm.

    Parameters
    ----------
    atm_interface : The ATM interface class. Existing interfaces include:

        ================================= ============================
        'ATMLucasKanadeStandardInterface' Suitable for holistic ATM
        'ATMLucasKanadeLinearInterface'   Suitable for linear ATM
        'ATMLucasKanadePatchInterface'    Suitable for patch-based ATM
        ================================= ============================

    eps : `float`, optional
        Value for checking the convergence of the optimization.
    """
    def __init__(self, atm_interface, eps=10**-5):
        self.eps = eps
        self.interface = atm_interface
        self._precompute()

    @property
    def transform(self):
        r"""
        Returns the model driven transform object of the ATM.

        :type: `menpofit.transform.ModelDrivenTransform` or subclass
        """
        return self.interface.transform

    @property
    def template(self):
        r"""
        Returns the template of the ATM.

        :type: `menpo.image.Image` or subclass
        """
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


class Compositional(LucasKanade):
    r"""
    Abstract class for defining Compositional ATM optimization algorithms.
    """
    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            map_inference=False):
        r"""
        Execute the optimization algorithm.

        Parameters
        ----------
        image : `menpo.image.Image`
            The input test image.
        initial_shape : `menpo.shape.PointCloud`
            The initial shape from which the optimization will start.
        gt_shape : `menpo.shape.PointCloud` or ``None``, optional
            The ground truth shape of the image. It is only needed in order
            to get passed in the optimization result object, which has the
            ability to compute the fitting error.
        max_iters : `int`, optional
            The maximum number of iterations. Note that the algorithm may
            converge, and thus stop, earlier.
        map_inference : `bool`, optional
            If ``True``, then the solution will be given after performing MAP
            inference.
        """
        # define cost closure
        def cost_closure(x):
            return lambda: x.T.dot(x)

        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]
        shapes = [self.transform.target]

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
            shapes.append(self.transform.target)

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
                image=image, shapes=shapes, shape_parameters=p_list,
                cost_functions=cost_functions, gt_shape=gt_shape)


class ForwardCompositional(Compositional):
    r"""
    Forward Compositional (FC) Gauss-Newton algorithm.
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


class InverseCompositional(Compositional):
    r"""
    Inverse Compositional (IC) Gauss-Newton algorithm.
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
