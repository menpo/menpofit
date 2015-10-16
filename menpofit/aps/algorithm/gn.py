import numpy as np

from menpo.feature import gradient as fast_gradient
from menpo.image import Image
from menpofit.modelinstance import OrthoPDM

from ..result import APSAlgorithmResult

class GaussNewtonBaseInterface(object):
    r"""
    Base interface for Gauss-Newton optimization of APS.

    Parameters
    ----------
    appearance_model : :map:`GMRFInstanceModel`
        The trained appearance GMRF model.
    deformation_model : :map:`GMRFInstanceModel`
        The trained deformation GMRF model.
    transform : :map:`OrhtoPDM` or `PDM`
        The motion model.
    use_deformation_cost : `bool`
        Whether to use the deformation model during the optimization.
    template : :map:`Image`
        The template (in this case it is the mean appearance).
    sampling : `ndarray`
        The sampling mask.
    patch_shape : (`int`, `int`)
        The patch shape.
    patch_normalisation : `callable`
        The method for normalizing the patches.
    """
    def __init__(self, appearance_model, deformation_model, transform,
                 use_deformation_cost, template, sampling, patch_shape,
                 patch_normalisation):
        self.appearance_model = appearance_model
        self.deformation_model = deformation_model
        self.use_deformation_cost = use_deformation_cost
        self.patch_shape = patch_shape
        self.patch_normalisation = patch_normalisation
        self.transform = transform
        self.template = template

        # build the sampling mask
        self._build_sampling_mask(sampling)

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

    def ds_dp(self):
        r"""
        Calculates the shape jacobian. That is

        .. math::

            \frac{d\mathcal{S}}{d\mathbf{p}} = \mathbf{J}_S = \mathbf{U}_S

        with size :math:`2 \times n \times n_S`.

        :type: `ndarray`
        """
        return np.rollaxis(self.transform.d_dp(None), -1)

    def ds_dp_vectorized(self):
        r"""
        Calculates the vectorized shape jacobian. That is

        .. math::

            \frac{d\mathcal{S}}{d\mathbf{p}} = \mathbf{J}_S = \mathbf{U}_S

        with size :math:`2n \times n_S`.

        :type: `ndarray`
        """
        n_params = self.ds_dp().shape[-1]
        return self.ds_dp().reshape([-1, n_params], order='F')

    def Q_d(self):
        r"""
        Returns the deformation precision matrix :math:`\mathbf{Q}_d` that
        has size :math:`2n \times 2n`.

        :type: `ndarray`
        """
        return self.deformation_model.precision

    def H_s(self):
        r"""
        Calculates the deformation Hessian matrix

        .. :math:

            \mathbf{H}_s = \mathbf{U}_S^T \mathbf{Q}_d \mathbf{U}_S

        that has size :math:`n_S \times n_S`.

        :type: `ndarray`
        """
        tmp = self.ds_dp_vectorized().T.dot(self.Q_d())
        return tmp.dot(self.ds_dp_vectorized())

    def warp(self, image):
        r"""
        Function that warps the input image, i.e. extracts the patches and
        normalizes them.

        Parameters
        ----------
        image : :map:`Image`
            The input image.

        Returns
        -------
        parts : :map:`Image`
            The part-based image.
        """
        parts = image.extract_patches(self.transform.target,
                                      patch_shape=self.patch_shape,
                                      as_single_array=True)
        parts = self.patch_normalisation(parts)
        return Image(parts, copy=False)

    def gradient(self, image):
        r"""
        Function that computes the gradient of the image.

        Parameters
        ----------
        image : :map:`Image`
            The input image.

        Returns
        -------
        gradient : `ndarray`
            The computed gradient.
        """
        pixels = image.pixels
        nabla = fast_gradient(pixels.reshape((-1,) + self.patch_shape))
        # remove 1st dimension gradient which corresponds to the gradient
        # between parts
        return nabla.reshape((2,) + pixels.shape)

    def steepest_descent_images(self, nabla, ds_dp):
        r"""
        Function that computes the steepest descent images, i.e.

        .. math::

            \mathbf{J}_{\mathbf{a}} = \nabla\mathbf{a} \frac{dS}{d\mathbf{p}}

        with size :math:`mn \times n_S`.

        Parameters
        ----------
        nabla : `ndarray`
            The image (or mean appearance) gradient.
        ds_dp : `ndarray`
            The shape jacobian.

        Returns
        -------
        steepest_descent_images : `ndarray`
            The computed steepest descent images.
        """
        # reshape nabla
        # nabla: dims x parts x off x ch x (h x w)
        nabla = nabla[self.gradient_mask].reshape(nabla.shape[:-2] + (-1,))
        # compute steepest descent images
        # nabla: dims x parts x off x ch x (h x w)
        # dS_dp: dims x parts x                             x params
        # sdi:          parts x off x ch x (h x w) x params
        sdi = 0
        a = nabla[..., None] * ds_dp[..., None, None, None, :]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (parts x offsets x ch x w x h) x params
        return sdi.reshape((-1, sdi.shape[-1]))

    def J_a_T_Q_a(self, J_a, Q_a):
        r"""
        Function that computes the dot product between the appearance
        jacobian and the precision matrix, i.e.

        .. math::

            \mathbf{J}_{\mathbf{a}}^T \mathbf{Q}_{a}

        with size :math:`n_S \times mn`.

        Parameters
        ----------
        J_a : `ndarray`
            The appearance jacobian (steepest descent images).
        Q_a : `scipy.sparse.bsr_matrix`
            The appearance precision matrix.

        Returns
        -------
        J_a_T_Q_a : `ndarray`
            The dot product.
        """
        # compute the dot product between the appearance jacobian (J_a^T) and
        # the precision matrix (Q_a)
        # J_a: (parts x offsets x ch x w x h) x params
        # Q_a: (parts x offsets x ch x w x h) x (parts x offsets x ch x w x h)
        return Q_a.dot(J_a).T

    def algorithm_result(self, image, shape_parameters, cost_functions=None,
                         gt_shape=None):
        r"""
        Returns an instance of the algorithm fitting result.

        Parameters
        ----------
        image : :map:`Image: or subclass
            The test image.
        shape_parameters : `list` of `ndarray`
            A `list` with the shape parameters per iteration. These are used to
            generate the fitted shapes.
        cost_functions : `list` of `callable` or ``None``, optional
            The `list` of `callable` that compute the cost per iteration.
        gt_shape : :map:`PointCloud` or ``None``, optional
            The ground truth shape of the image.
        """
        return APSAlgorithmResult(
            image, self, shape_parameters, cost_functions=cost_functions,
            gt_shape=gt_shape)


class GaussNewton(object):
    r"""
    Base algorithm for Gauss-Newton optimization of APS.

    Parameters
    ----------
    aps_interface : `GaussNewtonBaseInterface` or subclass
        The Gauss-Newton interface object.
    eps : `float`, optional
        The error threshold to stop the optimization.
    """
    def __init__(self, aps_interface, eps=10**-5):
        self.eps = eps
        self.interface = aps_interface
        self._precompute()

    @property
    def appearance_model(self):
        r"""
        Returns the appearance GMRF model.

        :type: :map:`GMRFInstanceModel`
        """
        return self.interface.appearance_model

    @property
    def deformation_model(self):
        r"""
        Returns the deformation GMRF model.

        :type: :map:`GMRFInstanceModel`
        """
        return self.interface.deformation_model

    @property
    def transform(self):
        r"""
        Returns the motion model.

        :type: :map:`OrthoPDM`
        """
        return self.interface.transform

    @property
    def template(self):
        r"""
        Returns the template (usually the mean appearance).

        :type: :map:`Image`
        """
        return self.interface.template

    def _precompute(self):
        # grab number of shape parameters
        self.n = self.transform.n_parameters

        # grab appearance model precision
        self.Q_a = self.appearance_model.precision
        # mask it
        # TODO: UNCOMMENT THIS TO ENABLE MASKING
        # x, y = np.meshgrid(self.interface.i_mask,
        #                    self.interface.i_mask)
        # self.Q_a = self.Q_a[x, y]

        # grab appearance model mean
        self.a_bar = self.appearance_model.mean()
        # vectorize it and mask it
        self.a_bar_m = self.a_bar.as_vector()[self.interface.i_mask]


class Inverse(GaussNewton):
    r"""
    Inverse Gauss-Newton optimization of APS.
    """
    def _precompute(self):
        # call super method
        super(Inverse, self)._precompute()
        # compute shape jacobian
        ds_dp = self.interface.ds_dp()
        # compute model's gradient
        nabla_a = self.interface.gradient(self.template)
        # compute appearance jacobian
        J_a = self.interface.steepest_descent_images(nabla_a, ds_dp)
        # transposed appearance jacobian and precision dot product
        self._J_a_T_Q_a = self.interface.J_a_T_Q_a(J_a, self.Q_a)
        # compute hessian inverse
        self._H_S = None
        H = self._J_a_T_Q_a.dot(J_a)
        if self.interface.use_deformation_cost:
            self._H_s = self.interface.H_s()
            H += self._H_s
        self._inv_H = np.linalg.inv(H)

    def _algorithm_str(self):
        return 'Inverse Gauss-Newton'

    def run(self, image, initial_shape, gt_shape=None, max_iters=20):
        r"""
        Run the optimization.

        Parameters
        ----------
        image : :map:`Image`
            The test image.
        initial_shape : :map:`PointCloud`
            The shape to start from.
        gt_shape : :map:`PointCloud` or ``None``
            The ground truth shape of the image. If ``None``, then the
            fitting errors are not computed.
        max_iters : `int` or `list` of `int`
            The maximum number of iterations. If `list`, then a value is
            specified per level. If `int`, then this value will be used for
            all levels.
        """
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Inverse Gauss-Newton loop -------------------------------------

        # warp image
        self.i = self.interface.warp(image)
        # vectorize it and mask it
        i_m = self.i.as_vector()[self.interface.i_mask]

        # compute masked error
        self.e_m = i_m - self.a_bar_m

        # update cost_functions
        #cost_functions = [cost_closure(self.e_m, self.project_out)]
        cost_functions = []

        while k < max_iters and eps > self.eps:
            # compute gauss-newton parameter updates
            b = self._J_a_T_Q_a.dot(self.e_m)
            p = p_list[-1].copy()
            if self._H_s is not None:
                if isinstance(self.transform, OrthoPDM):
                    p[0:4] = 0
                b += self._H_s.dot(p)
            dp = self._inv_H.dot(b)

            # update warp
            s_k = self.transform.target.points
            self.transform.from_vector_inplace(self.transform.as_vector() - dp)
            p_list.append(self.transform.as_vector())

            # warp image
            self.i = self.interface.warp(image)
            # vectorize it and mask it
            i_m = self.i.as_vector()[self.interface.i_mask]

            # compute masked error
            self.e_m = i_m - self.a_bar_m

            # update cost
            #cost_functions.append(cost_closure(self.e_m, self.project_out))

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return fitting result
        return self.interface.algorithm_result(
            image, p_list, cost_functions=cost_functions, gt_shape=gt_shape)


class Forward(GaussNewton):
    r"""
    Forward Gauss-Newton optimization of APS.
    """
    def _precompute(self):
        # call super method
        super(Forward, self)._precompute()
        # compute shape jacobian
        self._ds_dp = self.interface.ds_dp()
        # compute shape hessian
        self._H_s = None
        if self.interface.use_deformation_cost:
            self._H_s = self.interface.H_s()

    def _algorithm_str(self):
        return 'Forward Gauss-Newton'

    def run(self, image, initial_shape, gt_shape=None, max_iters=20):
        r"""
        Run the optimization.

        Parameters
        ----------
        image : :map:`Image`
            The test image.
        initial_shape : :map:`PointCloud`
            The shape to start from.
        gt_shape : :map:`PointCloud` or ``None``
            The ground truth shape of the image. If ``None``, then the
            fitting errors are not computed.
        max_iters : `int` or `list` of `int`
            The maximum number of iterations. If `list`, then a value is
            specified per level. If `int`, then this value will be used for
            all levels.
        """
        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Forward Gauss-Newton loop -------------------------------------

        # warp image
        i = self.interface.warp(image)
        # vectorize it and mask it
        i_m = i.as_vector()[self.interface.i_mask]

        # compute masked error
        self.e_m = i_m - self.a_bar_m

        # update cost_functions
        #cost_functions = [cost_closure(self.e_m, self.project_out)]
        cost_functions = []

        while k < max_iters and eps > self.eps:

            # compute image gradient
            nabla_i = self.interface.gradient(i)

            # compute appearance jacobian
            Ja = self.interface.steepest_descent_images(nabla_i, self._ds_dp)

            # transposed jacobian and precision dot product
            J_a_T_Q_a = self.interface.J_a_T_Q_a(Ja, self.Q_a)

            # compute hessian
            H = J_a_T_Q_a.dot(Ja)
            if self.interface.use_deformation_cost:
                H += self._H_s

            # compute gauss-newton parameter updates
            b = J_a_T_Q_a.dot(self.e_m)
            p = p_list[-1].copy()
            if self._H_s is not None:
                if isinstance(self.transform, OrthoPDM):
                    p[0:4] = 0
                b += self._H_s.dot(p)
            dp = -np.linalg.solve(H, b)

            # update warp
            s_k = self.transform.target.points
            self.transform.from_vector_inplace(self.transform.as_vector() + dp)
            p_list.append(self.transform.as_vector())

            # warp image
            i = self.interface.warp(image)
            # vectorize it and mask it
            i_m = i.as_vector()[self.interface.i_mask]

            # compute masked error
            self.e_m = i_m - self.a_bar_m

            # update cost
            #cost_functions.append(cost_closure(self.e_m, self.project_out))

            # test convergence
            eps = np.abs(np.linalg.norm(s_k - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return fitting result
        return self.interface.algorithm_result(
            image, p_list, cost_functions=cost_functions, gt_shape=gt_shape)
