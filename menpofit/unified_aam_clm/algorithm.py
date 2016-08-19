import numpy as np

from menpofit.base import build_grid
from menpofit.checks import check_model
from menpofit.modelinstance import OrthoPDM

from .result import UnifiedAAMCLMAlgorithmResult

multivariate_normal = None  # expensive, from scipy.stats


# Abstract Interface for AAM Algorithms ---------------------------------------

class UnifiedAlgorithm(object):
    r"""
    Base interface for optimization of a Unified AAM-CLM model.

    Parameters
    ----------
    aam_interface : : `subclass` of :map:`LucasKanadeBaseInterface`, 
        Concrete instantiation of an interface for Lucas-Kanade optimization of
        standard AAMs.
    expert_ensemble : `subclass` of :map:`ExpertEnsemble`, 
        A trained ensemble of experts.     
    patch_shape : (`int`, `int`)
        The shape of the patches.
    response_covariance : `int`, optional
        The covariance of the generated Gaussian response.
    eps : `float`, optional
        Value for checking the convergence of the optimization.
    """
    def __init__(self, aam_interface, expert_ensemble, patch_shape,
                 response_covariance, eps=10**-5, **kwargs):

        # AAM part ------------------------------------------------------------
        self.interface = aam_interface
        self.appearance_model = self.interface.appearance_model
        self.template = self.appearance_model.mean()
        self.transform = self.interface.transform
        check_model(self.transform.pdm, OrthoPDM)

        # CLM part ------------------------------------------------------------
        self.expert_ensemble = expert_ensemble
        self.patch_shape = patch_shape
        self.response_covariance = response_covariance
        self.pdm = self.transform.pdm

        # Unified part --------------------------------------------------------
        self.eps = eps
        self._precompute()

    def _precompute(self, **kwargs):
        # Mask Appearance Model
        self._U = self.appearance_model.components.T
        self._pinv_U = np.linalg.pinv(
            self._U[self.interface.i_mask, :]).T
        # grab appearance model mean
        self.a_bar = self.appearance_model.mean()
        # vectorize it and mask it
        self.a_bar_m = self.a_bar.as_vector()[self.interface.i_mask]

    def run(self, image, initial_shape, max_iters=20, gt_shape=None,
            return_costs=False, **kwargs):
        pass

    def _update_warp(self,dp):
        self.transform._from_vector_inplace(self.transform.as_vector() + dp)

    def _precompute_clm(self):
        global multivariate_normal
        if multivariate_normal is None:
            from scipy.stats import multivariate_normal  # expensive

        # set inverse rho2
        self._inv_rho2 = self.pdm.model.inverse_noise_variance()

        # compute Gaussian-KDE grid
        self._sampling_grid = build_grid(self.patch_shape)
        mean = np.zeros(self.transform.n_dims)
        response_covariance = self.response_covariance * self._inv_rho2
        mvn = multivariate_normal(mean=mean, cov=response_covariance)
        self._kernel_grid = mvn.pdf(self._sampling_grid)

        # compute CLM jacobian
        j_clm = np.rollaxis(self.pdm.d_dp(None), -1, 1)
        j_clm = j_clm.reshape((-1, j_clm.shape[-1]))
        self._j_clm = self._inv_rho2 * j_clm

        # compute CLM hessian
        self._h_clm = self._j_clm.T.dot(j_clm)

    def _compute_clm_error(self, image):
        target = self.transform.target
        # get all (x, y) pairs being considered
        yxs = target.points[:, None, None, ...] + self._sampling_grid

        # compute parts response
        parts_response = self.expert_ensemble.predict_probability(
            image, target).squeeze()
        parts_response[np.logical_not(np.isfinite(parts_response))] = .5

        # compute parts kernel
        parts_kernel = parts_response * self._kernel_grid
        parts_kernel /= np.sum(parts_kernel, axis=(-2, -1))[..., None, None]

        # compute mean shift target
        mean_shift_target = np.sum(parts_kernel[..., None] * yxs,
                                   axis=(-3, -2))

        # compute (shape) error term
        return mean_shift_target.ravel() - target.as_vector()


# Concrete Implementations of AAM Algorithm -----------------------------------


class ProjectOutRegularisedLandmarkMeanShift(UnifiedAlgorithm):
    r"""
    Project-Out Inverse Compositional + Regularized Landmark Mean Shift
    """
    def _precompute(self):
        super(ProjectOutRegularisedLandmarkMeanShift, self)._precompute()
        # AAM part ------------------------------------------------------------

        # sample appearance model
        self._U = self._U[self.interface.i_mask, :]

        # compute model's gradient
        nabla_t = self.interface.gradient(self.template)

        # compute warp jacobian
        dw_dp = self.interface.warp_jacobian()

        # set inverse sigma2
        self._inv_sigma2 = self.appearance_model.inverse_noise_variance()

        # compute AAM jacobian
        j = self.interface.steepest_descent_images(nabla_t, dw_dp)
        j_po = j - self._U.dot(self._pinv_U.T.dot(j))
        self._j_aam = self._inv_sigma2 * j_po

        # compute inverse hessian
        self._h_aam = self._j_aam.T.dot(j_po)

        # CLM part ------------------------------------------------------------
        self._precompute_clm()        

        # Unified part --------------------------------------------------------

        # set Prior
        sim_prior = np.zeros((4,))
        transform_prior = 1 / self.pdm.model.eigenvalues
        self._j_prior = np.hstack((sim_prior, transform_prior))

        # compute Unified hessian inverse and jacobian pseudo-inverse
        h = self._h_aam + self._h_clm
        self._pinv_j_aam = np.linalg.solve(h, self._j_aam.T)
        self._pinv_j_clm = np.linalg.solve(h, self._j_clm.T)
        self._inv_h_prior = np.linalg.inv(h + np.diag(self._j_prior))

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            return_costs=False, prior=False, a=0.5):
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
        return_costs : `bool`, optional
            If ``True``, then the cost function values will be computed
            during the fitting procedure. Then these cost values will be
            assigned to the returned `fitting_result`. Note that the costs
            computation increases the computational cost of the fitting. The
            additional computation cost depends on the fitting method. Only
            use this option for research purposes.
        prior : `bool`, optional
            If ``True``, use a Gaussian priors over the latent shape and
            appearance spaces.
            see the reference [1] section 3.1.1 for details.
        a : `float`, optional
            Ratio of the image noise variance and the shape noise variance.
            See [1] section 5 equations (25) & (26) and footnote 6.

        Returns
        -------
        fitting_result : :map:`UnifiedAAMCLMAlgorithmResult`
            The parametric iterative fitting result.

        References
        ----------
        .. [1] J. Alabort-i-Medina, and S. Zafeiriou. "Unifying holistic and
            parts-based deformable model fitting." Proceedings of the IEEE
            Conference on Computer Vision and Pattern Recognition. 2015.
        """
        # define cost closure
        def cost_closure(x, y, a):
            return a * x.T.dot(x) + y.T.dot(y)

        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]
        shapes = [self.transform.target]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # AAM part --------------------------------------------------------
        # warp image
        i = self.interface.warp(image)
        # vectorize it and mask it
        masked_i = i.as_vector()[self.interface.i_mask]

        # compute masked error
        e_aam = self.a_bar_m - masked_i

        # CLM part --------------------------------------------------------
        e_clm = self._compute_clm_error(image)

        # update costs
        costs = None
        if return_costs:
            costs = [cost_closure(e_aam, e_clm, a)]

        while k < max_iters and eps > self.eps:
            # compute gauss-newton parameter updates
            if prior:
                b = (self._j_prior * self.transform.as_vector() -
                     a * self._j_aam.T.dot(e_aam) -
                     (1 - a) * self._j_clm.T.dot(e_clm))
                dp = -self._inv_h_prior.dot(b)
            else:
                dp = self._pinv_j_aam.dot(e_aam) + self._pinv_j_clm.dot(e_clm)

            # update warp
            target = self.transform.target
            self._update_warp(dp)
            p_list.append(self.transform.as_vector())
            shapes.append(self.transform.target)

            # AAM part --------------------------------------------------------
            # warp image
            i = self.interface.warp(image)
            # vectorize it and mask it
            masked_i = i.as_vector()[self.interface.i_mask]

            # compute masked error
            e_aam = self.a_bar_m - masked_i

            # CLM part --------------------------------------------------------
            e_clm = self._compute_clm_error(image)

            # update costs
            if return_costs:
                costs.append(cost_closure(e_aam, e_clm, a))

            # test convergence
            eps = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return UnifiedAAMCLMAlgorithmResult(
            shapes=shapes, shape_parameters=p_list, appearance_parameters=None,
            initial_shape=initial_shape, image=image, gt_shape=gt_shape,
            costs=costs)


class AlternatingRegularisedLandmarkMeanShift(UnifiedAlgorithm):
    r"""
    Alternating Inverse Compositional + Regularized Landmark Mean Shift
    """
    def _precompute(self):
        super(AlternatingRegularisedLandmarkMeanShift, self)._precompute()
        # AAM part ------------------------------------------------------------

        # compute warp jacobian
        self._dw_dp = self.interface.warp_jacobian()

        # set inverse sigma2
        self._inv_sigma2 = self.appearance_model.inverse_noise_variance()

        # CLM part ------------------------------------------------------------
        self._precompute_clm()

        # Unified part --------------------------------------------------------

        # set Prior
        sim_prior = np.zeros((4,))
        transform_prior = 1 / self.pdm.model.eigenvalues
        self._j_prior = np.hstack((sim_prior, transform_prior))
        self._h_prior = np.diag(self._j_prior)

    def run(self, image, initial_shape, gt_shape=None, max_iters=20,
            return_costs=False, prior=False, a=0.5):
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
        return_costs : `bool`, optional
            If ``True``, then the cost function values will be computed
            during the fitting procedure. Then these cost values will be
            assigned to the returned `fitting_result`. Note that the costs
            computation increases the computational cost of the fitting. The
            additional computation cost depends on the fitting method. Only
            use this option for research purposes.        
        prior : `bool`, optional
            If ``True``, use a Gaussian priors over the latent shape and
            appearance spaces.
            see the reference [1] section 3.1.1 for details.
        a : `float`, optional
            Ratio of the image noise variance and the shape noise variance.
            See [1] section 5 equations (25) & (26) and footnote 6.

        Returns
        -------
        fitting_result : :map:`UnifiedAAMCLMAlgorithmResult`
            The parametric iterative fitting result.

        References
        ----------
        .. [1] J. Alabort-i-Medina, and S. Zafeiriou. "Unifying holistic and
            parts-based deformable model fitting." Proceedings of the IEEE
            Conference on Computer Vision and Pattern Recognition. 2015.
        """
        # define cost closure
        def cost_closure(x, y, a):
            return a * x.T.dot(x) + y.T.dot(y)

        # initialize transform
        self.transform.set_target(initial_shape)
        p_list = [self.transform.as_vector()]
        shapes = [self.transform.target]

        # initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # AAM part --------------------------------------------------------

        # warp image
        i = self.interface.warp(image)
        # mask warped image
        masked_i = i.as_vector()[self.interface.i_mask]

        # reconstruct appearance
        c = self._pinv_U.T.dot(masked_i - self.a_bar_m)
        t = self._U.dot(c) + self.a_bar.as_vector()
        self.template = self.template.from_vector(t)
        c_list = [c]

        # compute (image) error
        e_aam = (self.template.as_vector()[self.interface.i_mask] - masked_i)

        # CLM part --------------------------------------------------------
        e_clm = self._compute_clm_error(image)

        # update costs
        costs = None
        if return_costs:
            costs = [cost_closure(e_aam, e_clm, a)]

        while k < max_iters and eps > self.eps:
            # compute model gradient
            nabla_t = self.interface.gradient(self.template)

            # compute AAM jacobian
            j = self.interface.steepest_descent_images(nabla_t, self._dw_dp)
            j_aam = self._inv_sigma2 * j

            # compute AAM hessian
            h_aam = j_aam.T.dot(j)

            # compute Gauss-Newton parameter updates
            if prior:
                h = a * h_aam + (1 - a) * self._h_clm + self._h_prior
                b = (self._j_prior * self.transform.as_vector() -
                     a * j_aam.T.dot(e_aam) -
                     (1 - a) * self._j_clm.T.dot(e_clm))
                dp = -np.linalg.solve(h, b)
            else:
                dp = np.linalg.solve(a * h_aam + (1 - a) * self._h_clm,
                                     a * j_aam.T.dot(e_aam) +
                                     (1 - a) * self._j_clm.T.dot(e_clm))

            # update warp
            target = self.transform.target
            self._update_warp(dp)
            p_list.append(self.transform.as_vector())
            shapes.append(self.transform.target)

            # warp image
            i = self.interface.warp(image)
            # mask warped image
            masked_i = i.as_vector()[self.interface.i_mask]

            # update appearance parameters
            c = self._pinv_U.T.dot(masked_i - self.a_bar_m)
            t = self._U.dot(c) + self.a_bar.as_vector()
            self.template = self.template.from_vector(t)
            c_list.append(c)

            # compute (image) error
            e_aam = (self.template.as_vector()[self.interface.i_mask] -
                     masked_i)
            e_clm = self._compute_clm_error(image)

            # update costs
            if return_costs:
                costs.append(cost_closure(e_aam, e_clm, a))

            # test convergence
            eps = np.abs(np.linalg.norm(
                target.points - self.transform.target.points))

            # increase iteration counter
            k += 1

        # return algorithm result
        return UnifiedAAMCLMAlgorithmResult(
            shapes=shapes, shape_parameters=p_list, appearance_parameters=c_list,
            initial_shape=initial_shape, image=image, gt_shape=gt_shape,
            costs=costs)
