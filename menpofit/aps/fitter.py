from menpofit.fitter import MultiScaleParametricFitter
import menpofit.checks as checks

from .result import APSResult
from .algorithm.gn import GaussNewtonBaseInterface, Inverse


class APSFitter(MultiScaleParametricFitter):
    r"""
    Abstract class for defining an APS fitter.

    .. note:: When using a method with a parametric shape model, the first step
              is to **reconstruct the initial shape** using the shape model. The
              generated reconstructed shape is then used as initialisation for
              the iterative optimisation. This step takes place at each scale
              and it is not considered as an iteration, thus it is not counted
              for the provided `max_iters`.

    Parameters
    ----------
    aps : :map:`GenerativeAPS` or `subclass`
        The trained APS model.
    algorithms : `list` of `class`
        The list of algorithm objects that will perform the fitting per scale.
    """
    def __init__(self, aps, algorithms):
        self._model = aps
        # Call superclass
        super(APSFitter, self).__init__(
            scales=aps.scales, reference_shape=aps.reference_shape,
            holistic_features=aps.holistic_features, algorithms=algorithms)

    @property
    def aps(self):
        r"""
        The trained APS model.

        :type: :map:`GenerativeAPS` or subclass
        """
        return self._model

    def _fitter_result(self, image, algorithm_results, affine_transforms,
                       scale_transforms, gt_shape=None):
        r"""
        Function the creates the multi-scale fitting result object.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image that was fitted.
        algorithm_results : `list` of :map:`APSAlgorithmResult` or subclass
            The list of fitting result per scale.
        affine_transforms : `list` of `menpo.transform.Affine`
            The list of affine transforms per scale that are the inverses of the
            transformations introduced by the rescale wrt the reference shape as
            well as the feature extraction.
        scale_transforms : `list` of `menpo.shape.Scale`
            The list of inverse scaling transforms per scale.
        gt_shape : `menpo.shape.PointCloud`, optional
            The ground truth shape associated to the image.

        Returns
        -------
        fitting_result : :map:`APSResult` or subclass
            The multi-scale fitting result containing the result of the fitting
            procedure.
        """
        return APSResult(results=algorithm_results, scales=self.scales,
                         affine_transforms=affine_transforms,
                         scale_transforms=scale_transforms, image=image,
                         gt_shape=gt_shape)


class GaussNewtonAPSFitter(APSFitter):
    r"""
    A class for fitting an APS model with Gauss-Newton optimization.

    .. note:: When using a method with a parametric shape model, the first step
              is to **reconstruct the initial shape** using the shape model. The
              generated reconstructed shape is then used as initialisation for
              the iterative optimisation. This step takes place at each scale
              and it is not considered as an iteration, thus it is not counted
              for the provided `max_iters`.

    Parameters
    ----------
    aps : :map:`GenerativeAPS` or subclass
        The trained model.
    gn_algorithm_cls : `class`, optional
        The Gauss-Newton optimisation algorithm that will get applied. The
        possible algorithms are :map:`Inverse` and :map:`Forward`. Note that
        the :map:`Forward` algorithm is too slow. It is not recommended to be
        used for fitting an APS and is only included for comparison purposes.
    n_shape : `int` or `float` or `list` of those or ``None``, optional
        The number of shape components that will be used. If `int`, then it
        defines the exact number of active components. If `float`, then it
        defines the percentage of variance to keep. If `int` or `float`, then
        the provided value will be applied for all scales. If `list`, then it
        defines a value per scale. If ``None``, then all the available
        components will be used. Note that this simply sets the active
        components without trimming the unused ones. Also, the available
        components may have already been trimmed to `max_shape_components`
        during training.
    use_deformation_cost : `bool`, optional
        If ``True``, then the deformation cost is also included in the
        Hessian calculation.
    sampling : `list` of `int` or `ndarray` or ``None``
        It defines a sampling mask per scale. If `int`, then it defines the
        sub-sampling step of the sampling mask. If `ndarray`, then it
        explicitly defines the sampling mask. If ``None``, then no
        sub-sampling is applied.
    """
    def __init__(self, aps, gn_algorithm_cls=Inverse, n_shape=None,
                 use_deformation_cost=True, sampling=None):
        # Check parameters
        checks.set_models_components(aps.shape_models, n_shape)
        self._sampling = checks.check_sampling(sampling, aps.n_scales)

        # Get list of algorithm objects per scale
        algorithms = []
        for j in list(range(aps.n_scales)):
            # create the interface object
            interface = GaussNewtonBaseInterface(
                aps.appearance_models[j], aps.deformation_models[j],
                aps.shape_models[j], use_deformation_cost,
                aps.appearance_models[j].mean(), self._sampling[j],
                aps.patch_shape[j], aps.patch_normalisation[j])

            # create the algorithm object and append it
            algorithms.append(gn_algorithm_cls(interface))

        # Call superclass
        super(GaussNewtonAPSFitter, self).__init__(aps=aps,
                                                   algorithms=algorithms)
