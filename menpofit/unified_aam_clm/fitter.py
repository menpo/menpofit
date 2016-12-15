from menpo.base import name_of_callable
from menpofit import checks
from menpofit.fitter import MultiScaleParametricFitter

from .algorithm import AlternatingRegularisedLandmarkMeanShift
from .result import UnifiedAAMCLMResult


class UnifiedAAMCLMFitter(MultiScaleParametricFitter):
    r"""
    Class defining a Unified AAM - CLM fitter.

    .. note:: When using a method with a parametric shape model, the first step
              is to **reconstruct the initial shape** using the shape model. The
              generated reconstructed shape is then used as initialisation for
              the iterative optimisation. This step takes place at each scale
              and it is not considered as an iteration, thus it is not counted
              for the provided `max_iters`.

    Parameters
    ----------
    unified_aam_clm : :map:`UnifiedAAMCLM` or subclass
        The trained unified AAM-CLM model.
    algorithm_cls : `class`, optional
        The unified optimisation algorithm that will get applied. The
        possible algorithms are:

        ============================================== =====================
        Class                                          Method
        ============================================== =====================
        :map:`ProjectOutRegularisedLandmarkMeanShift`  Project-Out IC + RLMS
        :map:`AlternatingRegularisedLandmarkMeanShift` Alternating IC + RLMS
        ============================================== =====================

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
    n_appearance : `int` or `float` or `list` of those or ``None``, optional
        The number of appearance components that will be used. If `int`, then it
        defines the exact number of active components. If `float`, then it
        defines the percentage of variance to keep. If `int` or `float`, then
        the provided value will be applied for all scales. If `list`, then it
        defines a value per scale. If ``None``, then all the available
        components will be used. Note that this simply sets the active
        components without trimming the unused ones. Also, the available
        components may have already been trimmed to `max_appearance_components`
        during training.
    sampling : `list` of `int` or `ndarray` or ``None``
        It defines a sampling mask per scale. If `int`, then it defines the
        sub-sampling step of the sampling mask. If `ndarray`, then it
        explicitly defines the sampling mask. If ``None``, then no
        sub-sampling is applied.
    """
    def __init__(self, unified_aam_clm,
                 algorithm_cls=AlternatingRegularisedLandmarkMeanShift,
                 n_shape=None, n_appearance=None, sampling=None):
        self._model = unified_aam_clm
        # Check parameters
        checks.set_models_components(self._model.shape_models, n_shape)
        checks.set_models_components(self._model.appearance_models,
                                     n_appearance)
        self._sampling = checks.check_sampling(sampling, self._model.n_scales)

        # Get list of algorithm objects per scale
        interfaces = unified_aam_clm.build_fitter_interfaces(self._sampling)
        algorithms = [algorithm_cls(interface,
                                    self._model.expert_ensembles[k],
                                    self._model.patch_shape[k],
                                    self._model.response_covariance)
                      for k, interface in enumerate(interfaces)]

        # Call superclass
        super(UnifiedAAMCLMFitter, self).__init__(
            scales=self._model.scales,
            reference_shape=self._model.reference_shape,
            holistic_features=self._model.holistic_features,
            algorithms=algorithms)

    @property
    def unified_aam_clm(self):
        r"""
        The trained unified AAM-CLM model.

        :type: :map:`UnifiedAAMCLM` or `subclass`
        """
        return self._model

    def appearance_reconstructions(self, appearance_parameters,
                                   n_iters_per_scale):
        r"""
        Method that generates the appearance reconstructions given a set of
        appearance parameters. This is to be combined with a
        :map:`UnifiedAAMCLMResult` object, in order to generate the
        appearance reconstructions of a fitting procedure.

        Parameters
        ----------
        appearance_parameters : `list` of ``(n_params,)`` `ndarray`
            A set of appearance parameters per fitting iteration. It can be
            retrieved as a property of an :map:`UnifiedAAMCLMResult` object.
        n_iters_per_scale : `list` of `int`
            The number of iterations per scale. This is necessary in order to
            figure out which appearance parameters correspond to the model of
            each scale. It can be retrieved as a property of a
            :map:`UnifiedAAMCLMResult` object.

        Returns
        -------
        appearance_reconstructions : `list` of `menpo.image.Image`
            `List` of the appearance reconstructions that correspond to the
            provided parameters.
        """
        return self.unified_aam_clm.appearance_reconstructions(
            appearance_parameters=appearance_parameters,
            n_iters_per_scale=n_iters_per_scale)

    def warped_images(self, image, shapes):
        r"""
        Given an input test image and a list of shapes, it warps the image
        into the shapes. This is useful for generating the warped images of a
        fitting procedure stored within an :map:`UnifiedAAMCLMResult`.

        Parameters
        ----------
        image : `menpo.image.Image` or `subclass`
            The input image to be warped.
        shapes : `list` of `menpo.shape.PointCloud`
            The list of shapes in which the image will be warped. The shapes
            are obtained during the iterations of a fitting procedure.

        Returns
        -------
        warped_images : `list` of `menpo.image.MaskedImage` or `ndarray`
            The warped images.
        """
        return self.algorithms[-1].interface.warped_images(image=image,
                                                           shapes=shapes)

    @property
    def response_covariance(self):
        r"""
        Returns the covariance value of the desired Gaussian response used to
        train the ensemble of experts.

        :type: `int`
        """
        return self._model.response_covariance

    def _fitter_result(self, image, algorithm_results, affine_transforms,
                       scale_transforms, gt_shape=None):
        r"""
        Function the creates the multi-scale fitting result object.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image that was fitted.
        algorithm_results : `list` of :map:`AAMAlgorithmResult` or subclass
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
        fitting_result : :map:`UnifiedAAMCLMResult` or subclass
            The multi-scale fitting result containing the result of the fitting
            procedure.
        """
        return UnifiedAAMCLMResult(
            results=algorithm_results, scales=self.scales,
            affine_transforms=affine_transforms,
            scale_transforms=scale_transforms, image=image, gt_shape=gt_shape)

    def __str__(self):
        # Compute scale info strings
        scales_info = []
        lvl_str_tmplt = r"""  - Scale {}
     - {} active shape components
     - {} similarity transform components
     - {} active appearance components"""
        for k, s in enumerate(self.scales):
            scales_info.append(lvl_str_tmplt.format(
                    s,
                    self._model.shape_models[k].model.n_active_components,
                    self._model.shape_models[k].n_global_parameters,
                    self._model.appearance_models[k].n_active_components))
        scales_info = '\n'.join(scales_info)

        cls_str = r"""{class_title}
  - Scales: {scales}
{scales_info}
    """.format(class_title=name_of_callable(self.algorithms[0]),
               scales=self.scales,
               scales_info=scales_info)
        return self._model.__str__() + cls_str
