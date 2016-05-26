from menpo.base import name_of_callable
from menpofit import checks
from menpofit.fitter import MultiScaleParametricFitter

from .algorithm import AICRLMS


class UnifiedAAMCLMFitter(MultiScaleParametricFitter):
    r"""
    Class defining an unified AAM - CLM fitter 

    Parameters
    ----------
    unified_aam_clm : :map:`UnifiedAAMCLM`
        The trained unified model.
    algorithm_cls : `class`, optional
        The unified optimisation algorithm that will get applied.       
        The possible algorithms are:

        ================ ====================================================================
        Class            Method
        ================ ====================================================================
        :map:`PICRLMS`   Project-Out Inverse Compositional + Regularized Landmark Mean Shift
        :map:`AICRLMS`   Alternating Inverse Compositional + Regularized Landmark Mean Shift
        ================ ====================================================================

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
    def __init__(self, unified_aam_clm, algorithm_cls=AICRLMS,
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

    def warped_images(self, image, shapes):
        r"""
        Given an input test image and a list of shapes, it warps the image
        into the shapes. This is useful for generating the warped images of a
        fitting procedure stored within an :map:`AAMResult`.

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
        return self._model.response_covariance

    def __str__(self):
        # Compute scale info strings
        scales_info = []
        lvl_str_tmplt = r"""  - Scale {}
    - {} active shape components
    - {} active appearance components"""
        for k, s in enumerate(self.scales):
            scales_info.append(lvl_str_tmplt.format(
                    s,
                    self._model.shape_models[k].model.n_active_components,
                    self._model.appearance_models[k].n_active_components))
        scales_info = '\n'.join(scales_info)

        cls_str = r"""{class_title}
  - Scales: {scales}
{scales_info}
    """.format(class_title=name_of_callable(self.algorithms[0]),
               scales=self.scales,
               scales_info=scales_info)
        return self._model.__str__() + cls_str
