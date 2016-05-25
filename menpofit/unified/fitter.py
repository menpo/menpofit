import abc
import numpy as np
from menpo.transform import Scale, AlignmentAffine
from menpofit import checks
from menpofit.transform import OrthoMDTransform
from menpofit.modelinstance import OrthoPDM
from menpofit.aam.algorithm.lk import LucasKanadeStandardInterface
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

        ============================================== =====================
        Class                                          Method
        ============================================== =====================
        :map:`PICRLMS`                                 Project-Out Inverse Compositional + Regularized Landmark Mean Shift
        :map:`AICRLMS`                                 Alternating Inverse Compositional + Regularized Landmark Mean Shift  
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
    def __init__(self, unified_aam_clm, algorithm_cls=AICRLMS,
                 n_shape=None, n_appearance=None, sampling=None):
        # Check parameters
        checks.set_models_components(unified_aam_clm.shape_models, n_shape)
        checks.set_models_components(unified_aam_clm.appearance_models, n_appearance)
        self._sampling = checks.check_sampling(sampling, unified_aam_clm.n_scales)

        self.dm = unified_aam_clm
        algorithms = []

        # Get list of algorithm objects per scale
        for j, (am, ee, sm) in enumerate(zip(self.dm.appearance_models,
                                             self.dm.expert_ensembles,
                                             self.dm.shape_models)):

            # shape_model_cls must be OrthoPDM or a subclass
            pdm = self.dm.shape_model_cls[j](sm) 
            md_transform = OrthoMDTransform(
                pdm, self.dm.transform,
                source=am.mean().landmarks['source'].lms)

            template = am.mean()
            interface = LucasKanadeStandardInterface(am, md_transform, template, None)

            algorithm = algorithm_cls(
                interface, am, md_transform,
                ee, self.dm.patch_shape[j], 
                self.response_covariance)

            algorithms.append(algorithm)

        # Call superclass
        super(MultiScaleParametricFitter, self).__init__(
            scales=self.dm.scales, reference_shape=self.dm.reference_shape,
            holistic_features=self.dm.holistic_features, algorithms=algorithms)

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
        return self.dm.response_covariance

    def __str__(self):
        # Compute scale info strings
        scales_info = []
        lvl_str_tmplt = r"""   - Scale {}
     - {} active shape components
     - {} similarity transform components
     - {} active appearance components"""
        for k, s in enumerate(self.scales):
            scales_info.append(lvl_str_tmplt.format(
                    s,
                    self.dm.shape_models[k].n_active_components,
                    self.dm.shape_models[k].n_global_parameters,
                    self.dm.appearance_models[k].n_active_components))
        scales_info = '\n'.join(scales_info)

        cls_str = r"""{class_title}
 - Scales: {scales}
{scales_info}
    """.format(class_title=self.algorithms[0].__str__(),
               scales=self.scales,
               scales_info=scales_info)
        return self.aam.__str__() + cls_str

