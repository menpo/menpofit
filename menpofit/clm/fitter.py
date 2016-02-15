from menpofit.fitter import ModelFitter
from menpofit import checks

from .algorithm import RegularisedLandmarkMeanShift
from .result import CLMResult


class CLMFitter(ModelFitter):
    r"""
    Abstract class for defining a CLM fitter.
    """
    @property
    def clm(self):
        r"""
        The trained CLM model.

        :type: :map:`CLM` or `subclass`
        """
        return self._model

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return CLMResult(results=algorithm_results, scales=self.clm.scales,
                         affine_correction=affine_correction, image=image,
                         gt_shape=gt_shape)


class GradientDescentCLMFitter(CLMFitter):
    r"""
    Class for defining an CLM fitter using gradient descent optimization.

    Parameters
    ----------
    clm : :map:`CLM` or `subclass`
        The trained CLM model.
    gd_algorithm_cls : `class`, optional
        The gradient descent optimisation algorithm that will get applied. The
        possible options are :map:`RegularisedLandmarkMeanShift` and
        :map:`ActiveShapeModel`.
    n_shape : `int` or `list` or ``None``, optional
        The number of shape components that will be used. If `int`, then the
        provided value will be applied on all scales. If `list`, then it
        defines a value per scale. If ``None``, then all the components will
        be used.
    """
    def __init__(self, clm, gd_algorithm_cls=RegularisedLandmarkMeanShift,
                 n_shape=None):
        self._model = clm
        checks.set_models_components(clm.shape_models, n_shape)
        self._set_up(gd_algorithm_cls)

    def _set_up(self, gd_algorithm_cls ):
        self.algorithms = [gd_algorithm_cls(self.clm.expert_ensembles[i],
                                            self.clm.shape_models[i])
                           for i in range(self.clm.n_scales)]

    def __str__(self):
        # Compute scale info strings
        scales_info = []
        lvl_str_tmplt = r"""   - Scale {}
     - {} active shape components
     - {} similarity transform components"""
        for k, s in enumerate(self.scales):
            scales_info.append(lvl_str_tmplt.format(
                    s,
                    self.clm.shape_models[k].n_active_components,
                    self.clm.shape_models[k].n_global_parameters))
        scales_info = '\n'.join(scales_info)

        cls_str = r"""{class_title}
 - Scales: {scales}
{scales_info}
    """.format(class_title=self.algorithms[0].__str__(),
               scales=self.scales,
               scales_info=scales_info)
        return self.clm.__str__() + cls_str
