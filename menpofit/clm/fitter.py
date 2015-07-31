from menpofit import checks
from menpofit.fitter import ModelFitter
from menpofit.modelinstance import OrthoPDM
from .algorithm import CLMAlgorithm, RegularisedLandmarkMeanShift
from .result import CLMFitterResult


# TODO: Document me!
class CLMFitter(ModelFitter):
    r"""
    """
    @property
    def clm(self):
        return self._model

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return CLMFitterResult(image, self, algorithm_results,
                               affine_correction, gt_shape=gt_shape)


# TODO: Document me!
# TODO: Rethink shape model and OrthoPDM relation
class GradientDescentCLMFitter(CLMFitter):
    r"""
    """
    def __init__(self, clm, gd_algorithm_cls=RegularisedLandmarkMeanShift,
                 n_shape=None):
        self._model = clm
        self._gd_algorithms_cls = checks.check_algorithm_cls(
            gd_algorithm_cls, self.n_scales, CLMAlgorithm)
        self._check_n_shape(n_shape)

        # Construct algorithms
        self.algorithms = []
        for i in range(self.clm.n_scales):
            pdm = OrthoPDM(self.clm.shape_models[i])
            algorithm = self._gd_algorithms_cls[i](
                self.clm.expert_ensembles[i], pdm)
            self.algorithms.append(algorithm)


# TODO: Implement me!
# TODO: Document me!
class SupervisedDescentCLMFitter(CLMFitter):
    r"""
    """
