from __future__ import division
from menpofit import checks
from menpofit.fitter import ModelFitter
from .algorithm import InverseCompositional
from .result import ATMFitterResult


# TODO: document me!
class LucasKanadeATMFitter(ModelFitter):
    r"""
    """
    def __init__(self, atm, lk_algorithm_cls=InverseCompositional,
                 n_shape=None, sampling=None):
        self._model = atm
        checks.set_models_components(atm.shape_models, n_shape)
        self._sampling = checks.check_sampling(sampling, atm.n_scales)
        self._set_up(lk_algorithm_cls)

    @property
    def atm(self):
        return self._model

    def _set_up(self, lk_algorithm_cls):
        interfaces = self.atm.build_fitter_interfaces(self._sampling)
        self.algorithms = [lk_algorithm_cls(interface)
                           for interface in interfaces]

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return ATMFitterResult(image, self, algorithm_results,
                               affine_correction, gt_shape=gt_shape)
