from __future__ import division

from menpofit import checks
from menpofit.fitter import ModelFitter

from .algorithm import InverseCompositional
from .result import ATMResult


class LucasKanadeATMFitter(ModelFitter):
    r"""
    Class for defining an ATM fitter using the Lucas-Kanade optimization.

    Parameters
    ----------
    atm : `menpofit.atm.base.ATM` or subclass
        The trained ATM model.
    lk_algorithm_cls : `menpofit.atm.algorithm.LucasKanade` or subclass, optional
        The Lukas-Kanade optimization algorithm that will get applied. All
        possible algorithms are stored in `menpofit.atm.algorithm`.
    n_shape : `int` or `list` or ``None``, optional
        The number of shape components that will be used. If `int`, then the
        provided value will be applied on all scales. If `list`, then it
        defines a value per scale. If ``None``, then all the components will
        be used.
    sampling : `int` or ``None``, optional
        The sub-sampling step of the sampling mask. If ``None``, then no
        sampling is applied on the template.
    """
    def __init__(self, atm, lk_algorithm_cls=InverseCompositional,
                 n_shape=None, sampling=None):
        self._model = atm
        checks.set_models_components(atm.shape_models, n_shape)
        self._sampling = checks.check_sampling(sampling, atm.n_scales)
        self._set_up(lk_algorithm_cls)

    @property
    def atm(self):
        r"""
        The trained ATM model.

        :type: `menpofit.atm.ATM` or subclass
        """
        return self._model

    def _set_up(self, lk_algorithm_cls):
        interfaces = self.atm.build_fitter_interfaces(self._sampling)
        self.algorithms = [lk_algorithm_cls(interface)
                           for interface in interfaces]

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return ATMResult(results=algorithm_results, scales=self.atm.scales,
                         affine_correction=affine_correction, image=image,
                         gt_shape=gt_shape)
