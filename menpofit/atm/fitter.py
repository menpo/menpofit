from __future__ import division
from menpofit import checks
from menpofit.fitter import ModelFitter
from menpofit.modelinstance import OrthoPDM
from menpofit.transform import OrthoMDTransform, LinearOrthoMDTransform
from .base import ATM, MaskedATM, LinearATM, LinearMaskedATM, PatchATM
from .algorithm import (
    ATMLKStandardInterface, ATMLKPatchInterface, ATMLKLinearInterface,
    InverseCompositional)
from .result import ATMFitterResult


# TODO: document me!
class LucasKanadeATMFitter(ModelFitter):
    r"""
    """
    def __init__(self, atm, algorithm_cls=InverseCompositional,
                 n_shape=None, sampling=None):
        self._model = atm
        checks.set_models_components(atm.shape_models, n_shape)
        self._sampling = checks.check_sampling(sampling, atm.n_scales)
        self._set_up(algorithm_cls)

    @property
    def atm(self):
        return self._model

    def _set_up(self, algorithm_cls):
        self.algorithms = []
        for j, (wt, sm, s) in enumerate(zip(self.atm.warped_templates,
                                            self.atm.shape_models,
                                            self._sampling)):

            if type(self.atm) is ATM or type(self.atm) is MaskedATM:
                source_lmarks = wt.landmarks['source'].lms
                md_transform = OrthoMDTransform(sm, self.atm.transform,
                                                source=source_lmarks)
                interface = ATMLKStandardInterface(md_transform, wt, sampling=s)
                algorithm = algorithm_cls(interface)
            elif (type(self.atm) is LinearATM or
                  type(self.atm) is LinearMaskedATM):
                md_transform = LinearOrthoMDTransform(sm,
                                                      self.atm.reference_shape)
                interface = ATMLKLinearInterface(md_transform, wt, sampling=s)
                algorithm = algorithm_cls(interface)
            elif type(self.atm) is PatchATM:
                pdm = OrthoPDM(sm)
                interface = ATMLKPatchInterface(
                    pdm, wt, sampling=s, patch_size=self.atm.patch_size[j],
                    normalize_parts=self.atm.normalize_parts)
                algorithm = algorithm_cls(interface)
            else:
                raise ValueError("AAM object must be of one of the "
                                 "following classes: {}, {}, {}, {}, "
                                 "{}".format(ATM, MaskedATM, LinearATM,
                                             LinearMaskedATM, PatchATM))
            self.algorithms.append(algorithm)

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return ATMFitterResult(image, self, algorithm_results,
                               affine_correction, gt_shape=gt_shape)
