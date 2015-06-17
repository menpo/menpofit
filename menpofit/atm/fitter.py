from __future__ import division
from menpofit.fitter import ModelFitter
from menpofit.modelinstance import OrthoPDM
from menpofit.transform import OrthoMDTransform, LinearOrthoMDTransform
from .base import ATM, PatchATM, LinearATM, LinearPatchATM, PartsATM
from .algorithm import (
    LKATMInterface, LKLinearATMInterface, LKPartsATMInterface, IC)
from .result import ATMFitterResult


# TODO: document me!
class LKATMFitter(ModelFitter):
    r"""
    """
    def __init__(self, atm, algorithm_cls=IC, n_shape=None, sampling=None,
                 **kwargs):
        super(LKATMFitter, self).__init__(atm)
        self._algorithms = []
        self._check_n_shape(n_shape)
        self._set_up(algorithm_cls, sampling, **kwargs)

    def _set_up(self, algorithm_cls, sampling, **kwargs):
        for j, (wt, sm) in enumerate(zip(self.atm.warped_templates,
                                         self.atm.shape_models)):

            if type(self.atm) is ATM or type(self.atm) is PatchATM:
                # build orthonormal model driven transform
                md_transform = OrthoMDTransform(
                    sm, self.atm.transform,
                    source=wt.landmarks['source'].lms)
                # set up algorithm using standard aam interface
                algorithm = algorithm_cls(LKATMInterface, wt, md_transform,
                                          sampling=sampling, **kwargs)

            elif (type(self.atm) is LinearATM or
                  type(self.atm) is LinearPatchATM):
                # build linear version of orthogonal model driven transform
                md_transform = LinearOrthoMDTransform(
                    sm, self.atm.n_landmarks)
                # set up algorithm using linear aam interface
                algorithm = algorithm_cls(LKLinearATMInterface, wt,
                                          md_transform, sampling=sampling,
                                          **kwargs)

            elif type(self.atm) is PartsATM:
                # build orthogonal point distribution model
                pdm = OrthoPDM(sm)
                # set up algorithm using parts aam interface
                algorithm = algorithm_cls(
                    LKPartsATMInterface, wt, pdm, sampling=sampling,
                    patch_shape=self.atm.patch_shape[j],
                    normalize_parts=self.atm.normalize_parts)

            else:
                raise ValueError("AAM object must be of one of the "
                                 "following classes: {}, {}, {}, {}, "
                                 "{}".format(ATM, PatchATM, LinearATM,
                                             LinearPatchATM, PartsATM))

            # append algorithms to list
            self._algorithms.append(algorithm)

    @property
    def atm(self):
        return self._model

    @property
    def algorithms(self):
        return self._algorithms

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return ATMFitterResult(image, self, algorithm_results,
                               affine_correction, gt_shape=gt_shape)
