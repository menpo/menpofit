from __future__ import division
from menpofit.fitter import ModelFitter
from menpofit.modelinstance import OrthoPDM
from menpofit.transform import OrthoMDTransform, LinearOrthoMDTransform
from .base import AAM, PatchAAM, LinearAAM, LinearPatchAAM, PartsAAM
from .algorithm import AAMInterface, LinearAAMInterface, PartsAAMInterface, AIC
from .result import AAMFitterResult


# TODO: document me!
class LKAAMFitter(ModelFitter):
    r"""
    """
    def __init__(self, aam, algorithm_cls=AIC, n_shape=None,
                 n_appearance=None, **kwargs):
        super(LKAAMFitter, self).__init__()
        self.algorithms = []
        self._model = aam
        self._check_n_shape(n_shape)
        self._check_n_appearance(n_appearance)
        self._set_up(algorithm_cls, **kwargs)

    def _set_up(self, algorithm_cls, **kwargs):
        for j, (am, sm) in enumerate(zip(self._model.appearance_models,
                                         self._model.shape_models)):

            if type(self.aam) is AAM or type(self.aam) is PatchAAM:
                # build orthonormal model driven transform
                md_transform = OrthoMDTransform(
                    sm, self._model.transform,
                    source=am.mean().landmarks['source'].lms)
                # set up algorithm using standard aam interface
                algorithm = algorithm_cls(AAMInterface, am, md_transform,
                                          **kwargs)

            elif (type(self.aam) is LinearAAM or
                  type(self.aam) is LinearPatchAAM):
                # build linear version of orthogonal model driven transform
                md_transform = LinearOrthoMDTransform(
                    sm, self._model.n_landmarks)
                # set up algorithm using linear aam interface
                algorithm = algorithm_cls(LinearAAMInterface, am,
                                          md_transform, **kwargs)

            elif type(self.aam) is PartsAAM:
                # build orthogonal point distribution model
                pdm = OrthoPDM(sm)
                # set up algorithm using parts aam interface
                am.patch_shape = self._model.patch_shape[j]
                am.normalize_parts = self._model.normalize_parts
                algorithm = algorithm_cls(PartsAAMInterface, am, pdm, **kwargs)

            else:
                raise ValueError("AAM object must be of one of the "
                                 "following classes: {}, {}, {}, {}, "
                                 "{}".format(AAM, PatchAAM, LinearAAM,
                                             LinearPatchAAM, PartsAAM))

            # append algorithms to list
            self.algorithms.append(algorithm)

    @property
    def aam(self):
        return self._model

    def _check_n_appearance(self, n_appearance):
        if n_appearance is not None:
            if type(n_appearance) is int or type(n_appearance) is float:
                for am in self.aam.appearance_models:
                    am.n_active_components = n_appearance
            elif len(n_appearance) == 1 and self.aam.n_levels > 1:
                for am in self.aam.appearance_models:
                    am.n_active_components = n_appearance[0]
            elif len(n_appearance) == self.aam.n_levels:
                for am, n in zip(self.aam.appearance_models, n_appearance):
                    am.n_active_components = n
            else:
                raise ValueError('n_appearance can be an integer or a float '
                                 'or None or a list containing 1 or {} of '
                                 'those'.format(self.aam.n_levels))

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return AAMFitterResult(image, self, algorithm_results,
                               affine_correction, gt_shape=gt_shape)

