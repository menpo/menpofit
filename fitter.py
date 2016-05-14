

from alabortcvpr2015.fitter import Fitter
from alabortcvpr2015.pdm import OrthoPDM
from alabortcvpr2015.transform import OrthoMDTransform
from alabortcvpr2015.aam.algorithm import GlobalAAMInterface, PartsAAMInterface

from .algorithm import AICRLMS


# Abstract Interface for Unified Fitters  -------------------------------------

class UnifiedFitter(Fitter):

    def _check_n_appearance(self, n_appearance):
        if n_appearance is not None:
            if type(n_appearance) is int or type(n_appearance) is float:
                for am in self.dm.appearance_models:
                    am.n_active_components = n_appearance
            elif len(n_appearance) == 1 and self.dm.n_levels > 1:
                for am in self.dm.appearance_models:
                    am.n_active_components = n_appearance[0]
            elif len(n_appearance) == self.dm.n_levels:
                for am, n in zip(self.dm.appearance_models, n_appearance):
                    am.n_active_components = n
            else:
                raise ValueError('n_appearance can be an integer or a float '
                                 'or None or a list containing 1 or {} of '
                                 'those'.format(self.dm.n_levels))

    @property
    def covariance(self):
        return self.dm.covariance


# Concrete Implementations of Unified Fitters ---------------------------------

class GlobalUnifiedFitter(UnifiedFitter):

    def __init__(self, global_unified, algorithm_cls=AICRLMS,
                 n_shape=None, n_appearance=None, **kwargs):

        super(GlobalUnifiedFitter, self).__init__()

        self.dm = global_unified
        self._algorithms = []
        self._check_n_shape(n_shape)
        self._check_n_appearance(n_appearance)

        for j, (am, clf, sm) in enumerate(zip(self.dm.appearance_models,
                                              self.dm.classifiers,
                                              self.dm.shape_models)):

            md_transform = OrthoMDTransform(
                sm, self.dm.transform,
                source=am.mean().landmarks['source'].lms,
                sigma2=am.noise_variance())

            algorithm = algorithm_cls(
                GlobalAAMInterface, am, md_transform,
                clf, self.dm.parts_shape, self.dm.normalize_parts,
                self.covariance, md_transform.pdm, **kwargs)

            self._algorithms.append(algorithm)


class PartsUnifiedFitter(UnifiedFitter):

    def __init__(self, parts_aam, algorithm_cls=AICRLMS,
                 n_shape=None, n_appearance=None,  **kwargs):

        super(PartsUnifiedFitter, self).__init__()

        self.dm = parts_aam
        self._algorithms = []
        self._check_n_shape(n_shape)
        self._check_n_appearance(n_appearance)

        for j, (am, clf, sm) in enumerate(zip(self.dm.appearance_models,
                                              self.dm.classifiers,
                                              self.dm.shape_models)):

            pdm = OrthoPDM(sm, sigma2=am.noise_variance())

            algorithm = algorithm_cls(
                PartsAAMInterface, am, pdm,
                clf, self.dm.parts_shape, self.dm.normalize_parts,
                self.covariance, pdm, **kwargs)
            algorithm.parts_shape = self.dm.parts_shape
            algorithm.normalize_parts = self.dm.normalize_parts

            self._algorithms.append(algorithm)
