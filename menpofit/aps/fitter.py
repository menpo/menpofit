from __future__ import division

from menpofit.fitter import ModelFitter
from menpofit.modelinstance import OrthoPDM, PDM
import menpofit.checks as checks

from .result import APSFitterResult
from .algorithm.gn import GaussNewtonBaseInterface, Inverse

class APSFitter(ModelFitter):
    r"""
    Abstract class of an APS Fitter.
    """
    @property
    def aps(self):
        r"""
        The APS model.

        :type: :map:`GenerativeAPS` or subclass
        """
        return self._model

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return APSFitterResult(image, self, algorithm_results,
                               affine_correction, gt_shape=gt_shape)


class GaussNewtonAPSFitter(APSFitter):
    r"""
    A class for fitting an APS model with Gauss-Newton optimization.

    Parameters
    ----------
    aps : :map:`GenerativeAPS` or subclass
        The trained model.
    gn_algorithm_cls : subclass of :map:`GaussNewton`
        The Gauss-Newton algorithm class to be used.
    n_shape : `int`/`float` or `list` of `int`/`float` or ``None``, optional
        The number of active shape components. If `list`, then a value must
        be specified per level. If `int`/`float`, then this value is applied
        to all levels. If `int`, then the exact number of components is
        defined. If `float`, then the components are defined as a percentage
        of the variance. If ``None``, then all the components are employed.
    use_deformation_cost : `bool`, optional
        If ``True``, then the deformation cost is also included in the
        Hessian calculation.
    sampling : `ndarray` or ``None``
        Defines a sampling map to be applied on the patches.
    """
    def __init__(self, aps, gn_algorithm_cls=Inverse, n_shape=None,
                 use_deformation_cost=True, sampling=None):
        self._model = aps
        self._check_n_shape(n_shape)
        self._sampling = checks.check_sampling(sampling, aps.n_scales)
        self._set_up(gn_algorithm_cls, use_deformation_cost)

    def _set_up(self, gn_algorithm_cls, use_deformation_cost):
        self.algorithms = []
        for j, (am, sm, dm, s) in enumerate(zip(self.aps.appearance_models,
                                                self.aps.shape_models,
                                                self.aps.deformation_models,
                                                self._sampling)):
            template = am.mean()

            # build orthogonal point distribution model
            if self._model.use_procrustes:
                pdm = OrthoPDM(sm)
            else:
                pdm = PDM(sm)

            # create the interface object
            interface = GaussNewtonBaseInterface(
                am, dm, pdm, use_deformation_cost, template, s,
                self.aps.patch_shape[j], self.aps.patch_normalisation[j])

            # create the algorithm object and append it
            self.algorithms.append(gn_algorithm_cls(interface))
