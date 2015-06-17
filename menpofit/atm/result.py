from __future__ import division
from menpofit.result import ParametricAlgorithmResult, MultiFitterResult


# TODO: document me!
# TODO: handle costs
class ATMAlgorithmResult(ParametricAlgorithmResult):
    r"""
    """

# TODO: document me!
class LinearATMAlgorithmResult(ATMAlgorithmResult):
    r"""
    """
    @property
    def shapes(self):
        return [self.fitter.transform.from_vector(p).sparse_target
                for p in self.shape_parameters]

    @property
    def final_shape(self):
        return self.final_transform.sparse_target

    @property
    def initial_shape(self):
        return self.initial_transform.sparse_target


# TODO: document me!
# TODO: handle costs
class ATMFitterResult(MultiFitterResult):
    r"""
    """
    pass
