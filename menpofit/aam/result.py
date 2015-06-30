from __future__ import division
from menpofit.result import ParametricAlgorithmResult, MultiFitterResult


# TODO: handle costs!
# TODO: document me!
class AAMAlgorithmResult(ParametricAlgorithmResult):
    r"""
    """
    def __init__(self, image, fitter, shape_parameters,
                 appearance_parameters=None, gt_shape=None):
        super(AAMAlgorithmResult, self).__init__(
            image, fitter, shape_parameters, gt_shape=gt_shape)
        self.appearance_parameters = appearance_parameters


# TODO: handle costs!
# TODO: document me!
class LinearAAMAlgorithmResult(AAMAlgorithmResult):
    r"""
    """
    @property
    def shapes(self, as_points=False):
        return [self.fitter.transform.from_vector(p).sparse_target
                for p in self.shape_parameters]

    @property
    def final_shape(self):
        return self.final_transform.sparse_target

    @property
    def initial_shape(self):
        return self.initial_transform.sparse_target


# TODO: handle costs!
# TODO: document me!
class AAMFitterResult(MultiFitterResult):
    r"""
    """
    pass
