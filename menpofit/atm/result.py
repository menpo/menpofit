from __future__ import division
from menpofit.result import ParametricAlgorithmResult, MultiFitterResult


# TODO: document me!
class ATMAlgorithmResult(ParametricAlgorithmResult):
    r"""
    """
    def __init__(self, image, algorithm, shape_parameters, cost_functions=None,
                 gt_shape=None):
        super(ATMAlgorithmResult, self).__init__(
            image, algorithm, shape_parameters, gt_shape=gt_shape)
        self._cost_functions = cost_functions
        self._warped_images = None
        self._costs = None

    @property
    def warped_images(self):
        if self._warped_images is None:
            self._warped_images = []
            for p in self.shape_parameters:
                self.algorithm.transform._from_vector_inplace(p)
                self._warped_images.append(
                    self.algorithm.warp(self.image))
        return self._warped_images

    @property
    def costs(self):
        if self._costs is None:
            self._costs = [f() for f in self._cost_functions]
        return self._costs


# TODO: document me!
class LinearATMAlgorithmResult(ATMAlgorithmResult):
    r"""
    """
    @property
    def shapes(self):
        return [self.algorithm.transform.from_vector(p).sparse_target
                for p in self.shape_parameters]

    @property
    def final_shape(self):
        return self.final_transform.sparse_target

    @property
    def initial_shape(self):
        return self.initial_transform.sparse_target


# TODO: document me!
class ATMFitterResult(MultiFitterResult):
    r"""
    """
    def __init__(self, image, fitter, algorithm_results, affine_correction,
                 gt_shape=None):
        super(ATMFitterResult, self).__init__(
            image, fitter, algorithm_results, affine_correction,
            gt_shape=gt_shape)
        self._warped_images = None

    @property
    def warped_images(self):
        if self._warped_images is None:
            algorithm = self.algorithm_results[-1].algorithm
            self._warped_images = []
            for s in self.shapes:
                algorithm.transform.set_target(s)
                self._warped_images.append(
                    algorithm.warp(self.image))
        return self._warped_images

    @property
    def costs(self):
        costs = []
        for a in self.algorithm_results:
            costs += a.costs
        return costs
