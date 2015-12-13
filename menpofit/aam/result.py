from __future__ import division
from menpofit.result import ParametricAlgorithmResult, MultiFitterResult


# TODO: document me!
class AAMAlgorithmResult(ParametricAlgorithmResult):
    r"""
    """
    def __init__(self, image, algorithm, shape_parameters, cost_functions=None,
                 appearance_parameters=None, gt_shape=None):
        super(AAMAlgorithmResult, self).__init__(
            image, algorithm, shape_parameters, gt_shape=gt_shape)
        self._cost_functions = cost_functions
        self.appearance_parameters = appearance_parameters
        self._warped_images = None
        self._appearance_reconstructions = None
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
    def appearance_reconstructions(self):
        if self.appearance_parameters is not None:
            if self._appearance_reconstructions is None:
                self._appearance_reconstructions = []
                for c in self.appearance_parameters:
                    instance = self.algorithm.appearance_model.instance(c)
                    self._appearance_reconstructions.append(instance)
            return self._appearance_reconstructions
        else:
            raise ValueError('appearance_reconstructions is not well '
                             'defined for the chosen AAM algorithm: '
                             '{}'.format(self.algorithm.__class__))

    @property
    def costs(self):
        if self._cost_functions is not None:
            if self._costs is None:
                self._costs = [f() for f in self._cost_functions]
            return self._costs
        else:
            raise ValueError('costs is not well '
                             'defined for the chosen AAM algorithm: '
                             '{}'.format(self.algorithm.__class__))


# TODO: document me!
class LinearAAMAlgorithmResult(AAMAlgorithmResult):
    r"""
    """
    @property
    def shapes(self, as_points=False):
        return [self.algorithm.transform.from_vector(p).sparse_target
                for p in self.shape_parameters]

    @property
    def final_shape(self):
        return self.final_transform.sparse_target

    @property
    def initial_shape(self):
        return self.initial_transform.sparse_target


# TODO: document me!
class AAMFitterResult(MultiFitterResult):
    r"""
    """
    def __init__(self, image, fitter, algorithm_results, affine_correction,
                 gt_shape=None):
        super(AAMFitterResult, self).__init__(
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
    def appearance_reconstructions(self):
        reconstructions = self.algorithm_results[0].appearance_reconstructions
        if reconstructions is not None:
            for a in self.algorithm_results[1:]:
                reconstructions = (reconstructions +
                                   a.appearance_reconstructions)
        return reconstructions

    @property
    def costs(self):
        costs = []
        for a in self.algorithm_results:
            costs += a.costs
        return costs
