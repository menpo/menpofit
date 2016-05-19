

from alabortcvpr2015.result import AlgorithmResult, FitterResult


# Concrete Implementations of AAM Algorithm Results ---------------------------

class UnifiedAlgorithmResult(AlgorithmResult):

    def __init__(self, image, fitter, shape_parameters,
                 appearance_parameters=None, gt_shape=None):
        super(UnifiedAlgorithmResult, self).__init__()
        self.image = image
        self.fitter = fitter
        self.shape_parameters = shape_parameters
        self.appearance_parameters = appearance_parameters
        self._gt_shape = gt_shape


# Concrete Implementations of AAM Fitter Results  -----------------------------

class UnifiedFitterResult(FitterResult):

    @property
    def costs(self):
        r"""
        Returns a list containing the cost at each fitting iteration.

        :type: `list` of `float`
        """
        raise ValueError('costs not implemented yet.')

    @property
    def final_cost(self):
        r"""
        Returns the final fitting cost.

        :type: `float`
        """
        return self.algorithm_results[-1].final_cost

    @property
    def initial_cost(self):
        r"""
        Returns the initial fitting cost.

        :type: `float`
        """
        return self.algorithm_results[0].initial_cost
