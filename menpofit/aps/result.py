from __future__ import division

from menpofit.result import ParametricAlgorithmResult, MultiFitterResult


class APSAlgorithmResult(ParametricAlgorithmResult):
    r"""
    Class for creating the fitting result of a specific APS algorithm.

    Parameters
    ----------
    image : :map:`Image: or subclass
        The test image.
    algorithm : `GaussNewtonBaseInterface` or subclass
        The algorithm class of the APS.
    shape_parameters : `list` of `ndarray`
        A `list` with the shape parameters per iteration. These are used to
        generate the fitted shapes.
    cost_functions : `list` of `callable` or ``None``, optional
        The `list` of `callable` that compute the cost per iteration.
    gt_shape : :map:`PointCloud` or ``None``, optional
        The ground truth shape of the image.
    """
    def __init__(self, image, algorithm, shape_parameters, cost_functions=None,
                 gt_shape=None):
        super(APSAlgorithmResult, self).__init__(
            image, algorithm, shape_parameters, gt_shape=gt_shape)
        self._cost_functions = cost_functions
        self._warped_images = None
        self._appearance_reconstructions = None
        self._costs = None

    @property
    def warped_images(self):
        r"""
        Returns a `list` with the warped image per iteration. Note that the
        images are patch-based.

        :type: `list` of :map:`Image`
        """
        if self._warped_images is None:
            self._warped_images = []
            for p in self.shape_parameters:
                self.algorithm.transform.from_vector_inplace(p)
                self._warped_images.append(self.algorithm.warp(self.image))
        return self._warped_images

    @property
    def costs(self):
        r"""
        Returns a `list` with the cost per iteration.

        :type: `list` of `float`
        """
        if self._cost_functions is not None:
            if self._costs is None:
                self._costs = [f() for f in self._cost_functions]
            return self._costs
        else:
            raise ValueError('costs is not well defined for the chosen APS '
                             'algorithm: {}'.format(self.algorithm.__class__))


class APSFitterResult(MultiFitterResult):
    r"""
    Class for creating the fitting result of a specific APS fitter.

    Parameters
    ----------
    image : :map:`Image: or subclass
        The test image.
    fitter : `APSFitter` or subclass
        The fitter class of the APS.
    algorithm_results : `list` of `APSAlgorithmResult` or subclass
        A `list` with fitting result of the algorithm class per level.
    affine_correction :
        The affine transform to be applied to the shapes.
    gt_shape : :map:`PointCloud` or ``None``, optional
        The ground truth shape of the image.
    """
    def __init__(self, image, fitter, algorithm_results, affine_correction,
                 gt_shape=None):
        super(APSFitterResult, self).__init__(
            image, fitter, algorithm_results, affine_correction,
            gt_shape=gt_shape)
        self._warped_images = None

    @property
    def warped_images(self):
        r"""
        Returns a `list` with the warped image per iteration. Note that the
        images are patch-based.

        :type: `list` of :map:`Image`
        """
        if self._warped_images is None:
            algorithm = self.algorithm_results[-1].algorithm
            self._warped_images = []
            for s in self.shapes:
                algorithm.transform.set_target(s)
                self._warped_images.append(algorithm.warp(self.image))
        return self._warped_images

    @property
    def costs(self):
        r"""
        Returns a `list` with the cost per iteration.

        :type: `list` of `float`
        """
        costs = []
        for a in self.algorithm_results:
            costs += a.costs
        return costs
