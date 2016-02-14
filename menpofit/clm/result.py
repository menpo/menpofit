from menpofit.result import (ParametricIterativeResult,
                             MultiScaleParametricIterativeResult)


class CLMAlgorithmResult(ParametricIterativeResult):
    r"""
    Class for storing the iterative result of a CLM optimization algorithm.

    Parameters
    ----------
    shapes : `list` of `menpo.shape.PointCloud`
        The `list` of shapes per iteration. The first and last members
        correspond to the initial and final shapes, respectively.
    shape_parameters : `list` of ``(n_shape_parameters,)`` `ndarray`
        The `list` of shape parameters per iteration. The first and last members
        correspond to the initial and final shapes, respectively.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    gt_shape : `menpo.shape.PointCloud` or ``None``, optional
        The ground truth shape associated with the image. If ``None``, then no
        ground truth shape is assigned.
    """
    def __init__(self, shapes, shape_parameters, image=None, gt_shape=None):
        super(CLMAlgorithmResult, self).__init__(
                shapes=shapes[1:], initial_shape=shapes[0],
                shape_parameters=shape_parameters,  image=image,
                gt_shape=gt_shape)


class CLMResult(MultiScaleParametricIterativeResult):
    r"""
    Class for storing the multi-scale iterative fitting result of a CLM. It
    holds the shapes, shape parameters, appearance parameters and costs per
    iteration.

    Parameters
    ----------
    results : `list` of :map:`AAMAlgorithmResult`
        The `list` of optimization results per scale.
    scales : `list` or `tuple`
        The `list` of scale values per scale (low to high).
    affine_correction : `menpo.transform.homogeneous.affine.AlignmentAffine`
        The affine transform that transfers the per-scale shapes to the image
        scale.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    gt_shape : `menpo.shape.PointCloud` or ``None``, optional
        The ground truth shape associated with the image. If ``None``, then no
        ground truth shape is assigned.
    """
    def __init__(self, results, scales, affine_correction, image=None,
                 gt_shape=None):
        super(CLMResult, self).__init__(
                results=results, scales=scales,
                affine_correction=affine_correction, image=image,
                gt_shape=gt_shape)
