from __future__ import division
import numpy as np

from menpo.feature import no_op
from menpo.base import name_of_callable

from menpofit.transform import DifferentiableAlignmentAffine
from menpofit.fitter import MultiFitter, noisy_shape_from_bounding_box
from menpofit import checks

from .algorithm import InverseCompositional
from .residual import SSD
from .result import LucasKanadeResult


class LucasKanadeFitter(MultiFitter):
    r"""
    Class for defining a multi-scale Lucas-Kanade fitter with repect to an
    affine transform. Please see the references for a basic list of relevant
    papers.

    Parameters
    ----------
    template : `menpo.image.Image`
        The template image.
    group : `str` or ``None``, optional
        The landmark group that will be used.
    holistic_features : `closure`, optional
        The features that will be extracted. Please refer to `menpo.feature`
        for a list of potential features.
    diagonal : `int` or ``None``, optional
        This parameter is used to define the scale of the template. It
        defines the diagonal of the template's landmark group.
    scales : `tuple` of `float`, optional
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale.
    transform : `menpofit.transform.DifferentiableAlignmentAffine`, optional
        A differential affine transform object.
    algorithm_cls : `menpofit.lk.algorithm.LucasKanade` subclass, optional
        The Lukas-Kanade optimization algorithm that will get applied. All
        possible algorithms are stored in `menpofit.lk.algorithm`.
    residual_cls : `menpofit.lk.residual.Residual` subclass, optional
        The residual that will get applied. All possible residuals are
        stored in `menpofit.lk.residual`.

    References
    ----------
    .. [1] B.D. Lucas, and T. Kanade, "An iterative image registration
        technique with an application to stereo vision", International Joint
        Conference on Artificial Intelligence, pp. 674-679, 1981.
    .. [2] G.D. Evangelidis, and E.Z. Psarakis. "Parametric Image Alignment
        Using Enhanced Correlation Coefficient Maximization", IEEE Transactions
        on Pattern Analysis and Machine Intelligence, 30(10): 1858-1865, 2008.
    .. [3] A.B. Ashraf, S. Lucey, and T. Chen. "Fast Image Alignment in the
        Fourier Domain", IEEE Proceedings of International Conference on
        Computer Vision and Pattern Recognition, pp. 2480-2487, 2010.
    .. [4] G. Tzimiropoulos, S. Zafeiriou, and M. Pantic. "Robust and
        Efficient Parametric Face Alignment", IEEE Proceedings of International
        Conference on Computer Vision (ICCV), pp. 1847-1854, November 2011.
    """
    def __init__(self, template, group=None, holistic_features=no_op,
                 diagonal=None, transform=DifferentiableAlignmentAffine,
                 scales=(0.5, 1.0), algorithm_cls=InverseCompositional,
                 residual_cls=SSD):
        # Check arguments
        checks.check_diagonal(diagonal)
        scales = checks.check_scales(scales)
        holistic_features = checks.check_callable(holistic_features,
                                                  len(scales))
        # Assign attributes
        self.holistic_features = holistic_features
        self.transform_cls = transform
        self.diagonal = diagonal
        self.scales = list(scales)
        # Make template masked for warping
        template = template.as_masked(copy=False)

        if self.diagonal:
            template = template.rescale_landmarks_to_diagonal_range(
                self.diagonal, group=group)
        self.reference_shape = template.landmarks[group].lms

        self.templates, self.sources = self._prepare_template(template,
                                                              group=group)
        self._set_up(algorithm_cls, residual_cls)

    def _set_up(self, algorithm_cls, residual_cls):
        self.algorithms = []
        for j, (t, s) in enumerate(zip(self.templates, self.sources)):
            transform = self.transform_cls(s, s)
            residual = residual_cls()
            algorithm = algorithm_cls(t, transform, residual)
            self.algorithms.append(algorithm)

    def _prepare_template(self, template, group=None):
        gt_shape = template.landmarks[group].lms
        templates, _, sources = self._prepare_image(template, gt_shape,
                                                    gt_shape=gt_shape)
        return templates, sources

    def perturb_from_gt_bb(self, gt_bb,
                           perturb_func=noisy_shape_from_bounding_box):
        r"""
        Method for adding noise to the ground truth bounding box. This is
        useful for obtaining the initial bounding box of the fitting.

        Parameters
        ----------
        gt_bb : `menpo.shape.PointCloud`
            The ground truth bounding box.
        perturb_func : `closure`, optional
            The method to be used for adding noise to the ground truth
            bounding box.

        Returns
        -------
        perturbed_bb : `menpo.shape.PointCloud`
            The perturbed bounding box.
        """
        return perturb_func(gt_bb, gt_bb)

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return LucasKanadeResult(
                results=algorithm_results, scales=self.scales,
                affine_correction=affine_correction, image=image,
                gt_shape=gt_shape)

    def warped_images(self, image, shapes):
        r"""
        Given an input test image and a list of shapes, it warps the image
        into the shapes. This is useful for generating the warped images of a
        fitting procedure.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The input image to be warped.
        shapes : `list` of `menpo.shape.PointCloud`
            The list of shapes in which the image will be warped. The shapes
            are obtained during the iterations of a fitting procedure.

        Returns
        -------
        warped_images : `list` of `menpo.image.MaskedImage` or `ndarray`
            The warped images.
        """
        return self.algorithms[-1].warped_images(image=image, shapes=shapes)

    def __str__(self):
        if self.diagonal is not None:
            diagonal = self.diagonal
        else:
            y, x = self.reference_shape.range()
            diagonal = np.sqrt(x ** 2 + y ** 2)

        # Compute scale info strings
        scales_info = []
        lvl_str_tmplt = r"""   - Scale {}
     - Holistic feature: {}
     - Template shape: {}"""
        for k, s in enumerate(self.scales):
            scales_info.append(lvl_str_tmplt.format(
                    s, name_of_callable(self.holistic_features[k]),
                    self.templates[k].shape))
        scales_info = '\n'.join(scales_info)

        cls_str = r"""Lucas-Kanade {class_title}
 - {residual}
 - Images warped with {transform} transform
 - Images scaled to diagonal: {diagonal:.2f}
 - Scales: {scales}
{scales_info}
""".format(class_title=self.algorithms[0],
           residual=self.algorithms[0].residual,
           transform=name_of_callable(self.transform_cls),
           diagonal=diagonal,
           scales=self.scales,
           scales_info=scales_info)
        return cls_str
