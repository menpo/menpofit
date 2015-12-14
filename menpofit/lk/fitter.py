from __future__ import division
from menpo.feature import no_op
from menpofit.transform import DifferentiableAlignmentAffine
from menpofit.fitter import (MultiFitter, noisy_shape_from_shape,
                             noisy_shape_from_bounding_box)
from menpofit import checks
from .algorithm import InverseCompositional
from .residual import SSD
from .result import LucasKanadeFitterResult


# TODO: document me!
class LucasKanadeFitter(MultiFitter):
    r"""
    """
    def __init__(self, template, group=None, holistic_features=no_op,
                 transform_cls=DifferentiableAlignmentAffine, diagonal=None,
                 scales=(0.5, 1.0), algorithm_cls=InverseCompositional,
                 residual_cls=SSD):

        checks.check_diagonal(diagonal)
        scales = checks.check_scales(scales)
        holistic_features = checks.check_callable(holistic_features,
                                                  len(scales))

        self.holistic_features = holistic_features
        self.transform_cls = transform_cls
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
        return perturb_func(gt_bb, gt_bb)

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return LucasKanadeFitterResult(image, self, algorithm_results,
                                       affine_correction, gt_shape=gt_shape)
