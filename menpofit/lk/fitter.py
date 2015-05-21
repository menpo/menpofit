from __future__ import division
from menpo.feature import no_op
from menpofit.transform import DifferentiableAlignmentAffine
from menpofit.fitter import MultiFitter, noisy_align
from menpofit.result import MultiFitterResult
from .algorithm import IC
from .residual import SSD, FourierSSD


# TODO: document me!
class LKFitter(MultiFitter):
    r"""
    """
    def __init__(self, template, group=None, label=None, features=no_op,
                 transform_cls=DifferentiableAlignmentAffine, diagonal=None,
                 scales=(1, .5), scale_features=True, algorithm_cls=IC,
                 residual_cls=SSD, **kwargs):
        self._features = features
        self.transform_cls = transform_cls
        self.diagonal = diagonal
        self._scales = list(scales)
        self._scales.reverse()
        self._scale_features = scale_features

        self.templates, self.sources = self._prepare_template(
            template, group=group, label=label)

        self._reference_shape = self.sources[0]

        self._algorithms = []
        for j, (t, s) in enumerate(zip(self.templates, self.sources)):
            transform = self.transform_cls(s, s)
            if ('kernel_func' in kwargs and
                (residual_cls is SSD or
                 residual_cls is FourierSSD)):
                kernel_func = kwargs.pop('kernel_func')
                kernel = kernel_func(t.shape)
                residual = residual_cls(kernel=kernel)
            else:
                residual = residual_cls()
            algorithm = algorithm_cls(t, transform, residual, **kwargs)
            self._algorithms.append(algorithm)

    @property
    def algorithms(self):
        return self._algorithms

    @property
    def reference_shape(self):
        return self._reference_shape

    @property
    def features(self):
        return self._features

    @property
    def scales(self):
        return self._scales

    @property
    def scale_features(self):
        return self._scale_features

    def _prepare_template(self, template, group=None, label=None):
        # copy template
        template = template.copy()

        template = template.crop_to_landmarks_inplace(group=group, label=label)
        template = template.as_masked()

        # rescale template to diagonal range
        if self.diagonal:
            template = template.rescale_landmarks_to_diagonal_range(
                self.diagonal, group=group, label=label)

        # obtain image representation
        from copy import deepcopy
        scales = deepcopy(self.scales)
        scales.reverse()
        templates = []
        for j, s in enumerate(scales):
            if j == 0:
                # compute features at highest level
                feature_template = self.features(template)
            elif self.scale_features:
                # scale features at other levels
                feature_template = templates[0].rescale(s)
            else:
                # scale image and compute features at other levels
                scaled_template = template.rescale(s)
                feature_template = self.features(scaled_template)
            templates.append(feature_template)
        templates.reverse()

        # get sources per level
        sources = [i.landmarks[group][label] for i in templates]

        return templates, sources

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return MultiFitterResult(image, self, algorithm_results,
                                 affine_correction, gt_shape=gt_shape)

    def perturb_shape(self, gt_shape, noise_std=0.04):
        transform = noisy_align(self.transform_cls, self.reference_shape,
                                gt_shape, noise_std=noise_std)
        return transform.apply(self.reference_shape)
