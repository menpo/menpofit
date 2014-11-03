from __future__ import division


class Unified(object):

    def __init__(self, shape_models, classifiers, reference_shape,
                 parts_shape, features, sigma, scales, scale_shapes,
                 scale_features):

        self.shape_models = shape_models
        self.classifiers = classifiers
        self.parts_shape = parts_shape
        self.features = features
        self.sigma = sigma
        self.reference_shape = reference_shape
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features

    @property
    def n_levels(self):
        """
        The number of scale levels of the AAM.

        :type: `int`
        """
        return len(self.scales)
