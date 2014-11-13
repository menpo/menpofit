from __future__ import division
import abc

from serializablecallable import SerializableCallable

from menpofast.image import Image
from menpofast.utils import build_parts_image

from alabortcvpr2015.clm.classifier import MultipleMCF


# Abstract Interface for Unified Objects --------------------------------------

class Unified(object):

    __metaclass__ = abc.ABCMeta

    def __getstate__(self):
        import menpofast.feature as menpofast_feature
        d = self.__dict__.copy()

        features = d.pop('features')
        d['features'] = SerializableCallable(features, [menpofast_feature])

        return d

    def __setstate__(self, state):
        state['features'] = state['features'].callable
        self.__dict__.update(state)

    @property
    def n_levels(self):
        """
        The number of scale levels of the AAM.

        :type: `int`
        """
        return len(self.scales)

    def parts_filters(self):

        if isinstance(self.classifiers[0], MultipleMCF):
            return [[Image(f) for f in clf.invert_filters()]
                    for clf in self.classifiers]
        else:
            raise ValueError('Only suported MultipleMCF are supported')

    def parts_response(self, image, group=None, label=None):

        centers = image.landmarks[group][label].landmarks.lms

        parts_image = build_parts_image(
            image, centers, parts_shape=self.parts_shape,
            normalize_parts=self.normalize_parts)

        return [clf(parts_image)for clf in self.classifiers]


# Concrete Implementations of Unified Objects ---------------------------------

class GlobalUnified(Unified):

    def __init__(self, shape_models, appearance_models, classifiers,
                 reference_shape, transform, parts_shape, features,
                 normalize_parts, covariance, sigma, scales, scale_shapes,
                 scale_features):
        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.classifiers = classifiers
        self.reference_shape = reference_shape
        self.transform = transform
        self.parts_shape = parts_shape
        self.features = features
        self.normalize_parts = normalize_parts
        self.covariance = covariance
        self.sigma = sigma
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features


class PartsUnified(Unified):

    def __init__(self, shape_models, appearance_models, classifiers,
                 reference_shape, parts_shape, features, normalize_parts,
                 covariance, sigma, scales, scale_shapes, scale_features):
        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.classifiers = classifiers
        self.reference_shape = reference_shape
        self.parts_shape = parts_shape
        self.features = features
        self.normalize_parts = normalize_parts
        self.covariance = covariance
        self.sigma = sigma
        self.scales = scales
        self.scale_shapes = scale_shapes
        self.scale_features = scale_features