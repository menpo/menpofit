import numpy as np


def extract_parametric_features(appearance_model, warped_image,
                                rergession_features):
    r"""
    Extracts a particular parametric feature given an appearance model and
    a warped image.

    Parameters
    ----------
    appearance_model : :map:`PCAModel`
        The appearance model based on which the parametric features will be
        computed.
    warped_image : :map:`MaskedImage`
        The warped image.
    rergession_features : callable
        Defines the function from which the parametric features will be
        extracted.

        Non-default regression feature options and new experimental features
        can be used by defining a callable. In this case, the callable must
        define a constructor that receives as an input an appearance model and
        a warped masked image and on calling returns a particular parametric
        feature representation.

    Returns
    -------
    features : `ndarray`
        The resulting parametric features.
    """
    if rergession_features is None:
        features = weights(appearance_model, warped_image)
    elif hasattr(rergession_features, '__call__'):
        features = rergession_features(appearance_model, warped_image)
    else:
        raise ValueError("regression_features can only be: (1) None "
                         "or (2) a callable defining a non-standard "
                         "feature computation (see `menpo.fit.regression."
                         "parametricfeatures`")
    return features


def weights(appearance_model, warped_image):
    r"""
    Returns the resulting weights after projecting the warped image to the
    appearance PCA model.

    Parameters
    ----------
    appearance_model : :map:`PCAModel`
        The appearance model based on which the parametric features will be
        computed.
    warped_image : :map:`MaskedImage`
        The warped image.
    """
    return appearance_model.project(warped_image)


def whitened_weights(appearance_model, warped_image):
    r"""
    Returns the sheared weights after projecting the warped image to the
    appearance PCA model.

    Parameters
    ----------
    appearance_model : :map:`PCAModel`
        The appearance model based on which the parametric features will be
        computed.
    warped_image : :map:`MaskedImage`
        The warped image.
    """
    return appearance_model.project_whitened(warped_image)


def appearance(appearance_model, warped_image):
    r"""
    Projects the warped image onto the appearance model and rebuilds from the
    weights found.

    Parameters
    ----------
    appearance_model : :map:`PCAModel`
        The appearance model based on which the parametric features will be
        computed.
    warped_image : :map:`MaskedImage`
        The warped image.
    """
    return appearance_model.reconstruct(warped_image).as_vector()


def difference(appearance_model, warped_image):
    r"""
    Returns the difference between the warped image and the image constructed
    by projecting the warped image onto the appearance model and rebuilding it
    from the weights found.

    Parameters
    ----------
    appearance_model : :map:`PCAModel`
        The appearance model based on which the parametric features will be
        computed.
    warped_image : :map:`MaskedImage`
        The warped image.
    """
    return (warped_image.as_vector() -
            appearance(appearance_model, warped_image))


def project_out(appearance_model, warped_image):
    r"""
    Returns a version of the whitened warped image where all the basis of the
    model have been projected out and which has been scaled by the inverse of
    the appearance model's noise_variance.

    Parameters
    ----------
    appearance_model: :class:`menpo.model.pca`
        The appearance model based on which the parametric features will be
        computed.
    warped_image: :class:`menpo.image.masked`
        The warped image.
    """
    diff = warped_image.as_vector() - appearance_model.mean().as_vector()
    return appearance_model.distance_to_subspace_vector(diff).ravel()


class nonparametric_regression_features(object):

    def __init__(self, patch_shape, feature_patch_length, regression_features):
        self.patch_shape = patch_shape
        self.feature_patch_length = feature_patch_length
        self.regression_features = regression_features

    def __call__(self, image, shape):
        # extract patches
        patches = image.extract_patches(shape, patch_size=self.patch_shape)

        features = np.zeros((shape.n_points, self.feature_patch_length))
        for j, patch in enumerate(patches):
            # compute features
            features[j, ...] = self.regression_features(patch).as_vector()

        return np.hstack((features.ravel(), 1))


class parametric_regression_features(object):

    def __init__(self, transform, template, appearance_model,
                 regression_features):
        self.transform = transform
        self.template = template
        self.appearance_model = appearance_model
        self.regression_features = regression_features

    def __call__(self, image, shape):
        self.transform.set_target(shape)
        # TODO should the template be a mask or a shape? warp_to_shape here
        warped_image = image.warp_to_mask(self.template.mask, self.transform,
                                          warp_landmarks=False)
        features = extract_parametric_features(
            self.appearance_model, warped_image, self.regression_features)
        return np.hstack((features, 1))


class semiparametric_classifier_regression_features(object):

    def __init__(self, patch_shape, classifiers):
        self.patch_shape = patch_shape
        self.classifiers = classifiers

    def __call__(self, image, shape):
        patches = image.extract_patches(shape, patch_size=self.patch_shape)
        features = [clf(patch.as_vector(keep_channels=True))
                    for (clf, patch) in zip(self.classifiers, patches)]
        return np.hstack((np.asarray(features).ravel(), 1))
