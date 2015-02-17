from __future__ import division
from menpo.transform import AlignmentSimilarity, Similarity
import numpy as np
from menpo.visualize import progress_bar_str, print_dynamic


def name_of_callable(c):
    try:
        return c.__name__  # function
    except AttributeError:
        return c.__class__.__name__  # callable class


def is_pyramid_on_features(features):
    r"""
    True if feature extraction happens once and then a gaussian pyramid
    is taken. False if a gaussian pyramid is taken and then features are
    extracted at each level.
    """
    return callable(features)


def create_pyramid(images, n_levels, downscale, features, verbose=False):
    r"""
    Function that creates a generator function for Gaussian pyramid. The
    pyramid can be created either on the feature space or the original
    (intensities) space.

    Parameters
    ----------
    images: list of :map:`Image`
        The set of landmarked images from which to build the AAM.

    n_levels: int
        The number of multi-resolution pyramidal levels to be used.

    downscale: float
        The downscale factor that will be used to create the different
        pyramidal levels.

    features: ``callable`` ``[callable]``
        If a single callable, then the feature calculation will happen once
        followed by a gaussian pyramid. If a list of callables then a
        gaussian pyramid is generated with features extracted at each level
        (after downsizing and blurring).

    Returns
    -------
    list of generators :
        The generator function of the Gaussian pyramid.

    """
    will_take_a_while = is_pyramid_on_features(features)
    pyramids = []
    for i, img in enumerate(images):
        if will_take_a_while and verbose:
            print_dynamic(
                'Computing top level feature space - {}'.format(
                    progress_bar_str((i + 1.) / len(images),
                                     show_bar=False)))
        pyramids.append(pyramid_of_feature_images(n_levels, downscale,
                                                  features, img))
    return pyramids


def pyramid_of_feature_images(n_levels, downscale, features, image):
    r"""
    Generates a gaussian pyramid of feature images for a single image.
    """
    if is_pyramid_on_features(features):
        # compute feature image at the top
        feature_image = features(image)
        # create pyramid on the feature image
        return feature_image.gaussian_pyramid(n_levels=n_levels,
                                              downscale=downscale)
    else:
        # create pyramid on intensities image
        # feature will be computed per level
        pyramid = image.gaussian_pyramid(n_levels=n_levels,
                                         downscale=downscale)
        # add the feature generation here
        return feature_images(pyramid, features)


# adds feature extraction to a generator of images
def feature_images(images, features):
    for feature, level in zip(reversed(features), images):
        yield feature(level)


class DeformableModel(object):

    def __init__(self, features):
        self.features = features

    @property
    def pyramid_on_features(self):
        return is_pyramid_on_features(self.features)


# TODO: Should this be a method on Similarity? AlignableTransforms?
def noisy_align(source, target, noise_std=0.04, rotation=False):
    r"""
    Constructs and perturbs the optimal similarity transform between source
    to the target by adding white noise to its weights.

    Parameters
    ----------
    source: :class:`menpo.shape.PointCloud`
        The source pointcloud instance used in the alignment
    target: :class:`menpo.shape.PointCloud`
        The target pointcloud instance used in the alignment
    noise_std: float
        The standard deviation of the white noise

        Default: 0.04
    rotation: boolean
        If False the second parameter of the Similarity,
        which captures captures inplane rotations, is set to 0.

        Default:False

    Returns
    -------
    noisy_transform : :class: `menpo.transform.Similarity`
        The noisy Similarity Transform
    """
    transform = AlignmentSimilarity(source, target, rotation=rotation)
    parameters = transform.as_vector()
    parameter_range = np.hstack((parameters[:2], target.range()))
    noise = (parameter_range * noise_std *
             np.random.randn(transform.n_parameters))
    return Similarity.init_identity(source.n_dims).from_vector(parameters + noise)


def build_sampling_grid(patch_shape):
    r"""
    """
    patch_shape = np.array(patch_shape)
    patch_half_shape = np.require(np.floor(patch_shape / 2), dtype=int)
    start = -patch_half_shape
    end = patch_half_shape + 1
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    return sampling_grid.swapaxes(0, 2).swapaxes(0, 1)