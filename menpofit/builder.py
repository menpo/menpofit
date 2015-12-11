from __future__ import division
from functools import partial
import warnings
import numpy as np
from menpo.shape import mean_pointcloud, PointCloud, TriMesh
from menpo.image import Image, MaskedImage
from menpo.feature import no_op
from menpo.transform import Scale, Translation, GeneralizedProcrustesAnalysis
from menpo.visualize import print_dynamic
from menpofit.visualize import print_progress


class MenpoFitModelBuilderWarning(Warning):
    r"""
    A warning that the parameters chosen to build a given model may cause
    unexpected behaviour.
    """
    pass


def compute_reference_shape(shapes, normalization_diagonal, verbose=False):
    r"""
    Function that computes the reference shape as the mean shape of the provided
    shapes.

    Parameters
    ----------
    shapes : list of :map:`PointCloud`
        The set of shapes from which to build the reference shape.

    normalization_diagonal : `int`
        If int, it ensures that the mean shape is scaled so that the
        diagonal of the bounding box containing it matches the
        normalization_diagonal value.
        If None, the mean shape is not rescaled.

    verbose : `bool`, Optional
        Flag that controls information and progress printing.

    Returns
    -------
    reference_shape : :map:`PointCloud`
        The reference shape.
    """
    # the reference_shape is the mean shape of the images' landmarks
    if verbose:
        print_dynamic('- Computing reference shape')
    reference_shape = mean_pointcloud(shapes)

    # fix the reference_shape's diagonal length if asked
    if normalization_diagonal:
        x, y = reference_shape.range()
        scale = normalization_diagonal / np.sqrt(x**2 + y**2)
        reference_shape = Scale(scale, reference_shape.n_dims).apply(
            reference_shape)

    return reference_shape


# TODO: document me!
def rescale_images_to_reference_shape(images, group, reference_shape,
                                      verbose=False):
    r"""
    """
    wrap = partial(print_progress, prefix='- Normalizing images size',
                   end_with_newline=False, verbose=verbose)

    # Normalize the scaling of all images wrt the reference_shape size
    normalized_images = [i.rescale_to_pointcloud(reference_shape, group=group)
                         for i in wrap(images)]
    return normalized_images


def normalization_wrt_reference_shape(images, group, diagonal, verbose=False):
    r"""
    Function that normalizes the images sizes with respect to the reference
    shape (mean shape) scaling. This step is essential before building a
    deformable model.

    The normalization includes:
    1) Computation of the reference shape as the mean shape of the images'
       landmarks.
    2) Scaling of the reference shape using the diagonal.
    3) Rescaling of all the images so that their shape's scale is in
       correspondence with the reference shape's scale.

    Parameters
    ----------
    images : list of :class:`menpo.image.MaskedImage`
        The set of landmarked images to normalize.
    group : `str`
        The key of the landmark set that should be used. If None,
        and if there is only one set of landmarks, this set will be used.
    diagonal: `int`
        If int, it ensures that the mean shape is scaled so that the
        diagonal of the bounding box containing it matches the
        diagonal value.
        If None, the mean shape is not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).
    verbose : `bool`, Optional
        Flag that controls information and progress printing.

    Returns
    -------
    reference_shape : :map:`PointCloud`
        The reference shape that was used to resize all training images to
        a consistent object size.
    normalized_images : :map:`MaskedImage` list
        A list with the normalized images.
    """
    # get shapes
    shapes = [i.landmarks[group].lms for i in images]

    # compute the reference shape and fix its diagonal length
    reference_shape = compute_reference_shape(shapes, diagonal, verbose=verbose)

    # normalize the scaling of all images wrt the reference_shape size
    normalized_images = rescale_images_to_reference_shape(
        images, group, reference_shape, verbose=verbose)
    return reference_shape, normalized_images


# TODO: document me!
def compute_features(images, features, prefix='', verbose=False):
    wrap = partial(print_progress,
                   prefix='{}Computing feature space'.format(prefix),
                   end_with_newline=not prefix, verbose=verbose)

    return [features(i) for i in wrap(images)]


# TODO: document me!
def scale_images(images, scale, prefix='', verbose=False):
    wrap = partial(print_progress,
                   prefix='{}Scaling images'.format(prefix),
                   end_with_newline=not prefix, verbose=verbose)

    if not np.allclose(scale, 1):
        return [i.rescale(scale) for i in wrap(images)]
    else:
        return images


# TODO: document me!
def warp_images(images, shapes, reference_frame, transform, prefix='',
                verbose=None):
    wrap = partial(print_progress,
                   prefix='{}Warping images'.format(prefix),
                   end_with_newline=not prefix, verbose=verbose)

    warped_images = []
    # Build a dummy transform, use set_target for efficiency
    warp_transform = transform(reference_frame.landmarks['source'].lms,
                               reference_frame.landmarks['source'].lms)
    for i, s in wrap(zip(images, shapes)):
        # Update Transform Target
        warp_transform.set_target(s)
        # warp images
        warped_i = i.warp_to_mask(reference_frame.mask, warp_transform)
        # attach reference frame landmarks to images
        warped_i.landmarks['source'] = reference_frame.landmarks['source']
        warped_images.append(warped_i)
    return warped_images


# TODO: document me!
def extract_patches(images, shapes, patch_shape, normalise_function=no_op,
                    prefix='', verbose=False):
    wrap = partial(print_progress,
                   prefix='{}Extracting patches'.format(prefix),
                   end_with_newline=not prefix, verbose=verbose)

    parts_images = []
    for i, s in wrap(zip(images, shapes)):
        parts = i.extract_patches(s, patch_shape=patch_shape,
                                  as_single_array=True)
        parts = normalise_function(parts)
        parts_images.append(Image(parts, copy=False))
    return parts_images


def build_reference_frame(landmarks, boundary=3, group='source'):
    r"""
    Builds a reference frame from a particular set of landmarks.

    Parameters
    ----------
    landmarks : :map:`PointCloud`
        The landmarks that will be used to build the reference frame.
    boundary : `int`, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).
    group : `string`, optional
        Group that will be assigned to the provided set of landmarks on the
        reference frame.

    Returns
    -------
    reference_frame : :map:`Image`
        The reference frame.
    """
    reference_frame = _build_reference_frame(landmarks, boundary=boundary,
                                             group=group)
    source_landmarks = reference_frame.landmarks['source'].lms
    if not isinstance(source_landmarks, TriMesh):
        warnings.warn('The reference shape passed is not a TriMesh or '
                      'subclass and therefore the reference frame (mask) will '
                      'be calculated via a Delaunay triangulation. This may '
                      'cause small triangles and thus suboptimal warps.',
                      MenpoFitModelBuilderWarning)

    reference_frame.constrain_mask_to_landmarks(group=group)
    return reference_frame


def build_patch_reference_frame(landmarks, boundary=3, group='source',
                                patch_shape=(17, 17)):
    r"""
    Builds a reference frame from a particular set of landmarks.

    Parameters
    ----------
    landmarks : :map:`PointCloud`
        The landmarks that will be used to build the reference frame.

    boundary : `int`, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

    group : `string`, optional
        Group that will be assigned to the provided set of landmarks on the
        reference frame.

    patch_shape : tuple of ints, optional
        Tuple specifying the shape of the patches.

    Returns
    -------
    patch_based_reference_frame : :map:`Image`
        The patch based reference frame.
    """
    boundary = np.max(patch_shape) + boundary
    reference_frame = _build_reference_frame(landmarks, boundary=boundary,
                                             group=group)

    # mask reference frame
    reference_frame.build_mask_around_landmarks(patch_shape, group=group)

    return reference_frame


def _build_reference_frame(landmarks, boundary=3, group='source'):
    # translate landmarks to the origin
    minimum = landmarks.bounds(boundary=boundary)[0]
    landmarks = Translation(-minimum).apply(landmarks)

    resolution = landmarks.range(boundary=boundary)
    reference_frame = MaskedImage.init_blank(resolution)
    reference_frame.landmarks[group] = landmarks

    return reference_frame


# TODO: document me!
def densify_shapes(shapes, reference_frame, transform):
    # compute non-linear transforms
    transforms = [transform(reference_frame.landmarks['source'].lms, s)
                  for s in shapes]
    # build dense shapes
    dense_shapes = []
    for (t, s) in zip(transforms, shapes):
        warped_points = t.apply(reference_frame.mask.true_indices())
        dense_shape = PointCloud(np.vstack((s.points, warped_points)))
        dense_shapes.append(dense_shape)

    return dense_shapes


# TODO: document me!
def align_shapes(shapes):
    r"""
    """
    # centralize shapes
    centered_shapes = [Translation(-s.centre()).apply(s) for s in shapes]
    # align centralized shape using Procrustes Analysis
    gpa = GeneralizedProcrustesAnalysis(centered_shapes)
    return [s.aligned_source() for s in gpa.transforms]


class MenpoFitBuilderWarning(Warning):
    r"""
    A warning that some part of building the model may cause issues.
    """
    pass
