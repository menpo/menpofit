import numpy as np
from menpo.image import Image, MaskedImage, BooleanImage, extract_patches
from menpo.transform import AlignmentUniformScale

def convert_from_menpo(menpo_image):

    cls = eval(type(menpo_image).__name__)

    if cls is Image:
        image = cls(np.rollaxis(menpo_image.pixels, -1), copy=True)
    elif cls is MaskedImage:
        image = cls(np.rollaxis(menpo_image.pixels, -1),
                    mask=menpo_image.mask.pixels[..., 0], copy=True)
    elif cls is BooleanImage:
        image = cls(menpo_image.pixels[..., 0], copy=True)
    else:
        raise ValueError('{} is not a Menpo image class'.format(cls))

    if menpo_image.has_landmarks:
        image.landmarks = menpo_image.landmarks

    return image


def convert_to_menpo(image):

    cls = eval(type(image).__name__)

    if cls is Image:
        menpo_image = cls(np.rollaxis(image.pixels,  0, image.n_dims+1),
                          copy=True)
    elif cls is MaskedImage:
        menpo_image = cls(np.rollaxis(image.pixels, 0, image.n_dims+1),
                          mask=image.mask.pixels[0, ...], copy=True)
    elif cls is BooleanImage:
        menpo_image = cls(image.pixels[0, ...], copy=True)
    else:
        raise ValueError('{} is not a cvpr2015 image class'.format(cls))

    if image.has_landmarks:
        menpo_image.landmarks = image.landmarks

    return menpo_image


def build_parts_image(image, centres, parts_shape, offsets=np.array([[0, 0]]),
                      normalize_parts=False):

    # extract patches
    parts = extract_patches(image.pixels, np.round(centres.points),
                            np.array(parts_shape), offsets)

    # build parts image
    # img.pixels: n_centres x n_offsets x n_channels x height x width
    img = Image(parts)

    if normalize_parts:
        # normalize parts if required
        img.normalize_norm_inplace(mode='per_channel')

    return img

def build_sampling_grid(patch_shape):
    r"""
    """
    patch_shape = np.array(patch_shape)
    patch_half_shape = np.require(np.round(patch_shape / 2), dtype=int)
    start = -patch_half_shape
    end = patch_half_shape + 1
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    return sampling_grid.swapaxes(0, 2).swapaxes(0, 1)

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
    return Similarity.identity(source.n_dims).from_vector(parameters + noise)

def align_shape_with_bb(shape, bounding_box):
    r"""
    Returns the Similarity transform that aligns the provided shape with the
    provided bounding box.

    Parameters
    ----------
    shape: :class:`menpo.shape.PointCloud`
        The shape to be aligned.
    bounding_box: (2, 2) ndarray
        The bounding box specified as:

            np.array([[x_min, y_min], [x_max, y_max]])

    Returns
    -------
    transform : :class: `menpo.transform.Similarity`
        The align transform
    """
    shape_box = PointCloud(shape.bounds())
    bounding_box = PointCloud(bounding_box)
    return AlignmentSimilarity(shape_box, bounding_box, rotation=False)

def rescale_to_reference_shape(image, reference_shape, group=None, label=None, round='ceil', order=1):
    pc = image.landmarks[group][label]
    scale = AlignmentUniformScale(pc, reference_shape).as_vector().copy()
    return image.rescale(scale, round=round, order=order)