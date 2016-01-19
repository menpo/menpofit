from functools import wraps, partial
import numpy as np

from menpo.shape import PointCloud


def pointcloud_to_points(wrapped):
    @wraps(wrapped)
    def wrapper(*args, **kwargs):
        args = list(args)
        for index, arg in enumerate(args):
            if isinstance(arg, PointCloud):
                args[index] = arg.points
        for key in kwargs:
            if isinstance(kwargs[key], PointCloud):
                kwargs[key] = kwargs[key].points
        return wrapped(*args, **kwargs)
    return wrapper


def bb_area(shape):
    # Area = w + h
    height, width = np.max(shape, axis=0) - np.min(shape, axis=0)
    return height * width


def bb_perimeter(shape):
    # Area = 2(w + h)
    height, width = np.max(shape, axis=0) - np.min(shape, axis=0)
    return 2 * (height + width)


def bb_avg_edge_length(shape):
    # 0.5(w + h) = (2w + 2h) / 4
    height, width = np.max(shape, axis=0) - np.min(shape, axis=0)
    return 0.5 * (height + width)


def bb_diagonal(shape):
    # sqrt(w**2 + h**2)
    height, width = np.max(shape, axis=0) - np.min(shape, axis=0)
    return np.sqrt(width ** 2 + height ** 2)


bb_norm_types = {
    'avg_edge_length': bb_avg_edge_length,
    'perimeter': bb_perimeter,
    'diagonal': bb_diagonal,
    'area': bb_area
}


@pointcloud_to_points
def bb_normalised_error(shape_error_f, shape, gt_shape,
                        norm_shape=None, norm_type='avg_edge_length'):
    if norm_type not in bb_norm_types:
        raise ValueError('norm_type must be one of '
                         '{avg_edge_length, perimeter, diagonal, area}.')
    if norm_shape is None:
        norm_shape = gt_shape
    return (shape_error_f(shape, gt_shape) /
            bb_norm_types[norm_type](norm_shape))


def distance_two_indices(index1, index2, gt_shape):
    return euclidean_error(gt_shape[index1], gt_shape[index2])


@pointcloud_to_points
def distance_normalised_error(shape_error_f, distance_norm_f, shape, gt_shape):
    return shape_error_f(shape, gt_shape) / distance_norm_f(shape, gt_shape)


@pointcloud_to_points
def distance_indexed_normalised_error(shape_error_f, index1, index2, shape,
                                      gt_shape):
    return shape_error_f(shape, gt_shape) / distance_two_indices(index1, index2,
                                                                 gt_shape)


# TODO: Document me!
@pointcloud_to_points
def root_mean_square_error(shape, gt_shape):
    r"""
    """
    return np.sqrt(np.mean((shape.ravel() - gt_shape.ravel()) ** 2))


# TODO: Document me!
@pointcloud_to_points
def euclidean_error(shape, gt_shape):
    r"""
    """
    return np.mean(np.sqrt(np.sum((shape - gt_shape) ** 2, axis=-1)))


# TODO: Document me!
root_mean_square_bb_normalised_error = partial(bb_normalised_error,
                                               root_mean_square_error)

# TODO: Document me!
euclidean_bb_normalised_error = partial(bb_normalised_error, euclidean_error)


# TODO: Document me!
root_mean_square_distance_normalised_error = partial(distance_normalised_error,
                                                     root_mean_square_error)

# TODO: Document me!
euclidean_distance_normalised_error = partial(distance_normalised_error,
                                              euclidean_error)


compute_normalise_point_to_point_error = euclidean_bb_normalised_error
compute_root_mean_square_error = root_mean_square_error
compute_point_to_point_error = euclidean_error


# TODO: Document me!
def compute_cumulative_error(errors, boundaries):
    r"""
    """
    n_errors = len(errors)
    return [np.count_nonzero([errors <= x]) / n_errors for x in boundaries]