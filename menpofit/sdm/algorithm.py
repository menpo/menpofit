from __future__ import division
import numpy as np
from menpo.feature import no_op
from menpo.visualize import print_dynamic
from menpofit.result import NonParametricAlgorithmResult


# TODO document me!
class CRAlgorithm(object):
    r"""
    """
    def train(self, images, gt_shapes, current_shapes, verbose=False,
              **kwargs):
        n_images = len(images)
        n_samples_image = len(current_shapes[0])

        self._features_patch_length = compute_features_info(
            images[0], gt_shapes[0], self.features,
            patch_shape=self.patch_shape)[1]

        # obtain delta_x and gt_x
        delta_x, gt_x = obtain_delta_x(gt_shapes, current_shapes)

        # initialize iteration counter and list of regressors
        k = 0
        self.regressors = []

        # Cascaded Regression loop
        while k < self.iterations:
            # generate regression data
            features = obtain_patch_features(
                images, current_shapes, self.patch_shape, self.features,
                features_patch_length=self._features_patch_length)

            # perform regression
            if verbose:
                print_dynamic('- Performing regression...')
            regressor = self._perform_regression(features, delta_x, **kwargs)
            # add regressor to list
            self.regressors.append(regressor)

            # estimate delta_points
            estimated_delta_x = regressor(features)
            if verbose:
                error = _compute_rmse(delta_x, estimated_delta_x)
                print_dynamic('- Training Error is {0:.4f}.\n'.format(error))

            j = 0
            for shapes in current_shapes:
                for s in shapes:
                    # update current x
                    current_x = s.as_vector() + estimated_delta_x[j]
                    # update current shape inplace
                    s.from_vector_inplace(current_x)
                    # update delta_x
                    delta_x[j] = gt_x[j] - current_x
                    # increase index
                    j += 1
            # increase iteration counter
            k += 1

        # rearrange current shapes into their original list of list form
        return current_shapes

    def run(self, image, initial_shape, gt_shape=None, **kwargs):
        # set current shape and initialize list of shapes
        current_shape = initial_shape
        shapes = [initial_shape]

        # Cascaded Regression loop
        for r in self.regressors:
            # compute regression features
            features = compute_patch_features(
                image, current_shape, self.patch_shape, self.features,
                features_patch_length=self._features_patch_length)

            # solve for increments on the shape vector
            dx = r(features)

            # update current shape
            current_shape = current_shape.from_vector(
                current_shape.as_vector() + dx)
            shapes.append(current_shape)

        # return algorithm result
        return NonParametricAlgorithmResult(image, self, shapes,
                                            gt_shape=gt_shape)


# TODO: document me!
class SN(CRAlgorithm):
    r"""
    Supervised Newton.

    This class implements the Supervised Descent Method technique, proposed
    by Xiong and De la Torre in [XiongD13].

    References
    ----------
    .. [XiongD13] Supervised Descent Method and its Applications to
       Face Alignment
       Xuehan Xiong and Fernando De la Torre Fernando
       IEEE International Conference on Computer Vision and Pattern Recognition
       May, 2013
    """
    def __init__(self, features=no_op, patch_shape=(17, 17), iterations=3,
                 eps=10 ** -5):
        self.patch_shape = patch_shape
        self.features = features
        self.patch_shape = patch_shape
        self.iterations = iterations
        self.eps = eps
        # wire regression callable
        self._perform_regression = _supervised_newton


# TODO: document me!
class SGN(CRAlgorithm):
    r"""
    Supervised Gauss-Newton

    This class implements a variation of the Supervised Descent Method
    [XiongD13] by some of the ideas incorporating ideas...

    References
    ----------
    .. [XiongD13] Supervised Descent Method and its Applications to
       Face Alignment
       Xuehan Xiong and Fernando De la Torre Fernando
       IEEE International Conference on Computer Vision and Pattern Recognition
       May, 2013
    .. [Tzimiropoulos15] Supervised Descent Method and its Applications to
       Face Alignment
       Xuehan Xiong and Fernando De la Torre Fernando
       IEEE International Conference on Computer Vision and Pattern Recognition
       May, 2013
    """
    def __init__(self, features=no_op, patch_shape=(17, 17), iterations=3,
                 eps=10 ** -5):
        self.patch_shape = patch_shape
        self.features = features
        self.patch_shape = patch_shape
        self.iterations = iterations
        self.eps = eps
        # wire regression callable
        self._perform_regression = _supervised_gauss_newton


# TODO: document me!
class _supervised_newton(object):
    r"""
    """
    def __init__(self, features, deltas, gamma=None):
        # ridge regression
        XX = features.T.dot(features)
        XT = features.T.dot(deltas)
        if gamma:
            XX += gamma * np.eye(features.shape[1])
        # descent direction
        self.R = np.linalg.solve(XX, XT)

    def __call__(self, features):
        return np.dot(features, self.R)


# TODO: document me!
class _supervised_gauss_newton(object):
    r"""
    """

    def __init__(self, features, deltas, gamma=None):
        # ridge regression
        XX = deltas.T.dot(deltas)
        XT = deltas.T.dot(features)
        if gamma:
            XX += gamma * np.eye(deltas.shape[1])
        # average Jacobian
        self.J = np.linalg.solve(XX, XT)
        # average Hessian
        self.H = self.J.dot(self.J.T)
        # descent direction
        self.R = np.linalg.solve(self.H, self.J).T

    def __call__(self, features):
        return np.dot(features, self.R)


# TODO: document me!
def _compute_rmse(x1, x2):
    return np.sqrt(np.mean(np.sum((x1 - x2) ** 2, axis=1)))


# TODO: docment me!
def compute_patch_features(image, shape, patch_shape, features_callable,
                           features_patch_length=None):
    """r
    """
    patches = image.extract_patches(shape, patch_size=patch_shape,
                                    as_single_array=True)

    if features_patch_length:
        patch_features = np.empty((shape.n_points, features_patch_length))
        for j, p in enumerate(patches):
            patch_features[j] = features_callable(p[0]).ravel()
    else:
        patch_features = []
        for j, p in enumerate(patches):
            patch_features.append(features_callable(p[0]).ravel())
        patch_features = np.asarray(patch_features)

    return patch_features.ravel()


# TODO: docment me!
def generate_patch_features(image, shapes, patch_shape, features_callable,
                            features_patch_length=None):
    """r
    """
    if features_patch_length:
        patch_features = np.empty((len(shapes),
                                   shapes[0].n_points * features_patch_length))
        for j, s in enumerate(shapes):
            patch_features[j] = compute_patch_features(
                image, s, patch_shape, features_callable,
                features_patch_length=features_patch_length)
    else:
        patch_features = []
        for j, s in enumerate(shapes):
            patch_features.append(compute_patch_features(
                image, s, patch_shape, features_callable,
                features_patch_length=features_patch_length))
        patch_features = np.asarray(patch_features)

    return patch_features.ravel()


# TODO: docment me!
def obtain_patch_features(images, shapes, patch_shape, features_callable,
                          features_patch_length=None):
    """r
    """
    n_images = len(images)
    n_shapes = len(shapes[0])
    n_points = shapes[0][0].n_points

    if features_patch_length:

        patch_features = np.empty((n_images, (n_shapes * n_points *
                                              features_patch_length)))
        for j, i in enumerate(images):
            patch_features[j] = generate_patch_features(
                i, shapes[j], patch_shape, features_callable,
                features_patch_length=features_patch_length)
    else:
        patch_features = []
        for j, i in images:
            patch_features.append(generate_patch_features(
                i, shapes[j], patch_shape, features_callable,
                features_patch_length=features_patch_length))
        patch_features = np.asarray(patch_features)

    return patch_features.reshape((-1, n_points * features_patch_length))


def compute_delta_x(gt_shape, current_shapes):
    r"""
    """
    n_x = gt_shape.n_parameters
    n_current_shapes = len(current_shapes)

    # initialize ground truth and delta shape vectors
    gt_x = np.empty((n_current_shapes, n_x))
    delta_x = np.empty((n_current_shapes, n_x))

    for j, s in enumerate(current_shapes):
        # compute ground truth shape vector
        gt_x[j] = gt_shape.as_vector()
        # compute delta shape vector
        delta_x[j] = gt_x[j] - s.as_vector()

    return delta_x, gt_x


def obtain_delta_x(gt_shapes, current_shapes):
    r"""
    """
    n_x = gt_shapes[0].n_parameters
    n_gt_shapes = len(gt_shapes)
    n_current_shapes = len(current_shapes[0])

    # initialize current, ground truth and delta parameters
    gt_x = np.empty((n_gt_shapes, n_current_shapes, n_x))
    delta_x = np.empty((n_gt_shapes, n_current_shapes, n_x))

    # obtain ground truth points and compute delta points
    for j, (gt_s, shapes) in enumerate(zip(gt_shapes, current_shapes)):
        # compute ground truth par
        delta_x[j], gt_x[j] = compute_delta_x(gt_s, shapes)

    return delta_x.reshape((-1, n_x)), gt_x.reshape((-1, n_x))


def compute_features_info(image, shape, features_callable,
                          patch_shape=(17, 17)):
    # TODO: include offsets support?
    patches = image.extract_patches(shape, patch_size=patch_shape,
                                    as_single_array=True)

    # TODO: include offsets support?
    features_patch_shape = features_callable(patches[0, 0]).shape
    features_patch_length = np.prod(features_patch_shape)
    features_shape = patches.shape[:1] + features_patch_shape
    features_length = np.prod(features_shape)

    return (features_patch_shape, features_patch_length,
            features_shape, features_length)

# def initialize_sampling(self, image, group=None, label=None):
#         if self._sampling is None:
#             sampling = np.ones(self.patch_shape, dtype=np.bool)
#         else:
#             sampling = self._sampling
#
#         # TODO: include offsets support?
#         patches = image.extract_patches_around_landmarks(
#             group=group, label=label, patch_size=self.patch_shape,
#             as_single_array=True)
#
#         # TODO: include offsets support?
#         features_patch_shape = self.features(patches[0, 0]).shape
#         self._features_patch_length = np.prod(features_patch_shape)
#         self._features_shape = (patches.shape[0], features_patch_shape)
#         self._features_length = np.prod(self._features_shape)
#
#         feature_mask = np.tile(sampling[None, None, None, ...],
#                                self._feature_shape[:3] + (1, 1))
#         self._feature_mask = np.nonzero(feature_mask.flatten())[0]
