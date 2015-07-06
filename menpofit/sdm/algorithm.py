from __future__ import division
import numpy as np
from menpo.feature import no_op
from menpo.visualize import print_dynamic
from menpofit.result import NonParametricAlgorithmResult


# TODO: compute more meaningful error
# TODO: document me!
class SupervisedDescentAlgorithm(object):
    r"""
    """
    def train(self, images, gt_shapes, current_shapes, verbose=False,
              **kwargs):
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
                print_dynamic('- Performing regression.')
            r = self._regressor_cls(**kwargs)
            r.train(features, delta_x)
            # add regressor to list
            self.regressors.append(r)

            # estimate delta_points
            estimated_delta_x = r.predict(features)
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

    def increment(self, images, gt_shapes, current_shapes, verbose=False,
                  **kwarg):
        # obtain delta_x and gt_x
        delta_x, gt_x = obtain_delta_x(gt_shapes, current_shapes)

        # Cascaded Regression loop
        for r in self.regressors:
            # generate regression data
            features = obtain_patch_features(
                images, current_shapes, self.patch_shape, self.features,
                features_patch_length=self._features_patch_length)

            # update regression
            if verbose:
                print_dynamic('- Updating regression')
            r.increment(features, delta_x)

            # estimate delta_points
            estimated_delta_x = r.predict(features)
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
            dx = r.predict(features)

            # update current shape
            current_shape = current_shape.from_vector(
                current_shape.as_vector() + dx)
            shapes.append(current_shape)

        # return algorithm result
        return NonParametricAlgorithmResult(image, self, shapes,
                                            gt_shape=gt_shape)


# TODO: document me!
class Newton(SupervisedDescentAlgorithm):
    r"""
    """
    def __init__(self, features=no_op, patch_shape=(17, 17), iterations=3,
                 eps=10**-5):
        self.patch_shape = patch_shape
        self.features = features
        self.patch_shape = patch_shape
        self.iterations = iterations
        self.eps = eps
        self._regressor_cls = _incremental_least_squares


# TODO: document me!
class GaussNewton(SupervisedDescentAlgorithm):
    r"""
    """
    def __init__(self, features=no_op, patch_shape=(17, 17), iterations=3,
                 eps=10**-5):
        self.patch_shape = patch_shape
        self.features = features
        self.patch_shape = patch_shape
        self.iterations = iterations
        self.eps = eps
        self._perform_regression = _incremental_indirect_least_squares


# TODO: document me!
class _incremental_least_squares(object):
    r"""
    """
    def __init__(self, l=0):
        self.l = l

    def train(self, X, Y):
        # regularized least squares
        XX = X.T.dot(X)
        np.fill_diagonal(XX, self.l + np.diag(XX))
        self.V = np.linalg.inv(XX)
        self.W = self.V.dot(X.T.dot(Y))

    def increment(self, X, Y):
        # incremental regularized least squares
        U = X.dot(self.V).dot(X.T)
        np.fill_diagonal(U, 1 + np.diag(U))
        U = np.linalg.inv(U)
        Q = self.V.dot(X.T).dot(U).dot(X)
        self.V = self.V - Q.dot(self.V)
        self.W = self.W - Q.dot(self.W) + self.V.dot(X.T.dot(Y))

    def predict(self, x):
        return np.dot(x, self.W)


# TODO: document me!
class _incremental_indirect_least_squares(object):
    r"""
    """
    def __init__(self, l=0, d=0):
        self._ils = _incremental_least_squares(l)
        self.d = d

    def train(self, X, Y):
        # regularized least squares exchanging the roles of X and Y
        self._ils.train(Y, X)
        J = self._ils.W
        # solve the original problem by computing the pseudo-inverse of the
        # previous solution
        H = J.T.dot(J)
        np.fill_diagonal(H, self.d + np.diag(H))
        self.W = np.linalg.solve(H, J.T)

    def predict(self, x):
        return np.dot(x, self.W)


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
