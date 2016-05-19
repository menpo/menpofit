
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from sklearn import svm
from sklearn import linear_model


class MCF(object):
    r"""
    Multi-channel Correlation Filter
    """
    def __init__(self, X, Y, l=0, cosine_mask=False):

        if (X[0].shape[0],) + X[0].shape[-2:] != (len(Y),) + Y[0].shape:
            raise ValueError('')

        n_offsets, n_channels, height, width = X[0].shape

        self._cosine_mask = 1
        if cosine_mask:
            c1 = np.cos(np.linspace(-np.pi/2, np.pi/2, height))
            c2 = np.cos(np.linspace(-np.pi/2, np.pi/2, width))
            self._cosine_mask = c1[..., None].dot(c2[None, ...])


        X_hat = []
        for x in X:
            x_hat = np.require(fft2(self._cosine_mask * x),
                               dtype=np.complex64)
            X_hat.append(x_hat)

        Y_hat = []
        for y in Y:
            y_hat = np.require(fft2(y),
                               dtype=np.complex64)
            Y_hat.append(y_hat)

        self.f = np.zeros((n_channels, height, width), dtype=np.complex64)
        for j in range(height):
            for k in range(width):
                H_hat = np.zeros((n_channels, n_channels), dtype=np.complex64)
                J_hat = np.zeros((n_channels,), dtype=np.complex64)
                for x_hat in X_hat:
                    for o, y_hat in enumerate(Y_hat):
                        x_hat_ij = x_hat[o, :, j, k][:]
                        H_hat += np.conj(x_hat_ij[..., None]).dot(
                            x_hat_ij[None, ...])
                        J_hat += np.conj(x_hat_ij) * y_hat[j, k]
                H_hat += l * np.eye(H_hat.shape[0])
                self.f[..., j, k] = np.linalg.solve(H_hat, J_hat)

    def __call__(self, x):
        return np.real(
            ifft2(self.f * np.require(fft2(self._cosine_mask * x),
                                      dtype=np.complex64)))

    def _compute_fft2s(self, X):
        X_hat = []
        for x in X:
            x_hat = np.require(fft2(self._cosine_mask * x),
                               dtype=np.complex64)
            X_hat.append(x_hat)
        return X_hat


class MultipleMCF(object):
    r"""
    Multiple of Multi-channel Correlation Filter
    """
    def __init__(self, clfs):

        self._cosine_mask = clfs[0]._cosine_mask

        # concatenate all filters
        n_channels, height, width = clfs[0].f.shape
        n_landmarks = len(clfs)
        self.F = np.zeros((n_landmarks, n_channels, height, width),
                          dtype=np.complex64)
        for j, clf in enumerate(clfs):
            self.F[j, ...] = clf.f

    def __call__(self, parts_image):

        # compute responses
        parts_response = np.sum(np.real(ifft2(
            self.F * np.require(fft2(self._cosine_mask *
                                     parts_image.pixels[:, 0, :, ...]),
                                dtype=np.complex64))), axis=-3)

        # normalize
        min_parts_response = np.min(parts_response,
                                    axis=(-2, -1))[..., None, None]
        parts_response -= min_parts_response
        parts_response /= np.max(parts_response,
                                 axis=(-2, -1))[..., None, None]

        return parts_response

    def invert_filters(self):
        return np.real(fftshift(ifft2(self.F), axes=(-2, -1)))


class LinearSVMLR(object):
    r"""
    Binary classifier that combines Linear Support Vector Machines and
    Logistic Regression.
    """
    def __init__(self, samples, mask, threshold=0.05, **kwarg):

        mask = mask[0]

        n_samples = len(samples)
        n_offsets, n_channels, height, width = samples[0].shape

        true_mask = mask >= threshold
        false_mask = mask < threshold

        n_true = len(mask[true_mask])
        n_false = len(mask[false_mask][::])

        pos_labels = np.ones((n_true * n_samples,))
        neg_labels = -np.ones((n_false * n_samples,))

        pos_samples = np.zeros((n_channels, n_true * n_samples))
        neg_samples = np.zeros((n_channels, n_false * n_samples))
        for j, x in enumerate(samples):
            pos_index = j*n_true
            pos_samples[:, pos_index:pos_index+n_true] = x[0, :, true_mask].T
            neg_index = j*n_false
            neg_samples[:, neg_index:neg_index+n_false] = x[0, :, false_mask].T

        X = np.vstack((pos_samples.T, neg_samples.T))
        t = np.hstack((pos_labels, neg_labels))

        self.clf1 = svm.LinearSVC(class_weight='auto', **kwarg)
        self.clf1.fit(X, t)
        t1 = self.clf1.decision_function(X)
        self.clf2 = linear_model.LogisticRegression(class_weight='auto')
        self.clf2.fit(t1[..., None], t)

    def __call__(self, x):
        t1_pred = self.clf1.decision_function(x)
        return self.clf2.predict_proba(t1_pred[..., None])[:, 1]


class MultipleLinearSVMLR(object):
    r"""
    Multiple Binary classifier that combines Linear Support Vector Machines
    and Logistic Regression.
    """
    def __init__(self, clfs):

        self.classifiers = clfs
        self.n_clfs = len(clfs)

    def __call__(self, parts_image):

        h, w = parts_image.shape[-2:]
        parts_pixels = parts_image.pixels

        parts_response = np.zeros((self.n_clfs, h, w))
        for j, clf in enumerate(self.classifiers):
            i = parts_pixels[j, ...].reshape((parts_image.shape[-3], -1))
            parts_response[j, ...] = clf(i.T).reshape((h, w))

        # normalize
        min_parts_response = np.min(parts_response,
                                    axis=(-2, -1))[..., None, None]
        parts_response -= min_parts_response
        parts_response /= np.max(parts_response,
                                 axis=(-2, -1))[..., None, None]

        return parts_response
