import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.blas import get_blas_funcs

from menpo.math import pca


class IRLRegression(object):
    r"""
    Class for training and applying Incremental Regularized Linear Regression.

    Parameters
    ----------
    alpha : `float`, optional
        The regularization parameter of the features.
    bias : `bool`, optional
        If ``True``, a bias term is used.
    incrementable : `bool`, optional
        If ``True``, then the regression model will have the ability to get
        incremented.
    """

    def __init__(self, alpha=0, bias=True, incrementable=False):
        self.alpha = alpha
        self.bias = bias
        self.incrementable = incrementable
        self.V = None
        self.W = None

    def train(self, X, Y):
        r"""
        Train the regression model.

        Parameters
        ----------
        X : ``(n_features, n_samples)`` `ndarray`
            The array of feature vectors.
        Y : ``(n_dims, n_samples)`` `ndarray`
            The array of target vectors.
        """
        self._XX = self._XY = None
        if not self.incrementable:
            # Cholesky solve of the (symmetric) normal equations: half the
            # arithmetic of the LU solve, in place, and no copy of X for the bias
            try:
                self.accumulate(X, Y)
                self.solve()
                return
            except np.linalg.LinAlgError:
                # the matrix is not positive definite (for example, no
                # regularization and fewer samples than features)
                self._XX = self._XY = None

        if self.bias:
            # add bias
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        # regularized linear regression
        XX = X.T.dot(X)
        # ensure covariance is perfectly symmetric for inversion
        XX = (XX + XX.T) / 2.0
        if self.alpha:
            np.fill_diagonal(XX, self.alpha + np.diag(XX))
        if self.incrementable:
            self.V = np.linalg.inv(XX)
        self.W = np.linalg.solve(XX, X.T.dot(Y))

    # ``accumulate`` and ``solve`` train the model on batches of samples
    _supports_batches = True

    def accumulate(self, X, Y):
        r"""
        Add a batch of samples to the model, which is trained with `solve`.

        The solution of ridge regression only depends on ``X^T X`` and
        ``X^T Y``, which are sums over the samples. Accumulating them over
        batches and solving once gives exactly the model of `train` on all the
        samples, while only one batch of samples is in memory at a time. The
        memory is that of the ``(n_features, n_features)`` matrix ``X^T X``, of
        which only the lower triangle is computed (a symmetric rank-k update,
        half the arithmetic of the full product).

        Parameters
        ----------
        X : ``(n_samples, n_features)`` `ndarray`
            The batch of feature vectors.
        Y : ``(n_samples, n_dims)`` `ndarray`
            The batch of target vectors.
        """
        if self.incrementable:
            raise ValueError("A model that is trained in batches cannot be incremented")
        # the matrix is of the precision of the features: float32 features
        # (half the memory of float64) give a float32 matrix
        X = np.asarray(X)
        if X.dtype not in (np.float32, np.float64):
            X = X.astype(np.float64)
        X = np.ascontiguousarray(X)
        Y = np.asarray(Y).astype(X.dtype, copy=False)
        if getattr(self, "_XX", None) is None:
            n_features = X.shape[1]
            self._XX = np.zeros((n_features, n_features), dtype=X.dtype, order="F")
            self._XY = np.zeros((n_features, Y.shape[1]), dtype=X.dtype)
            self._x_sum = np.zeros(n_features, dtype=np.float64)
            self._y_sum = np.zeros(Y.shape[1], dtype=np.float64)
            self._n_samples = 0
        # X.T is Fortran ordered, so BLAS does not need a copy of the batch
        syrk = get_blas_funcs("syrk", (X,))
        self._XX = syrk(1.0, X.T, beta=1.0, c=self._XX, lower=1, overwrite_c=1)
        self._XY += X.T.dot(Y)
        if self.bias:
            # the column of ones is not appended to X: it only adds these sums
            self._x_sum += X.sum(axis=0, dtype=np.float64)
            self._y_sum += Y.sum(axis=0, dtype=np.float64)
            self._n_samples += X.shape[0]

    def solve(self):
        r"""
        Solve the regression for all the batches that were added with
        `accumulate`. The Cholesky factorisation is done in place, so that the
        memory needed is that of one ``(n_features, n_features)`` matrix. The
        bias is solved for with the Schur complement, so that this matrix is
        not extended by a row and a column.
        """
        if getattr(self, "_XX", None) is None:
            raise ValueError("No samples were accumulated")
        XX, XY = self._XX, self._XY
        x_sum, y_sum, n_samples = self._x_sum, self._y_sum, self._n_samples
        self._XX = self._XY = self._x_sum = self._y_sum = None
        if self.alpha:
            XX[np.diag_indices_from(XX)] += self.alpha
        right_hand_side = XY
        if self.bias:
            right_hand_side = np.hstack((XY, x_sum[:, None].astype(XY.dtype)))
        factor = cho_factor(XX, lower=True, overwrite_a=True, check_finite=False)
        solution = cho_solve(factor, right_hand_side, check_finite=False)
        if not self.bias:
            self.W = solution
            return
        weights, from_ones = solution[:, :-1], solution[:, -1]
        # [[XX, s], [s^T, n + alpha]] [W; w0] = [XY; y_sum]
        schur = n_samples + self.alpha - x_sum.dot(from_ones)
        w0 = (y_sum - x_sum.dot(weights)) / schur
        self.W = np.vstack((weights - np.outer(from_ones, w0), w0)).astype(
            weights.dtype, copy=False
        )

    def increment(self, X, Y):
        r"""
        Incrementally update the regression model.

        Parameters
        ----------
        X : ``(n_features, n_samples)`` `ndarray`
            The array of feature vectors.
        Y : ``(n_dims, n_samples)`` `ndarray`
            The array of target vectors.

        Raises
        ------
        ValueError
            Model is not incrementable
        """
        if not self.incrementable:
            raise ValueError("Model is not incrementable")

        if self.bias:
            # add bias
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        # incremental regularized linear regression
        U = X.dot(self.V).dot(X.T)
        if self.alpha:
            np.fill_diagonal(U, self.alpha + np.diag(U))
        U = np.linalg.inv(U)
        Q = self.V.dot(X.T).dot(U).dot(X)
        self.V = self.V - Q.dot(self.V)
        self.W = self.W - Q.dot(self.W) + self.V.dot(X.T.dot(Y))

    def predict(self, x):
        r"""
        Makes a prediction using the trained regression model.

        Parameters
        ----------
        x : ``(n_features,)`` `ndarray`
            The input feature vector.

        Returns
        -------
        prediction : ``(n_dims,)`` `ndarray`
            The prediction vector.
        """
        if self.bias:
            if len(x.shape) == 1:
                x = np.hstack((x, np.ones(1)))
            else:
                x = np.hstack((x, np.ones((x.shape[0], 1))))
        return np.dot(x, self.W)


class IIRLRegression(IRLRegression):
    r"""
    Class for training and applying Indirect Incremental Regularized Linear
    Regression.

    Parameters
    ----------
    alpha : `float`, optional
        The regularization parameter.
    bias : `bool`, optional
        If ``True``, a bias term is used.
    alpha2 : `float`, optional
        The regularization parameter of the Hessian.
    """

    # trains with a different cost: no accumulate and solve
    _supports_batches = False

    def __init__(self, alpha=0, bias=False, alpha2=0):
        # TODO: Can we model the bias? May need to slice off of prediction?
        super(IIRLRegression, self).__init__(alpha=alpha, bias=False)
        self.alpha2 = alpha2

    def train(self, X, Y):
        r"""
        Train the regression model.

        Parameters
        ----------
        X : ``(n_features, n_samples)`` `ndarray`
            The array of feature vectors.
        Y : ``(n_dims, n_samples)`` `ndarray`
            The array of target vectors.
        """
        # regularized linear regression exchanging the roles of X and Y
        super(IIRLRegression, self).train(Y, X)
        J = self.W
        # solve the original problem by computing the pseudo-inverse of the
        # previous solution
        # Note that everything is transposed from the above exchanging of roles
        H = J.dot(J.T)
        if self.alpha2:
            np.fill_diagonal(H, self.alpha2 + np.diag(H))
        self.W = np.linalg.solve(H, J).T

    def increment(self, X, Y):
        r"""
        Incrementally update the regression model.

        Parameters
        ----------
        X : ``(n_features, n_samples)`` `ndarray`
            The array of feature vectors.
        Y : ``(n_dims, n_samples)`` `ndarray`
            The array of target vectors.

        Raises
        ------
        ValueError
            Model is not incrementable
        """
        # incremental least squares exchanging the roles of X and Y
        super(IIRLRegression, self).increment(Y, X)
        J = self.W
        # solve the original problem by computing the pseudo-inverse of the
        # previous solution
        # Note that everything is transposed from the above exchanging of roles
        H = J.dot(J.T)
        if self.alpha2:
            np.fill_diagonal(H, self.alpha2 + np.diag(H))
        self.W = np.linalg.solve(H, J)

    def predict(self, x):
        r"""
        Makes a prediction using the trained regression model.

        Parameters
        ----------
        x : ``(n_features,)`` `ndarray`
            The input feature vector.

        Returns
        -------
        prediction : ``(n_dims,)`` `ndarray`
            The prediction vector.
        """
        return np.dot(x, self.W)


class PCRRegression(object):
    r"""
    Class for training and applying Multivariate Linear Regression using
    Principal Component Regression.

    Parameters
    ----------
    variance : `float` or ``None``, optional
        The SVD variance.
    bias : `bool`, optional
        If ``True``, a bias term is used.
    """

    def __init__(self, variance=None, bias=True):
        self.variance = variance
        self.bias = bias
        self.R = None
        self.V = None

    def train(self, X, Y):
        r"""
        Train the regression model.

        Parameters
        ----------
        X : ``(n_features, n_samples)`` `ndarray`
            The array of feature vectors.
        Y : ``(n_dims, n_samples)`` `ndarray`
            The array of target vectors.
        """
        if self.bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        # Reduce variance
        U, s, self.V = np.linalg.svd(X, full_matrices=False)
        if self.variance:
            variation = np.cumsum(s) / np.sum(s)
            # Inverted for easier parameter semantics
            k = np.sum(variation < self.variance)
            U = U[:, :k]
            self.V = self.V[:k, :]
            s = s[:k]
        S = np.diag(s)

        # Perform PCR
        self.R = self.V.T.dot(np.linalg.inv(S)).dot(U.T).dot(Y)

    def increment(self, X, Y):
        r"""
        Incrementally update the regression model.

        Parameters
        ----------
        X : ``(n_features, n_samples)`` `ndarray`
            The array of feature vectors.
        Y : ``(n_dims, n_samples)`` `ndarray`
            The array of target vectors.

        Raises
        ------
        ValueError
            Model is not incrementable
        """
        raise NotImplementedError()

    def predict(self, x):
        r"""
        Makes a prediction using the trained regression model.

        Parameters
        ----------
        x : ``(n_features,)`` `ndarray`
            The input feature vector.

        Returns
        -------
        prediction : ``(n_dims,)`` `ndarray`
            The prediction vector.
        """
        if self.bias:
            if len(x.shape) == 1:
                x = np.hstack((x, np.ones(1)))
            else:
                x = np.hstack((x, np.ones((x.shape[0], 1))))
        x = np.dot(np.dot(x, self.V.T), self.V)
        return np.dot(x, self.R)


class OptimalLinearRegression(object):
    r"""
    Class for training and applying Multivariate Linear Regression using optimal
    reconstructions.

    Parameters
    ----------
    variance : `float` or ``None``, optional
        The SVD variance.
    bias : `bool`, optional
        If ``True``, a bias term is used.
    """

    def __init__(self, variance=None, bias=True):
        self.variance = variance
        self.bias = bias
        self.R = None

    def train(self, X, Y):
        r"""
        Train the regression model.

        Parameters
        ----------
        X : ``(n_features, n_samples)`` `ndarray`
            The array of feature vectors.
        Y : ``(n_dims, n_samples)`` `ndarray`
            The array of target vectors.
        """
        if self.bias:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        # Reduce variance of X
        U, l, _ = pca(X, centre=False)
        k = U.shape[0]
        if self.variance is not None:
            variation = np.cumsum(l) / np.sum(l)
            # Inverted for easier parameter semantics
            k = np.sum(variation < self.variance)
            U = U[:k, :]

        # Whitened components
        inv_eig = np.sqrt(np.linalg.inv(np.diag(l[:k])))
        U = inv_eig.dot(U)

        A = X.T.dot(Y).dot(Y.T).dot(X)
        A_tilde = U.dot(A).dot(U.T)

        V, l2, _ = pca(A_tilde, centre=False)
        H = V.dot(U)

        self.R = H.T.dot(np.linalg.pinv(X.dot(H.T)).dot(Y))

    def increment(self, X, Y):
        r"""
        Incrementally update the regression model.

        Parameters
        ----------
        X : ``(n_features, n_samples)`` `ndarray`
            The array of feature vectors.
        Y : ``(n_dims, n_samples)`` `ndarray`
            The array of target vectors.

        Raises
        ------
        ValueError
            Model is not incrementable
        """
        raise NotImplementedError()

    def predict(self, x):
        r"""
        Makes a prediction using the trained regression model.

        Parameters
        ----------
        x : ``(n_features,)`` `ndarray`
            The input feature vector.

        Returns
        -------
        prediction : ``(n_dims,)`` `ndarray`
            The prediction vector.
        """
        if self.bias:
            if len(x.shape) == 1:
                x = np.hstack((x, np.ones(1)))
            else:
                x = np.hstack((x, np.ones((x.shape[0], 1))))
        return np.dot(x, self.R)


class OPPRegression(object):
    r"""
    Class for training and applying Multivariate Linear Regression using
    Orthogonal Procrustes Problem reconstructions.

    Parameters
    ----------
    bias : `bool`, optional
        If ``True``, a bias term is used.
    whiten : `bool`, optional
        Whether to use a whitened PCA model.
    """

    def __init__(self, bias=True, whiten=False):
        self.bias = bias
        self.R = None
        self.whiten = whiten

    def train(self, X, Y):
        r"""
        Train the regression model.

        Parameters
        ----------
        X : ``(n_features, n_samples)`` `ndarray`
            The array of feature vectors.
        Y : ``(n_dims, n_samples)`` `ndarray`
            The array of target vectors.
        """
        if self.bias:
            # add bias
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        # Whiten
        if self.whiten:
            U, l, _ = pca(X, centre=False)
            inv_eig = np.sqrt(np.linalg.inv(np.diag(l)))
            U = inv_eig.dot(U)
            X = X.dot(U.T).dot(U)

        U, _, V = np.linalg.svd(X.T.dot(Y), full_matrices=False)
        # Skinny SVD
        self.R = U.dot(V)

    def increment(self, X, Y):
        r"""
        Incrementally update the regression model.

        Parameters
        ----------
        X : ``(n_features, n_samples)`` `ndarray`
            The array of feature vectors.
        Y : ``(n_dims, n_samples)`` `ndarray`
            The array of target vectors.

        Raises
        ------
        ValueError
            Model is not incrementable
        """
        raise NotImplementedError()

    def predict(self, x):
        r"""
        Makes a prediction using the trained regression model.

        Parameters
        ----------
        x : ``(n_features,)`` `ndarray`
            The input feature vector.

        Returns
        -------
        prediction : ``(n_dims,)`` `ndarray`
            The prediction vector.
        """
        if self.bias:
            if len(x.shape) == 1:
                x = np.hstack((x, np.ones(1)))
            else:
                x = np.hstack((x, np.ones((x.shape[0], 1))))
        return np.dot(x, self.R)
