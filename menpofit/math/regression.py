import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.blas import get_blas_funcs

from menpo.math import pca


class IRLRegression(object):
    r"""
    Class for training and applying Incremental Regularized Linear Regression.

    The solution of ridge regression only depends on the products ``X^T X`` and
    ``X^T Y``, which are sums over the samples. They are accumulated (so
    incrementing adds the new samples to them) and the normal equations are
    solved with a Cholesky factorisation, which needs ``X^T X`` to be positive
    definite: use a ridge penalty (``alpha > 0``) when there are fewer samples
    than features.

    Parameters
    ----------
    alpha : `float`, optional
        The regularization parameter of the features.
    bias : `bool`, optional
        If ``True``, a bias term is used.
    incrementable : `bool`, optional
        If ``True``, then the regression model will have the ability to get
        incremented: ``X^T X`` is kept (``n_features ** 2`` values, twice that
        while solving).
    """

    def __init__(self, alpha=0, bias=True, incrementable=False):
        self.alpha = alpha
        self.bias = bias
        self.incrementable = incrementable
        self._W = None
        self._XX = None

    @property
    def W(self):
        # an incrementable model is solved when its weights are needed, after
        # all the batches that were added
        if self._W is None and self._XX is not None:
            self._solve()
        return self._W

    @W.setter
    def W(self, value):
        self._W = value

    def __getstate__(self):
        self.W  # a model is saved solved
        return self.__dict__

    def train(self, X, Y):
        r"""
        Train the regression model.

        Parameters
        ----------
        X : ``(n_samples, n_features)`` `ndarray`
            The array of feature vectors.
        Y : ``(n_samples, n_dims)`` `ndarray`
            The array of target vectors.
        """
        self._XX = self._W = None
        self._accumulate(X, Y)
        if not self.incrementable:
            self._solve()

    def increment(self, X, Y):
        r"""
        Incrementally update the regression model.

        Parameters
        ----------
        X : ``(n_samples, n_features)`` `ndarray`
            The array of feature vectors.
        Y : ``(n_samples, n_dims)`` `ndarray`
            The array of target vectors.

        Raises
        ------
        ValueError
            Model is not incrementable
        """
        if not self.incrementable:
            raise ValueError("Model is not incrementable")
        self._accumulate(X, Y)
        self._W = None

    def _accumulate(self, X, Y):
        # Ridge regression is W = (X^T X + alpha I)^-1 X^T Y, with X the features
        # (n_samples x n_features) and Y the targets (n_samples x n_dims). Both
        # products are sums over the samples, so they are accumulated over batches:
        #   _XX = sum of X^T X  (n_features x n_features, symmetric: only its lower
        #                        triangle is computed)
        #   _XY = sum of X^T Y  (n_features x n_dims)
        # With a bias, X has a column of ones appended, which is not built (it would
        # copy X): the entries of the products that it adds are
        #   _x_sum = sum of X^T 1 (n_features), _y_sum = sum of Y^T 1 (n_dims) and
        #   _n_samples = 1^T 1.
        # The matrix has the precision of the features: float32 features give a
        # float32 matrix, half the memory of float64.
        X = np.asarray(X)
        if X.dtype not in (np.float32, np.float64):
            X = X.astype(np.float64)
        X = np.ascontiguousarray(X)
        Y = np.asarray(Y).astype(X.dtype, copy=False)
        if self._XX is None:
            n_features = X.shape[1]
            self._XX = np.zeros((n_features, n_features), dtype=X.dtype, order="F")
            self._XY = np.zeros((n_features, Y.shape[1]), dtype=X.dtype)
            self._x_sum = np.zeros(n_features, dtype=np.float64)
            self._y_sum = np.zeros(Y.shape[1], dtype=np.float64)
            self._n_samples = 0
        # X.T is Fortran ordered, so BLAS needs no copy of X
        syrk = get_blas_funcs("syrk", (X,))
        self._XX = syrk(1.0, X.T, beta=1.0, c=self._XX, lower=1, overwrite_c=1)
        self._XY += X.T.dot(Y)
        if self.bias:
            self._x_sum += X.sum(axis=0, dtype=np.float64)
            self._y_sum += Y.sum(axis=0, dtype=np.float64)
            self._n_samples += X.shape[0]

    def _solve(self):
        # The factorisation overwrites its matrix: when the model can be
        # incremented, the sums are kept and a copy is factorised.
        XX, XY = self._XX, self._XY
        if self.incrementable:
            XX = XX.copy(order="F")
        else:
            self._XX = self._XY = None
        if self.alpha:
            XX[np.diag_indices_from(XX)] += self.alpha
        right_hand_side = XY
        if self.bias:
            right_hand_side = np.hstack((XY, self._x_sum[:, None].astype(XY.dtype)))
        try:
            factor = cho_factor(XX, lower=True, overwrite_a=True, check_finite=False)
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "The normal equations of the regression are not positive "
                "definite (for example, with fewer samples than features): "
                "use a ridge penalty (alpha > 0)."
            ) from error
        solution = cho_solve(factor, right_hand_side, check_finite=False)
        if not self.bias:
            self._W = solution
            return
        # With the column of ones, the system is
        #   [[XX + alpha I, x_sum], [x_sum^T, n + alpha]] [W; w0] = [XY; y_sum]
        # and its Schur complement for w0 is n + alpha - x_sum^T (XX + alpha I)^-1 x_sum
        weights, from_ones = solution[:, :-1], solution[:, -1]
        schur = self._n_samples + self.alpha - self._x_sum.dot(from_ones)
        w0 = (self._y_sum - self._x_sum.dot(weights)) / schur
        self._W = np.vstack((weights - np.outer(from_ones, w0), w0)).astype(
            weights.dtype, copy=False
        )

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
