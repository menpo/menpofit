import numpy as np


# TODO: document me!
class incremental_least_squares(object):
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
class incremental_indirect_least_squares(object):
    r"""
    """
    def __init__(self, l=0, d=0):
        self._ils = incremental_least_squares(l)
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

    def increment(self, X, Y):
        # incremental least squares exchanging the roles of X and Y
        self._ils.increment(Y, X)
        J = self._ils.W
        # solve the original problem by computing the pseudo-inverse of the
        # previous solution
        H = J.T.dot(J)
        np.fill_diagonal(H, self.d + np.diag(H))
        self.W = np.linalg.solve(H, J.T)

    def predict(self, x):
        return np.dot(x, self.W)