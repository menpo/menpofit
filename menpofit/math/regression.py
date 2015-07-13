import numpy as np


# TODO: document me!
class IRLRegression(object):
    r"""
    Incremental Regularized Linear Regression
    """
    def __init__(self, l=0, bias=True):
        self.l = l
        self.bias = bias

    def train(self, X, Y):
        if self.bias:
            # add bias
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        # regularized linear regression
        XX = X.T.dot(X)
        np.fill_diagonal(XX, self.l + np.diag(XX))
        self.V = np.linalg.inv(XX)
        self.W = self.V.dot(X.T.dot(Y))

    def increment(self, X, Y):
        if self.bias:
            # add bias
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        # incremental regularized linear regression
        U = X.dot(self.V).dot(X.T)
        np.fill_diagonal(U, 1 + np.diag(U))
        U = np.linalg.inv(U)
        Q = self.V.dot(X.T).dot(U).dot(X)
        self.V = self.V - Q.dot(self.V)
        self.W = self.W - Q.dot(self.W) + self.V.dot(X.T.dot(Y))

    def predict(self, x):
        if self.bias:
            if len(x.shape) == 1:
                x = np.hstack((x, np.ones(1)))
            else:
                x = np.hstack((x, np.ones((x.shape[0], 1))))
        return np.dot(x, self.W)


# TODO: document me!
class IIRLRegression(IRLRegression):
    r"""
    Indirect Incremental Regularized Linear Regression
    """
    def __init__(self, l=0, bias=True, d=0):
        super(IIRLRegression, self).__init__(l=l, bias=bias)
        self.d = d

    def train(self, X, Y):
        # regularized linear regression exchanging the roles of X and Y
        super(IIRLRegression, self).train(Y, X)
        J = self.W
        # solve the original problem by computing the pseudo-inverse of the
        # previous solution
        H = J.T.dot(J)
        np.fill_diagonal(H, self.d + np.diag(H))
        self.W = np.linalg.solve(H, J.T)

    def increment(self, X, Y):
        # incremental least squares exchanging the roles of X and Y
        super(IIRLRegression, self).increment(Y, X)
        J = self.W
        # solve the original problem by computing the pseudo-inverse of the
        # previous solution
        H = J.T.dot(J)
        np.fill_diagonal(H, self.d + np.diag(H))
        self.W = np.linalg.solve(H, J.T)
