import numpy as np
from menpofit.math.correlationfilter import mccf, imccf


# TODO: document me!
class IncrementalCorrelationFilterThinWrapper(object):
    r"""
    """
    def __init__(self, cf_callable=mccf, icf_callable=imccf):
        self.cf_callable = cf_callable
        self.icf_callable = icf_callable

    def increment(self, A, B, n_x, Z, t):
        r"""
        """
        # Turn list of X into ndarray
        if isinstance(Z, list):
            Z = np.asarray(Z)
        return self.icf_callable(A, B, n_x, Z, t)

    def train(self, X, t):
        r"""
        """
        # Turn list of X into ndarray
        if isinstance(X, list):
            X = np.asarray(X)
        # Return linear svm filter and bias
        return self.cf_callable(X, t)
