import numpy as np
from menpo.transform import PiecewiseAffine
from menpofit.differentiable import DL, DX


class DifferentiablePiecewiseAffine(PiecewiseAffine, DL, DX):

    def d_dl(self, points):
        """
        Returns the jacobian of the warp at each point given in relation to the
        source points for a PiecewiseAffine transform

        Parameters
        ----------
        points : (n_points, 2) ndarray
            The points to calculate the Jacobian for.

        Returns
        -------
        d_dl : (n_points, n_centres, 2) ndarray
            The Jacobian for each of the given points over each point in
            the source points.

        """
        tri_index, alpha_i, beta_i = self.index_alpha_beta(points)
        # for the jacobian we only need
        # gamma = 1 - alpha - beta
        # for each vertex (i, j, & k)
        # gamma is the 'far edge' weighting wrt the vertex in question.
        # given gamma implicitly for the first vertex in our trilist,
        # we can permute around to get the others. (e.g. rotate CW around
        # the triangle to get the j'th vertex-as-prime variant,
        # and once again to the kth).
        #
        # alpha_j = 1 - alpha_i - beta_i
        # gamma_j = alpha_i
        # gamma_k = beta_i
        #
        # TODO this ordering is empirically correct but I don't know why..
        #
        # we stack all the gamma's together
        # so gamma_ijk.shape = (n_sample_points, 3)
        gamma_ijk = np.hstack(((1 - alpha_i - beta_i)[:, None],
                               alpha_i[:, None],
                               beta_i[:, None]))
        # the jacobian wrt source is of shape
        # (n_sample_points, n_source_points, 2)
        jac = np.zeros((points.shape[0], self.n_points, 2))
        # per sample point, find the source points for the ijk vertices of
        # the containing triangle - only these points will get a non 0
        # jacobian value
        ijk_per_point = self.trilist[tri_index]
        # to index into the jacobian, we just need a linear iterator for the
        # first axis - literally [0, 1, ... , n_sample_points]. The
        # reshape is needed to make it broadcastable with the other indexing
        # term, ijk_per_point.
        linear_iterator = np.arange(points.shape[0]).reshape((-1, 1))
        # in one line, we are done.
        jac[linear_iterator, ijk_per_point] = gamma_ijk[..., None]
        return jac

    def d_dx(self, points):
        """
        Calculates the first order spatial derivative of PWA at points.

        The nature of this derivative is complicated by the piecewise nature
        of this transform. For points close to the source points of the
        transform the derivative is ill-defined. In these cases, an identity
        jacobian is returned.

        In all other cases the jacobian is equal to the containing triangle's
        d_dx.

        WARNING - presently the above behavior is only valid at the source
        points.

        Returns
        -------
        d_dx: (n_points, n_dims, n_dims) ndarray
            The first order spatial derivative of this transform

        Raises
        ------
        TriangleContainmentError:
            If any point is outside any triangle of this PWA.


        """
        # TODO check for position and return true d_dx (see docstring)
        # for the time being we assume the points are on the source landmarks
        return np.eye(2, 2)[None, ...]
