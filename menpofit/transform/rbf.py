import numpy as np
from scipy.spatial.distance import cdist
from menpo.transform.rbf import R2LogR2RBF, R2LogRRBF
from menpofit.differentiable import DL


class DifferentiableR2LogRRBF(R2LogRRBF, DL):

    def d_dl(self, points):
        """
        The derivative of the basis function wrt the coordinate system
        evaluated at `points`.

        :math:`(x - c)^T (1 + 2 \log{r_{x, l}})`.

        .. note::

            :math:`r_{x, l} = \lVert x - c \rVert`

        Parameters
        ----------
        points : (n_points, n_dims) ndarray
            Set of points to apply the basis to.

        Returns
        -------
        d_dl : (n_points, n_centres, n_dims) ndarray
            The jacobian tensor representing the first order partial derivative
            of each points wrt the centres
        """
        euclidean_distance = cdist(points, self.c)
        component_distances = points[..., None, ...] - self.c
        # Avoid log(0) and set to 1 so that log(1) = 0
        euclidean_distance[euclidean_distance == 0] = 1
        d_dl = (component_distances *
                (1 + 2 * np.log(euclidean_distance[..., None])))
        return d_dl


class DifferentiableR2LogR2RBF(R2LogR2RBF, DL):

    def d_dl(self, points):
        """
        Apply the derivative of the basis function wrt the centres and the
        points given by `points`.

        .. note::

            Let `points` be `x`, then

            ..math::

                2 (x - c)^T (\log{r^2_{x, l}} + 1) ===
                2 (x - c)^T (2 \log{r_{x, l}} + 1)

            where:

            :math:`r_{x, l} = \lVert x - c \rVert`


        Parameters
        ----------
        x : (n_points, n_dims) ndarray
            Set of points to apply the basis to.

        Returns
        -------
        d_dl : (n_points, n_centres, n_dims) ndarray
            The jacobian tensor representing the first order derivative
            of the radius from each centre wrt the centre's position, evaluated
            at each point.
        """
        euclidean_distance = cdist(points, self.c)
        component_distances = points[..., None, ...] - self.c
        # Avoid log(0) and set to 1 so that log(1) = 0
        euclidean_distance[euclidean_distance == 0] = 1
        d_dl = (2 * component_distances *
                (2 * np.log(euclidean_distance[..., None]) + 1))
        return d_dl
