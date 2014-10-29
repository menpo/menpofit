import abc


class DP(object):
    r"""Object that is able to take it's own derivative wrt parametrization.

    The parametrization of objects is typically defined by the
    :map:`Vectorizable` interface. As a result, :map:`DP` is a mix-in that
    should be inherited along with :map:`Vectorizable`.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def d_dp(self, points):
        r"""The derivative of this spatial object wrt parametrization changes
        evaluated at points.

        Parameters
        ----------
        points : ``(n_points, n_dims)`` `ndarray`
            The spatial points at which the derivative should be evaluated.

        Returns
        -------
        d_dp : ``(n_points, n_parameters, n_dims)`` `ndarray`
            The jacobian wrt parametrization.

            ``d_dp[i, j, k]`` is the scalar differential change that the
            k'th dimension of the i'th point experiences due to a first order
            change in the j'th scalar in the parametrization vector.
        """


class DX(object):
    r"""Object that is able to take it's own derivative wrt spatial changes.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def d_dx(self, points):
        r"""The first order derivative of this spatial object wrt spatial
        changes evaluated at points.

        Parameters
        ----------
        points : ``(n_points, n_dims)`` `ndarray`
            The spatial points at which the derivative should be evaluated.

        Returns
        -------
        d_dx : ``(n_points, n_dims, n_dims)`` `ndarray`
            The jacobian wrt spatial changes.

            ``d_dx[i, j, k]`` is the scalar differential change that the
            j'th dimension of the i'th point experiences due to a first order
            change in the k'th dimension.

            It may be the case that the jacobian is constant across space -
            in this case axis zero may have length ``1`` to allow for
            broadcasting.
        """


class DL(object):
    r"""Object that is able to take it's own derivative wrt landmark changes.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def d_dl(self, points):
        r"""The derivative of this spatial object wrt spatial changes in anchor
        landmark points or centres, evaluated at points.

        Parameters
        ----------
        points : ``(n_points, n_dims)`` `ndarray`
            The spatial points at which the derivative should be evaluated.

        Returns
        -------
        d_dl : ``(n_points, n_centres, n_dims)`` `ndarray`
            The jacobian wrt landmark changes.

            ``d_dl[i, k, m]`` is the scalar differential change that the
            any dimension of the i'th point experiences due to a first order
            change in the m'th dimension of the k'th landmark point.

            Note that at present this assumes that the change in every
            dimension is equal.
        """
