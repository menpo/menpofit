import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
from scipy.sparse import spdiags, eye as speye
from scipy.sparse.linalg import spsolve
from menpofit.math.fft_utils import pad, crop


# TODO: Document me!
def mosse(X, y, l=0.01, boundary='constant', crop_filter=True):
    r"""
    Minimum Output Sum of Squared Errors (MOSSE) filter.

    Parameters
    ----------
    X : ``(n_images, n_channels, height, width)`` `ndarray`
        Training images.
    y : ``(1, height, width)`` `ndarray`
        Desired response.
    l: `float`, optional
        Regularization parameter.
    boundary: str {`constant`, `symmetric`}, optional
        Determines how the image is padded.
    crop_filter: `bool`, optional
        If ``True``, the shape of the MOSSE filter is the same as the shape
        of the desired response. If ``False``, the filter's shape is equal to:
        ``X[0].shape + y.shape - 1``

    Returns
    -------
    mosse: ``(1, height, width)`` `ndarray`
        Minimum Output Sum od Squared Errors (MOSSE) filter associated to
        the training images.

    References
    ----------
    .. [1] David S. Bolme, J. Ross Beveridge,  Bruce A. Draper and Yui Man Lui.
    "Visual Object Tracking using Adaptive Correlation Filters". CVPR, 2010.
    """
    # number of images, number of channels, height and width
    n, k, hx, wx = X.shape

    # height and width of desired responses
    _, hy, wy = y.shape
    y_shape = (hy, wy)

    # extended shape
    ext_h = hx + hy - 1
    ext_w = wx + wy - 1
    ext_shape = (ext_h, ext_w)

    # extend desired response
    ext_y = pad(y, ext_shape)
    # fft of extended desired response
    fft_ext_y = fft2(ext_y)

    # auto and cross spectral energy matrices
    sXX = 0
    sXY = 0
    # for each training image and desired response
    for x in X:
        # extend image
        ext_x = pad(x, ext_shape, boundary=boundary)
        # fft of extended image
        fft_ext_x = fft2(ext_x)

        # update auto and cross spectral energy matrices
        sXX += fft_ext_x.conj() * fft_ext_x
        sXY += fft_ext_x.conj() * fft_ext_y

    # compute desired correlation filter
    fft_ext_f = sXY / (sXX + l)
    # reshape extended filter to extended image shape
    fft_ext_f = fft_ext_f.reshape((k, ext_h, ext_w))

    # compute extended filter inverse fft
    f = np.real(ifftshift(ifft2(fft_ext_f), axes=(-2, -1)))

    if crop_filter:
        # crop extended filter to match desired response shape
        f = crop(f, y_shape)

    return f, sXY, sXX


def imosse(A, B, n_ab, X, y, l=0.01, boundary='constant',
           crop_filter=True, f=1.0):
    r"""
    Incremental Minimum Output Sum od Squared Errors (iMOSSE) filter

    Parameters
    ----------
    A :
    B :
    n_ab : `int`
        Total number of samples used to produce A and B.
    X : ``(n_images, n_channels, height, width)`` `ndarray`
        Training images.
    y : ``(1, height, width)`` `ndarray`
        Desired response.
    l : `float`, optional
        Regularization parameter.
    boundary : str {`constant`, `symmetric`}, optional
        Determines how the image is padded.
    crop_filter : `bool`, optional
    f : ``[0, 1]`` `float`, optional
        Forgetting factor that weights the relative contribution of new
        samples vs old samples. If 1.0, all samples are weighted equally.
        If <1.0, more emphasis is put on the new samples.

    Returns
    -------
    mccf : ``(1, height, width)`` `ndarray`
        Multi-Channel Correlation Filter (MCCF) filter associated to the
        training images.
    sXY :
    sXX :

    References
    ----------
    .. [1] David S. Bolme, J. Ross Beveridge,  Bruce A. Draper and Yui Man Lui.
    "Visual Object Tracking using Adaptive Correlation Filters". CVPR, 2010.
    """
    # number of images; number of channels, height and width
    n_x, k, hz, wz = X.shape

    # height and width of desired responses
    _, hy, wy = y.shape
    y_shape = (hy, wy)

    # multiply the number of samples used to produce the auto and cross
    # spectral energy matrices A and B by forgetting factor
    n_ab *= f
    # total number of samples
    n = n_ab + n_x
    # compute weighting factors
    nu_ab = n_ab / n
    nu_x = n_x / n

    # extended shape
    ext_h = hz + hy - 1
    ext_w = wz + wy - 1
    ext_shape = (ext_h, ext_w)

    # extend desired response
    ext_y = pad(y, ext_shape)
    # fft of extended desired response
    fft_ext_y = fft2(ext_y)

    # extend images
    ext_X = pad(X, ext_shape, boundary=boundary)

    # auto and cross spectral energy matrices
    sXX = 0
    sXY = 0
    # for each training image and desired response
    for ext_x in ext_X:
        # fft of extended image
        fft_ext_x = fft2(ext_x)

        # update auto and cross spectral energy matrices
        sXX += fft_ext_x.conj() * fft_ext_x
        sXY += fft_ext_x.conj() * fft_ext_y

    # combine old and new auto and cross spectral energy matrices
    sXY = nu_ab * A + nu_x * sXY
    sXX = nu_ab * B + nu_x * sXX
    # compute desired correlation filter
    fft_ext_f = sXY / (sXX + l)
    # reshape extended filter to extended image shape
    fft_ext_f = fft_ext_f.reshape((k, ext_h, ext_w))

    # compute filter inverse fft
    f = np.real(ifftshift(ifft2(fft_ext_f), axes=(-2, -1)))

    if crop_filter:
        # crop extended filter to match desired response shape
        f = crop(f, y_shape)

    return f, sXY, sXX


# TODO: Document me!
def mccf(X, y, l=0.01, boundary='constant', crop_filter=True):
    r"""
    Multi-Channel Correlation Filter (MCCF).

    Parameters
    ----------
    X : ``(n_images, n_channels, height, width)`` `ndarray`
        Training images.
    y : ``(1, height, width)`` `ndarray`
        Desired response.
    l : `float`, optional
        Regularization parameter.
    boundary : str {`constant`, `symmetric`}, optional
        Determines how the image is padded.
    crop_filter : `bool`, optional

    Returns
    -------
    mccf: ``(1, height, width)`` `ndarray`
        Multi-Channel Correlation Filter (MCCF) filter associated to the
        training images.
    sXY :
    sXX :

    References
    ----------
    .. [1] Hamed Kiani Galoogahi, Terence Sim,  Simon Lucey. "Multi-Channel
    Correlation Filters". ICCV, 2013.
    """
    # number of images; number of channels, height and width
    n, k, hx, wx = X.shape

    # height and width of desired responses
    _, hy, wy = y.shape
    y_shape = (hy, wy)

    # extended shape
    ext_h = hx + hy - 1
    ext_w = wx + wy - 1
    ext_shape = (ext_h, ext_w)
    # extended dimensionality
    ext_d = ext_h * ext_w

    # extend desired response
    ext_y = pad(y, ext_shape)
    # fft of extended desired response
    fft_ext_y = fft2(ext_y)

    # extend images
    ext_X = pad(X, ext_shape, boundary=boundary)

    # auto and cross spectral energy matrices
    sXX = 0
    sXY = 0
    # for each training image and desired response
    for ext_x in ext_X:
        # fft of extended image
        fft_ext_x = fft2(ext_x)

        # store extended image fft as sparse diagonal matrix
        diag_fft_x = spdiags(fft_ext_x.reshape((k, -1)),
                             -np.arange(0, k) * ext_d, ext_d * k, ext_d).T
        # vectorize extended desired response fft
        diag_fft_y = fft_ext_y.ravel()

        # update auto and cross spectral energy matrices
        sXX += diag_fft_x.conj().T.dot(diag_fft_x)
        sXY += diag_fft_x.conj().T.dot(diag_fft_y)

    # solve ext_d independent k x k linear systems (with regularization)
    # to obtain desired extended multi-channel correlation filter
    fft_ext_f = spsolve(sXX + l * speye(sXX.shape[-1]), sXY)
    # reshape extended filter to extended image shape
    fft_ext_f = fft_ext_f.reshape((k, ext_h, ext_w))

    # compute filter inverse fft
    f = np.real(ifftshift(ifft2(fft_ext_f), axes=(-2, -1)))

    if crop_filter:
        # crop extended filter to match desired response shape
        f = crop(f, y_shape)

    return f, sXY, sXX


# TODO: Document me!
def imccf(A, B, n_ab, X, y, l=0.01, boundary='constant', crop_filter=True,
          f=1.0):
    r"""
    Incremental Multi-Channel Correlation Filter (MCCF)

    Parameters
    ----------
    A :
    B :
    n_ab : `int`
        Total number of samples used to produce A and B.
    X : ``(n_images, n_channels, height, width)`` `ndarray`
        Training images.
    y : ``(1, height, width)`` `ndarray`
        Desired response.
    l : `float`, optional
        Regularization parameter.
    boundary : str {`constant`, `symmetric`}, optional
        Determines how the image is padded.
    crop_filter : `bool`, optional
    f : ``[0, 1]`` `float`, optional
        Forgetting factor that weights the relative contribution of new
        samples vs old samples. If 1.0, all samples are weighted equally.
        If <1.0, more emphasis is put on the new samples.

    Returns
    -------
    mccf : ``(1, height, width)`` `ndarray`
        Multi-Channel Correlation Filter (MCCF) filter associated to the
        training images.
    sXY :
    sXX :

    References
    ----------
    .. [1] David S. Bolme, J. Ross Beveridge,  Bruce A. Draper and Yui Man Lui.
    "Visual Object Tracking using Adaptive Correlation Filters". CVPR, 2010.
    .. [2] Hamed Kiani Galoogahi, Terence Sim,  Simon Lucey. "Multi-Channel
    Correlation Filters". ICCV, 2013.
    """
    # number of images; number of channels, height and width
    n_x, k, hz, wz = X.shape

    # height and width of desired responses
    _, hy, wy = y.shape
    y_shape = (hy, wy)

    # multiply the number of samples used to produce the auto and cross
    # spectral energy matrices A and B by forgetting factor
    n_ab *= f
    # total number of samples
    n = n_ab + n_x
    # compute weighting factors
    nu_ab = n_ab / n
    nu_x = n_x / n

    # extended shape
    ext_h = hz + hy - 1
    ext_w = wz + wy - 1
    ext_shape = (ext_h, ext_w)
    # extended dimensionality
    ext_d = ext_h * ext_w

    # extend desired response
    ext_y = pad(y, ext_shape)
    # fft of extended desired response
    fft_ext_y = fft2(ext_y)

    # extend images
    ext_X = pad(X, ext_shape, boundary=boundary)

    # auto and cross spectral energy matrices
    sXX = 0
    sXY = 0
    # for each training image and desired response
    for ext_x in ext_X:
        # fft of extended image
        fft_ext_x = fft2(ext_x)

        # store extended image fft as sparse diagonal matrix
        diag_fft_x = spdiags(fft_ext_x.reshape((k, -1)),
                             -np.arange(0, k) * ext_d, ext_d * k, ext_d).T
        # vectorize extended desired response fft
        diag_fft_y = fft_ext_y.ravel()

        # update auto and cross spectral energy matrices
        sXX += diag_fft_x.conj().T.dot(diag_fft_x)
        sXY += diag_fft_x.conj().T.dot(diag_fft_y)

    # combine old and new auto and cross spectral energy matrices
    sXY = nu_ab * A + nu_x * sXY
    sXX = nu_ab * B + nu_x * sXX
    # solve ext_d independent k x k linear systems (with regularization)
    # to obtain desired extended multi-channel correlation filter
    fft_ext_f = spsolve(sXX + l * speye(sXX.shape[-1]), sXY)
    # reshape extended filter to extended image shape
    fft_ext_f = fft_ext_f.reshape((k, ext_h, ext_w))

    # compute filter inverse fft
    f = np.real(ifftshift(ifft2(fft_ext_f), axes=(-2, -1)))
    if crop_filter:
        # crop extended filter to match desired response shape
        f = crop(f, y_shape)

    return f, sXY, sXX
