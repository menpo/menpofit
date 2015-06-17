import abc
import numpy as np
from numpy.fft import fftn, ifftn, fft2
import scipy.linalg
from menpo.feature import gradient


class Residual(object):
    """
    An abstract base class for calculating the residual between two images
    within the Lucas-Kanade algorithm. The classes were designed
    specifically to work within the Lucas-Kanade framework and so no
    guarantee is made that calling methods on these subclasses will generate
    correct results.
    """
    @classmethod
    def gradient(cls, image, forward=None):
        r"""
        Calculates the gradients of the given method.

        If `forward` is provided, then the gradients are warped
        (as required in the forward additive algorithm)

        Parameters
        ----------
        image : :class:`menpo.image.base.Image`
            The image to calculate the gradients for
        forward : (:map:`Image`, :map:`AlignableTransform>`), optional
            A tuple containing the extra weights required for the function
            `warp` (which should be passed as a function handle).

            Default: `None`
        """
        if forward:
            # Calculate the gradient over the image
            # grad:  (dims x ch) x H x W
            grad = gradient(image)
            # Warp gradient for forward additive using the given transform
            # grad:  (dims x ch) x h x w
            template, transform = forward
            grad = grad.warp_to_mask(template.mask, transform,
                                     warp_landmarks=False)
        else:
            # Calculate the gradient over the image and set one pixels along
            # the boundary of the image mask to zero (no reliable gradient
            # can be computed there!)
            # grad:  (dims x ch) x h x w
            grad = gradient(image)
            grad.set_boundary_pixels()
        return grad

    @abc.abstractmethod
    def steepest_descent_images(self, image, dW_dp, **kwargs):
        r"""
        Calculates the standard steepest descent images.

        Within the forward additive framework this is defined as

        .. math::
             \nabla I \frac{\partial W}{\partial p}

        The input image is vectorised (`N`-pixels) so that masked images can
        be handled.

        Parameters
        ----------
        image : :class:`menpo.image.base.Image`
            The image to calculate the steepest descent images from, could be
            either the template or input image depending on which framework is
            used.
        dW_dp : ndarray
            The Jacobian of the warp.

        Returns
        -------
        VT_dW_dp : (N, n_params) ndarray
            The steepest descent images
        """
        pass

    @abc.abstractmethod
    def hessian(self, sdi):
        r"""
        Calculates the Gauss-Newton approximation to the Hessian.

        This is abstracted because some residuals expect the Hessian to be
        pre-processed. The Gauss-Newton approximation to the Hessian is
        defined as:

        .. math::
            \mathbf{J J^T}

        Parameters
        ----------
        J : (N, n_params) ndarray
            The steepest descent images.

        Returns
        -------
        H : (n_params, n_params) ndarray
            The approximation to the Hessian
        """
        pass

    @abc.abstractmethod
    def steepest_descent_update(self, sdi, image, template):
        r"""
        Calculates the steepest descent parameter updates.

        These are defined, for the forward additive algorithm, as:

        .. math::
            \sum_x [ \nabla I \frac{\partial W}{\partial p} ]^T [ T(x) - I(W(x;p)) ]

        Parameters
        ----------
        J : (N, n_params) ndarray
            The steepest descent images.
        image : :class:`menpo.image.base.Image`
            Either the warped image or the template
            (depending on the framework)
        template : :class:`menpo.image.base.Image`
            Either the warped image or the template
            (depending on the framework)

        Returns
        -------
        sd_delta_p : (n_params,) ndarray
            The steepest descent parameter updates.
        """
        pass


class SSD(Residual):

    def __init__(self, kernel=None):
        self.kernel = kernel

    def steepest_descent_images(self, image, dW_dp, forward=None):
        # compute gradient
        # grad:  dims x ch x h x w
        nabla = self.gradient(image, forward=forward)
        nabla = nabla.as_vector().reshape((image.n_dims, image.n_channels) +
                                          image.shape)

        # compute steepest descent images
        # gradient: dims x ch x h x w
        # dw_dp:    dims x    x h x w x params
        # sdi:             ch x h x w x params
        sdi = 0
        a = nabla[..., None] * dW_dp[:, None, ...]
        for d in a:
            sdi += d

        if self.kernel is not None:
            # if required, filter steepest descent images
            # fft_sdi:  ch x h x w x params
            filtered_sdi = ifftn(self.kernel[..., None] *
                                 fftn(sdi, axes=(-3, -2)),
                                 axes=(-3, -2))
            # reshape steepest descent images
            # sdi:           (ch x h x w) x params
            # filtered_sdi:  (ch x h x w) x params
            sdi = sdi.reshape((-1, sdi.shape[-1]))
            filtered_sdi = filtered_sdi.reshape(sdi.shape)
        else:
            # reshape steepest descent images
            # sdi:           (ch x h x w) x params
            # filtered_sdi:  (ch x h x w) x params
            sdi = sdi.reshape((-1, sdi.shape[-1]))
            filtered_sdi = sdi

        return filtered_sdi, sdi

    def hessian(self, sdi, sdi2=None):
        # compute hessian
        # sdi.T:   params x (ch x h x w)
        # sdi:              (ch x h x w) x params
        # hessian: params x               x params
        if sdi2 is None:
            H = sdi.T.dot(sdi)
        else:
            H = sdi.T.dot(sdi2)
        return H

    def steepest_descent_update(self, sdi, image, template):
        error_img = image.as_vector() - template.as_vector()
        return sdi.T.dot(error_img)


class FourierSSD(Residual):

    def __init__(self, kernel=None):
        self.kernel = kernel

    def steepest_descent_images(self, image, dW_dp, forward=None):
        # compute gradient
        # grad:  dims x ch x h x w
        nabla = self.gradient(image, forward=forward)
        nabla = nabla.as_vector().reshape((image.n_dims, image.n_channels) +
                                          image.shape)

        # compute steepest descent images
        # gradient: dims x ch x h x w
        # dw_dp:    dims x    x h x w x params
        # sdi:             ch x h x w x params
        sdi = 0
        a = nabla[..., None] * dW_dp[:, None, ...]
        for d in a:
            sdi += d

        # compute steepest descent images fft
        # fft_sdi:  ch x h x w x params
        fft_sdi = fftn(sdi, axes=(-3, -2))

        if self.kernel is not None:
            # if required, filter steepest descent images
            filtered_fft_sdi = self.kernel[..., None] * fft_sdi
            # reshape steepest descent images
            # fft_sdi:           (ch x h x w) x params
            # filtered_fft_sdi:  (ch x h x w) x params
            fft_sdi = fft_sdi.reshape((-1, fft_sdi.shape[-1]))
            filtered_fft_sdi = filtered_fft_sdi.reshape(fft_sdi.shape)
        else:
            # reshape steepest descent images
            # fft_sdi:           (ch x h x w) x params
            # filtered_fft_sdi:  (ch x h x w) x params
            fft_sdi = fft_sdi.reshape((-1, fft_sdi.shape[-1]))
            filtered_fft_sdi = fft_sdi

        return filtered_fft_sdi, fft_sdi

    def hessian(self, sdi, sdi2=None):
        if sdi2 is None:
            H = sdi.conjugate().T.dot(sdi)
        else:
            H = sdi.conjugate().T.dot(sdi2)
        return H

    def steepest_descent_update(self, sdi, image, template):
        # compute error image
        # error_img:  ch x h x w
        error_img = image.pixels - template.pixels

        # compute error image fft
        # fft_error_img:  ch x (h x w)
        fft_error_img = fft2(error_img)

        # compute steepest descent update
        # fft_sdi:        params x (ch x h x w)
        # fft_error_img:           (ch x h x w)
        # fft_sdu:        params
        return sdi.conjugate().T.dot(fft_error_img.ravel())


class ECC(Residual):

    def _normalise_images(self, image):
        # TODO: do we need to copy the image?
        # TODO: is this supposed to be per channel normalization?
        norm_image = image.copy()
        norm_image.normalize_norm_inplace()
        return norm_image

    def steepest_descent_images(self, image, dW_dp, forward=None):
        # normalize image
        norm_image = self._normalise_images(image)

        # compute gradient
        # gradient:  dims x ch x pixels
        grad = self.gradient(norm_image, forward=forward)
        grad = grad.as_vector().reshape((image.n_dims, image.n_channels, -1))

        # compute steepest descent images
        # gradient: dims x ch x pixels
        # dw_dp:    dims x    x pixels x params
        # sdi:             ch x pixels x params
        sdi = 0
        a = grad[..., None] * dW_dp[:, None, ...]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (ch x pixels) x params
        return sdi.reshape((-1, sdi.shape[-1]))

    def hessian(self, sdi):
        # compute hessian
        # sdi.T:   params x (ch x pixels)
        # sdi:              (ch x pixels) x params
        # hessian: params x               x params
        H = sdi.T.dot(sdi)
        self._H_inv = scipy.linalg.inv(H)
        return H

    def steepest_descent_update(self, sdi, image, template):
        normalised_IWxp = self._normalise_images(image).as_vector()
        normalised_template = self._normalise_images(template).as_vector()

        Gt = sdi.T.dot(normalised_template)
        Gw = sdi.T.dot(normalised_IWxp)

        # Calculate the numerator
        IWxp_norm = scipy.linalg.norm(normalised_IWxp)
        num1 = IWxp_norm ** 2
        num2 = np.dot(Gw.T, np.dot(self._H_inv, Gw))
        num = num1 - num2

        # Calculate the denominator
        den1 = np.dot(normalised_template, normalised_IWxp)
        den2 = np.dot(Gt.T, np.dot(self._H_inv, Gw))
        den = den1 - den2

        # Calculate lambda to choose the step size
        # Avoid division by zero
        if den > 0:
            l = num / den
        else:
            den3 = np.dot(Gt.T, np.dot(self._H_inv, Gt))
            l1 = np.sqrt(num2 / den3)
            l2 = - den / den3
            l = np.maximum(l1, l2)

        self._error_img = l * normalised_IWxp - normalised_template

        return sdi.T.dot(self._error_img)


class GradientImages(Residual):

    def _regularise_gradients(self, grad):
        pixels = grad.pixels
        ab = np.sqrt(np.sum(pixels**2, axis=0))
        m_ab = np.median(ab)
        ab = ab + m_ab
        grad.pixels = pixels / ab
        return grad

    def steepest_descent_images(self, image, dW_dp, forward=None):
        n_dims = image.n_dims
        n_channels = image.n_channels

        # compute gradient
        first_grad = self.gradient(image, forward=forward)
        self._template_grad = self._regularise_gradients(first_grad)

        # compute gradient
        # second_grad:  dims x dims x ch x pixels
        second_grad = self.gradient(self._template_grad)
        second_grad = second_grad.masked_pixels().flatten().reshape(
            (n_dims, n_dims,  n_channels, -1))

        # Fix crossed derivatives: dydx = dxdy
        second_grad[1, 0, ...] = second_grad[0, 1, ...]

        # compute steepest descent images
        # gradient: dims x dims x ch x (h x w)
        # dw_dp:    dims x           x (h x w) x params
        # sdi:             dims x ch x (h x w) x params
        sdi = 0
        a = second_grad[..., None] * dW_dp[:, None, None, ...]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (dims x ch x h x w) x params
        return sdi.reshape((-1, sdi.shape[-1]))

    def hessian(self, sdi):
        # compute hessian
        # sdi.T:   params x (dims x ch x pixels)
        # sdi:              (dims x ch x pixels) x params
        # hessian: params x                     x params
        return sdi.T.dot(sdi)

    def steepest_descent_update(self, sdi, image, template):
        # compute image regularized gradient
        IWxp_grad = self.gradient(image)
        IWxp_grad = self._regularise_gradients(IWxp_grad)

        # compute vectorized error_image
        # error_img: (dims x ch x pixels)
        self._error_img = (IWxp_grad.as_vector() -
                           self._template_grad.as_vector())

        # compute steepest descent update
        # sdi.T:      params x (dims x ch x pixels)
        # error_img:           (dims x ch x pixels)
        # sdu:        params
        return sdi.T.dot(self._error_img)


class GradientCorrelation(Residual):

    def steepest_descent_images(self, image, dW_dp, forward=None):
        n_dims = image.n_dims
        n_channels = image.n_channels

        # compute gradient
        # grad:  dims x ch x pixels
        grad = self.gradient(image, forward=forward)
        grad2 = grad.as_vector().reshape((n_dims, n_channels) + image.shape)

        # compute IGOs (remember axis 0 is y, axis 1 is x)
        # grad:    dims x ch x pixels
        # phi:            ch x pixels
        # cos_phi:        ch x pixels
        # sin_phi:        ch x pixels
        phi = np.angle(grad2[1, ...] + 1j * grad2[0, ...])
        self._cos_phi = np.cos(phi)
        self._sin_phi = np.sin(phi)

        # concatenate sin and cos terms so that we can take the second
        # derivatives correctly. sin(phi) = y and cos(phi) = x which is the
        # correct ordering when multiplying against the warp Jacobian
        # cos_phi:         ch  x pixels
        # sin_phi:         ch  x pixels
        # grad:    (dims x ch) x pixels
        grad.from_vector_inplace(
            np.concatenate((self._sin_phi[None, ...],
                            self._cos_phi[None, ...]), axis=0).ravel())

        # compute IGOs gradient
        # second_grad:  dims x dims x ch x pixels
        second_grad = self.gradient(grad)
        second_grad = second_grad.masked_pixels().flatten().reshape(
            (n_dims, n_dims,  n_channels) + image.shape)

        # Fix crossed derivatives: dydx = dxdy
        second_grad[1, 0, ...] = second_grad[0, 1, ...]

        # complete full IGOs gradient computation
        # second_grad:  dims x dims x ch x pixels
        second_grad[1, ...] = (-self._sin_phi[None, ...] * second_grad[1, ...])
        second_grad[0, ...] = (self._cos_phi[None, ...] * second_grad[0, ...])

        # compute steepest descent images
        # gradient: dims x dims x ch x pixels
        # dw_dp:    dims x           x pixels x params
        # sdi:                    ch x pixels x params
        sdi = 0
        aux = second_grad[..., None] * dW_dp[None, :, None, ...]
        for a in aux.reshape(((-1,) + aux.shape[2:])):
                sdi += a

        # compute constant N
        # N:  1
        self._N = grad.n_parameters / 2

        # reshape steepest descent images
        # sdi: (ch x pixels) x params
        sdi = sdi.reshape((-1, sdi.shape[-1]))

        return sdi, sdi

    def hessian(self, sdi, sdi2=None):
        # compute hessian
        # sdi.T:   params x (ch x h x w)
        # sdi:              (ch x h x w) x params
        # hessian: params x               x params
        if sdi2 is None:
            H = sdi.T.dot(sdi)
        else:
            H = sdi.T.dot(sdi2)
        return H

    def steepest_descent_update(self, sdi, image, template):
        n_dims = image.n_dims
        n_channels = image.n_channels

        # compute image gradient
        IWxp_grad = self.gradient(image)
        IWxp_grad = IWxp_grad.as_vector().reshape(
            (n_dims, n_channels) + image.shape)

        # compute IGOs (remember axis 0 is y, axis 1 is x)
        # IWxp_grad:     dims x ch x pixels
        # phi:                  ch x pixels
        # IWxp_cos_phi:         ch x pixels
        # IWxp_sin_phi:         ch x pixels
        phi = np.angle(IWxp_grad[1, ...] + 1j * IWxp_grad[0, ...])
        IWxp_cos_phi = np.cos(phi)
        IWxp_sin_phi = np.sin(phi)

        # compute error image
        # error_img:  (ch x h x w)
        self._error_img = (self._cos_phi * IWxp_sin_phi -
                           self._sin_phi * IWxp_cos_phi).ravel()

        # compute steepest descent update
        # sdi:       (ch x pixels) x params
        # error_img: (ch x pixels)
        # sdu:                      params
        sdu = sdi.T.dot(self._error_img)

        # compute step size
        qp = np.sum(self._cos_phi * IWxp_cos_phi +
                    self._sin_phi * IWxp_sin_phi)
        l = self._N / qp
        return l * sdu
