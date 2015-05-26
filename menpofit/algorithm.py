from __future__ import division
import numpy as np
from menpo.image import Image
from menpo.feature import no_op
from menpo.feature import gradient as fast_gradient


# TODO: implement more clever sampling?
class LKInterface(object):

    def __init__(self, lk_algorithm, sampling=None):
        self.algorithm = lk_algorithm

        n_true_pixels = self.template.n_true_pixels()
        n_channels = self.template.n_channels
        n_parameters = self.transform.n_parameters
        sampling_mask = np.zeros(n_true_pixels, dtype=np.bool)

        if sampling is None:
            sampling = 1
        sampling_pattern = xrange(0, n_true_pixels, sampling)
        sampling_mask[sampling_pattern] = 1

        self.i_mask = np.nonzero(np.tile(
            sampling_mask[None, ...], (n_channels, 1)).flatten())[0]
        self.dW_dp_mask = np.nonzero(np.tile(
            sampling_mask[None, ..., None], (2, 1, n_parameters)))
        self.nabla_mask = np.nonzero(np.tile(
            sampling_mask[None, None, ...], (2, n_channels, 1)))
        self.nabla2_mask = np.nonzero(np.tile(
            sampling_mask[None, None, None, ...], (2, 2, n_channels, 1)))

    @property
    def template(self):
        return self.algorithm.template

    @property
    def transform(self):
        return self.algorithm.transform

    @property
    def n(self):
        return self.transform.n_parameters

    @property
    def true_indices(self):
        return self.template.mask.true_indices()

    def warp_jacobian(self):
        dW_dp = np.rollaxis(self.transform.d_dp(self.true_indices), -1)
        return dW_dp[self.dW_dp_mask].reshape((dW_dp.shape[0], -1,
                                               dW_dp.shape[2]))

    def warp(self, image):
        return image.warp_to_mask(self.template.mask,
                                  self.transform)

    def gradient(self, img):
        nabla = fast_gradient(img)
        nabla.set_boundary_pixels()
        return nabla.as_vector().reshape((2, img.n_channels, -1))

    def steepest_descent_images(self, nabla, dW_dp):
        # reshape gradient
        # nabla: n_dims x n_channels x n_pixels
        nabla = nabla[self.nabla_mask].reshape(nabla.shape[:2] + (-1,))
        # compute steepest descent images
        # nabla: n_dims x n_channels x n_pixels
        # warp_jacobian: n_dims x            x n_pixels x n_params
        # sdi:            n_channels x n_pixels x n_params
        sdi = 0
        a = nabla[..., None] * dW_dp[:, None, ...]
        for d in a:
            sdi += d
        # reshape steepest descent images
        # sdi: (n_channels x n_pixels) x n_params
        return sdi.reshape((-1, sdi.shape[2]))


class LKPartsInterface(LKInterface):

    def __init__(self, lk_algorithm, patch_shape=(17, 17),
                 normalize_parts=no_op, sampling=None):
        self.algorithm = lk_algorithm
        self.patch_shape = patch_shape
        self.normalize_parts = normalize_parts

        if sampling is None:
            sampling = np.ones(self.patch_shape, dtype=np.bool)

        image_shape = self.algorithm.template.pixels.shape
        image_mask = np.tile(sampling[None, None, None, ...],
                             image_shape[:3] + (1, 1))
        self.i_mask = np.nonzero(image_mask.flatten())[0]
        self.nabla_mask = np.nonzero(np.tile(
            image_mask[None, ...], (2, 1, 1, 1, 1, 1)))
        self.nabla2_mask = np.nonzero(np.tile(
            image_mask[None, None, ...], (2, 2, 1, 1, 1, 1, 1)))

    def warp_jacobian(self):
        return np.rollaxis(self.transform.d_dp(None), -1)

    # TODO: add parts normalization
    def warp(self, image):
        parts = image.extract_patches(self.transform.target,
                                      patch_size=self.patch_shape,
                                      as_single_array=True)
        parts = self.normalize_parts(parts)
        return Image(parts)

    def gradient(self, image):
        pixels = image.pixels
        g = fast_gradient(pixels.reshape((-1,) + self.patch_shape))
        # remove 1st dimension gradient which corresponds to the gradient
        # between parts
        return g.reshape((2,) + pixels.shape)

    def steepest_descent_images(self, nabla, dw_dp):
        # reshape nabla
        # nabla: dims x parts x off x ch x (h x w)
        nabla = nabla[self.nabla_mask].reshape(
            nabla.shape[:-2] + (-1,))
        # compute steepest descent images
        # nabla: dims x parts x off x ch x (h x w)
        # ds_dp:    dims x parts x                             x params
        # sdi:             parts x off x ch x (h x w) x params
        sdi = 0
        a = nabla[..., None] * dw_dp[..., None, None, None, :]
        for d in a:
            sdi += d

        # reshape steepest descent images
        # sdi: (parts x offsets x ch x w x h) x params
        return sdi.reshape((-1, sdi.shape[-1]))
