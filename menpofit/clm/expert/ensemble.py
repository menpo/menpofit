from __future__ import division
from functools import partial
import numpy as np
from scipy.stats import multivariate_normal
from menpo.shape import PointCloud
from menpo.image import Image
from menpofit.base import build_grid
from menpofit.feature import normalize_norm, probability_map
from menpofit.math.fft_utils import (
    fft2, ifft2, fftshift, pad, crop, fft_convolve2d_sum)
from menpofit.visualize import print_progress
from .base import IncrementalCorrelationFilterThinWrapper


# TODO: Document me!
class ExpertEnsemble(object):
    r"""
    """


# TODO: Document me!
# TODO: Should convolutional experts of ensembles support patch features?
class ConvolutionBasedExpertEnsemble(ExpertEnsemble):
    r"""
    """
    @property
    def n_experts(self):
        r"""
        """
        return self.fft_padded_filters.shape[0]

    @property
    def n_sample_offsets(self):
        r"""
        """
        if self.sample_offsets:
            return self.sample_offsets.n_points
        else:
            return 1

    @property
    def padded_size(self):
        r"""
        """
        pad_size = np.floor(1.5 * np.asarray(self.patch_shape) - 1).astype(int)
        return tuple(pad_size)

    @property
    def search_size(self):
        r"""
        """
        return self.patch_shape

    def increment(self, images, shapes, prefix='', verbose=False):
        r"""
        """
        self._train(images, shapes, prefix=prefix, verbose=verbose,
                    increment=True)

    @property
    def spatial_filter_images(self):
        r"""
        """
        filter_images = []
        for fft_padded_filter in self.fft_padded_filters:
            spatial_filter = np.real(ifft2(fft_padded_filter))
            spatial_filter = crop(spatial_filter,
                                  self.patch_shape)[:, ::-1, ::-1]
            filter_images.append(Image(spatial_filter))
        return filter_images

    @property
    def frequency_filter_images(self):
        r"""
        """
        filter_images = []
        for fft_padded_filter in self.fft_padded_filters:
            spatial_filter = np.real(ifft2(fft_padded_filter))
            spatial_filter = crop(spatial_filter,
                                  self.patch_shape)[:, ::-1, ::-1]
            frequency_filter = np.abs(fftshift(fft2(spatial_filter)))
            filter_images.append(Image(frequency_filter))
        return filter_images

    def _extract_patch(self, image, landmark):
        r"""
        """
        # Extract patch from image
        patch = image.extract_patches(
            landmark, patch_shape=self.patch_shape,
            sample_offsets=self.sample_offsets, as_single_array=True)
        # Reshape patch
        # patch: (offsets x ch) x h x w
        patch = patch.reshape((-1,) + patch.shape[-2:])
        # Normalise patch
        return self.patch_normalisation(patch)

    def _extract_patches(self, image, shape):
        r"""
        """
        # Obtain patch ensemble, the whole shape is used to extract patches
        # from all landmarks at once
        patches = image.extract_patches(shape, patch_shape=self.patch_shape,
                                        sample_offsets=self.sample_offsets,
                                        as_single_array=True)
        # Reshape patches
        # patches: n_patches x (n_offsets x n_channels) x height x width
        patches = patches.reshape((patches.shape[0], -1) + patches.shape[-2:])
        # Normalise patches
        return self.patch_normalisation(patches)

    def predict_response(self, image, shape):
        r"""
        """
        # Extract patches
        patches = self._extract_patches(image, shape)
        # Predict responses
        return fft_convolve2d_sum(patches, self.fft_padded_filters,
                                  fft_filter=True, axis=1)

    def predict_probability(self, image, shape):
        r"""
        """
        # Predict responses
        responses = self.predict_response(image, shape)
        # Turn them into proper probability maps
        return probability_map(responses)


# TODO: Document me!
class CorrelationFilterExpertEnsemble(ConvolutionBasedExpertEnsemble):
    r"""
    """
    def __init__(self, images, shapes, verbose=False, prefix='',
                 icf_cls=IncrementalCorrelationFilterThinWrapper,
                 patch_shape=(17, 17), context_size=(34, 34),
                 response_covariance=3, patch_normalisation=normalize_norm,
                 cosine_mask=True, sample_offsets=None):
        # TODO: check parameters?
        # Set parameters
        self._icf = icf_cls()
        self.patch_shape = patch_shape
        self.context_size = context_size
        self.response_covariance = response_covariance
        self.patch_normalisation = patch_normalisation
        self.cosine_mask = cosine_mask
        self.sample_offsets = sample_offsets

        # Generate cosine mask
        self._cosine_mask = generate_cosine_mask(self.context_size)

        # Generate desired response, i.e. a Gaussian response with the
        # specified covariance centred at the middle of the patch
        self.response = generate_gaussian_response(
            self.patch_shape, self.response_covariance)[None, ...]

        # Train ensemble of correlation filter experts
        self._train(images, shapes, verbose=verbose, prefix=prefix)

    def _extract_patch(self, image, landmark):
        r"""
        """
        # Extract patch from image
        patch = image.extract_patches(
            landmark, patch_shape=self.context_size,
            sample_offsets=self.sample_offsets, as_single_array=True)
        # Reshape patch
        # patch: (offsets x ch) x h x w
        patch = patch.reshape((-1,) + patch.shape[-2:])
        # Normalise patch
        patch = self.patch_normalisation(patch)
        if self.cosine_mask:
            # Apply cosine mask if require
            patch = self._cosine_mask * patch
        return patch

    def _train(self, images, shapes, prefix='', verbose=False,
               increment=False):
        r"""
        """
        # Define print_progress partial
        wrap = partial(print_progress,
                       prefix='{}Training experts'
                              .format(prefix),
                       end_with_newline=not prefix,
                       verbose=verbose)

        # If increment is False, we need to initialise/reset the ensemble of
        # experts
        if not increment:
            self.fft_padded_filters = []
            self.auto_correlations = []
            self.cross_correlations = []
            # Set number of images
            self.n_images = len(images)
        else:
            # Update number of images
            self.n_images += len(images)

        # Obtain total number of experts
        n_experts = shapes[0].n_points

        # Train ensemble of correlation filter experts
        fft_padded_filters = []
        auto_correlations = []
        cross_correlations = []
        for i in wrap(range(n_experts)):
            patches = []
            for image, shape in zip(images, shapes):
                # Select the appropriate landmark
                landmark = PointCloud([shape.points[i]])
                # Extract patch
                patch = self._extract_patch(image, landmark)
                # Add patch to the list
                patches.append(patch)

            if increment:
                # Increment correlation filter
                correlation_filter, auto_correlation, cross_correlation = (
                    self._icf.increment(self.auto_correlations[i],
                                        self.cross_correlations[i],
                                        self.n_images,
                                        patches,
                                        self.response))
            else:
                # Train correlation filter
                correlation_filter, auto_correlation, cross_correlation = (
                    self._icf.train(patches, self.response))

            # Pad filter with zeros
            padded_filter = pad(correlation_filter, self.padded_size)
            # Compute fft of padded filter
            fft_padded_filter = fft2(padded_filter)
            # Add fft padded filter to list
            fft_padded_filters.append(fft_padded_filter)
            auto_correlations.append(auto_correlation)
            cross_correlations.append(cross_correlation)

        # Turn list into ndarray
        self.fft_padded_filters = np.asarray(fft_padded_filters)
        self.auto_correlations = np.asarray(auto_correlations)
        self.cross_correlations = np.asarray(cross_correlations)


# TODO: Document me!
def generate_gaussian_response(patch_shape, response_covariance):
    r"""
    """
    grid = build_grid(patch_shape)
    mvn = multivariate_normal(mean=np.zeros(2), cov=response_covariance)
    return mvn.pdf(grid)


# TODO: Document me!
def generate_cosine_mask(patch_shape):
    r"""
    """
    cy = np.hanning(patch_shape[0])
    cx = np.hanning(patch_shape[1])
    return cy[..., None].dot(cx[None, ...])
