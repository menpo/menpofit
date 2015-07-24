from __future__ import division
from functools import partial
import numpy as np
import warnings
from menpo.transform import Scale
from menpo.feature import no_op
from menpo.visualize import print_progress
from menpofit.builder import (normalization_wrt_reference_shape, scale_images,
                              rescale_images_to_reference_shape)
from menpofit.fitter import (
    MultiFitter, noisy_shape_from_shape, noisy_shape_from_bounding_box,
    align_shape_with_bounding_box)
from menpofit.result import MultiFitterResult
import menpofit.checks as checks
from .algorithm import Newton


# TODO: document me!
class SupervisedDescentFitter(MultiFitter):
    r"""
    """
    def __init__(self, sd_algorithm_cls=Newton, features=no_op,
                 patch_shape=(17, 17), diagonal=None, scales=(1, 0.5),
                 iterations=6, n_perturbations=30,
                 perturb_from_bounding_box=noisy_shape_from_bounding_box,
                 **kwargs):
        # check parameters
        checks.check_diagonal(diagonal)
        scales, n_levels = checks.check_scales(scales)
        features = checks.check_features(features, n_levels)
        patch_shape = checks.check_patch_shape(patch_shape, n_levels)
        # set parameters
        self.algorithms = []
        self.reference_shape = None
        self._sd_algorithm_cls = sd_algorithm_cls
        self._features = features
        self._patch_shape = patch_shape
        self.diagonal = diagonal
        self.scales = list(scales)[::-1]
        self.n_perturbations = n_perturbations
        self.iterations = checks.check_max_iters(iterations, n_levels)
        self._perturb_from_bounding_box = perturb_from_bounding_box
        # set up algorithms
        self._reset_algorithms(**kwargs)

    def _reset_algorithms(self, **kwargs):
        if len(self.algorithms) > 0:
            for j in range(len(self.algorithms) - 1, -1, -1):
                del self.algorithms[j]
        for j in range(self.n_levels):
            self.algorithms.append(self._sd_algorithm_cls(
                features=self._holistic_features[j], patch_shape=self._patch_shape[j],
                iterations=self.iterations[j], **kwargs))

    def perturb_from_bounding_box(self, bounding_box):
        return self._perturb_from_bounding_box(self.reference_shape,
                                               bounding_box)

    def train(self, images, group=None, label=None, bounding_box_group=None,
              verbose=False, increment=False, **kwargs):
        # In the case where group is None, we need to get the only key so that
        # we can add landmarks below and not get a complaint about using None
        first_image = images[0]
        if group is None:
            group = first_image.landmarks.group_labels[0]

        if not increment:
            # Reset the algorithm classes
            self._reset_algorithms()
            # Normalize images and compute reference shape
            self.reference_shape, images = normalization_wrt_reference_shape(
                images, group, label, self.diagonal, verbose=verbose)
        else:
            if len(self.algorithms) == 0:
                raise ValueError('Must train before training incrementally.')
            # We are incrementing, so rescale to existing reference shape
            images = rescale_images_to_reference_shape(images, group, label,
                                                       self.reference_shape,
                                                       verbose=verbose)

        # No bounding box is given, so we will use the ground truth box
        if bounding_box_group is None:
            bounding_box_group = '__gt_bb_'
            for i in images:
                gt_s = i.landmarks[group][label]
                perturb_bbox_group = bounding_box_group + '0'
                i.landmarks[perturb_bbox_group] = gt_s.bounding_box()

        # Find all bounding boxes on the images with the given bounding box key
        all_bb_keys = list(first_image.landmarks.keys_matching(
            '*{}*'.format(bounding_box_group)))
        n_perturbations = len(all_bb_keys)

        # If there is only one example bounding box, then we will generate
        # more perturbations based on the bounding box.
        if n_perturbations == 1:
            if verbose:
                msg = '- Generating {} new initial bounding boxes ' \
                      'per image'.format(self.n_perturbations)
                wrap = partial(print_progress, prefix=msg)
            else:
                wrap = lambda x: x

            for i in wrap(images):
                # We assume that the first bounding box is a valid perturbation
                # thus create n_perturbations - 1 new bounding boxes
                for j in range(1, self.n_perturbations):
                    gt_s = i.landmarks[group][label].bounding_box()
                    bb = i.landmarks[all_bb_keys[0]].lms

                    # This is customizable by passing in the correct method
                    p_s = self.perturb_from_bounding_box(gt_s, bb)
                    perturb_bbox_group = bounding_box_group + '_{}'.format(j)
                    i.landmarks[perturb_bbox_group] = p_s
        elif n_perturbations != self.n_perturbations:
            warnings.warn('The original value of n_perturbation {} '
                          'will be reset to {} in order to agree with '
                          'the provided bounding_box_group.'.
                          format(self.n_perturbations, n_perturbations))
            self.n_perturbations = n_perturbations

        # Re-grab all the bounding box keys for iterating over when calculating
        # perturbations
        all_bb_keys = list(first_image.landmarks.keys_matching(
            '*{}*'.format(bounding_box_group)))

        # for each pyramid level (low --> high)
        for j in range(self.n_levels):
            if verbose:
                if len(self.scales) > 1:
                    level_str = '  - Level {}: '.format(j)
                else:
                    level_str = '  - '
            else:
                level_str = None

            # Scale images and compute features at other levels
            level_images = scale_images(images, self.scales[j],
                                        level_str=level_str, verbose=verbose)

            # Extract scaled ground truth shapes for current level
            level_gt_shapes = [i.landmarks[group][label] for i in level_images]

            if j == 0:
                if verbose:
                    msg = '{}Generating {} perturbations per image'.format(
                        level_str, self.n_perturbations)
                    wrap = partial(print_progress, prefix=msg,
                                   end_with_newline=False)
                else:
                    wrap = lambda x: x

                # Extract perturbations at the very bottom level
                current_shapes = []
                for i in wrap(level_images):
                    c_shapes = []
                    for perturb_bbox_group in all_bb_keys:
                        bbox = i.landmarks[perturb_bbox_group].lms
                        c_s = align_shape_with_bounding_box(
                            self.reference_shape, bbox)
                        c_shapes.append(c_s)
                    current_shapes.append(c_shapes)

            # train supervised descent algorithm
            if increment:
                current_shapes = self.algorithms[j].increment(
                    level_images, level_gt_shapes, current_shapes,
                    verbose=verbose, **kwargs)
            else:
                current_shapes = self.algorithms[j].train(
                    level_images, level_gt_shapes, current_shapes,
                    level_str=level_str, verbose=verbose, **kwargs)

            # Scale current shapes to next level resolution
            if self.scales[j] != (1 or self.scales[-1]):
                transform = Scale(self.scales[j + 1] / self.scales[j], n_dims=2)
                for image_shapes in current_shapes:
                    for shape in image_shapes:
                        transform.apply_inplace(shape)

    def increment(self, images, group=None, label=None,
                  bounding_box_group=None, verbose=False,
                  **kwargs):
        return self.train(images, group=group, label=label,
                          bounding_box_group=bounding_box_group,
                          verbose=verbose,
                          increment=True, **kwargs)

    def train_incrementally(self, images, group=None, label=None,
                            batch_size=100, verbose=False, **kwargs):
        # number of batches
        n_batches = np.int(np.ceil(len(images) / batch_size))

        # train first batch
        if verbose:
            print('Training batch 1.')
        self.train(images[:batch_size], group=group, label=label,
                   verbose=verbose, **kwargs)

        # train all other batches
        start = batch_size
        for j in range(1, n_batches):
            if verbose:
                print('Training batch {}.'.format(j + 1))
            end = start + batch_size
            self.increment(images[start:end], group=group, label=label,
                           verbose=verbose, **kwargs)
            start = end

    def _prepare_image(self, image, initial_shape, gt_shape=None,
                       crop_image=0.5):
        r"""
        Prepares the image to be fitted.

        The image is first rescaled wrt the ``reference_landmarks`` and then
        a gaussian pyramid is applied. Depending on the
        ``pyramid_on_features`` flag, the pyramid is either applied to the
        features image computed from the rescaled imaged or applied to the
        rescaled image and features extracted at each pyramidal level.

        Parameters
        ----------
        image : :map:`Image` or subclass
            The image to be fitted.
        initial_shape : :map:`PointCloud`
            The initial shape from which the fitting will start.
        gt_shape : :map:`PointCloud`, optional
            The original ground truth shape associated to the image.
        crop_image: `None` or float`, optional
            If `float`, it specifies the proportion of the border wrt the
            initial shape to which the image will be internally cropped around
            the initial shape range.
            If `None`, no cropping is performed.

            This will limit the fitting algorithm search region but is
            likely to speed up its running time, specially when the
            modeled object occupies a small portion of the image.

        Returns
        -------
        images : `list` of :map:`Image` or subclass
            The list of images that will be fitted by the fitters.
        initial_shapes : `list` of :map:`PointCloud`
            The initial shape for each one of the previous images.
        gt_shapes : `list` of :map:`PointCloud`
            The ground truth shape for each one of the previous images.
        """
        # attach landmarks to the image
        image.landmarks['initial_shape'] = initial_shape
        if gt_shape:
            image.landmarks['gt_shape'] = gt_shape

        # if specified, crop the image
        if crop_image:
            image = image.crop_to_landmarks_proportion(crop_image,
                                                       group='initial_shape')

        # rescale image wrt the scale factor between reference_shape and
        # initial_shape
        image = image.rescale_to_reference_shape(self.reference_shape,
                                                 group='initial_shape')

        # obtain image representation
        images = []
        for s in self.scales:
            if s != 1:
                # scale image
                scaled_image = image.rescale(s)
            else:
                scaled_image = image
            images.append(scaled_image)

        # get initial shapes per level
        initial_shapes = [i.landmarks['initial_shape'].lms for i in images]

        # get ground truth shapes per level
        if gt_shape:
            gt_shapes = [i.landmarks['gt_shape'].lms for i in images]
        else:
            gt_shapes = None

        return images, initial_shapes, gt_shapes

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return MultiFitterResult(image, self, algorithm_results,
                                 affine_correction, gt_shape=gt_shape)

    # TODO: fix me!
    def __str__(self):
        pass
        # out = "Supervised Descent Method\n" \
        #       " - Non-Parametric '{}' Regressor\n" \
        #       " - {} training images.\n".format(
        #     name_of_callable(self._fitters[0].regressor),
        #     self._n_training_images)
        # # small strings about number of channels, channels string and downscale
        # down_str = []
        # for j in range(self.n_levels):
        #     if j == self.n_levels - 1:
        #         down_str.append('(no downscale)')
        #     else:
        #         down_str.append('(downscale by {})'.format(
        #             self.downscale**(self.n_levels - j - 1)))
        # temp_img = Image(image_data=np.random.rand(40, 40))
        # if self.pyramid_on_features:
        #     temp = self.features(temp_img)
        #     n_channels = [temp.n_channels] * self.n_levels
        # else:
        #     n_channels = []
        #     for j in range(self.n_levels):
        #         temp = self.features[j](temp_img)
        #         n_channels.append(temp.n_channels)
        # # string about features and channels
        # if self.pyramid_on_features:
        #     feat_str = "- Feature is {} with ".format(
        #         name_of_callable(self.features))
        #     if n_channels[0] == 1:
        #         ch_str = ["channel"]
        #     else:
        #         ch_str = ["channels"]
        # else:
        #     feat_str = []
        #     ch_str = []
        #     for j in range(self.n_levels):
        #         if isinstance(self.features[j], str):
        #             feat_str.append("- Feature is {} with ".format(
        #                 self.features[j]))
        #         elif self.features[j] is None:
        #             feat_str.append("- No features extracted. ")
        #         else:
        #             feat_str.append("- Feature is {} with ".format(
        #                 self.features[j].__name__))
        #         if n_channels[j] == 1:
        #             ch_str.append("channel")
        #         else:
        #             ch_str.append("channels")
        # if self.n_levels > 1:
        #     out = "{} - Gaussian pyramid with {} levels and downscale " \
        #           "factor of {}.\n".format(out, self.n_levels,
        #                                    self.downscale)
        #     if self.pyramid_on_features:
        #         out = "{}   - Pyramid was applied on feature space.\n   " \
        #               "{}{} {} per image.\n".format(out, feat_str,
        #                                             n_channels[0], ch_str[0])
        #     else:
        #         out = "{}   - Features were extracted at each pyramid " \
        #               "level.\n".format(out)
        #         for i in range(self.n_levels - 1, -1, -1):
        #             out = "{}   - Level {} {}: \n     {}{} {} per " \
        #                   "image.\n".format(
        #                 out, self.n_levels - i, down_str[i], feat_str[i],
        #                 n_channels[i], ch_str[i])
        # else:
        #     if self.pyramid_on_features:
        #         feat_str = [feat_str]
        #     out = "{0} - No pyramid used:\n   {1}{2} {3} per image.\n".format(
        #         out, feat_str[0], n_channels[0], ch_str[0])
        # return out


# class CRFitter(MultiFitter):
#     r"""
#     """
#     def __init__(self, cr_algorithm_cls=SN, features=no_op, diagonal=None,
#                  scales=(1, 0.5), sampling=None, n_perturbations=10,
#                  iterations=6, **kwargs):
#         # check parameters
#         checks.check_diagonal(diagonal)
#         scales, n_levels = checks.check_scales(scales)
#         features = checks.check_features(features, n_levels)
#         sampling = checks.check_sampling(sampling, n_levels)
#         # set parameters
#         self._algorithms = []
#         self.diagonal = diagonal
#         self.scales = list(scales)
#         self.n_perturbations = n_perturbations
#         self.iterations = checks.check_iterations(iterations, n_levels)
#         # set up algorithms
#         self._reset_algorithms(cr_algorithm_cls, features, sampling, **kwargs)
#
#     @property
#     def algorithms(self):
#         return self._algorithms
#
#     def _reset_algorithms(self, cr_algorithm_cls, features, sampling, **kwargs):
#         for j, s in range(self.n_levels):
#             algorithm = cr_algorithm_cls(
#                 features=features[j], sampling=sampling[j],
#                 max_iters=self.iterations[j], **kwargs)
#             self._algorithms.append(algorithm)
#
#     def train(self, images, group=None, label=None, verbose=False, **kwargs):
#         # normalize images and compute reference shape
#         reference_shape, images = normalization_wrt_reference_shape(
#             images, group, label, self.diagonal, verbose=verbose)
#
#         # for each pyramid level (low --> high)
#         for j in range(self.n_levels):
#             if verbose:
#                 if len(self.scales) > 1:
#                     level_str = '  - Level {}: '.format(j)
#                 else:
#                     level_str = '  - '
#
#             # scale images and compute features at other levels
#             level_images = scale_images(images, self.scales[j],
#                                         level_str=level_str, verbose=verbose)
#
#             # extract ground truth shapes for current level
#             level_gt_shapes = [i.landmarks[group][label] for i in level_images]
#
#             if j == 0:
#                 # generate perturbed shapes
#                 current_shapes = []
#                 for gt_s in level_gt_shapes:
#                     perturbed_shapes = []
#                     for _ in range(self.n_perturbations):
#                         p_s = self.noisy_shape_from_shape(gt_s)
#                         perturbed_shapes.append(p_s)
#                     current_shapes.append(perturbed_shapes)
#
#             # train cascaded regression algorithm
#             current_shapes = self.algorithms[j].train(
#                 level_images, level_gt_shapes, current_shapes,
#                 verbose=verbose, **kwargs)
#
#             # scale current shapes to next level resolution
#             if self.scales[j] != self.scales[-1]:
#                 transform = Scale(self.scales[j+1]/self.scales[j], n_dims=2)
#                 for image_shapes in current_shapes:
#                     for shape in image_shapes:
#                         transform.apply_inplace(shape)
#
#     def _fitter_result(self, image, algorithm_results, affine_correction,
#                        gt_shape=None):
#         return MultiFitterResult(image, algorithm_results, affine_correction,
#                                  gt_shape=gt_shape)
#
#     # TODO: fix me!
#     def __str__(self):
#         pass
#         # out = "Supervised Descent Method\n" \
#         #       " - Non-Parametric '{}' Regressor\n" \
#         #       " - {} training images.\n".format(
#         #     name_of_callable(self._fitters[0].regressor),
#         #     self._n_training_images)
#         # # small strings about number of channels, channels string and downscale
#         # down_str = []
#         # for j in range(self.n_levels):
#         #     if j == self.n_levels - 1:
#         #         down_str.append('(no downscale)')
#         #     else:
#         #         down_str.append('(downscale by {})'.format(
#         #             self.downscale**(self.n_levels - j - 1)))
#         # temp_img = Image(image_data=np.random.rand(40, 40))
#         # if self.pyramid_on_features:
#         #     temp = self.features(temp_img)
#         #     n_channels = [temp.n_channels] * self.n_levels
#         # else:
#         #     n_channels = []
#         #     for j in range(self.n_levels):
#         #         temp = self.features[j](temp_img)
#         #         n_channels.append(temp.n_channels)
#         # # string about features and channels
#         # if self.pyramid_on_features:
#         #     feat_str = "- Feature is {} with ".format(
#         #         name_of_callable(self.features))
#         #     if n_channels[0] == 1:
#         #         ch_str = ["channel"]
#         #     else:
#         #         ch_str = ["channels"]
#         # else:
#         #     feat_str = []
#         #     ch_str = []
#         #     for j in range(self.n_levels):
#         #         if isinstance(self.features[j], str):
#         #             feat_str.append("- Feature is {} with ".format(
#         #                 self.features[j]))
#         #         elif self.features[j] is None:
#         #             feat_str.append("- No features extracted. ")
#         #         else:
#         #             feat_str.append("- Feature is {} with ".format(
#         #                 self.features[j].__name__))
#         #         if n_channels[j] == 1:
#         #             ch_str.append("channel")
#         #         else:
#         #             ch_str.append("channels")
#         # if self.n_levels > 1:
#         #     out = "{} - Gaussian pyramid with {} levels and downscale " \
#         #           "factor of {}.\n".format(out, self.n_levels,
#         #                                    self.downscale)
#         #     if self.pyramid_on_features:
#         #         out = "{}   - Pyramid was applied on feature space.\n   " \
#         #               "{}{} {} per image.\n".format(out, feat_str,
#         #                                             n_channels[0], ch_str[0])
#         #     else:
#         #         out = "{}   - Features were extracted at each pyramid " \
#         #               "level.\n".format(out)
#         #         for i in range(self.n_levels - 1, -1, -1):
#         #             out = "{}   - Level {} {}: \n     {}{} {} per " \
#         #                   "image.\n".format(
#         #                 out, self.n_levels - i, down_str[i], feat_str[i],
#         #                 n_channels[i], ch_str[i])
#         # else:
#         #     if self.pyramid_on_features:
#         #         feat_str = [feat_str]
#         #     out = "{0} - No pyramid used:\n   {1}{2} {3} per image.\n".format(
#         #         out, feat_str[0], n_channels[0], ch_str[0])
#         # return out
