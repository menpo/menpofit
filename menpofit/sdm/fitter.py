from __future__ import division
from functools import partial
from menpo.transform import Scale, AlignmentSimilarity
from menpo.feature import no_op
from menpofit.builder import normalization_wrt_reference_shape, scale_images
from menpofit.fitter import MultiFitter, noisy_align
from menpofit.result import MultiFitterResult
import menpofit.checks as checks
from .algorithm import SN


# TODO: document me!
class CRFitter(MultiFitter):
    r"""
    """
    def __init__(self, cr_algorithm_cls=SN, features=no_op,
                 patch_shape=(17, 17), diagonal=None, scales=(1, 0.5),
                 iterations=6, n_perturbations=10, **kwargs):
        # check parameters
        checks.check_diagonal(diagonal)
        scales, n_levels = checks.check_scales(scales)
        features = checks.check_features(features, n_levels)
        patch_shape = checks.check_patch_shape(patch_shape, n_levels)
        # set parameters
        self._algorithms = []
        self.diagonal = diagonal
        self.scales = list(scales)[::-1]
        self.n_perturbations = n_perturbations
        self.iterations = checks.check_max_iters(iterations, n_levels)
        # set up algorithms
        self._set_up(cr_algorithm_cls, features, patch_shape, **kwargs)

    @property
    def algorithms(self):
        return self._algorithms

    @property
    def reference_bounding_box(self):
        return self.reference_shape.bounding_box()

    def _set_up(self, cr_algorithm_cls, features, patch_shape, **kwargs):
        for j in range(self.n_levels):
            algorithm = cr_algorithm_cls(
                features=features[j], patch_shape=patch_shape[j],
                iterations=self.iterations[j], **kwargs)
            self._algorithms.append(algorithm)

    def train(self, images, group=None, label=None, verbose=False, **kwargs):
        # normalize images and compute reference shape
        self.reference_shape, images = normalization_wrt_reference_shape(
            images, group, label, self.diagonal, verbose=verbose)

        # for each pyramid level (low --> high)
        for j in range(self.n_levels):
            if verbose:
                if len(self.scales) > 1:
                    level_str = '  - Level {}: '.format(j)
                else:
                    level_str = '  - '

            # scale images and compute features at other levels
            level_images = scale_images(images, self.scales[j],
                                        level_str=level_str, verbose=verbose)

            # extract ground truth shapes for current level
            level_gt_shapes = [i.landmarks[group][label] for i in level_images]

            if j == 0:
                # generate perturbed shapes
                current_shapes = []
                for gt_s in level_gt_shapes:
                    perturbed_shapes = []
                    for _ in range(self.n_perturbations):
                        p_s = self.noisy_shape_from_shape(gt_s)
                        perturbed_shapes.append(p_s)
                    current_shapes.append(perturbed_shapes)

            # train cascaded regression algorithm
            current_shapes = self.algorithms[j].train(
                level_images, level_gt_shapes, current_shapes,
                verbose=verbose, **kwargs)

            # scale current shapes to next level resolution
            if self.scales[j] != (1 or self.scales[-1]):
                transform = Scale(self.scales[j+1]/self.scales[j], n_dims=2)
                for image_shapes in current_shapes:
                    for shape in image_shapes:
                        transform.apply_inplace(shape)

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

        gt_shape : class : :map:`PointCloud`, optional
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
            image = image.copy()
            image.crop_to_landmarks_proportion_inplace(crop_image,
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

    def noisy_shape_from_bounding_box(self, bounding_box, noise_std=0.04,
                                      rotation=False):
        transform = noisy_align(AlignmentSimilarity,
                                self.reference_bounding_box, bounding_box,
                                noise_std=noise_std, rotation=rotation)
        return transform.apply(self.reference_shape)

    def noisy_shape_from_shape(self, shape, noise_std=0.04, rotation=False):
        return self.noisy_shape_from_bounding_box(
            shape.bounding_box(), noise_std=noise_std, rotation=rotation)

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


# TODO: document me!
SDMFitter = partial(CRFitter, cr_algorithm_cls=SN)


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
#         self._set_up(cr_algorithm_cls, features, sampling, **kwargs)
#
#     @property
#     def algorithms(self):
#         return self._algorithms
#
#     def _set_up(self, cr_algorithm_cls, features, sampling, **kwargs):
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