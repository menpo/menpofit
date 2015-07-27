from __future__ import division
from itertools import chain
import numpy as np
from functools import partial
import warnings
from menpo.transform import Scale
from menpo.feature import no_op
from menpofit.visualize import print_progress
from menpofit.base import batch, name_of_callable
from menpofit.builder import (normalization_wrt_reference_shape, scale_images,
                              rescale_images_to_reference_shape)
from menpofit.fitter import (MultiFitter, noisy_shape_from_bounding_box,
                             align_shape_with_bounding_box)
from menpofit.result import MultiFitterResult
import menpofit.checks as checks
from .algorithm import Newton


# TODO: document me!
class SupervisedDescentFitter(MultiFitter):
    r"""
    """
    def __init__(self, images, group=None, bounding_box_group=None,
                 sd_algorithm_cls=Newton, holistic_feature=no_op,
                 patch_features=no_op, patch_shape=(17, 17), diagonal=None,
                 scales=(1, 0.5), iterations=6, n_perturbations=30,
                 perturb_from_bounding_box=noisy_shape_from_bounding_box,
                 batch_size=None, verbose=False):
        # check parameters
        checks.check_diagonal(diagonal)
        scales, n_levels = checks.check_scales(scales)
        patch_features = checks.check_features(patch_features, n_levels)
        patch_shape = checks.check_patch_shape(patch_shape, n_levels)
        # set parameters
        self.algorithms = []
        self.reference_shape = None
        self._sd_algorithm_cls = sd_algorithm_cls
        self._holistic_feature = holistic_feature
        self._patch_features = patch_features
        self._patch_shape = patch_shape
        self.diagonal = diagonal
        self.scales = list(scales)[::-1]
        self.n_perturbations = n_perturbations
        self.iterations = checks.check_max_iters(iterations, n_levels)
        self._perturb_from_bounding_box = perturb_from_bounding_box
        # set up algorithms
        self._setup_algorithms()

        # Now, train the model!
        self._train(images, group=group, bounding_box_group=bounding_box_group,
                    verbose=verbose, increment=False, batch_size=batch_size)

    def _setup_algorithms(self):
        for j in range(self.n_levels):
            self.algorithms.append(self._sd_algorithm_cls(
                features=self._patch_features[j],
                patch_shape=self._patch_shape[j],
                iterations=self.iterations[j]))

    def perturb_from_bounding_box(self, bounding_box):
        return self._perturb_from_bounding_box(self.reference_shape,
                                               bounding_box)

    def _train(self, images, group=None, bounding_box_group=None,
               verbose=False, increment=False, batch_size=None):

        # If batch_size is not None, then we may have a generator, else we
        # assume we have a list.
        if batch_size is not None:
            # Create a generator of fixed sized batches. Will still work even
            # on an infinite list.
            image_batches = batch(images, batch_size)
            first_batch = next(image_batches)
        else:
            image_batches = []
            first_batch = list(images)

        # In the case where group is None, we need to get the only key so that
        # we can attach landmarks below and not get a complaint about using None
        first_image = first_batch[0]
        if group is None:
            group = first_image.landmarks.group_labels[0]

        for k, image_batch in enumerate(chain([first_batch], image_batches)):
            # After the first batch, we are incrementing the model
            if k > 0:
                increment = True

            if verbose:
                print('Computing batch {}'.format(k))

            if not increment:
                # Normalize images and compute reference shape
                self.reference_shape, image_batch = normalization_wrt_reference_shape(
                    image_batch, group, self.diagonal, verbose=verbose)
            else:
                # We are incrementing, so rescale to existing reference shape
                image_batch = rescale_images_to_reference_shape(
                    image_batch, group, self.reference_shape,
                    verbose=verbose)

            # No bounding box is given, so we will use the ground truth box
            if bounding_box_group is None:
                bounding_box_group = '__gt_bb_'
                for i in image_batch:
                    gt_s = i.landmarks[group].lms
                    perturb_bbox_group = bounding_box_group + '0'
                    i.landmarks[perturb_bbox_group] = gt_s.bounding_box()

            # Find all bounding boxes on the images with the given bounding
            # box key
            all_bb_keys = list(first_image.landmarks.keys_matching(
                '*{}*'.format(bounding_box_group)))
            n_perturbations = len(all_bb_keys)

            # If there is only one example bounding box, then we will generate
            # more perturbations based on the bounding box.
            if n_perturbations == 1:
                msg = '- Generating {} new initial bounding boxes ' \
                      'per image'.format(self.n_perturbations)
                wrap = partial(print_progress, prefix=msg, verbose=verbose)

                for i in wrap(image_batch):
                    # We assume that the first bounding box is a valid
                    # perturbation thus create n_perturbations - 1 new bounding
                    # boxes
                    for j in range(1, self.n_perturbations):
                        gt_s = i.landmarks[group].lms.bounding_box()
                        bb = i.landmarks[all_bb_keys[0]].lms

                        # This is customizable by passing in the correct method
                        p_s = self._perturb_from_bounding_box(gt_s, bb)
                        perturb_bbox_group = '{}_{}'.format(bounding_box_group,
                                                            j)
                        i.landmarks[perturb_bbox_group] = p_s
            elif n_perturbations != self.n_perturbations:
                warnings.warn('The original value of n_perturbation {} '
                              'will be reset to {} in order to agree with '
                              'the provided bounding_box_group.'.
                              format(self.n_perturbations, n_perturbations))
                self.n_perturbations = n_perturbations

            # Re-grab all the bounding box keys for iterating over when
            # calculating perturbations
            all_bb_keys = list(first_image.landmarks.keys_matching(
                '*{}*'.format(bounding_box_group)))

            # Before scaling, we compute the holistic feature on the whole image
            msg = '- Computing holistic features ({})'.format(
                name_of_callable(self._holistic_feature))
            wrap = partial(print_progress, prefix=msg, verbose=verbose)
            image_batch = [self._holistic_feature(im)
                           for im in wrap(image_batch)]

            # for each pyramid level (low --> high)
            current_shapes = []
            for j in range(self.n_levels):
                if verbose:
                    if len(self.scales) > 1:
                        level_str = '  - Level {}: '.format(j)
                    else:
                        level_str = '  - '
                else:
                    level_str = None

                # Scale images
                level_images = scale_images(image_batch, self.scales[j],
                                            level_str=level_str,
                                            verbose=verbose)

                # Extract scaled ground truth shapes for current level
                level_gt_shapes = [i.landmarks[group].lms
                                   for i in level_images]

                if j == 0:
                    msg = '{}Generating {} perturbations per image'.format(
                        level_str, self.n_perturbations)
                    wrap = partial(print_progress, prefix=msg,
                                   end_with_newline=False, verbose=verbose)

                    # Extract perturbations at the very bottom level
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
                        verbose=verbose)
                else:
                    current_shapes = self.algorithms[j].train(
                        level_images, level_gt_shapes, current_shapes,
                        level_str=level_str, verbose=verbose)

                # Scale current shapes to next level resolution
                if self.scales[j] != (1 or self.scales[-1]):
                    transform = Scale(self.scales[j + 1] / self.scales[j],
                                      n_dims=2)
                    for image_shapes in current_shapes:
                        for shape in image_shapes:
                            transform.apply_inplace(shape)

    def increment(self, images, group=None, bounding_box_group=None,
                  verbose=False):
        return self._train(images, group=group,
                           bounding_box_group=bounding_box_group,
                           verbose=verbose,
                           increment=True)

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
        # Attach landmarks to the image
        image.landmarks['initial_shape'] = initial_shape
        if gt_shape:
            image.landmarks['gt_shape'] = gt_shape

        # If specified, crop the image
        if crop_image:
            image = image.crop_to_landmarks_proportion(crop_image,
                                                       group='initial_shape')

        # Rescale image w.r.t the scale factor between reference_shape and
        # initial_shape
        image = image.rescale_to_reference_shape(self.reference_shape,
                                                 group='initial_shape')

        # Compute the holistic feature on the normalized image
        image = self._holistic_feature(image)

        # Obtain image representation
        images = []
        for s in self.scales:
            if s != 1:
                # scale image
                scaled_image = image.rescale(s)
            else:
                scaled_image = image
            images.append(scaled_image)

        # Get initial shapes per level
        initial_shapes = [i.landmarks['initial_shape'].lms for i in images]

        # Get ground truth shapes per level
        if gt_shape:
            gt_shapes = [i.landmarks['gt_shape'].lms for i in images]
        else:
            gt_shapes = None

        return images, initial_shapes, gt_shapes

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return MultiFitterResult(image, self, algorithm_results,
                                 affine_correction, gt_shape=gt_shape)

    def __str__(self):
        if self.diagonal is not None:
            diagonal = self.diagonal
        else:
            y, x = self.reference_shape.range()
            diagonal = np.sqrt(x ** 2 + y ** 2)
        is_custom_perturb_func = (self._perturb_from_bounding_box !=
                                  noisy_shape_from_bounding_box)
        regressor_cls = self.algorithms[0]._regressor_cls

        # Compute level info strings
        level_info = []
        lvl_str_tmplt = r"""  - Level {} (Scale {})
   - {} iterations
   - Patch shape: {}"""
        for k, s in enumerate(self.scales):
            level_info.append(lvl_str_tmplt.format(k, s,
                                                   self.iterations[k],
                                                   self._patch_shape[k]))
        level_info = '\n'.join(level_info)

        cls_str = r"""Supervised Descent Method
 - Regression performed using the {reg_alg} algorithm
   - Regression class: {reg_cls}
 - Levels: {levels}
{level_info}
 - Perturbations generated per shape: {n_perturbations}
 - Images scaled to diagonal: {diagonal:.2f}
 - Custom perturbation scheme used: {is_custom_perturb_func}""".format(
            reg_alg=name_of_callable(self._sd_algorithm_cls),
            reg_cls=name_of_callable(regressor_cls),
            n_levels=len(self.scales),
            levels=self.scales,
            level_info=level_info,
            n_perturbations=self.n_perturbations,
            diagonal=diagonal,
            is_custom_perturb_func=is_custom_perturb_func)
        return cls_str
