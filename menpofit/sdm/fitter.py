from __future__ import division
import numpy as np
from functools import partial
import warnings

from menpo.transform import Scale
from menpo.feature import no_op
from menpo.base import name_of_callable

from menpofit.visualize import print_progress
from menpofit.base import batch
from menpofit.builder import (scale_images, rescale_images_to_reference_shape,
                              compute_reference_shape, MenpoFitBuilderWarning,
                              compute_features)
from menpofit.fitter import (MultiFitter, noisy_shape_from_bounding_box,
                             align_shape_with_bounding_box,
                             generate_perturbations_from_gt)
from menpofit.result import MultiFitterResult
import menpofit.checks as checks


# TODO: document me!
class SupervisedDescentFitter(MultiFitter):
    r"""
    """
    def __init__(self, images, group=None, bounding_box_group_glob=None,
                 reference_shape=None, sd_algorithm_cls=None,
                 holistic_features=no_op, patch_features=no_op,
                 patch_shape=(17, 17), diagonal=None, scales=(0.5, 1.0),
                 n_iterations=3, n_perturbations=30,
                 perturb_from_gt_bounding_box=noisy_shape_from_bounding_box,
                 batch_size=None, verbose=False):

        if batch_size is not None:
            raise NotImplementedError('Training an SDM with a batch size '
                                      '(incrementally) is not implemented yet.')
        # check parameters
        checks.check_diagonal(diagonal)
        scales = checks.check_scales(scales)
        n_scales = len(scales)
        patch_features = checks.check_callable(patch_features, n_scales)
        sd_algorithm_cls = checks.check_callable(sd_algorithm_cls, n_scales)
        holistic_features = checks.check_callable(holistic_features, n_scales)
        patch_shape = checks.check_patch_shape(patch_shape, n_scales)
        # set parameters
        self.algorithms = []
        self.reference_shape = reference_shape
        self._sd_algorithm_cls = sd_algorithm_cls
        self.holistic_features = holistic_features
        self.patch_features = patch_features
        self.patch_shape = patch_shape
        self.diagonal = diagonal
        self.scales = scales
        self.n_perturbations = n_perturbations
        self.n_iterations = checks.check_max_iters(n_iterations, n_scales)
        self._perturb_from_gt_bounding_box = perturb_from_gt_bounding_box
        # set up algorithms
        self._setup_algorithms()

        # Now, train the model!
        self._train(images, increment=False,  group=group,
                    bounding_box_group_glob=bounding_box_group_glob,
                    verbose=verbose, batch_size=batch_size)

    def _setup_algorithms(self):
        for j in range(self.n_scales):
            self.algorithms.append(self._sd_algorithm_cls[j](
                patch_features=self.patch_features[j],
                patch_shape=self.patch_shape[j],
                n_iterations=self.n_iterations[j]))

    def _train(self, images, increment=False, group=None,
               bounding_box_group_glob=None, verbose=False, batch_size=None):
        r"""
        """
        # If batch_size is not None, then we may have a generator, else we
        # assume we have a list.
        # If batch_size is not None, then we may have a generator, else we
        # assume we have a list.
        if batch_size is not None:
            # Create a generator of fixed sized batches. Will still work even
            # on an infinite list.
            image_batches = batch(images, batch_size)
        else:
            image_batches = [list(images)]

        for k, image_batch in enumerate(image_batches):
            if k == 0:
                if self.reference_shape is None:
                    # If no reference shape was given, use the mean of the first
                    # batch
                    if batch_size is not None:
                        warnings.warn('No reference shape was provided. The '
                                      'mean of the first batch will be the '
                                      'reference shape. If the batch mean is '
                                      'not representative of the true mean, '
                                      'this may cause issues.',
                                      MenpoFitBuilderWarning)
                    self.reference_shape = compute_reference_shape(
                        [i.landmarks[group].lms for i in image_batch],
                        self.diagonal, verbose=verbose)
            # We set landmarks on the images to archive the perturbations, so
            # when the default 'None' is used, we need to grab the actual
            # label to sort out the ambiguity
            if group is None:
                group = image_batch[0].landmarks.group_labels[0]

            # After the first batch, we are incrementing the model
            if k > 0:
                increment = True

            if verbose:
                print('Computing batch {}'.format(k))

            # Train each batch
            self._train_batch(
                image_batch, increment=increment, group=group,
                bounding_box_group_glob=bounding_box_group_glob,
                verbose=verbose)

    def _train_batch(self, image_batch, increment=False, group=None,
                     bounding_box_group_glob=None, verbose=False):
        # Rescale to existing reference shape
        image_batch = rescale_images_to_reference_shape(
            image_batch, group, self.reference_shape,
            verbose=verbose)

        generated_bb_func = generate_perturbations_from_gt(
            image_batch, self.n_perturbations,
            self._perturb_from_gt_bounding_box, gt_group=group,
            bb_group_glob=bounding_box_group_glob, verbose=verbose)

        # for each scale (low --> high)
        current_shapes = []
        for j in range(self.n_scales):
            if verbose:
                if len(self.scales) > 1:
                    scale_prefix = '  - Scale {}: '.format(j)
                else:
                    scale_prefix = '  - '
            else:
                scale_prefix = None

            # Handle holistic features
            if j == 0 and self.holistic_features[j] == no_op:
                # Saves a lot of memory
                feature_images = image_batch
            elif j == 0 or self.holistic_features[j] is not self.holistic_features[j - 1]:
                # Compute features only if this is the first pass through
                # the loop or the features at this scale are different from
                # the features at the previous scale
                feature_images = compute_features(image_batch,
                                                  self.holistic_features[j],
                                                  prefix=scale_prefix,
                                                  verbose=verbose)
            # handle scales
            if self.scales[j] != 1:
                # Scale feature images only if scale is different than 1
                scaled_images = scale_images(feature_images, self.scales[j],
                                             prefix=scale_prefix,
                                             verbose=verbose)
            else:
                scaled_images = feature_images

            # Extract scaled ground truth shapes for current scale
            scaled_shapes = [i.landmarks[group].lms for i in scaled_images]

            if j == 0:
                msg = '{}Aligning reference shape with bounding boxes.'.format(
                    scale_prefix)
                wrap = partial(print_progress, prefix=msg,
                               end_with_newline=False, verbose=verbose)

                # Extract perturbations at the very bottom level
                for ii in wrap(scaled_images):
                    c_shapes = []
                    for bbox in generated_bb_func(ii):
                        c_s = align_shape_with_bounding_box(
                            self.reference_shape, bbox)
                        c_shapes.append(c_s)
                    current_shapes.append(c_shapes)

            # train supervised descent algorithm
            if not increment:
                current_shapes = self.algorithms[j].train(
                    scaled_images, scaled_shapes, current_shapes,
                    prefix=scale_prefix, verbose=verbose)
            else:
                current_shapes = self.algorithms[j].increment(
                    scaled_images, scaled_shapes, current_shapes,
                    prefix=scale_prefix, verbose=verbose)

            # Scale current shapes to next resolution, don't bother
            # scaling final level
            if j != (self.n_scales - 1):
                transform = Scale(self.scales[j + 1] / self.scales[j],
                                  n_dims=2)
                for image_shapes in current_shapes:
                    for k, shape in enumerate(image_shapes):
                        image_shapes[k] = transform.apply(shape)

    def increment(self, images, group=None, bounding_box_group=None,
                  verbose=False, batch_size=None):
        raise NotImplementedError('Incrementing SDM methods is not yet '
                                  'implemented as careful attention must '
                                  'be taken when considering the relationships '
                                  'between cascade levels.')

    def perturb_from_bb(self, gt_shape, bb):
        return self._perturb_from_gt_bounding_box(gt_shape, bb)

    def perturb_from_gt_bb(self, gt_bb):
        return self._perturb_from_gt_bounding_box(gt_bb, gt_bb)

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
        is_custom_perturb_func = (self._perturb_from_gt_bounding_box !=
                                  noisy_shape_from_bounding_box)
        regressor_cls = self.algorithms[0]._regressor_cls

        # Compute scale info strings
        scales_info = []
        lvl_str_tmplt = r"""  - Scale {}
   - {} iterations
   - Patch shape: {}
   - Holistic feature: {}
   - Patch feature: {}"""
        for k, s in enumerate(self.scales):
            scales_info.append(lvl_str_tmplt.format(
                s, self.n_iterations[k], self.patch_shape[k],
                name_of_callable(self.holistic_features[k]),
                name_of_callable(self.patch_features[k])))
        scales_info = '\n'.join(scales_info)

        cls_str = r"""Supervised Descent Method
 - Regression performed using the {reg_alg} algorithm
   - Regression class: {reg_cls}
 - Perturbations generated per shape: {n_perturbations}
 - Images scaled to diagonal: {diagonal:.2f}
 - Custom perturbation scheme used: {is_custom_perturb_func}
 - Scales: {scales}
{scales_info}
""".format(
            reg_alg=name_of_callable(self._sd_algorithm_cls),
            reg_cls=name_of_callable(regressor_cls),
            n_perturbations=self.n_perturbations,
            diagonal=diagonal,
            is_custom_perturb_func=is_custom_perturb_func,
            scales=self.scales,
            scales_info=scales_info)
        return cls_str

# *
# ************************* Non-Parametric Fitters *****************************
# *
from .algorithm import NonParametricNewton

# Aliases for common combinations of supervised descent fitting
SDM = partial(SupervisedDescentFitter, sd_algorithm_cls=NonParametricNewton)


class RegularizedSDM(SupervisedDescentFitter):
    def __init__(self, images, group=None, bounding_box_group_glob=None,
                 alpha=0.0001, reference_shape=None,
                 holistic_features=no_op, patch_features=no_op,
                 patch_shape=(17, 17), diagonal=None, scales=(0.5, 1.0),
                 n_iterations=6, n_perturbations=30,
                 perturb_from_gt_bounding_box=noisy_shape_from_bounding_box,
                 batch_size=None, verbose=False):
        super(RegularizedSDM, self).__init__(
            images, group=group,
            bounding_box_group_glob=bounding_box_group_glob,
            reference_shape=reference_shape,
            sd_algorithm_cls=partial(NonParametricNewton, alpha=alpha),
            holistic_features=holistic_features, patch_features=patch_features,
            patch_shape=patch_shape, diagonal=diagonal, scales=scales,
            n_iterations=n_iterations, n_perturbations=n_perturbations,
            perturb_from_gt_bounding_box=perturb_from_gt_bounding_box,
            batch_size=batch_size, verbose=verbose)
