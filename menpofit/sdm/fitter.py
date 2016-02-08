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
import menpofit.checks as checks

from .algorithm import NonParametricNewton


class SupervisedDescentFitter(MultiFitter):
    r"""
    Class for training a multi-scale Supervised Descent model.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The `list` of training images.
    group : `str` or ``None``, optional
        The landmark group that will be used to train ERT. Note that all
        the training images need to have the specified landmark group.
    bounding_box_group_glob : `glob` or ``None``, optional
        Glob that defines the bounding boxes to be used for training. If
        ``None``, then the bounding boxes of the ground truth shapes are used.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for building the ERT. If
        ``None``, then the mean shape will be used.
    sd_algorithm_cls : `class` from `menpofit.sdm.algorithm`, optional
        The Supervised Descent algorithm to be used. For a `list` of
        available algorithms please refer to `menpofit.sdm.algorithm`.
    holistic_features : `function`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before extracting the patches. Please
        refer to `menpo.feature` for a list of potential features. If a `list`
        is provided, then it defines a value per scale.
    patch_features : `function`, optional
        The features that will be extracted from the patches of the training
        images. Note that, as opposed to `holistic_features`, these features
        are extracted after extracting the patches. Please refer to
        `menpo.feature` and `menpofit.feature` for a list of potential features.
         If a `list` is provided, then it defines a value per scale.
    patch_shape : ``(int, int)`` or `list` of ``(int, int)``, optional
        The shape of the patches to be extracted. If a `list` is provided,
        then it defines a patch shape per scale.
    diagonal : `int` or ``None``, optional
        This parameter is used to normalize the scale of the training images
        so that the extracted features are in correspondence. The
        normalization is performed by rescaling all the training images so
        that the diagonal of their ground truth shapes' bounding boxes
        equals to the provided value. The reference scale gets rescaled as
        well. If ``None``, then the images are rescaled with respect to the
        reference shape's diagonal.
    scales : `float` or `tuple` of `float`, optional
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale. If `float`, then a single scale is
        assumed.
    n_iterations : `int` or `list` of `int`, optional
        The number of iterations (cascades) of each level. If `list`, it must
        specify a value per scale. If `int`, then it defines the total number of
        iterations (cascades) over all scales.
    n_perturbations : `int` or ``None``, optional
        The number of perturbations to be generated from the provided
        bounding boxes.
    perturb_from_gt_bounding_box : `function`, optional
        The function that will be used to generate the perturbations.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.
    verbose : `bool`, optional
        If ``True``, then the progress of building ERT will be printed.

    References
    ----------
    .. [1] X. Xiong, and F. De la Torre. "Supervised Descent Method and its
        applications to face alignment", Proceedings of the IEEE Conference on
        Computer Vision and Pattern Recognition (CVPR), 2013.
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
        r"""
        Method to increment the trained SDM with a new set of training images.

        Parameters
        ----------
        images : `list` of `menpo.image.Image`
            The `list` of training images.
        group : `str` or ``None``, optional
            The landmark group that will be used to train the SDM. Note that all
            the training images need to have the specified landmark group.
        bounding_box_group : `str` or ``None``, optional
            The landmark group of the bounding boxes used for initialisation.
        verbose : `bool`, optional
            If ``True``, then the progress of building the AAM will be printed.
        batch_size : `int` or ``None``, optional
            If an `int` is provided, then the training is performed in an
            incremental fashion on image batches of size equal to the provided
            value. If ``None``, then the training is performed directly on the
            all the images.
        """
        raise NotImplementedError('Incrementing SDM methods is not yet '
                                  'implemented as careful attention must '
                                  'be taken when considering the relationships '
                                  'between cascade levels.')

    def perturb_from_bb(self, gt_shape, bb):
        """
        Returns a perturbed version of the ground truth shape. The perturbation
        is applied on the alignment between the ground truth bounding box and
        the provided bounding box. This is useful for obtaining the initial
        bounding box of the fitting.

        Parameters
        ----------
        gt_shape : `menpo.shape.PointCloud`
            The ground truth shape.
        bb : `menpo.shape.PointDirectedGraph`
            The target bounding box.

        Returns
        -------
        perturbed_shape : `menpo.shape.PointCloud`
            The perturbed shape.
        """
        return self._perturb_from_gt_bounding_box(gt_shape, bb)

    def perturb_from_gt_bb(self, gt_bb):
        """
        Returns a perturbed version of the ground truth bounding box. This is
        useful for obtaining the initial bounding box of the fitting.

        Parameters
        ----------
        gt_bb : `menpo.shape.PointDirectedGraph`
            The ground truth bounding box.

        Returns
        -------
        perturbed_bb : `menpo.shape.PointDirectedGraph`
            The perturbed ground truth bounding box.
        """
        return self._perturb_from_gt_bounding_box(gt_bb, gt_bb)

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return self.algorithms[0]._multi_scale_fitter_result(
                results=algorithm_results, scales=self.scales,
                affine_correction=affine_correction, image=image,
                gt_shape=gt_shape)

    def __str__(self):
        if self.diagonal is not None:
            diagonal = self.diagonal
        else:
            y, x = self.reference_shape.range()
            diagonal = np.sqrt(x ** 2 + y ** 2)
        is_custom_perturb_func = (self._perturb_from_gt_bounding_box !=
                                  noisy_shape_from_bounding_box)
        if is_custom_perturb_func:
            is_custom_perturb_func = name_of_callable(
                    self._perturb_from_gt_bounding_box)
        regressor_cls = self.algorithms[0]._regressor_cls

        # Compute scale info strings
        scales_info = []
        lvl_str_tmplt = r"""   - Scale {}
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
            reg_alg=name_of_callable(self._sd_algorithm_cls[0]),
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
# Aliases for common combinations of supervised descent fitting
SDM = partial(SupervisedDescentFitter, sd_algorithm_cls=NonParametricNewton)


class RegularizedSDM(SupervisedDescentFitter):
    r"""
    Class for training a multi-scale Regularized Supervised Descent Method model.
    The model uses `menpofit.sdm.algorithm.NonParametricNewton` by default with
    the specified regularization parameter.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The `list` of training images.
    group : `str` or ``None``, optional
        The landmark group that will be used to train ERT. Note that all
        the training images need to have the specified landmark group.
    bounding_box_group_glob : `glob` or ``None``, optional
        Glob that defines the bounding boxes to be used for training. If
        ``None``, then the bounding boxes of the ground truth shapes are used.
    alpha : `float`, optional
        The regularization parameter.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for building the ERT. If
        ``None``, then the mean shape will be used.
    holistic_features : `function`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before extracting the patches. Please
        refer to `menpo.feature` for a list of potential features. If a `list`
        is provided, then it defines a value per scale.
    patch_features : `function`, optional
        The features that will be extracted from the patches of the training
        images. Note that, as opposed to `holistic_features`, these features
        are extracted after extracting the patches. Please refer to
        `menpo.feature` and `menpofit.feature` for a list of potential features.
         If a `list` is provided, then it defines a value per scale.
    patch_shape : ``(int, int)`` or `list` of ``(int, int)``, optional
        The shape of the patches to be extracted. If a `list` is provided,
        then it defines a patch shape per scale.
    diagonal : `int` or ``None``, optional
        This parameter is used to normalize the scale of the training images
        so that the extracted features are in correspondence. The
        normalization is performed by rescaling all the training images so
        that the diagonal of their ground truth shapes' bounding boxes
        equals to the provided value. The reference scale gets rescaled as
        well. If ``None``, then the images are rescaled with respect to the
        reference shape's diagonal.
    scales : `float` or `tuple` of `float`, optional
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale. If `float`, then a single scale is
        assumed.
    n_iterations : `int` or `list` of `int`, optional
        The number of iterations (cascades) of each level. If `list`, it must
        specify a value per scale. If `int`, then it defines the total number of
        iterations (cascades) over all scales.
    n_perturbations : `int` or ``None``, optional
        The number of perturbations to be generated from the provided
        bounding boxes.
    perturb_from_gt_bounding_box : `function`, optional
        The function that will be used to generate the perturbations.
    batch_size : `int` or ``None``, optional
        If an `int` is provided, then the training is performed in an
        incremental fashion on image batches of size equal to the provided
        value. If ``None``, then the training is performed directly on the
        all the images.
    verbose : `bool`, optional
        If ``True``, then the progress of building ERT will be printed.

    References
    ----------
    .. [1] X. Xiong, and F. De la Torre. "Supervised Descent Method and its
        applications to face alignment", Proceedings of the IEEE Conference on
        Computer Vision and Pattern Recognition (CVPR), 2013.
    """
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
