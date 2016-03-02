from __future__ import division
from functools import partial
import warnings
import dlib
from pathlib import Path

from .algorithm import DlibAlgorithm

from menpo.feature import no_op
from menpo.transform import Scale, AlignmentAffine, Affine

from menpofit import checks
from menpofit.compatibility import STRING_TYPES
from menpofit.fitter import noisy_shape_from_bounding_box, MultiFitter, \
    generate_perturbations_from_gt
from menpofit.result import MultiFitterResult
from menpofit.builder import (
    scale_images, rescale_images_to_reference_shape,
    compute_reference_shape)


class DlibERT(MultiFitter):
    r"""
    Multiscale Dlib wrapper class. Trains models over multiple scales.
    """
    def __init__(self, images, group=None, bounding_box_group_glob=None,
                 verbose=False, reference_shape=None, diagonal=None,
                 scales=(0.5, 1.0), n_perturbations=30, n_dlib_perturbations=1,
                 perturb_from_gt_bounding_box=noisy_shape_from_bounding_box,
                 n_iterations=10, feature_padding=0, n_pixel_pairs=400,
                 distance_prior_weighting=0.1, regularisation_weight=0.1,
                 n_split_tests=20, n_trees=500, n_tree_levels=5):

        checks.check_diagonal(diagonal)

        self.diagonal = diagonal
        self.scales = checks.check_scales(scales)
        self.holistic_features = checks.check_callable(no_op, self.n_scales)
        self.reference_shape = reference_shape
        self.n_perturbations = n_perturbations
        self.n_iterations = checks.check_max_iters(n_iterations, self.n_scales)
        self._perturb_from_gt_bounding_box = perturb_from_gt_bounding_box
        self._setup_dlib_options(feature_padding, n_pixel_pairs,
                                 distance_prior_weighting,
                                 regularisation_weight, n_split_tests, n_trees,
                                 n_dlib_perturbations, n_tree_levels)
        self._setup_algorithms()

        # Train DLIB over multiple scales
        self._train(images, group=group,
                    bounding_box_group_glob=bounding_box_group_glob,
                    verbose=verbose)

    def _setup_algorithms(self):
        self.algorithms = []
        for j in range(self.n_scales):
            self.algorithms.append(DlibAlgorithm(
                self._dlib_options_templates[j],
                n_iterations=self.n_iterations[j]))

    def _setup_dlib_options(self, feature_padding, n_pixel_pairs,
                            distance_prior_weighting, regularisation_weight,
                            n_split_tests, n_trees, n_dlib_perturbations,
                            n_tree_levels):
        check_int = partial(checks.check_multi_scale_param, self.n_scales,
                            (int,))
        check_float = partial(checks.check_multi_scale_param, self.n_scales,
                              (float,))
        feature_padding = check_int('feature_padding', feature_padding)
        n_pixel_pairs = check_int('n_pixel_pairs', n_pixel_pairs)
        distance_prior_weighting = check_float('distance_prior_weighting',
                                               distance_prior_weighting)
        regularisation_weight = check_float('regularisation_weight',
                                            regularisation_weight)
        n_split_tests = check_int('n_split_tests', n_split_tests)
        n_trees = check_int('n_trees', n_trees)
        n_dlib_perturbations = check_int('n_dlib_perturbations',
                                         n_dlib_perturbations)
        n_tree_levels = check_int('n_tree_levels', n_tree_levels)
        self._dlib_options_templates = []
        for j in range(self.n_scales):
            new_opts = dlib.shape_predictor_training_options()

            # Size of region within which to sample features for the feature
            # pool, e.g a padding of 0.5 would cause the algorithm to sample
            # pixels from a box that was 2x2 pixels
            new_opts.feature_pool_region_padding = feature_padding[j]
            # P parameter form Kazemi paper
            new_opts.feature_pool_size = n_pixel_pairs[j]
            # Controls how tight the feature sampling should be. Lower values
            # enforce closer features. Opposite of explanation from Kazemi
            # paper, lambda
            new_opts.lambda_param = distance_prior_weighting[j]
            # Boosting regularization parameter - nu from Kazemi paper, larger
            # values may cause overfitting but improve performance on training
            # data
            new_opts.nu = regularisation_weight[j]
            # S from Kazemi paper - Number of split features at each node to
            # sample. The one that gives the best split is chosen.
            new_opts.num_test_splits = n_split_tests[j]
            # K from Kazemi paper - number of weak regressors
            new_opts.num_trees_per_cascade_level = n_trees[j]
            # R from Kazemi paper - amount of times other shapes are sampled
            # as example initialisations
            new_opts.oversampling_amount = n_dlib_perturbations[j]
            # F from Kazemi paper - number of levels in the tree (depth of tree)
            new_opts.tree_depth = n_tree_levels[j]

            self._dlib_options_templates.append(new_opts)

    def _train(self, original_images, group=None, bounding_box_group_glob=None,
               verbose=False):
        r"""
        """
        # Dlib does not support incremental builds, so we must be passed a list
        if not isinstance(original_images, list):
            original_images = list(original_images)
        # We use temporary landmark groups - so we need the group key to not be
        # None
        if group is None:
            group = original_images[0].landmarks.group_labels[0]

        # Temporarily store all the bounding boxes for rescaling
        for i in original_images:
            i.landmarks['__gt_bb'] = i.landmarks[group].lms.bounding_box()

        if self.reference_shape is None:
            # If no reference shape was given, use the mean of the first batch
            self.reference_shape = compute_reference_shape(
                [i.landmarks['__gt_bb'].lms for i in original_images],
                self.diagonal, verbose=verbose)

        # Rescale to existing reference shape
        images = rescale_images_to_reference_shape(
            original_images, '__gt_bb', self.reference_shape,
            verbose=verbose)

        # Scaling is done - remove temporary gt bounding boxes
        for i, i2 in zip(original_images, images):
            del i.landmarks['__gt_bb']
            del i2.landmarks['__gt_bb']

        generated_bb_func = generate_perturbations_from_gt(
            images, self.n_perturbations, self._perturb_from_gt_bounding_box,
            gt_group=group, bb_group_glob=bounding_box_group_glob,
            verbose=verbose)

        # for each scale (low --> high)
        current_bounding_boxes = []
        for j in range(self.n_scales):
            if verbose:
                if len(self.scales) > 1:
                    scale_prefix = '  - Scale {}: '.format(j)
                else:
                    scale_prefix = '  - '
            else:
                scale_prefix = None

            # handle scales
            if self.scales[j] != 1:
                # Scale feature images only if scale is different than 1
                scaled_images = scale_images(images, self.scales[j],
                                             prefix=scale_prefix,
                                             verbose=verbose)
            else:
                scaled_images = images

            if j == 0:
                current_bounding_boxes = [generated_bb_func(im)
                                          for im in scaled_images]

            # Extract scaled ground truth shapes for current scale
            scaled_gt_shapes = [i.landmarks[group].lms for i in scaled_images]

            # Train the Dlib model
            current_bounding_boxes = self.algorithms[j].train(
                scaled_images, scaled_gt_shapes, current_bounding_boxes,
                prefix=scale_prefix, verbose=verbose)

            # Scale current shapes to next resolution, don't bother
            # scaling final level
            if j != (self.n_scales - 1):
                transform = Scale(self.scales[j + 1] / self.scales[j],
                                  n_dims=2)
                for bboxes in current_bounding_boxes:
                    for bb in enumerate(bboxes):
                        bboxes[k] = transform.apply(bb)

    @property
    def n_scales(self):
        """
        The number of scales of the Dlib fitter.

        :type: `int`
        """
        return len(self.scales)

    def perturb_from_bb(self, gt_shape, bb):
        return self._perturb_from_gt_bounding_box(gt_shape, bb)

    def perturb_from_gt_bb(self, gt_bb):
            return self._perturb_from_gt_bounding_box(gt_bb, gt_bb)

    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        return MultiFitterResult(image, self, algorithm_results,
                                 affine_correction, gt_shape=gt_shape)

    def fit_from_shape(self, image, initial_shape, max_iters=20, gt_shape=None,
                       crop_image=None, **kwargs):

        warnings.warn('Fitting from an initial shape is not supported by '
                      'Dlib - therefore we are falling back to the tightest '
                      'bounding box from the given initial_shape')
        tightest_bb = initial_shape.bounding_box()
        return self.fit_from_bb(image, tightest_bb, max_iters=max_iters,
                                gt_shape=gt_shape, crop_image=crop_image,
                                **kwargs)

    def fit_from_bb(self, image, bounding_box, max_iters=20, gt_shape=None,
                    crop_image=None, **kwargs):
        # generate the list of images to be fitted
        images, bounding_boxes, gt_shapes = self._prepare_image(
            image, bounding_box, gt_shape=gt_shape, crop_image=crop_image)

        # work out the affine transform between the initial shape of the
        # highest pyramidal level and the initial shape of the original image
        affine_correction = AlignmentAffine(bounding_boxes[-1], bounding_box)

        # run multilevel fitting
        algorithm_results = self._fit(images, bounding_boxes[0],
                                      max_iters=max_iters,
                                      gt_shapes=gt_shapes, **kwargs)

        # build multilevel fitting result
        fitter_result = self._fitter_result(
            image, algorithm_results, affine_correction, gt_shape=gt_shape)

        return fitter_result

    def __str__(self):
        return ''


class DlibWrapper(object):
    r"""
    Multiscale Dlib wrapper class. Trains models over multiple scales.
    """
    def __init__(self, model):
        if isinstance(model, STRING_TYPES) or isinstance(model, Path):
            m_path = Path(model)
            if not Path(m_path).exists():
                raise ValueError('Model {} does not exist.'.format(m_path))
            model = dlib.shape_predictor(str(m_path))

        # Dlib doesn't expose any information about how the model was buit,
        # so we just create dummy options
        self.algorithm = DlibAlgorithm(dlib.shape_predictor_training_options(),
                                       n_iterations=0)
        self.algorithm.dlib_model = model
        self.scales = [1]

    def fit_from_shape(self, image, initial_shape, gt_shape=None, **kwargs):

        warnings.warn('Fitting from an initial shape is not supported by '
                      'Dlib - therefore we are falling back to the tightest '
                      'bounding box from the given initial_shape')
        tightest_bb = initial_shape.bounding_box()
        return self.fit_from_bb(image, tightest_bb, gt_shape=gt_shape, **kwargs)

    def fit_from_bb(self, image, bounding_box, gt_shape=None, **kwargs):
        algo_result = self.algorithm.run(image, bounding_box, gt_shape=gt_shape)

        # TODO: This should be a basic result instead.
        return MultiFitterResult(image, self, [algo_result],
                                 Affine.init_identity(2),
                                 gt_shape=gt_shape)
