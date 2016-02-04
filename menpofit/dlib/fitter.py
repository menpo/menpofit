from __future__ import division
from functools import partial
import warnings
import dlib
from pathlib import Path
import numpy as np

from menpo.feature import no_op
from menpo.transform import Scale, AlignmentAffine
from menpo.base import name_of_callable

from menpofit import checks
from menpofit.compatibility import STRING_TYPES
from menpofit.fitter import (noisy_shape_from_bounding_box, MultiFitter,
                             generate_perturbations_from_gt)
from menpofit.builder import (scale_images, rescale_images_to_reference_shape,
                              compute_reference_shape)
from menpofit.result import MultiScaleNonParametricIterativeResult, Result

from .algorithm import DlibAlgorithm


class DlibERT(MultiFitter):
    r"""
    Class for training a multi-scale Ensemble of Regression Trees model.

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
    diagonal : `int` or ``None``, optional
        This parameter is used to normalize the scale of the training images
        so that the extracted features are in correspondence. The
        normalization is performed by rescaling all the training images so
        that the diagonal of their ground truth shapes' bounding boxes
        equals to the provided value. The reference scale gets rescaled as
        well. If ``None``, then the images are rescaled with respect to the
        reference shape's diagonal.
    scales : `tuple` of `float`, optional
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale.
    n_perturbations : `int` or ``None``, optional
        The number of perturbations to be generated from the provided
        bounding boxes.
    n_dlib_perturbations : `int` or ``None`` or `list` of those, optional
        The number of perturbations to be generated from the part of DLib. DLib
        calls this "oversampling amount". If `list`, it must specify a value per
        scale.
    perturb_from_gt_bounding_box : `function`, optional
        The function that will be used to generate the perturbations.
    n_iterations : `int` of `list` of `int`, optional
        The number of iterations (cascades) of each level. If `list`, it must
        specify a value per scale.
    feature_padding : `float` or `list` of `float`, optional
        When we randomly sample the pixels for the feature pool we do so in a
        box fit around the provided training landmarks. By default, this box
        is the tightest box that contains the landmarks. However, you can
        expand or shrink the size of the pixel sampling region by setting a
        different value of padding. To explain this precisely, for a padding
        of 0 we say that the pixels are sampled from a box of size 1x1.  The
        padding value is added to each side of the box.  So a padding of 0.5
        would cause the algorithm to sample pixels from a box that was 2x2,
        effectively multiplying the area pixels are sampled from by 4.
        Similarly, setting the padding to -0.2 would cause it to sample from
        a box 0.6x0.6 in size. If `list`, it must specify a value per scale.
    n_pixel_pairs : `int` or `list` of `int`, optional
        `P` parameter from [1]. At each level of the cascade we randomly sample
        pixels from the image. These pixels are used to generate features for
        the random trees. So in general larger settings of this parameter
        give better accuracy but make the algorithm run slower. If `list`, it
        must specify a value per scale.
    distance_prior_weighting : `float` or `list` of `float`, optional
        To decide how to split nodes in the regression trees the algorithm
        looks at pairs of pixels in the image. These pixel pairs are sampled
        randomly but with a preference for selecting pixels that are near
        each other. This parameter controls this "nearness" preference. In
        particular, smaller values will make the algorithm prefer to select
        pixels close together and larger values will make it care less about
        picking nearby pixel pairs. Note that this is the inverse of how it is
        defined in [1]. For this object, you should think of
        `distance_prior_weighting` as "the fraction of the bounding box will
        we traverse to find a neighboring pixel".  Nominally, this is
        normalized between 0 and 1.  So reasonable settings are values in the
        range (0, 1). If `list`, it must specify a value per scale.
    regularisation_weight : `float` or `list` of `float`, optional
        Boosting regularization parameter - `nu` from [1]. Larger values may
        cause overfitting but improve performance on training data. If `list`,
        it must specify a value per scale.
    n_split_tests : `int` or `list` of `int`, optional
        When generating the random trees we randomly sample `n_split_tests`
        possible split features at each node and pick the one that gives the
        best split.  Larger values of this parameter will usually give more
        accurate outputs but take longer to train. It is equivalent of `S`
        from [1]. If `list`, it must specify a value per scale.
    n_trees : `int` or `list` of `int`, optional
        Number of trees created for each cascade. The total number of trees
        in the learned model is equal n_trees * n_tree_levels. Equivalent to
        `K` from [1]. If `list`, it must specify a value per scale.
    n_tree_levels : `int` or `list` of `int`, optional
        The number of levels in the tree (depth of tree). In particular,
        there are pow(2, n_tree_levels) leaves in each tree. Equivalent to
        `F` from [1]. If `list`, it must specify a value per scale.
    verbose : `bool`, optional
        If ``True``, then the progress of building ERT will be printed.

    References
    ----------
    .. [1] V. Kazemi, and J. Sullivan. "One millisecond face alignment with
        an ensemble of regression trees." Proceedings of the IEEE Conference
        on Computer Vision and Pattern Recognition. 2014.
    """
    def __init__(self, images, group=None, bounding_box_group_glob=None,
                 reference_shape=None, diagonal=None, scales=(0.5, 1.0),
                 n_perturbations=30, n_dlib_perturbations=1,
                 perturb_from_gt_bounding_box=noisy_shape_from_bounding_box,
                 n_iterations=10, feature_padding=0, n_pixel_pairs=400,
                 distance_prior_weighting=0.1, regularisation_weight=0.1,
                 n_split_tests=20, n_trees=500, n_tree_levels=5, verbose=False):
        checks.check_diagonal(diagonal)
        self.diagonal = diagonal
        self.scales = checks.check_scales(scales)
        # Dummy option that is required by _prepare_image of MultiFitter.
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
            # P parameter from Kazemi paper
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
                    for k, bb in enumerate(bboxes):
                        bboxes[k] = transform.apply(bb)

    @property
    def n_scales(self):
        """
        Returns the number of scales.

        :type: `int`
        """
        return len(self.scales)

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
        return MultiScaleNonParametricIterativeResult(
                algorithm_results, self.scales,
                affine_correction=affine_correction, image=image,
                gt_shape=gt_shape)

    def fit_from_shape(self, image, initial_shape, gt_shape=None,
                       crop_image=None):
        r"""
        Fits the model to an image. Note that it is not possible to
        initialise the fitting process from a shape. Thus, this method raises a
        warning and calls `fit_from_bb` with the bounding box of the provided
        `initial_shape`.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image to be fitted.
        initial_shape : :map:`PointCloud`
            The initial shape estimate from which the fitting procedure
            will start. Note that the shape won't actually be used, only its
            bounding box.
        gt_shape : class : :map:`PointCloud`, optional
            The ground truth shape associated to the image.
        crop_image : ``None`` or `float`, optional
            If `float`, it specifies the proportion of the border wrt the
            initial shape to which the image will be internally cropped around
            the initial shape range. If ``None``, no cropping is performed.
            This will limit the fitting algorithm search region but is
            likely to speed up its running time, specially when the
            modeled object occupies a small portion of the image.

        Returns
        -------
        fitting_result: `menpofit.result.MultiScaleNonParametricIterativeResult`
            The result of the fitting procedure.
        """
        warnings.warn('Fitting from an initial shape is not supported by '
                      'Dlib - therefore we are falling back to the tightest '
                      'bounding box from the given initial_shape')
        tightest_bb = initial_shape.bounding_box()
        return self.fit_from_bb(image, tightest_bb, gt_shape=gt_shape,
                                crop_image=crop_image)

    def fit_from_bb(self, image, bounding_box, gt_shape=None, crop_image=None):
        r"""
        Fits the model to an image given an initial bounding box.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image to be fitted.
        bounding_box : `menpo.shape.PointDirectedGraph`
            The initial bounding box from which the fitting procedure
            will start.
        gt_shape : class : :map:`PointCloud`, optional
            The ground truth shape associated to the image.
        crop_image : ``None`` or `float`, optional
            If `float`, it specifies the proportion of the border wrt the
            initial shape to which the image will be internally cropped around
            the initial shape range. If ``None``, no cropping is performed.
            This will limit the fitting algorithm search region but is
            likely to speed up its running time, specially when the
            modeled object occupies a small portion of the image.

        Returns
        -------
        fitting_result: `menpofit.result.MultiScaleNonParametricIterativeResult`
            The result of the fitting procedure.
        """
        # generate the list of images to be fitted
        images, bounding_boxes, gt_shapes = self._prepare_image(
            image, bounding_box, gt_shape=gt_shape, crop_image=crop_image)

        # work out the affine transform between the initial shape of the
        # highest pyramidal level and the initial shape of the original image
        affine_correction = AlignmentAffine(bounding_boxes[-1], bounding_box)

        # run multilevel fitting
        algorithm_results = self._fit(images, bounding_boxes[0],
                                      gt_shapes=gt_shapes)

        # build multilevel fitting result
        fitter_result = self._fitter_result(
            image, algorithm_results, affine_correction, gt_shape=gt_shape)

        return fitter_result

    def __str__(self):
        if self.diagonal is not None:
            diagonal = self.diagonal
        else:
            y, x = self.reference_shape.range()
            diagonal = np.sqrt(x ** 2 + y ** 2)

        # Compute scale info strings
        scales_info = []
        lvl_str_tmplt = r"""   - Scale {0}
     - Cascade of depth {1}
     - Each tree has depth {2}
     - {3} trees per cascade level
     - Regularisation parameter: {4:.1f}
     - Feature pool of size {5} and padding {6:.1f}
     - Lambda: {7:.1f}
     - {8} split tests
     - {9} oversampling perturbations"""
        for k, s in enumerate(self.scales):
            scales_info.append(lvl_str_tmplt.format(
                    s,
                    self._dlib_options_templates[k].cascade_depth,
                    self._dlib_options_templates[k].tree_depth,
                    self._dlib_options_templates[k].num_trees_per_cascade_level,
                    self._dlib_options_templates[k].nu,
                    self._dlib_options_templates[k].feature_pool_size,
                    self._dlib_options_templates[k].feature_pool_region_padding,
                    self._dlib_options_templates[k].lambda_param,
                    self._dlib_options_templates[k].num_test_splits,
                    self._dlib_options_templates[k].oversampling_amount))
        scales_info = '\n'.join(scales_info)

        cls_str = r"""{class_title}
 - Images scaled to diagonal: {diagonal:.2f}
 - {n_pert} perturbations with {pert}
 - Scales: {scales}
{scales_info}
""".format(class_title='Ensemble of Regression Trees',
           diagonal=diagonal,
           n_pert=self.n_perturbations,
           pert=name_of_callable(self._perturb_from_gt_bounding_box),
           scales=self.scales,
           scales_info=scales_info)
        return cls_str


class DlibWrapper(object):
    r"""
    Wrapper class for fitting a pre-trained ERT model. Pre-trained models are
    provided by the official DLib package (http://dlib.net/).

    Parameters
    ----------
    model : `Path` or `str`
        Path to the pre-trained model.
    """
    def __init__(self, model):
        if isinstance(model, STRING_TYPES) or isinstance(model, Path):
            m_path = Path(model)
            if not Path(m_path).exists():
                raise ValueError('Model {} does not exist.'.format(m_path))
            model = dlib.shape_predictor(str(m_path))

        # Dlib doesn't expose any information about how the model was built,
        # so we just create dummy options
        self.algorithm = DlibAlgorithm(dlib.shape_predictor_training_options(),
                                       n_iterations=0)
        self.algorithm.dlib_model = model
        self.scales = [1]

    def fit_from_shape(self, image, initial_shape, gt_shape=None):
        r"""
        Fits the model to an image. Note that it is not possible to
        initialise the fitting process from a shape. Thus, this method raises a
        warning and calls `fit_from_bb` with the bounding box of the provided
        `initial_shape`.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image to be fitted.
        initial_shape : `menpo.shape.PointCloud`
            The initial shape estimate from which the fitting procedure
            will start. Note that the shape won't actually be used, only its
            bounding box.
        gt_shape : :map:`PointCloud`
            The ground truth shape associated to the image.

        Returns
        -------
        fitting_result: `menpofit.result.Result`
            The result of the fitting procedure.
        """
        warnings.warn('Fitting from an initial shape is not supported by '
                      'Dlib - therefore we are falling back to the tightest '
                      'bounding box from the given initial_shape')
        tightest_bb = initial_shape.bounding_box()
        return self.fit_from_bb(image, tightest_bb, gt_shape=gt_shape)

    def fit_from_bb(self, image, bounding_box, gt_shape=None):
        r"""
        Fits the model to an image given an initial bounding box.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image to be fitted.
        bounding_box : `menpo.shape.PointDirectedGraph`
            The initial bounding box.
        gt_shape : :map:`PointCloud`
            The ground truth shape associated to the image.

        Returns
        -------
        fitting_result: `menpofit.result.Result`
            The result of the fitting procedure.
        """
        # We get back a NonParametricIterativeResult with one iteration,
        # which is pointless. Simply convert it to a Result instance without
        # passing in an initial shape.
        fit_result = self.algorithm.run(image, bounding_box, gt_shape=gt_shape)
        return Result(final_shape=fit_result.final_shape, image=image,
                      initial_shape=None, gt_shape=gt_shape)

    def __str__(self):
        return "Pre-trained DLib Ensemble of Regression Trees model"
