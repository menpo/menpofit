from __future__ import division
from functools import partial
import numpy as np
from menpo.shape import PointCloud
from menpo.transform import (
    scale_about_centre, rotate_ccw_about_centre, Translation,
    Scale, AlignmentAffine, AlignmentSimilarity)
import menpofit.checks as checks
from menpofit.visualize import print_progress


def noisy_alignment_similarity_transform(source, target, noise_type='uniform',
                                         noise_percentage=0.1, rotation=False):
    r"""
    Constructs and perturbs the optimal similarity transform between source
    and target by adding noise to its parameters.

    Parameters
    ----------
    source: :class:`menpo.shape.PointCloud`
        The source pointcloud instance used in the alignment
    target: :class:`menpo.shape.PointCloud`
        The target pointcloud instance used in the alignment
    noise_type: str, optional
        The type of noise to be added, 'uniform' or 'gaussian'.
    noise_percentage: 0 < float < 1 or triplet of 0 < float < 1, optional
        The standard percentage of noise to be added. If float the same amount
        of noise is applied to the scale, rotation and translation
        parameters of the true similarity transform. If triplet of
        floats, the first, second and third elements denote the amount of
        noise to be applied to the scale, rotation and translation
        parameters respectively.
    rotation: boolean, optional
        If False rotation is not considered when computing the optimal
        similarity transform between source and target.

    Returns
    -------
    noisy_alignment_similarity_transform : :class: `menpo.transform.Similarity`
        The noisy Similarity Transform between source and target.
    """
    if isinstance(noise_percentage, float):
        noise_percentage = [noise_percentage] * 3
    elif len(noise_percentage) == 1:
        noise_percentage *= 3

    similarity = AlignmentSimilarity(source, target, rotation=rotation)

    if noise_type is 'gaussian':
        s = noise_percentage[0] * (0.5 / 3) * np.asscalar(np.random.randn(1))
        r = noise_percentage[1] * (180 / 3) * np.asscalar(np.random.randn(1))
        t = noise_percentage[2] * (target.range() / 3) * np.random.randn(2)

        s = scale_about_centre(target, 1 + s)
        r = rotate_ccw_about_centre(target, r)
        t = Translation(t, source.n_dims)
    elif noise_type is 'uniform':
        s = noise_percentage[0] * 0.5 * (2 * np.asscalar(np.random.randn(1)) - 1)
        r = noise_percentage[1] * 180 * (2 * np.asscalar(np.random.rand(1)) - 1)
        t = noise_percentage[2] * target.range() * (2 * np.random.rand(2) - 1)

        s = scale_about_centre(target, 1. + s)
        r = rotate_ccw_about_centre(target, r)
        t = Translation(t, source.n_dims)
    else:
        raise ValueError('Unexpected noise type. '
                         'Supported values are {gaussian, uniform}')

    return similarity.compose_after(t.compose_after(s.compose_after(r)))


def noisy_target_alignment_transform(source, target,
                                     alignment_transform_cls=AlignmentAffine,
                                     noise_std=0.1, **kwargs):
    r"""
    Constructs and the optimal alignment transform between the source and
    a noisy version of the target obtained by adding white noise to each of
    its points.

    Parameters
    ----------
    source: :class:`menpo.shape.PointCloud`
        The source pointcloud instance used in the alignment
    target: :class:`menpo.shape.PointCloud`
        The target pointcloud instance used in the alignment
    alignment_transform_cls: :class:`menpo.transform.Alignment`, optional
        The alignment transform class used to perform the alignment.
    noise_std: float or triplet of floats, optional
        The standard deviation of the white noise to be added to each one of
        the target points.

    Returns
    -------
    noisy_transform : :class: `menpo.transform.Alignment`
        The noisy Similarity Transform
    """
    noise = noise_std * target.range() * np.random.randn(target.n_points,
                                                         target.n_dims)
    noisy_target = PointCloud(target.points + noise)
    return alignment_transform_cls(source, noisy_target, **kwargs)


def noisy_shape_from_bounding_box(shape, bounding_box, noise_type='uniform',
                                  noise_percentage=0.05, rotation=False):
    transform = noisy_alignment_similarity_transform(
        shape.bounding_box(), bounding_box, noise_type=noise_type,
        noise_percentage=noise_percentage, rotation=rotation)
    return transform.apply(shape)


def noisy_shape_from_shape(reference_shape, shape, noise_type='uniform',
                           noise_percentage=0.05, rotation=False):
    transform = noisy_alignment_similarity_transform(
        reference_shape, shape, noise_type=noise_type,
        noise_percentage=noise_percentage, rotation=rotation)
    return transform.apply(reference_shape)


def align_shape_with_bounding_box(shape, bounding_box,
                                  alignment_transform_cls=AlignmentSimilarity,
                                  **kwargs):
    r"""
    Aligns the shape with the bounding box using a particular ali .

    Parameters
    ----------
    source: :class:`menpo.shape.PointCloud`
        The shape instance used in the alignment.
    bounding_box: :class:`menpo.shape.PointCloud`
        The bounding box instance used in the alignment.
    alignment_transform_cls: :class:`menpo.transform.Alignment`, optional
        The class of the alignment transform used to perform the alignment.

    Returns
    -------
    noisy_transform : :class: `menpo.transform.Alignment`
        The noisy Alignment Transform
    """
    shape_bb = shape.bounding_box()
    transform = alignment_transform_cls(shape_bb, bounding_box, **kwargs)
    return transform.apply(shape)


class MultiFitter(object):
    r"""
    """
    @property
    def n_scales(self):
        r"""
        The number of scales used during alignment.

        :type: `int`
        """
        return len(self.scales)

    def fit_from_shape(self, image, initial_shape, max_iters=20, gt_shape=None,
                       crop_image=None, **kwargs):
        r"""
        Fits the multilevel fitter to an image.

        Parameters
        -----------
        image: :map:`Image` or subclass
            The image to be fitted.
        initial_shape: :map:`PointCloud`
            The initial shape estimate from which the fitting procedure
            will start.
        max_iters: `int` or `list` of `int`, optional
            The maximum number of iterations.
            If `int`, specifies the overall maximum number of iterations.
            If `list` of `int`, specifies the maximum number of iterations per
            level.
        gt_shape: :map:`PointCloud`
            The ground truth shape associated to the image.
        crop_image: `None` or float`, optional
            If `float`, it specifies the proportion of the border wrt the
            initial shape to which the image will be internally cropped around
            the initial shape range.
            If `None`, no cropping is performed.

            This will limit the fitting algorithm search region but is
            likely to speed up its running time, specially when the
            modeled object occupies a small portion of the image.
        **kwargs:
            Additional keyword arguments that can be passed to specific
            implementations of ``_fit`` method.

        Returns
        -------
        multi_fitting_result: :map:`MultilevelFittingResult`
            The multilevel fitting result containing the result of
            fitting procedure.
        """
        # generate the list of images to be fitted
        images, initial_shapes, gt_shapes = self._prepare_image(
            image, initial_shape, gt_shape=gt_shape, crop_image=crop_image)

        # work out the affine transform between the initial shape of the
        # highest pyramidal level and the initial shape of the original image
        affine_correction = AlignmentAffine(initial_shapes[-1], initial_shape)

        # run multilevel fitting
        algorithm_results = self._fit(images, initial_shapes[0],
                                      max_iters=max_iters,
                                      gt_shapes=gt_shapes, **kwargs)

        # build multilevel fitting result
        fitter_result = self._fitter_result(
            image, algorithm_results, affine_correction, gt_shape=gt_shape)

        return fitter_result

    def fit_from_bb(self, image, bounding_box, max_iters=20, gt_shape=None,
                    crop_image=None, **kwargs):
        initial_shape = align_shape_with_bounding_box(self.reference_shape,
                                                      bounding_box)
        return self.fit_from_shape(image, initial_shape, max_iters=max_iters,
                                   gt_shape=gt_shape, crop_image=crop_image,
                                   **kwargs)

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
        # Attach landmarks to the image
        image.landmarks['__initial_shape'] = initial_shape
        if gt_shape:
            image.landmarks['__gt_shape'] = gt_shape

        if crop_image:
            # If specified, crop the image
            image = image.crop_to_landmarks_proportion(crop_image,
                                                       group='__initial_shape')

        # Rescale image wrt the scale factor between reference_shape and
        # initial_shape
        image = image.rescale_to_pointcloud(self.reference_shape,
                                            group='__initial_shape')

        # Compute image representation
        images = []
        for i in range(self.n_scales):
            # Handle features
            if i == 0 or self.holistic_features[i] is not self.holistic_features[i - 1]:
                # Compute features only if this is the first pass through
                # the loop or the features at this scale are different from
                # the features at the previous scale
                feature_image = self.holistic_features[i](image)

            # Handle scales
            if self.scales[i] != 1:
                # Scale feature images only if scale is different than 1
                scaled_image = feature_image.rescale(self.scales[i])
            else:
                scaled_image = feature_image

            # Add scaled image to list
            images.append(scaled_image)

        # Get initial shapes per level
        initial_shapes = [i.landmarks['__initial_shape'].lms for i in images]

        # Get ground truth shapes per level
        if gt_shape:
            gt_shapes = [i.landmarks['__gt_shape'].lms for i in images]
        else:
            gt_shapes = None

        # detach added landmarks from image
        del image.landmarks['__initial_shape']
        if gt_shape:
            del image.landmarks['__gt_shape']

        return images, initial_shapes, gt_shapes

    def _fit(self, images, initial_shape, gt_shapes=None, max_iters=20,
             **kwargs):
        r"""
        Fits the fitter to the multilevel pyramidal images.

        Parameters
        -----------
        images: :class:`menpo.image.masked.MaskedImage` list
            The images to be fitted.
        initial_shape: :class:`menpo.shape.PointCloud`
            The initial shape from which the fitting will start.
        gt_shapes: :class:`menpo.shape.PointCloud` list, optional
            The original ground truth shapes associated to the multilevel
            images.
        max_iters: int or list, optional
            The maximum number of iterations.
            If int, then this will be the overall maximum number of iterations
            for all the pyramidal levels.
            If list, then a maximum number of iterations is specified for each
            pyramidal level.

        Returns
        -------
        algorithm_results: :class:`FittingResult` list
            The fitting object containing the state of the whole fitting
            procedure.
        """
        # Perform check
        max_iters = checks.check_max_iters(max_iters, self.n_scales)

        # Set initial and ground truth shapes
        shape = initial_shape
        gt_shape = None

        # Initialize list of algorithm results
        algorithm_results = []
        for i in range(self.n_scales):
            # Handle ground truth shape
            if gt_shapes is not None:
                gt_shape = gt_shapes[i]

            # Run algorithm
            algorithm_result = self.algorithms[i].run(images[i], shape,
                                                      gt_shape=gt_shape,
                                                      max_iters=max_iters[i],
                                                      **kwargs)
            # Add algorithm result to the list
            algorithm_results.append(algorithm_result)

            # Prepare this scale's final shape for the next scale
            shape = algorithm_result.final_shape
            if self.scales[i] != self.scales[-1]:
                shape = Scale(self.scales[i + 1] / self.scales[i],
                              n_dims=shape.n_dims).apply(shape)

        # Return list of algorithm results
        return algorithm_results


# TODO: document me!
class ModelFitter(MultiFitter):
    r"""
    """
    @property
    def reference_shape(self):
        r"""
        The reference shape of the AAM.

        :type: :map:`PointCloud`
        """
        return self._model.reference_shape

    @property
    def holistic_features(self):
        r"""
        """
        return self._model.holistic_features

    @property
    def scales(self):
        return self._model.scales

    def _check_n_shape(self, n_shape):
        checks.set_models_components(self._model.shape_models, n_shape)

    def perturb_from_bb(self, gt_shape, bb,
                        perturb_func=noisy_shape_from_bounding_box):
        return perturb_func(gt_shape, bb)

    def perturb_from_gt_bb(self, gt_bb,
                           perturb_func=noisy_shape_from_bounding_box):
        return perturb_func(gt_bb, gt_bb)


def generate_perturbations_from_gt(images, n_perturbations, perturb_func,
                                   gt_group=None, bb_group_glob=None,
                                   verbose=False):

    if bb_group_glob is None:
        bb_generator = lambda im: [im.landmarks[gt_group].lms.bounding_box()]
        n_bbs = 1
    else:
        def bb_glob(im):
            for k, v in im.landmarks.items_matching(bb_group_glob):
                yield v.lms.bounding_box()
        bb_generator = bb_glob
        n_bbs = len(list(bb_glob(images[0])))

    if n_bbs == 0:
        raise ValueError('Must provide a valid bounding box glob - no bounding '
                         'boxes matched the following '
                         'glob: {}'.format(bb_group_glob))

    # If we have multiple boxes - we didn't just throw them away, we re-add them
    # to the end
    if bb_group_glob is not None:
        msg = '- Generating {0} ({1} perturbations * {2} provided boxes) new ' \
              'initial bounding boxes + {2} provided boxes per image'.format(
            n_perturbations * n_bbs, n_perturbations, n_bbs)
    else:
        msg = '- Generating {} new bounding boxes directly from the ' \
              'ground truth shape'.format(n_perturbations)

    wrap = partial(print_progress, prefix=msg, verbose=verbose)
    for im in wrap(images):
        gt_s = im.landmarks[gt_group].lms.bounding_box()

        k = 0
        for bb in bb_generator(im):
            for _ in range(n_perturbations):
                p_s = perturb_func(gt_s, bb).bounding_box()
                perturb_bbox_group = '__generated_bb_{}'.format(k)
                im.landmarks[perturb_bbox_group] = p_s
                k += 1

            if bb_group_glob is not None:
                perturb_bbox_group = '__generated_bb_{}'.format(k)
                im.landmarks[perturb_bbox_group] = bb
                k += 1

        if im.has_landmarks_outside_bounds:
            im.constrain_landmarks_to_bounds()

    generated_bb_func = lambda x: [v.lms for k, v in x.landmarks.items_matching(
        '__generated_bb_*')]
    return generated_bb_func
