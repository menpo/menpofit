from __future__ import division
import numpy as np
from copy import deepcopy
from menpo.shape import PointCloud
from menpo.transform import (
    scale_about_centre, rotate_ccw_about_centre, Translation,
    Scale, Similarity, AlignmentAffine, AlignmentSimilarity)
import menpofit.checks as checks


# TODO: document me!
class MultiFitter(object):
    r"""
    """
    @property
    def n_levels(self):
        r"""
        The number of pyramidal levels used during alignment.

        :type: `int`
        """
        return len(self.scales)

    def fit(self, image, initial_shape, max_iters=50, gt_shape=None,
            crop_image=0.5, **kwargs):
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

        # detach added landmarks from image
        del image.landmarks['initial_shape']
        if gt_shape:
            del image.landmarks['gt_shape']

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
            image = image.crop_to_landmarks_proportion(crop_image,
                                                       group='initial_shape')

        # rescale image wrt the scale factor between reference_shape and
        # initial_shape
        image = image.rescale_to_reference_shape(self.reference_shape,
                                                 group='initial_shape')

        # obtain image representation
        images = []
        for j, s in enumerate(self.scales[::-1]):
            if j == 0:
                # compute features at highest level
                feature_image = self.features[j](image)
            elif self.scale_features:
                # scale features at other levels
                feature_image = images[0].rescale(s)
            else:
                # scale image and compute features at other levels
                scaled_image = image.rescale(s)
                feature_image = self.features[j](scaled_image)
            images.append(feature_image)
        images.reverse()

        # get initial shapes per level
        initial_shapes = [i.landmarks['initial_shape'].lms for i in images]

        # get ground truth shapes per level
        if gt_shape:
            gt_shapes = [i.landmarks['gt_shape'].lms for i in images]
        else:
            gt_shapes = None

        return images, initial_shapes, gt_shapes

    def _fit(self, images, initial_shape, gt_shapes=None, max_iters=50,
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

            Default: None
        max_iters: int or list, optional
            The maximum number of iterations.
            If int, then this will be the overall maximum number of iterations
            for all the pyramidal levels.
            If list, then a maximum number of iterations is specified for each
            pyramidal level.

            Default: 50

        Returns
        -------
        algorithm_results: :class:`menpo.fg2015.fittingresult.FittingResult` list
            The fitting object containing the state of the whole fitting
            procedure.
        """
        max_iters = checks.check_max_iters(max_iters, self.n_levels)
        shape = initial_shape
        gt_shape = None
        algorithm_results = []
        for j, (i, alg, it, s) in enumerate(zip(images, self.algorithms,
                                                max_iters, self.scales)):
            if gt_shapes:
                gt_shape = gt_shapes[j]

            algorithm_result = alg.run(i, shape, gt_shape=gt_shape,
                                       max_iters=it, **kwargs)
            algorithm_results.append(algorithm_result)

            shape = algorithm_result.final_shape
            if s != self.scales[-1]:
                shape = Scale(self.scales[j+1]/s,
                              n_dims=shape.n_dims).apply(shape)

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
    def reference_bounding_box(self):
        return self.reference_shape.bounding_box()

    @property
    def features(self):
        r"""
        The feature extracted at each pyramidal level during AAM building.
        Stored in ascending pyramidal order.

        :type: `list`
        """
        return self._model.features

    @property
    def scales(self):
        return self._model.scales

    @property
    def scale_features(self):
        r"""
        Flag that defined the nature of Gaussian pyramid used to build the
        AAM.
        If ``True``, the feature space is computed once at the highest scale
        and the Gaussian pyramid is applied to the feature images.
        If ``False``, the Gaussian pyramid is applied to the original images
        and features are extracted at each level.

        :type: `boolean`
        """
        return self._model.scale_features

    def _check_n_shape(self, n_shape):
        if n_shape is not None:
            if type(n_shape) is int or type(n_shape) is float:
                for sm in self._model.shape_models:
                    sm.n_active_components = n_shape
            elif len(n_shape) == 1 and self._model.n_levels > 1:
                for sm in self._model.shape_models:
                    sm.n_active_components = n_shape[0]
            elif len(n_shape) == self._model.n_levels:
                for sm, n in zip(self._model.shape_models, n_shape):
                    sm.n_active_components = n
            else:
                raise ValueError('n_shape can be an integer or a float or None'
                                 'or a list containing 1 or {} of '
                                 'those'.format(self._model.n_levels))

    def noisy_shape_from_bounding_box(self, bounding_box, noise_std=0.05):
        transform = noisy_alignment_similarity_transform(
            self.reference_bounding_box, bounding_box, noise_std=noise_std)
        return transform.apply(self.reference_shape)

    def noisy_shape_from_shape(self, shape, noise_std=0.05):
        return self.noisy_shape_from_bounding_box(
            shape.bounding_box(), noise_std=noise_std)


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
        noise_percentage= [noise_percentage] * 3
    elif len(noise_percentage) == 1:
        noise_percentage *= 3

    similarity = AlignmentSimilarity(source, target, rotation=rotation)

    if noise_type is 'normal':
        #
        s = noise_percentage[0] * (0.5 / 3) * np.asscalar(np.random.randn(1))
        #
        r = noise_percentage[1] * (180 / 3) * np.asscalar(np.random.randn(1))
        #
        t = noise_percentage[2] * (target.range() / 3) * np.random.randn(2)

        s = scale_about_centre(target, 1 + s)
        r = rotate_ccw_about_centre(target, r)
        t = Translation(t, source.n_dims)
    elif noise_type is 'uniform':
        #
        s = noise_percentage[0] * 0.5 * (2 * np.asscalar(np.random.randn(1)) - 1)
        #
        r = noise_percentage[1] * 180 * (2 * np.asscalar(np.random.rand(1)) - 1)
        #
        t = noise_percentage[2] * target.range() * (2 * np.random.rand(2) - 1)

        s = scale_about_centre(target, 1. + s)
        r = rotate_ccw_about_centre(target, r)
        t = Translation(t, source.n_dims)

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
                                  noise_percentage=0.1, rotation=False):
    transform = noisy_alignment_similarity_transform(
        shape.bounding_box(), bounding_box, noise_type=noise_type,
        noise_percentage=noise_percentage, rotation=rotation)
    return transform.apply(shape)


def noisy_shape_from_shape(reference_shape, shape, noise_type='uniform',
                           noise_percentage=0.1, rotation=False):
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
