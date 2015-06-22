from __future__ import division
import abc
import numpy as np
from menpo.shape import PointCloud
from menpo.transform import Scale, AlignmentAffine, AlignmentSimilarity
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

    @abc.abstractproperty
    def algorithms(self):
        pass

    # @abc.abstractproperty
    # def reference_shape(self):
    #     pass
    #
    # @abc.abstractproperty
    # def features(self):
    #     pass
    #
    # @abc.abstractproperty
    # def scales(self):
    #     pass
    #
    # @abc.abstractproperty
    # def scale_features(self):
    #     pass

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
        from copy import deepcopy
        scales = deepcopy(self.scales)
        scales.reverse()
        images = []
        for j, s in enumerate(scales):
            if j == 0:
                # compute features at highest level
                feature_image = self.features(image)
            elif self.scale_features:
                # scale features at other levels
                feature_image = images[0].rescale(s)
            else:
                # scale image and compute features at other levels
                scaled_image = image.rescale(s)
                feature_image = self.features(scaled_image)
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
                Scale(self.scales[j+1]/s,
                      n_dims=shape.n_dims).apply_inplace(shape)

        return algorithm_results

    @abc.abstractmethod
    def _fitter_result(self, image, algorithm_results, affine_correction,
                       gt_shape=None):
        pass


# TODO: correctly implement initialization from bounding box
# TODO: document me!
class ModelFitter(MultiFitter):
    r"""
    """
    def __init__(self, model):
        self._model = model

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

    # TODO: Bounding boxes should be PointGraphs
    def noisy_shape_from_bounding_box(self, bounding_box, noise_std=0.04,
                                      rotation=False):
        transform = noisy_align(AlignmentSimilarity,
                                self.reference_bounding_box, bounding_box,
                                noise_std=noise_std, rotation=rotation)
        return transform.apply(self.reference_shape)

    def noisy_shape_from_shape(self, shape, noise_std=0.04, rotation=False):
        return self.noisy_shape_from_bounding_box(
            shape.bounding_box(), noise_std=noise_std, rotation=rotation)


# TODO: document me!
def noisy_align(alignment_transform_cls, source, target, noise_std=0.1,
                **kwargs):
    r"""
    """
    noise = noise_std * target.range() * np.random.randn(target.n_points,
                                                         target.n_dims)
    noisy_target = PointCloud(target.points + noise)
    return alignment_transform_cls(source, noisy_target, **kwargs)


# TODO: document me!
def align_shape_with_bounding_box(alignment_transform_cls, shape,
                                  bounding_box, **kwargs):
    r"""
    """
    shape_bb = shape.bounding_box()
    return alignment_transform_cls(shape_bb, bounding_box, **kwargs)

