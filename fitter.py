from __future__ import division
import abc

import numpy as np

from menpo.transform import Scale, AlignmentAffine
from menpo.fitmultilevel.functions import noisy_align, align_shape_with_bb
from cvpr2015.utils import fsmooth
from cvpr2015.deformablemodel.pdm import OrthoPDM
from cvpr2015.deformablemodel.transform import OrthoMDTransform
from .alabortcvpr2015.alabortcvpr2015.unified.algorithm import GlobalAAMInterface, PartsAAMInterface, AIC
from .result import AAMFitterResult



# Abstract Interface for AAM Fitters ------------------------------------------

class AAMFitter(object):
    r"""
    Abstract Interface for defining Active Appearance Models Fitters.

    Parameters
    -----------
    dm : :map:`AAM`
        The Active Appearance Model to be used.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, aam, n_shape, n_appearance):
        self.aam = aam
        self._algorithms = []

        self._check_n_shape(n_shape)
        self._check_n_appearance(n_appearance)

    @property
    def reference_shape(self):
        r"""
        The reference shape of the AAM.

        :type: :map:`PointCloud`
        """
        return self.aam.reference_shape

    @property
    def features(self):
        r"""
        The feature extracted at each pyramidal level during AAM building.
        Stored in ascending pyramidal order.

        :type: `list`
        """
        return self.aam.features

    @property
    def n_levels(self):
        r"""
        The number of pyramidal levels used during AAM building.

        :type: `int`
        """
        return self.aam.n_levels

    @property
    def sigma(self):
        return self.aam.sigma

    @property
    def scales(self):
        return self.aam.scales

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
        return self.aam.scale_features

    def _check_n_shape(self, n_shape):
        if n_shape is not None:
            if type(n_shape) is int or type(n_shape) is float:
                for sm in self.aam.shape_models:
                    sm.n_active_components = n_shape
            elif len(n_shape) == 1 and self.aam.n_levels > 1:
                for sm in self.aam.shape_models:
                    sm.n_active_components = n_shape[0]
            elif len(n_shape) == self.aam.n_levels:
                for sm, n in zip(self.aam.shape_models, n_shape):
                    sm.n_active_components = n
            else:
                raise ValueError('n_shape can be an integer or a float or None'
                                 'or a list containing 1 or {} of '
                                 'those'.format(self.aam.n_levels))

    def _check_n_appearance(self, n_appearance):
        if n_appearance is not None:
            if type(n_appearance) is int or type(n_appearance) is float:
                for am in self.aam.appearance_models:
                    am.n_active_components = n_appearance
            elif len(n_appearance) == 1 and self.aam.n_levels > 1:
                for am in self.aam.appearance_models:
                    am.n_active_components = n_appearance[0]
            elif len(n_appearance) == self.aam.n_levels:
                for am, n in zip(self.aam.appearance_models, n_appearance):
                    am.n_active_components = n
            else:
                raise ValueError('n_appearance can be an integer or a float '
                                 'or None or a list containing 1 or {} of '
                                 'those'.format(self.aam.n_levels))

    def fit(self, image, initial_shape, max_iters=50, gt_shape=None,
            **kwargs):
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
            image, initial_shape, gt_shape=gt_shape)

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
        fitter_result = AAMFitterResult(
            image, self, algorithm_results, affine_correction,
            gt_shape=gt_shape)

        return fitter_result

    def perturb_shape(self, gt_shape, noise_std=0.04, rotation=False):
        r"""
        Generates an initial shape by adding gaussian noise to the perfect
        similarity alignment between the ground truth and reference_shape.

        Parameters
        -----------
        gt_shape: :class:`menpo.shape.PointCloud`
            The ground truth shape.
        noise_std: float, optional
            The standard deviation of the gaussian noise used to produce the
            initial shape.

            Default: 0.04
        rotation: boolean, optional
            Specifies whether ground truth in-plane rotation is to be used
            to produce the initial shape.

            Default: False

        Returns
        -------
        initial_shape: :class:`menpo.shape.PointCloud`
            The initial shape.
        """
        reference_shape = self.reference_shape
        return noisy_align(reference_shape, gt_shape, noise_std=noise_std,
                           rotation=rotation).apply(reference_shape)

    def obtain_shape_from_bb(self, bounding_box):
        r"""
        Generates an initial shape given a bounding box detection.

        Parameters
        -----------
        bounding_box: (2, 2) ndarray
            The bounding box specified as:

                np.array([[x_min, y_min], [x_max, y_max]])

        Returns
        -------
        initial_shape: :class:`menpo.shape.PointCloud`
            The initial shape.
        """

        reference_shape = self.reference_shape
        return align_shape_with_bb(reference_shape,
                                   bounding_box).apply(reference_shape)

    def _prepare_image(self, image, initial_shape, gt_shape=None):
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

        # rescale image wrt the scale factor between reference_shape and
        # initial_shape
        image = image.rescale_to_reference_shape(self.reference_shape,
                                                 group='initial_shape')
        if self.sigma:
            image.pixels = fsmooth(image.pixels, self.sigma)

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

        max_iters = self._prepare_max_iters(max_iters)

        shape = initial_shape
        gt_shape = None
        algorithm_results = []
        for j, (i, alg, it, s) in enumerate(zip(images, self._algorithms,
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

    def _prepare_max_iters(self, max_iters):

        n_levels = self.n_levels

        # check max_iters parameter
        if type(max_iters) is int:
            max_iters = [np.round(max_iters/n_levels)
                         for _ in range(n_levels)]
        elif len(max_iters) == 1 and n_levels > 1:
            max_iters = [np.round(max_iters[0]/n_levels)
                         for _ in range(n_levels)]
        elif len(max_iters) != n_levels:
            raise ValueError('max_iters can be integer, integer list '
                             'containing 1 or {} elements or '
                             'None'.format(self.n_levels))
        return np.require(max_iters, dtype=np.int)


# Concrete Implementations of AAM Fitters -------------------------------------

class GlobalAAMFitter(AAMFitter):

    def __init__(self, aam, algorithm=AIC,
                 n_shape=None, n_appearance=None, **kwargs):

        super(GlobalAAMFitter, self).__init__(
            aam, n_shape, n_appearance)

        for j, (am, sm) in enumerate(zip(self.aam.appearance_models,
                                         self.aam.shape_models)):

            md_transform = OrthoMDTransform(
                sm, self.aam.transform,
                source=am.mean.landmarks['source'].lms,
                sigma2=am.noise_variance)

            self._algorithms.append(
                algorithm(GlobalAAMInterface, am, md_transform, **kwargs))


class PartsAAMFitter(AAMFitter):

    def __init__(self, parts_aam, algorithm=AIC_GN,
                 n_shape=None, n_appearance=None, **kwargs):

        super(PartsAAMFitter, self).__init__(
            parts_aam, n_shape, n_appearance)

        for j, (am, sm) in enumerate(zip(self.aam.appearance_models,
                                         self.aam.shape_models)):

            am.parts_shape = self.aam.parts_shape
            pdm = OrthoPDM(sm, sigma2=am.noise_variance)

            self._algorithms.append(
                algorithm(PartsAAMInterface, am, pdm, **kwargs))
