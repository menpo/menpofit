from __future__ import division
import abc
from menpo.transform import AlignmentAffine, Scale, AlignmentSimilarity
import numpy as np
from menpo.shape import PointCloud
from menpofit.base import is_pyramid_on_features, pyramid_of_feature_images, \
    noisy_align
from menpofit.fittingresult import MultilevelFittingResult


class Fitter(object):
    r"""
    Abstract interface that all :map:`Fitter` objects must implement.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _set_up(self, **kwargs):
        r"""
        Abstract method that sets up the fitter object.
        """
        pass

    def fit(self, image, initial_parameters, gt_shape=None, **kwargs):
        r"""
        Fits the fitter to an image.

        Parameters
        -----------
        image: :map:`Image` or subclass
            The image to be fitted.
        initial_parameters: list
            The initial parameters of the model.
        gt_shape: :map:`PointCloud`
            The ground truth shape associated to the image.

        Returns
        -------
        fitting_result: :map:`FittingResult`
            The fitting result containing the result of fitting procedure.
        """
        fitting_result = self._create_fitting_result(
            image, initial_parameters, gt_shape=gt_shape)
        return self._fit(fitting_result, **kwargs)

    @abc.abstractmethod
    def _create_fitting_result(self, **kwargs):
        r"""
        Abstract method that defines the fitting result object associated to
        the fitter object.
        """
        pass

    @abc.abstractmethod
    def _fit(self, **kwargs):
        r"""
        Abstract method implements a particular alignment algorithm.
        """
        pass

    def get_parameters(self, shape):
        r"""
        Abstract method that gets the parameters.
        """
        pass


class MultilevelFitter(Fitter):
    r"""
    Abstract interface that all :map:`MultilevelFitter` must implement.
    """

    @abc.abstractproperty
    def reference_shape(self):
        r"""
        The reference shape of the multilevel fitter.
        """
        pass

    @abc.abstractproperty
    def features(self):
        r"""
        Returns the feature computation functions applied at each pyramidal
        level.
        """
        pass

    @abc.abstractproperty
    def n_levels(self):
        r"""
        The number of pyramidal levels.
        """
        pass

    @abc.abstractproperty
    def downscale(self):
        r"""
        The downscale factor used by the multiple fitter.
        """
        pass

    @property
    def pyramid_on_features(self):
        r"""
        Returns True if the pyramid is computed on the feature image and False
        if it is computed on the original (intensities) image and features are
        extracted at each level.
        """
        return is_pyramid_on_features(self.features)

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
        fitting_results = self._fit(images, initial_shapes[0],
                                    max_iters=max_iters,
                                    gt_shapes=gt_shapes, **kwargs)

        # build multilevel fitting result
        multi_fitting_result = self._create_fitting_result(
            image, fitting_results, affine_correction, gt_shape=gt_shape)

        return multi_fitting_result

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

        images = list(reversed(list(pyramid_of_feature_images(
            self.n_levels, self.downscale, self.features, image))))

        # get initial shapes per level
        initial_shapes = [i.landmarks['initial_shape'].lms for i in images]

        # get ground truth shapes per level
        if gt_shape:
            gt_shapes = [i.landmarks['gt_shape'].lms for i in images]
            del image.landmarks['gt_shape']
        else:
            gt_shapes = None

        return images, initial_shapes, gt_shapes

    def _create_fitting_result(self, image, fitting_results, affine_correction,
                               gt_shape=None):
        r"""
        Creates the :class: `menpo.aam.fitting.MultipleFitting` object
        associated with a particular Fitter object.

        Parameters
        -----------
        image: :class:`menpo.image.masked.MaskedImage`
            The original image to be fitted.
        fitting_results: :class:`menpo.fit.fittingresult.FittingResultList`
            A list of basic fitting objects containing the state of the
            different fitting levels.
        affine_correction: :class: `menpo.transforms.affine.Affine`
            An affine transform that maps the result of the top resolution
            fitting level to the space scale of the original image.
        gt_shape: class:`menpo.shape.PointCloud`, optional
            The ground truth shape associated to the image.

            Default: None
        error_type: 'me_norm', 'me' or 'rmse', optional
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

            Default: 'me_norm'

        Returns
        -------
        fitting: :class:`menpo.fitmultilevel.fittingresult.MultilevelFittingResult`
            The fitting object that will hold the state of the fitter.
        """
        return MultilevelFittingResult(image, self, fitting_results,
                                       affine_correction, gt_shape=gt_shape)

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
        fitting_results: :class:`menpo.fit.fittingresult.FittingResult` list
            The fitting object containing the state of the whole fitting
            procedure.
        """
        shape = initial_shape
        gt_shape = None
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

        # fit images
        fitting_results = []
        for j, (i, f, it) in enumerate(zip(images, self._fitters, max_iters)):
            if gt_shapes is not None:
                gt_shape = gt_shapes[j]

            parameters = f.get_parameters(shape)
            fitting_result = f.fit(i, parameters, gt_shape=gt_shape,
                                   max_iters=it, **kwargs)
            fitting_results.append(fitting_result)

            shape = fitting_result.final_shape
            Scale(self.downscale, n_dims=shape.n_dims).apply_inplace(shape)

        return fitting_results


def align_shape_with_bb(shape, bounding_box):
    r"""
    Returns the Similarity transform that aligns the provided shape with the
    provided bounding box.

    Parameters
    ----------
    shape: :class:`menpo.shape.PointCloud`
        The shape to be aligned.
    bounding_box: (2, 2) ndarray
        The bounding box specified as:

            np.array([[x_min, y_min], [x_max, y_max]])

    Returns
    -------
    transform : :class: `menpo.transform.Similarity`
        The align transform
    """
    shape_box = PointCloud(shape.bounds())
    bounding_box = PointCloud(bounding_box)
    return AlignmentSimilarity(shape_box, bounding_box, rotation=False)