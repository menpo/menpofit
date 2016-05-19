import abc
import menpo.io as mio
from menpo.transform import Scale
from menpofit.error import euclidean_bb_normalised_error
from menpo.image import Image

# Abstract Interface for Results ----------------------------------------------

class Result(object, metaclass=abc.ABCMeta):

    @abc.abstractproperty
    def n_iters(self):
        r"""
        Returns the number of iterations.
        """

    @abc.abstractmethod
    def shapes(self, as_points=False):
        r"""
        Generates a list containing the shapes obtained at each fitting
        iteration.

        Parameters
        -----------
        as_points : boolean, optional
            Whether the results is returned as a list of :map:`PointCloud`s or
            ndarrays.

            Default: `False`

        Returns
        -------
        shapes : :map:`PointCloud`s or ndarray list
            A list containing the shapes obtained at each fitting iteration.
        """

    @abc.abstractproperty
    def final_shape(self):
        r"""
        Returns the final fitted shape.
        """

    @abc.abstractproperty
    def initial_shape(self):
        r"""
        Returns the initial shape from which the fitting started.
        """

    @property
    def gt_shape(self):
        r"""
        Returns the original ground truth shape associated to the image.
        """
        return self._gt_shape

    @property
    def fitted_image(self):
        r"""
        Returns a copy of the fitted image with the following landmark
        groups attached to it:
            - ``initial``, containing the initial fitted shape .
            - ``final``, containing the final shape.
            - ``ground``, containing the ground truth shape. Only returned if
            the ground truth shape was provided.

        :type: :map:`Image`
        """
        image = Image(self.image.pixels)

        image.landmarks['initial'] = self.initial_shape
        image.landmarks['final'] = self.final_shape
        if self.gt_shape is not None:
            image.landmarks['ground'] = self.gt_shape
        return image

    @property
    def iter_image(self):
        r"""
        Returns a copy of the fitted image with a as many landmark groups as
        iteration run by fitting procedure:
            - ``iter_0``, containing the initial shape.
            - ``iter_1``, containing the the fitted shape at the first
            iteration.
            - ``...``
            - ``iter_n``, containing the final fitted shape.

        :type: :map:`Image`
        """
        image = Image(self.image.pixels)
        for j, s in enumerate(self.shapes()):
            image.landmarks['iter_'+str(j)] = s
        return image

    def errors(self, error_type='me_norm'):
        r"""
        Returns a list containing the error at each fitting iteration.

        Parameters
        -----------
        error_type : `str` ``{'me_norm', 'me', 'rmse'}``, optional
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

        Returns
        -------
        errors : `list` of `float`
            The errors at each iteration of the fitting process.
        """
        if self.gt_shape is not None:
            return [euclidean_bb_normalised_error(t, self.gt_shape, error_type)
                    for t in self.shapes()]
        else:
            raise ValueError('Ground truth has not been set, errors cannot '
                             'be computed')

    def final_error(self, error_type='me_norm'):
        r"""
        Returns the final fitting error.

        Parameters
        -----------
        error_type : `str` ``{'me_norm', 'me', 'rmse'}``, optional
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

        Returns
        -------
        final_error : `float`
            The final error at the end of the fitting procedure.
        """
        if self.gt_shape is not None:
            return euclidean_bb_normalised_error(self.final_shape, self.gt_shape)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    def initial_error(self, error_type='me_norm'):
        r"""
        Returns the initial fitting error.

        Parameters
        -----------
        error_type : `str` ``{'me_norm', 'me', 'rmse'}``, optional
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

        Returns
        -------
        initial_error : `float`
            The initial error at the start of the fitting procedure.
        """
        if self.gt_shape is not None:
            return euclidean_bb_normalised_error(self.initial_shape, self.gt_shape)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    def __str__(self):
        out = "Initial error: {0:.4f}\nFinal error: {1:.4f}".format(
            self.initial_error(), self.final_error())
        return out


# Abstract Interfaces for Algorithm Results -----------------------------------

class AlgorithmResult(Result):

    @property
    def n_iters(self):
        return len(self.shapes()) - 1

    @property
    def transforms(self):
        r"""
        Generates a list containing the transforms obtained at each fitting
        iteration.
        """
        return [self.fitter.transform.from_vector(p)
                for p in self.shape_parameters]

    @property
    def final_transform(self):
        r"""
        Returns the final transform.
        """
        return self.fitter.transform.from_vector(self.shape_parameters[-1])

    @property
    def initial_transform(self):
        r"""
        Returns the initial transform from which the fitting started.
        """
        return self.fitter.transform.from_vector(self.shape_parameters[0])

    def shapes(self, as_points=False):
        if as_points:
            return [self.fitter.transform.from_vector(p).target.points
                    for p in self.shape_parameters]

        else:
            return [self.fitter.transform.from_vector(p).target
                    for p in self.shape_parameters]

    @property
    def final_shape(self):
        return self.final_transform.target

    @property
    def initial_shape(self):
        return self.initial_transform.target


# Abstract Interface for Fitter Results ---------------------------------------

class FitterResult(Result):

    def __init__(self, image, fitter, algorithm_results, affine_correction,
                 gt_shape=None):
        super(FitterResult, self).__init__()
        self.image = image
        self.fitter = fitter
        self.algorithm_results = algorithm_results
        self._affine_correction = affine_correction
        self._gt_shape = gt_shape

    @property
    def n_levels(self):
        r"""
        The number of levels of the fitter object.

        :type: `int`
        """
        return self.fitter.n_levels

    @property
    def scales(self):
        return self.fitter.scales

    @property
    def n_iters(self):
        r"""
        The total number of iterations used to fitter the image.

        :type: `int`
        """
        n_iters = 0
        for f in self.algorithm_results:
            n_iters += f.n_iters
        return n_iters

    def shapes(self, as_points=False):
        r"""
        Generates a list containing the shapes obtained at each fitting
        iteration.

        Parameters
        -----------
        as_points : `boolean`, optional
            Whether the result is returned as a `list` of :map:`PointCloud` or
            a `list` of `ndarrays`.

        Returns
        -------
        shapes : `list` of :map:`PointCoulds` or `list` of `ndarray`
            A list containing the fitted shapes at each iteration of
            the fitting procedure.
        """
        shapes = []
        for j, (alg, s) in enumerate(zip(self.algorithm_results, self.scales)):
            transform = Scale(self.scales[-1]/s, alg.final_shape.n_dims)
            for t in alg.shapes(as_points=as_points):
                t = transform.apply(t)
                shapes.append(self._affine_correction.apply(t))

        return shapes

    @property
    def final_shape(self):
        r"""
        The final fitted shape.

        :type: :map:`PointCloud`
        """
        final_shape = self.algorithm_results[-1].final_shape
        return self._affine_correction.apply(final_shape)

    @property
    def initial_shape(self):
        initial_shape = self.algorithm_results[0].initial_shape
        initial_shape = Scale(self.scales[-1]/self.scales[0],initial_shape.n_dims).apply(initial_shape)
        return self._affine_correction.apply(initial_shape)


# Serializable Result --------------------------------------------------------

class SerializableResult(Result):

    def __init__(self, image_path, shapes, n_iters, algorithm, gt_shape=None):
        self._image_path = image_path
        self._image = None
        self._gt_shape = gt_shape
        self._shapes = shapes
        self._n_iters = n_iters
        self.algorithm = str(algorithm)

    @property
    def n_iters(self):
        return self._n_iters

    def shapes(self, as_points=False):
        if as_points:
            return [s.points for s in self._shapes]
        else:
            return self._shapes

    @property
    def initial_shape(self):
        return self._shapes[0]

    @property
    def final_shape(self):
        return self._shapes[-1]

    @property
    def image(self):
        if self._image is None:
            image = mio.import_image(self._image_path)
            image.crop_to_landmarks_proportion_inplace(0.5)
            self._image = image

        return self._image


# Concrete Implementations of AAM Algorithm Results ---------------------------

class UnifiedAlgorithmResult(AlgorithmResult):

    def __init__(self, image, fitter, shape_parameters,
                 appearance_parameters=None, gt_shape=None):
        super(UnifiedAlgorithmResult, self).__init__()
        self.image = image
        self.fitter = fitter
        self.shape_parameters = shape_parameters
        self.appearance_parameters = appearance_parameters
        self._gt_shape = gt_shape


# Concrete Implementations of AAM Fitter Results  -----------------------------

class UnifiedFitterResult(FitterResult):

    @property
    def costs(self):
        r"""
        Returns a list containing the cost at each fitting iteration.

        :type: `list` of `float`
        """
        raise ValueError('costs not implemented yet.')

    @property
    def final_cost(self):
        r"""
        Returns the final fitting cost.

        :type: `float`
        """
        return self.algorithm_results[-1].final_cost

    @property
    def initial_cost(self):
        r"""
        Returns the initial fitting cost.

        :type: `float`
        """
        return self.algorithm_results[0].initial_cost
