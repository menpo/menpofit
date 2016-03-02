from __future__ import division
import abc
from functools import wraps, partial
import numpy as np
from menpo.transform import Scale
from menpo.shape import PointCloud
from menpo.image import Image


# TODO: document me!
class Result(object):
    r"""
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

    def final_error(self, compute_error=None):
        r"""
        Returns the final fitting error.

        Parameters
        -----------
        compute_error: `callable`, optional
            Callable that computes the error between the fitted and
            ground truth shapes.

        Returns
        -------
        final_error : `float`
            The final error at the end of the fitting procedure.
        """
        if compute_error is None:
            compute_error = euclidean_bb_normalised_error
        if self.gt_shape is not None:
            return compute_error(self.final_shape, self.gt_shape)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    def initial_error(self, compute_error=None):
        r"""
        Returns the initial fitting error.

        Parameters
        -----------
        compute_error: `callable`, optional
            Callable that computes the error between the fitted and
            ground truth shapes.

        Returns
        -------
        initial_error : `float`
            The initial error at the start of the fitting procedure.
        """
        if compute_error is None:
            compute_error = euclidean_bb_normalised_error
        if self.gt_shape is not None:
            return compute_error(self.initial_shape, self.gt_shape)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    def as_serializableresult(self):
        return SerializableIterativeResult(
            self.image, self.initial_shape, self.final_shape,
            gt_shape=self.gt_shape)

    def __str__(self):
        out = "Initial error: {0:.4f}\nFinal error: {1:.4f}".format(
            self.initial_error(), self.final_error())
        return out


# TODO: document me!
class IterativeResult(Result):
    r"""
    """
    @abc.abstractproperty
    def n_iters(self):
        r"""
        Returns the number of iterations.
        """

    @abc.abstractproperty
    def shapes(self):
        r"""
        Generates a list containing the shapes obtained at each fitting
        iteration.

        Returns
        -------
        shapes : :map:`PointCloud`s or ndarray list
            A list containing the shapes obtained at each fitting iteration.
        """

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
        for j, s in enumerate(self.shapes):
            image.landmarks['iter_'+str(j)] = s
        return image

    def errors(self, compute_error=None):
        r"""
        Returns a list containing the error at each fitting iteration.

        Parameters
        -----------
        compute_error: `callable`, optional
            Callable that computes the error between the fitted and
            ground truth shapes.

        Returns
        -------
        errors : `list` of `float`
            The errors at each iteration of the fitting process.
        """
        if compute_error is None:
            compute_error = euclidean_bb_normalised_error
        if self.gt_shape is not None:
            return [compute_error(t, self.gt_shape)
                    for t in self.shapes]
        else:
            raise ValueError('Ground truth has not been set, errors cannot '
                             'be computed')

    def plot_errors(self, error_type=None, figure_id=None,
                    new_figure=False, render_lines=True, line_colour='b',
                    line_style='-', line_width=2, render_markers=True,
                    marker_style='o', marker_size=4, marker_face_colour='b',
                    marker_edge_colour='k', marker_edge_width=1.,
                    render_axes=True, axes_font_name='sans-serif',
                    axes_font_size=10, axes_font_style='normal',
                    axes_font_weight='normal', figure_size=(10, 6),
                    render_grid=True, grid_line_style='--',
                    grid_line_width=0.5):
        r"""
        Plot of the error evolution at each fitting iteration.
        Parameters
        ----------
        error_type : {``me_norm``, ``me``, ``rmse``}, optional
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_lines : `bool`, optional
            If ``True``, the line will be rendered.
        line_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``} or
                      ``(3, )`` `ndarray`, optional
            The colour of the lines.
        line_style : {``-``, ``--``, ``-.``, ``:``}, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : {``.``, ``,``, ``o``, ``v``, ``^``, ``<``, ``>``, ``+``,
                        ``x``, ``D``, ``d``, ``s``, ``p``, ``*``, ``h``, ``H``,
                        ``1``, ``2``, ``3``, ``4``, ``8``}, optional
            The style of the markers.
        marker_size : `int`, optional
            The size of the markers in points^2.
        marker_face_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                             or ``(3, )`` `ndarray`, optional
            The face (filling) colour of the markers.
        marker_edge_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                             or ``(3, )`` `ndarray`, optional
            The edge colour of the markers.
        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : {``serif``, ``sans-serif``, ``cursive``, ``fantasy``,
                          ``monospace``}, optional
            The font of the axes.
        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the axes.
        axes_font_weight : {``ultralight``, ``light``, ``normal``, ``regular``,
                            ``book``, ``medium``, ``roman``, ``semibold``,
                            ``demibold``, ``demi``, ``bold``, ``heavy``,
                            ``extra bold``, ``black``}, optional
            The font weight of the axes.
        figure_size : (`float`, `float`) or `None`, optional
            The size of the figure in inches.
        render_grid : `bool`, optional
            If ``True``, the grid will be rendered.
        grid_line_style : {``-``, ``--``, ``-.``, ``:``}, optional
            The style of the grid lines.
        grid_line_width : `float`, optional
            The width of the grid lines.
        Returns
        -------
        viewer : :map:`GraphPlotter`
            The viewer object.
        """
        from menpo.visualize import GraphPlotter
        errors_list = self.errors(compute_error=error_type)
        return GraphPlotter(figure_id=figure_id, new_figure=new_figure,
                            x_axis=range(len(errors_list)),
                            y_axis=[errors_list],
                            title='Fitting Errors per Iteration',
                            x_label='Iteration', y_label='Fitting Error',
                            x_axis_limits=(0, len(errors_list)-1),
                            y_axis_limits=None).render(
            render_lines=render_lines, line_colour=line_colour,
            line_style=line_style, line_width=line_width,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width, render_legend=False,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, render_grid=render_grid,
            grid_line_style=grid_line_style, grid_line_width=grid_line_width,
            figure_size=figure_size)

    def displacements(self):
        r"""
        A list containing the displacement between the shape of each iteration
        and the shape of the previous one.
        :type: `list` of ndarray
        """
        return [np.linalg.norm(s1.points - s2.points, axis=1)
                for s1, s2 in zip(self.shapes, self.shapes[1:])]

    def displacements_stats(self, stat_type='mean'):
        r"""
        A list containing the a statistical metric on the displacement between
        the shape of each iteration and the shape of the previous one.
        Parameters
        -----------
        stat_type : `str` ``{'mean', 'median', 'min', 'max'}``, optional
            Specifies a statistic metric to be extracted from the displacements.
        Returns
        -------
        :type: `list` of `float`
            The statistical metric on the points displacements for each
            iteration.
        """
        if stat_type == 'mean':
            return [np.mean(d) for d in self.displacements()]
        elif stat_type == 'median':
            return [np.median(d) for d in self.displacements()]
        elif stat_type == 'max':
            return [np.max(d) for d in self.displacements()]
        elif stat_type == 'min':
            return [np.min(d) for d in self.displacements()]
        else:
            raise ValueError("type must be 'mean', 'median', 'min' or 'max'")

    def plot_displacements(self, stat_type='mean', figure_id=None,
                           new_figure=False, render_lines=True, line_colour='b',
                           line_style='-', line_width=2, render_markers=True,
                           marker_style='o', marker_size=4,
                           marker_face_colour='b', marker_edge_colour='k',
                           marker_edge_width=1., render_axes=True,
                           axes_font_name='sans-serif', axes_font_size=10,
                           axes_font_style='normal', axes_font_weight='normal',
                           figure_size=(10, 6), render_grid=True,
                           grid_line_style='--', grid_line_width=0.5):
        r"""
        Plot of a statistical metric of the displacement between the shape of
        each iteration and the shape of the previous one.
        Parameters
        ----------
        stat_type : {``mean``, ``median``, ``min``, ``max``}, optional
            Specifies a statistic metric to be extracted from the displacements
            (see also `displacements_stats()` method).
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_lines : `bool`, optional
            If ``True``, the line will be rendered.
        line_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``} or
                      ``(3, )`` `ndarray`, optional
            The colour of the lines.
        line_style : {``-``, ``--``, ``-.``, ``:``}, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : {``.``, ``,``, ``o``, ``v``, ``^``, ``<``, ``>``, ``+``,
                        ``x``, ``D``, ``d``, ``s``, ``p``, ``*``, ``h``, ``H``,
                        ``1``, ``2``, ``3``, ``4``, ``8``}, optional
            The style of the markers.
        marker_size : `int`, optional
            The size of the markers in points^2.
        marker_face_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                             or ``(3, )`` `ndarray`, optional
            The face (filling) colour of the markers.
        marker_edge_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                             or ``(3, )`` `ndarray`, optional
            The edge colour of the markers.
        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : {``serif``, ``sans-serif``, ``cursive``, ``fantasy``,
                          ``monospace``}, optional
            The font of the axes.
        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the axes.
        axes_font_weight : {``ultralight``, ``light``, ``normal``, ``regular``,
                            ``book``, ``medium``, ``roman``, ``semibold``,
                            ``demibold``, ``demi``, ``bold``, ``heavy``,
                            ``extra bold``, ``black``}, optional
            The font weight of the axes.
        figure_size : (`float`, `float`) or `None`, optional
            The size of the figure in inches.
        render_grid : `bool`, optional
            If ``True``, the grid will be rendered.
        grid_line_style : {``-``, ``--``, ``-.``, ``:``}, optional
            The style of the grid lines.
        grid_line_width : `float`, optional
            The width of the grid lines.
        Returns
        -------
        viewer : :map:`GraphPlotter`
            The viewer object.
        """
        from menpo.visualize import GraphPlotter
        # set labels
        if stat_type == 'max':
            ylabel = 'Maximum Displacement'
            title = 'Maximum displacement per Iteration'
        elif stat_type == 'min':
            ylabel = 'Minimum Displacement'
            title = 'Minimum displacement per Iteration'
        elif stat_type == 'mean':
            ylabel = 'Mean Displacement'
            title = 'Mean displacement per Iteration'
        elif stat_type == 'median':
            ylabel = 'Median Displacement'
            title = 'Median displacement per Iteration'
        else:
            raise ValueError('stat_type must be one of {max, min, mean, '
                             'median}.')
        # plot
        displacements_list = self.displacements_stats(stat_type=stat_type)
        return GraphPlotter(figure_id=figure_id, new_figure=new_figure,
                            x_axis=range(len(displacements_list)),
                            y_axis=[displacements_list],
                            title=title,
                            x_label='Iteration', y_label=ylabel,
                            x_axis_limits=(0, len(displacements_list)-1),
                            y_axis_limits=None).render(
            render_lines=render_lines, line_colour=line_colour,
            line_style=line_style, line_width=line_width,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width, render_legend=False,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, render_grid=render_grid,
            grid_line_style=grid_line_style, grid_line_width=grid_line_width,
            figure_size=figure_size)

    def as_serializableresult(self):
        return SerializableIterativeResult(
            self.image, self.shapes, self.n_iters, gt_shape=self.gt_shape)


# TODO: document me!
class ParametricAlgorithmResult(IterativeResult):
    r"""
    """
    def __init__(self, image, algorithm, shape_parameters, gt_shape=None):
        self.image = image
        self.algorithm = algorithm
        self.shape_parameters = shape_parameters
        self._gt_shape = gt_shape

    @property
    def n_iters(self):
        return len(self.shapes) - 1

    @property
    def transforms(self):
        r"""
        Generates a list containing the transforms obtained at each fitting
        iteration.
        """
        return [self.algorithm.transform.from_vector(p)
                for p in self.shape_parameters]

    @property
    def final_transform(self):
        r"""
        Returns the final transform.
        """
        return self.algorithm.transform.from_vector(self.shape_parameters[-1])

    @property
    def initial_transform(self):
        r"""
        Returns the initial transform from which the fitting started.
        """
        return self.algorithm.transform.from_vector(self.shape_parameters[0])

    @property
    def shapes(self):
        return [self.algorithm.transform.from_vector(p).target
                for p in self.shape_parameters]

    @property
    def final_shape(self):
        return self.final_transform.target

    @property
    def initial_shape(self):
        return self.initial_transform.target


# TODO: document me!
class NonParametricAlgorithmResult(IterativeResult):
    r"""
    """
    def __init__(self, image, shapes, gt_shape=None):
        self.image = image
        self._shapes = shapes
        self._gt_shape = gt_shape

    @property
    def n_iters(self):
        return len(self.shapes) - 1

    @property
    def shapes(self):
        return self._shapes

    @property
    def final_shape(self):
        return self.shapes[-1]

    @property
    def initial_shape(self):
        return self.shapes[0]


# TODO: document me!
class MultiFitterResult(IterativeResult):
    r"""
    """
    def __init__(self, image, fitter, algorithm_results, affine_correction,
                 gt_shape=None):
        super(MultiFitterResult, self).__init__()
        self.image = image
        self.fitter = fitter
        self.algorithm_results = algorithm_results
        self._affine_correction = affine_correction
        self._gt_shape = gt_shape

    @property
    def n_scales(self):
        r"""
        The number of levels of the fitter object.

        :type: `int`
        """
        return self.fitter.n_scales

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

    @property
    def shapes(self):
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
        return _rescale_shapes_to_reference(
            self.algorithm_results, self.scales, self._affine_correction)

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
        initial_shape = Scale(self.scales[-1]/self.scales[0],
                              initial_shape.n_dims).apply(initial_shape)
        return self._affine_correction.apply(initial_shape)


# TODO: document me!
class SerializableIterativeResult(IterativeResult):
    r"""
    """
    def __init__(self, image, shapes, n_iters, gt_shape=None):
        self.image = image
        self._gt_shape = gt_shape
        self._shapes = shapes
        self._n_iters = n_iters

    @property
    def n_iters(self):
        return self._n_iters

    @property
    def shapes(self):
        return self._shapes

    @property
    def initial_shape(self):
        return self._shapes[0]

    @property
    def final_shape(self):
        return self._shapes[-1]


# TODO: Document me!
def _rescale_shapes_to_reference(algorithm_results, scales, affine_correction):
    r"""
    """
    shapes = []
    for j, (alg, scale) in enumerate(zip(algorithm_results, scales)):
        transform = Scale(scales[-1]/scale, alg.final_shape.n_dims)
        for shape in alg.shapes:
            shape = transform.apply(shape)
            shapes.append(affine_correction.apply(shape))
    return shapes


# TODO: Document me!
def pointcloud_to_points(wrapped):

    @wraps(wrapped)
    def wrapper(*args, **kwargs):
        args = list(args)
        for index, arg in enumerate(args):
            if isinstance(arg, PointCloud):
                args[index] = arg.points
        for key in kwargs:
            if isinstance(kwargs[key], PointCloud):
                kwargs[key] = kwargs[key].points
        return wrapped(*args, **kwargs)
    return wrapper


def bb_area(shape):
    # Area = w + h
    height, width = np.max(shape, axis=0) - np.min(shape, axis=0)
    return height * width


def bb_perimeter(shape):
    # Area = 2(w + h)
    height, width = np.max(shape, axis=0) - np.min(shape, axis=0)
    return 2 * (height + width)


def bb_avg_edge_length(shape):
    # 0.5(w + h) = (2w + 2h) / 4
    height, width = np.max(shape, axis=0) - np.min(shape, axis=0)
    return 0.5 * (height + width)


def bb_diagonal(shape):
    # sqrt(w**2 + h**2)
    height, width = np.max(shape, axis=0) - np.min(shape, axis=0)
    return np.sqrt(width ** 2 + height ** 2)


bb_norm_types = {
    'avg_edge_length': bb_avg_edge_length,
    'perimeter': bb_perimeter,
    'diagonal': bb_diagonal,
    'area': bb_area
}


@pointcloud_to_points
def bb_normalised_error(shape_error_f, shape, gt_shape,
                        norm_shape=None, norm_type='avg_edge_length'):
    if norm_type not in bb_norm_types:
        raise ValueError('norm_type must be one of '
                         '{avg_edge_length, perimeter, diagonal, area}.')
    if norm_shape is None:
        norm_shape = gt_shape
    return (shape_error_f(shape, gt_shape) /
            bb_norm_types[norm_type](norm_shape))


def distance_two_indices(index1, index2, gt_shape):
    return euclidean_error(gt_shape[index1], gt_shape[index2])


@pointcloud_to_points
def distance_normalised_error(shape_error_f, distance_norm_f, shape, gt_shape):
    return shape_error_f(shape, gt_shape) / distance_norm_f(shape, gt_shape)


@pointcloud_to_points
def distance_indexed_normalised_error(shape_error_f, index1, index2, shape,
                                      gt_shape):
    return shape_error_f(shape, gt_shape) / distance_two_indices(index1, index2,
                                                                 gt_shape)


# TODO: Document me!
@pointcloud_to_points
def root_mean_square_error(shape, gt_shape):
    r"""
    """
    return np.sqrt(np.mean((shape.ravel() - gt_shape.ravel()) ** 2))


# TODO: Document me!
@pointcloud_to_points
def euclidean_error(shape, gt_shape):
    r"""
    """
    return np.mean(np.sqrt(np.sum((shape - gt_shape) ** 2, axis=-1)))


# TODO: Document me!
root_mean_square_bb_normalised_error = partial(bb_normalised_error,
                                               root_mean_square_error)

# TODO: Document me!
euclidean_bb_normalised_error = partial(bb_normalised_error, euclidean_error)


# TODO: Document me!
root_mean_square_distance_normalised_error = partial(distance_normalised_error,
                                                     root_mean_square_error)

# TODO: Document me!
euclidean_distance_normalised_error = partial(distance_normalised_error,
                                              euclidean_error)


compute_normalise_point_to_point_error = euclidean_bb_normalised_error
compute_root_mean_square_error = root_mean_square_error
compute_point_to_point_error = euclidean_error


# TODO: Document me!
def compute_cumulative_error(errors, boundaries):
    r"""
    """
    n_errors = len(errors)
    return [np.count_nonzero([errors <= x]) / n_errors for x in boundaries]


def plot_cumulative_error_distribution(
        errors, error_range=None, figure_id=None, new_figure=False,
        title='Cumulative Error Distribution',
        x_label='Normalized Point-to-Point Error', y_label='Images Proportion',
        legend_entries=None, render_lines=True, line_colour=None,
        line_style='-', line_width=2, render_markers=True, marker_style='s',
        marker_size=10, marker_face_colour='w', marker_edge_colour=None,
        marker_edge_width=2, render_legend=True, legend_title=None,
        legend_font_name='sans-serif', legend_font_style='normal',
        legend_font_size=10, legend_font_weight='normal',
        legend_marker_scale=1., legend_location=2,
        legend_bbox_to_anchor=(1.05, 1.), legend_border_axes_pad=1.,
        legend_n_columns=1, legend_horizontal_spacing=1.,
        legend_vertical_spacing=1., legend_border=True,
        legend_border_padding=0.5, legend_shadow=False,
        legend_rounded_corners=False, render_axes=True,
        axes_font_name='sans-serif', axes_font_size=10,
        axes_font_style='normal', axes_font_weight='normal', axes_x_limits=None,
        axes_y_limits=None, axes_x_ticks=None, axes_y_ticks=None,
        figure_size=(10, 8), render_grid=True, grid_line_style='--',
        grid_line_width=0.5):
    r"""
    Plot the cumulative error distribution (CED) of the provided fitting errors.

    Parameters
    ----------
    errors : `list` of `lists`
        A `list` with `lists` of fitting errors. A separate CED curve will be
        rendered for each errors `list`.
    error_range : `list` of `float` with length 3, optional
        Specifies the horizontal axis range, i.e.

        ::

        error_range[0] = min_error
        error_range[1] = max_error
        error_range[2] = error_step

        If ``None``, then ``'error_range = [0., 0.101, 0.005]'``.
    figure_id : `object`, optional
        The id of the figure to be used.
    new_figure : `bool`, optional
        If ``True``, a new figure is created.
    title : `str`, optional
        The figure's title.
    x_label : `str`, optional
        The label of the horizontal axis.
    y_label : `str`, optional
        The label of the vertical axis.
    legend_entries : `list of `str` or ``None``, optional
        If `list` of `str`, it must have the same length as `errors` `list` and
        each `str` will be used to name each curve. If ``None``, the CED curves
        will be named as `'Curve %d'`.
    render_lines : `bool` or `list` of `bool`, optional
        If ``True``, the line will be rendered. If `bool`, this value will be
        used for all curves. If `list`, a value must be specified for each
        fitting errors curve, thus it must have the same length as `errors`.
    line_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``} or
                  ``(3, )`` `ndarray` or `list` of those or ``None``, optional
        The colour of the lines. If not a `list`, this value will be
        used for all curves. If `list`, a value must be specified for each
        fitting errors curve, thus it must have the same length as `errors`. If
        ``None``, the colours will be linearly sampled from jet colormap.
    line_style : {``-``, ``--``, ``-.``, ``:``} or `list` of those, optional
        The style of the lines. If not a `list`, this value will be used for all
        curves. If `list`, a value must be specified for each fitting errors
        curve, thus it must have the same length as `errors`.
    line_width : `float` or `list` of `float`, optional
        The width of the lines. If `float`, this value will be used for all
        curves. If `list`, a value must be specified for each fitting errors
        curve, thus it must have the same length as `errors`.
    render_markers : `bool` or `list` of `bool`, optional
        If ``True``, the markers will be rendered. If `bool`, this value will be
        used for all curves. If `list`, a value must be specified for each
        fitting errors curve, thus it must have the same length as `errors`.
    marker_style : {``.``, ``,``, ``o``, ``v``, ``^``, ``<``, ``>``, ``+``,
                    ``x``, ``D``, ``d``, ``s``, ``p``, ``*``, ``h``, ``H``,
                    ``1``, ``2``, ``3``, ``4``, ``8``} or `list` of those, optional
        The style of the markers. If not a `list`, this value will be used for
        all curves. If `list`, a value must be specified for each fitting errors
        curve, thus it must have the same length as `errors`.
    marker_size : `int` or `list` of `int`, optional
        The size of the markers in points^2. If `int`, this value will be used
        for all curves. If `list`, a value must be specified for each fitting
        errors curve, thus it must have the same length as `errors`.
    marker_face_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                         or ``(3, )`` `ndarray` or `list` of those or ``None``, optional
        The face (filling) colour of the markers. If not a `list`, this value
        will be used for all curves. If `list`, a value must be specified for
        each fitting errors curve, thus it must have the same length as
        `errors`. If ``None``, the colours will be linearly sampled from jet
        colormap.
    marker_edge_colour : {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                         or ``(3, )`` `ndarray` or `list` of those or ``None``, optional
        The edge colour of the markers. If not a `list`, this value will be used
        for all curves. If `list`, a value must be specified for each fitting
        errors curve, thus it must have the same length as `errors`. If
        ``None``, the colours will be linearly sampled from jet colormap.
    marker_edge_width : `float` or `list` of `float`, optional
        The width of the markers' edge. If `float`, this value will be used for
        all curves. If `list`, a value must be specified for each fitting errors
        curve, thus it must have the same length as `errors`.
    render_legend : `bool`, optional
        If ``True``, the legend will be rendered.
    legend_title : `str`, optional
        The title of the legend.
    legend_font_name : {``serif``, ``sans-serif``, ``cursive``, ``fantasy``,
                        ``monospace``}, optional
        The font of the legend.
    legend_font_style : {``normal``, ``italic``, ``oblique``}, optional
        The font style of the legend.
    legend_font_size : `int`, optional
        The font size of the legend.
    legend_font_weight : {``ultralight``, ``light``, ``normal``,
                          ``regular``, ``book``, ``medium``, ``roman``,
                          ``semibold``, ``demibold``, ``demi``, ``bold``,
                          ``heavy``, ``extra bold``, ``black``}, optional
        The font weight of the legend.
    legend_marker_scale : `float`, optional
        The relative size of the legend markers with respect to the original
    legend_location : `int`, optional
        The location of the legend. The predefined values are:

        =============== ===
        'best'          0
        'upper right'   1
        'upper left'    2
        'lower left'    3
        'lower right'   4
        'right'         5
        'center left'   6
        'center right'  7
        'lower center'  8
        'upper center'  9
        'center'        10
        =============== ===

    legend_bbox_to_anchor : (`float`, `float`), optional
        The bbox that the legend will be anchored.
    legend_border_axes_pad : `float`, optional
        The pad between the axes and legend border.
    legend_n_columns : `int`, optional
        The number of the legend's columns.
    legend_horizontal_spacing : `float`, optional
        The spacing between the columns.
    legend_vertical_spacing : `float`, optional
        The vertical space between the legend entries.
    legend_border : `bool`, optional
        If ``True``, a frame will be drawn around the legend.
    legend_border_padding : `float`, optional
        The fractional whitespace inside the legend border.
    legend_shadow : `bool`, optional
        If ``True``, a shadow will be drawn behind legend.
    legend_rounded_corners : `bool`, optional
        If ``True``, the frame's corners will be rounded (fancybox).
    render_axes : `bool`, optional
        If ``True``, the axes will be rendered.
    axes_font_name : {``serif``, ``sans-serif``, ``cursive``, ``fantasy``,
                      ``monospace``}, optional
        The font of the axes.
    axes_font_size : `int`, optional
        The font size of the axes.
    axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
        The font style of the axes.
    axes_font_weight : {``ultralight``, ``light``, ``normal``, ``regular``,
                        ``book``, ``medium``, ``roman``, ``semibold``,
                        ``demibold``, ``demi``, ``bold``, ``heavy``,
                        ``extra bold``, ``black``}, optional
        The font weight of the axes.
    axes_x_limits : (`float`, `float`) or ``None``, optional
        The limits of the x axis. If ``None``, it is set to
        ``(0., 'errors_max')``.
    axes_y_limits : (`float`, `float`) or ``None``, optional
        The limits of the y axis. If ``None``, it is set to ``(0., 1.)``.
    axes_x_ticks : `list` or `tuple` or ``None``, optional
        The ticks of the x axis.
    axes_y_ticks : `list` or `tuple` or ``None``, optional
        The ticks of the y axis.
    figure_size : (`float`, `float`) or ``None``, optional
        The size of the figure in inches.
    render_grid : `bool`, optional
        If ``True``, the grid will be rendered.
    grid_line_style : {``-``, ``--``, ``-.``, ``:``}, optional
        The style of the grid lines.
    grid_line_width : `float`, optional
        The width of the grid lines.

    Raises
    ------
    ValueError
        legend_entries list has different length than errors list

    Returns
    -------
    viewer : :map:`GraphPlotter`
        The viewer object.
    """
    from menpo.visualize import plot_curve

    # make sure that errors is a list even with one list member
    if not isinstance(errors[0], list):
        errors = [errors]

    # create x and y axes lists
    x_axis = list(np.arange(error_range[0], error_range[1], error_range[2]))
    ceds = [compute_cumulative_error(e, x_axis) for e in errors]

    # parse legend_entries, axes_x_limits and axes_y_limits
    if legend_entries is None:
        legend_entries = ["Curve {}".format(k) for k in range(len(ceds))]
    if len(legend_entries) != len(ceds):
        raise ValueError('legend_entries list has different length than errors '
                         'list')
    if axes_x_limits is None:
        axes_x_limits = (0., x_axis[-1])
    if axes_y_limits is None:
        axes_y_limits = (0., 1.)

    # render
    return plot_curve(
        x_axis=x_axis, y_axis=ceds, figure_id=figure_id, new_figure=new_figure,
        legend_entries=legend_entries, title=title, x_label=x_label,
        y_label=y_label, axes_x_limits=axes_x_limits,
        axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
        axes_y_ticks=axes_y_ticks, render_lines=render_lines,
        line_colour=line_colour, line_style=line_style, line_width=line_width,
        render_markers=render_markers, marker_style=marker_style,
        marker_size=marker_size, marker_face_colour=marker_face_colour,
        marker_edge_colour=marker_edge_colour,
        marker_edge_width=marker_edge_width, render_legend=render_legend,
        legend_title=legend_title, legend_font_name=legend_font_name,
        legend_font_style=legend_font_style, legend_font_size=legend_font_size,
        legend_font_weight=legend_font_weight,
        legend_marker_scale=legend_marker_scale,
        legend_location=legend_location,
        legend_bbox_to_anchor=legend_bbox_to_anchor,
        legend_border_axes_pad=legend_border_axes_pad,
        legend_n_columns=legend_n_columns,
        legend_horizontal_spacing=legend_horizontal_spacing,
        legend_vertical_spacing=legend_vertical_spacing,
        legend_border=legend_border,
        legend_border_padding=legend_border_padding,
        legend_shadow=legend_shadow,
        legend_rounded_corners=legend_rounded_corners,
        render_axes=render_axes,
        axes_font_name=axes_font_name, axes_font_size=axes_font_size,
        axes_font_style=axes_font_style, axes_font_weight=axes_font_weight,
        figure_size=figure_size, render_grid=render_grid,
        grid_line_style=grid_line_style, grid_line_width=grid_line_width)
