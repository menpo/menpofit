from __future__ import division

import abc
from itertools import chain
import numpy as np

from menpo.shape.pointcloud import PointCloud
from menpo.image import Image
from menpo.transform import Scale
from menpofit.base import name_of_callable


class FittingResult(object):
    r"""
    Object that holds the state of a single fitting object, during and after it
    has fitted a particular image.

    Parameters
    -----------
    image : :map:`Image` or subclass
        The fitted image.
    gt_shape : :map:`PointCloud`
        The ground truth shape associated to the image.
    """

    def __init__(self, image, gt_shape=None):
        self.image = image
        self._gt_shape = gt_shape

    @property
    def n_iters(self):
        return len(self.shapes) - 1

    @abc.abstractproperty
    def shapes(self):
        r"""
        A list containing the shapes obtained at each fitting iteration.

        :type: `list` of :map:`PointCloud`
        """

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
        Returns a copy of the fitted image with as many landmark groups as
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
            key = 'iter_{}'.format(j)
            image.landmarks[key] = s
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
            return [compute_error(t, self.gt_shape, error_type)
                    for t in self.shapes]
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
            return compute_error(self.final_shape, self.gt_shape, error_type)
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
            return compute_error(self.initial_shape, self.gt_shape, error_type)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    def view_widget(self, popup=False):
        r"""
        Visualizes the multilevel fitting result object using the
        menpo.visualize.widgets.visualize_fitting_results widget.

        Parameters
        -----------
        popup : `boolean`, optional
            If enabled, the widget will appear as a popup window.
        """
        from menpofit.visualize import visualize_fitting_results
        visualize_fitting_results(self, figure_size=(7, 7), popup=popup)

    def plot_errors(self, error_type='me_norm', figure_id=None,
                    new_figure=False, render_lines=True, line_colour='b',
                    line_style='-', line_width=2, render_markers=True,
                    marker_style='o', marker_size=4, marker_face_colour='b',
                    marker_edge_colour='k', marker_edge_width=1.,
                    render_axes=True, axes_font_name='sans-serif',
                    axes_font_size=10, axes_font_style='normal',
                    axes_font_weight='normal', figure_size=(6, 4),
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
        errors_list = self.errors(error_type=error_type)
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

    def plot_displacements(self, stat_type='mean', figure_id=None,
                           new_figure=False, render_lines=True, line_colour='b',
                           line_style='-', line_width=2, render_markers=True,
                           marker_style='o', marker_size=4,
                           marker_face_colour='b', marker_edge_colour='k',
                           marker_edge_width=1., render_axes=True,
                           axes_font_name='sans-serif', axes_font_size=10,
                           axes_font_style='normal', axes_font_weight='normal',
                           figure_size=(6, 4), render_grid=True,
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

    def as_serializable(self):
        r""""
        Returns a serializable version of the fitting result. This is a much
        lighter weight object than the initial fitting result. For example,
        it won't contain the original fitting object.

        Returns
        -------
        serializable_fitting_result : :map:`SerializableFittingResult`
            The lightweight serializable version of this fitting result.
        """
        if self.parameters is not None:
            parameters = [p.copy() for p in self.parameters]
        else:
            parameters = []
        gt_shape = self.gt_shape.copy() if self.gt_shape else None
        return SerializableFittingResult(self.image.copy(),
                                         parameters,
                                         [s.copy() for s in self.shapes],
                                         gt_shape)


class NonParametricFittingResult(FittingResult):
    r"""
    Object that holds the state of a Non Parametric :map:`Fitter` object
    before, during and after it has fitted a particular image.

    Parameters
    -----------
    image : :map:`Image`
        The fitted image.
    fitter : :map:`Fitter`
        The Fitter object used to fitter the image.
    shapes : `list` of :map:`PointCloud`
        The list of fitted shapes per iteration of the fitting procedure.
    gt_shape : :map:`PointCloud`
        The ground truth shape associated to the image.
    """

    def __init__(self, image, fitter, parameters=None, gt_shape=None):
        super(NonParametricFittingResult, self).__init__(image,
                                                         gt_shape=gt_shape)
        self.fitter = fitter
        # The parameters are the shapes for Non-Parametric algorithms
        self.parameters = parameters

    @property
    def shapes(self):
        return self.parameters

    @property
    def final_shape(self):
        return self.parameters[-1].copy()

    @property
    def initial_shape(self):
        return self.parameters[0].copy()

    @FittingResult.gt_shape.setter
    def gt_shape(self, value):
        r"""
        Setter for the ground truth shape associated to the image.
        """
        if isinstance(value, PointCloud):
            self._gt_shape = value
        else:
            raise ValueError("Accepted values for gt_shape setter are "
                             "PointClouds.")


class SemiParametricFittingResult(FittingResult):
    r"""
    Object that holds the state of a Semi Parametric :map:`Fitter` object
    before, during and after it has fitted a particular image.

    Parameters
    -----------
    image : :map:`Image`
        The fitted image.
    fitter : :map:`Fitter`
        The Fitter object used to fitter the image.
    parameters : `list` of `ndarray`
        The list of optimal transform parameters per iteration of the fitting
        procedure.
    gt_shape : :map:`PointCloud`
        The ground truth shape associated to the image.
    """

    def __init__(self, image, fitter, parameters=None, gt_shape=None):
        FittingResult.__init__(self, image, gt_shape=gt_shape)
        self.fitter = fitter
        self.parameters = parameters

    @property
    def transforms(self):
        r"""
        Generates a list containing the transforms obtained at each fitting
        iteration.
        """
        return [self.fitter.transform.from_vector(p) for p in self.parameters]

    @property
    def final_transform(self):
        r"""
        Returns the final transform.
        """
        return self.fitter.transform.from_vector(self.parameters[-1])

    @property
    def initial_transform(self):
        r"""
        Returns the initial transform from which the fitting started.
        """
        return self.fitter.transform.from_vector(self.parameters[0])

    @property
    def shapes(self):
        return [self.fitter.transform.from_vector(p).target
                for p in self.parameters]

    @property
    def final_shape(self):
        return self.final_transform.target

    @property
    def initial_shape(self):
        return self.initial_transform.target

    @FittingResult.gt_shape.setter
    def gt_shape(self, value):
        r"""
        Setter for the ground truth shape associated to the image.
        """
        if type(value) is PointCloud:
            self._gt_shape = value
        elif type(value) is list and value[0] is float:
            transform = self.fitter.transform.from_vector(value)
            self._gt_shape = transform.target
        else:
            raise ValueError("Accepted values for gt_shape setter are "
                             "PointClouds or float lists "
                             "specifying transform shapes.")


class ParametricFittingResult(SemiParametricFittingResult):
    r"""
    Object that holds the state of a Fully Parametric :map:`Fitter` object
    before, during and after it has fitted a particular image.

    Parameters
    -----------
    image : :map:`Image`
        The fitted image.
    fitter : :map:`Fitter`
        The Fitter object used to fitter the image.
    parameters : `list` of `ndarray`
        The list of optimal transform parameters per iteration of the fitting
        procedure.
    weights : `list` of `ndarray`
        The list of optimal appearance parameters per iteration of the fitting
        procedure.
    gt_shape : :map:`PointCloud`
        The ground truth shape associated to the image.
    """
    def __init__(self, image, fitter, parameters=None, weights=None,
                 gt_shape=None):
        SemiParametricFittingResult.__init__(self, image, fitter, parameters,
                                             gt_shape=gt_shape)
        self.weights = weights

    @property
    def warped_images(self):
        r"""
        The list containing the warped images obtained at each fitting
        iteration.

        :type: `list` of :map:`Image` or subclass
        """
        mask = self.fitter.template.mask
        transform = self.fitter.transform
        return [self.image.warp_to_mask(mask, transform.from_vector(p))
                for p in self.parameters]

    @property
    def appearance_reconstructions(self):
        r"""
        The list containing the appearance reconstruction obtained at
        each fitting iteration.

        :type: list` of :map:`Image` or subclass
        """
        if self.weights:
            return [self.fitter.appearance_model.instance(w)
                    for w in self.weights]
        else:
            return [self.fitter.template for _ in self.shapes]

    @property
    def error_images(self):
        r"""
        The list containing the error images obtained at
        each fitting iteration.

        :type: list` of :map:`Image` or subclass
        """
        template = self.fitter.template
        warped_images = self.warped_images
        appearances = self.appearance_reconstructions

        error_images = []
        for a, i in zip(appearances, warped_images):
            error = a.as_vector() - i.as_vector()
            error_image = template.from_vector(error)
            error_images.append(error_image)

        return error_images


class SerializableFittingResult(FittingResult):
    r"""
    Designed to allow the fitting results to be easily serializable. In
    comparison to the other fitting result objects, the serializable fitting
    results contain a much stricter set of data. For example, the major data
    components of a serializable fitting result are the fitted shapes, the
    parameters and the fitted image.

    Parameters
    -----------
    image : :map:`Image`
        The fitted image.
    parameters : `list` of `ndarray`
        The list of optimal transform parameters per iteration of the fitting
        procedure.
    shapes : `list` of :map:`PointCloud`
        The list of fitted shapes per iteration of the fitting procedure.
    gt_shape : :map:`PointCloud`
        The ground truth shape associated to the image.
    """
    def __init__(self, image, parameters, shapes, gt_shape):
        FittingResult.__init__(self, image, gt_shape=gt_shape)

        self.parameters = parameters
        self._shapes = shapes

    @property
    def shapes(self):
        return self._shapes

    @property
    def initial_shape(self):
        return self._shapes[0]

    @property
    def final_shape(self):
        return self._shapes[-1]


class MultilevelFittingResult(FittingResult):
    r"""
    Class that holds the state of a :map:`MultilevelFitter` object before,
    during and after it has fitted a particular image.

    Parameters
    -----------
    image : :map:`Image` or subclass
        The fitted image.
    multilevel_fitter : :map:`MultilevelFitter`
        The multilevel fitter object used to fit the image.
    fitting_results : `list` of :map:`FittingResult`
        The list of fitting results.
    affine_correction : :map:`Affine`
        The affine transform between the initial shape of the highest
        pyramidal level and the initial shape of the original image
    gt_shape : class:`PointCloud`, optional
        The ground truth shape associated to the image.
    """
    def __init__(self, image, multiple_fitter, fitting_results,
                 affine_correction, gt_shape=None):
        super(MultilevelFittingResult, self).__init__(image, gt_shape=gt_shape)
        self.fitter = multiple_fitter
        self.fitting_results = fitting_results
        self._affine_correction = affine_correction

    @property
    def n_levels(self):
        r"""
        The number of levels of the fitter object.

        :type: `int`
        """
        return self.fitter.n_levels

    @property
    def downscale(self):
        r"""
        The downscale factor used by the multiple fitter.

        :type: `float`
        """
        return self.fitter.downscale

    @property
    def n_iters(self):
        r"""
        The total number of iterations used to fitter the image.

        :type: `int`
        """
        n_iters = 0
        for f in self.fitting_results:
            n_iters += f.n_iters
        return n_iters

    @property
    def shapes(self):
        r"""
        A list containing the shapes obtained at each fitting iteration.

        :type: `list` of :map:`PointCloud`
        """
        return _rescale_shapes_to_reference(self.fitting_results, self.n_levels,
                                            self.downscale,
                                            self._affine_correction)

    @property
    def final_shape(self):
        r"""
        The final fitted shape.

        :type: :map:`PointCloud`
        """
        return self._affine_correction.apply(
            self.fitting_results[-1].final_shape)

    @property
    def initial_shape(self):
        r"""
        The initial shape from which the fitting started.

        :type: :map:`PointCloud`
        """
        n = self.n_levels - 1
        initial_shape = self.fitting_results[0].initial_shape
        Scale(self.downscale ** n, initial_shape.n_dims).apply_inplace(
            initial_shape)

        return self._affine_correction.apply(initial_shape)

    @FittingResult.gt_shape.setter
    def gt_shape(self, value):
        r"""
        Setter for the ground truth shape associated to the image.

        type: :map:`PointCloud`
        """
        self._gt_shape = value

    def __str__(self):
        if self.fitter.pyramid_on_features:
            feat_str = name_of_callable(self.fitter.features)
        else:
            feat_str = []
            for j in range(self.n_levels):
                if isinstance(self.fitter.features[j], str):
                    feat_str.append(self.fitter.features[j])
                elif self.fitter.features[j] is None:
                    feat_str.append("none")
                else:
                    feat_str.append(name_of_callable(self.fitter.features[j]))
        out = "Fitting Result\n" \
              " - Initial error: {0:.4f}\n" \
              " - Final error: {1:.4f}\n" \
              " - {2} method with {3} pyramid levels, {4} iterations " \
              "and using {5} features.".format(
              self.initial_error(), self.final_error(), self.fitter.algorithm,
              self.n_levels, self.n_iters, feat_str)
        return out

    def as_serializable(self):
        r""""
        Returns a serializable version of the fitting result. This is a much
        lighter weight object than the initial fitting result. For example,
        it won't contain the original fitting object.

        Returns
        -------
        serializable_fitting_result : :map:`SerializableFittingResult`
            The lightweight serializable version of this fitting result.
        """
        gt_shape = self.gt_shape.copy() if self.gt_shape else None
        fr_copies = [fr.as_serializable() for fr in self.fitting_results]

        return SerializableMultilevelFittingResult(
            self.image.copy(), fr_copies,
            gt_shape, self.n_levels, self.downscale, self.n_iters,
            self._affine_correction.copy())


class AMMultilevelFittingResult(MultilevelFittingResult):
    r"""
    Class that holds the state of an Active Model (either AAM or ATM).
    """
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
        raise ValueError('costs not implemented yet.')

    @property
    def initial_cost(self):
        r"""
        Returns the initial fitting cost.

        :type: `float`
        """
        raise ValueError('costs not implemented yet.')

    @property
    def warped_images(self):
        r"""
        The list containing the warped images obtained at each fitting
        iteration.

        :type: `list` of :map:`Image` or subclass
        """
        mask = self.fitting_results[-1].fitter.template.mask
        transform = self.fitting_results[-1].fitter.transform
        warped_images = []
        for s in self.shapes():
            transform.set_target(s)
            image = self.image.warp_to_mask(mask, transform)
            warped_images.append(image)

        return warped_images

    @property
    def error_images(self):
        r"""
        The list containing the error images obtained at each fitting
        iteration.

        :type: `list` of :map:`Image` or subclass
        """
        return list(chain(
            *[f.error_images for f in self.fitting_results]))


class SerializableMultilevelFittingResult(FittingResult):
    r"""
    Designed to allow the fitting results to be easily serializable. In
    comparison to the other fitting result objects, the serializable fitting
    results contain a much stricter set of data. For example, the major data
    components of a serializable fitting result are the fitted shapes, the
    parameters and the fitted image.

    Parameters
    -----------
    image : :map:`Image`
        The fitted image.
    shapes : `list` of :map:`PointCloud`
        The list of fitted shapes per iteration of the fitting procedure.
    gt_shape : :map:`PointCloud`
        The ground truth shape associated to the image.
    n_levels : `int`
        Number of levels within the multilevel fitter.
    downscale : `int`
        Scale of downscaling applied to the image.
    n_iters : `int`
        Number of iterations the fitter performed.
    """
    def __init__(self, image, fitting_results, gt_shape, n_levels,
                 downscale, n_iters, affine_correction):
        FittingResult.__init__(self, image, gt_shape=gt_shape)
        self.fitting_results = fitting_results
        self.n_levels = n_levels
        self._n_iters = n_iters
        self.downscale = downscale
        self.affine_correction = affine_correction

    @property
    def n_iters(self):
        return self._n_iters

    @property
    def final_shape(self):
        return self.shapes[-1]

    @property
    def initial_shape(self):
        return self.shapes[0]

    @property
    def shapes(self):
        return _rescale_shapes_to_reference(self.fitting_results, self.n_levels,
                                            self.downscale,
                                            self.affine_correction)


def _rescale_shapes_to_reference(fitting_results, n_levels, downscale,
                                 affine_correction):
    n = n_levels - 1
    shapes = []
    for j, f in enumerate(fitting_results):
        transform = Scale(downscale ** (n - j), f.final_shape.n_dims)
        for t in f.shapes:
            t = transform.apply(t)
            shapes.append(affine_correction.apply(t))
    return shapes


def compute_error(target, ground_truth, error_type='me_norm'):
    r"""
    """
    gt_points = ground_truth.points
    target_points = target.points

    if error_type == 'me_norm':
        return _compute_me_norm(target_points, gt_points)
    elif error_type == 'me':
        return _compute_me(target_points, gt_points)
    elif error_type == 'rmse':
        return _compute_rmse(target_points, gt_points)
    else:
        raise ValueError("Unknown error_type string selected. Valid options "
                         "are: me_norm, me, rmse'")


def _compute_me(target, ground_truth):
    r"""
    """
    return np.mean(np.sqrt(np.sum((target - ground_truth) ** 2, axis=-1)))


def _compute_rmse(target, ground_truth):
    r"""
    """
    return np.sqrt(np.mean((target.flatten() - ground_truth.flatten()) ** 2))


def _compute_me_norm(target, ground_truth):
    r"""
    """
    normalizer = np.mean(np.max(ground_truth, axis=0) -
                         np.min(ground_truth, axis=0))
    return _compute_me(target, ground_truth) / normalizer


def compute_cumulative_error(errors, x_axis):
    r"""
    """
    n_errors = len(errors)
    cumulative_error = [np.count_nonzero([errors <= x])
                        for x in x_axis]
    return np.array(cumulative_error) / n_errors
