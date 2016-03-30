import numpy as np

from menpo.image import Image

from menpofit.visualize import view_image_multiple_landmarks
from menpofit.result import (NonParametricIterativeResult,
                             MultiScaleNonParametricIterativeResult,
                             _parse_iters, _get_scale_of_iter)


class LucasKanadeAlgorithmResult(NonParametricIterativeResult):
    r"""
    Class for storing the iterative result of a Lucas-Kanade Image Alignment
    optimization algorithm.

    Parameters
    ----------
    shapes : `list` of `menpo.shape.PointCloud`
        The `list` of shapes per iteration. The first and last members
        correspond to the initial and final shapes, respectively.
    homogeneous_parameters : `list` of ``(n_parameters,)`` `ndarray`
        The `list` of parameters of the homogeneous transform per iteration.
        The first and last members correspond to the initial and final
        shapes, respectively.
    initial_shape : `menpo.shape.PointCloud` or ``None``, optional
        The initial shape from which the fitting process started. If
        ``None``, then no initial shape is assigned.
    cost_functions : `list` of `callable` or ``None``, optional
        The `list` of methods that compute the cost per iteration.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    gt_shape : `menpo.shape.PointCloud` or ``None``, optional
        The ground truth shape associated with the image. If ``None``, then no
        ground truth shape is assigned.
    """
    def __init__(self, shapes, homogeneous_parameters, initial_shape=None,
                 cost_functions=None, image=None, gt_shape=None):
        super(LucasKanadeAlgorithmResult, self).__init__(
            shapes=shapes, initial_shape=initial_shape, image=image,
            gt_shape=gt_shape)
        self._homogeneous_parameters = homogeneous_parameters
        self._cost_functions = cost_functions

    @property
    def homogeneous_parameters(self):
        r"""
        Returns the `list` of parameters of the homogeneous transform
        obtained at each iteration of the fitting process. The `list`
        includes the parameters of the `initial_shape` (if it exists) and
        `final_shape`.

        :type: `list` of ``(n_params,)`` `ndarray`
        """
        return self._homogeneous_parameters

    @property
    def costs(self):
        r"""
        Returns a `list` with the cost per iteration.

        :type: `list` of `float`
        """
        if self._cost_functions is not None:
            return [f() for f in self._cost_functions]
        else:
            return None

    def plot_costs(self, figure_id=None, new_figure=False, render_lines=True,
                   line_colour='b', line_style='-', line_width=2,
                   render_markers=True, marker_style='o', marker_size=4,
                   marker_face_colour='b', marker_edge_colour='k',
                   marker_edge_width=1., render_axes=True,
                   axes_font_name='sans-serif', axes_font_size=10,
                   axes_font_style='normal', axes_font_weight='normal',
                   axes_x_limits=0., axes_y_limits=None, axes_x_ticks=None,
                   axes_y_ticks=None, figure_size=(10, 6),
                   render_grid=True, grid_line_style='--',
                   grid_line_width=0.5):
        r"""
        Plots the cost function evolution at each fitting iteration.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_lines : `bool`, optional
            If ``True``, the line will be rendered.
        line_colour : `colour` or ``None``, optional
            The colour of the line. If ``None``, the colour is sampled from
            the jet colormap.
            Example `colour` options are ::

                    {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                    or
                    (3, ) ndarray

        line_style : ``{'-', '--', '-.', ':'}``, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : `marker`, optional
            The style of the markers.
            Example `marker` options ::

                    {'.', ',', 'o', 'v', '^', '<', '>', '+', 'x', 'D', 'd', 's',
                     'p', '*', 'h', 'H', '1', '2', '3', '4', '8'}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : `colour` or ``None``, optional
            The face (filling) colour of the markers. If ``None``, the colour
            is sampled from the jet colormap.
            Example `colour` options are ::

                    {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                    or
                    (3, ) ndarray

        marker_edge_colour : `colour` or ``None``, optional
            The edge colour of the markers.If ``None``, the colour
            is sampled from the jet colormap.
            Example `colour` options are ::

                    {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                    or
                    (3, ) ndarray

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See below, optional
            The font of the axes.
            Example options ::

                {'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{'normal', 'italic', 'oblique'}``, optional
            The font style of the axes.
        axes_font_weight : See below, optional
            The font weight of the axes.
            Example options ::

                {'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
                 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
                 'extra bold', 'black'}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the graph as a percentage of the curves' width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_y_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the graph as a percentage of the curves' height.
            If `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) or ``None``, optional
            The size of the figure in inches.
        render_grid : `bool`, optional
            If ``True``, the grid will be rendered.
        grid_line_style : ``{'-', '--', '-.', ':'}``, optional
            The style of the grid lines.
        grid_line_width : `float`, optional
            The width of the grid lines.

        Returns
        -------
        renderer : `menpo.visualize.GraphPlotter`
            The renderer object.
        """
        from menpo.visualize import plot_curve
        costs = self.costs
        if costs is not None:
            return plot_curve(
                    x_axis=list(range(len(costs))), y_axis=[costs],
                    figure_id=figure_id, new_figure=new_figure,
                    title='Cost per Iteration', x_label='Iteration',
                    y_label='Cost Function', axes_x_limits=axes_x_limits,
                    axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
                    axes_y_ticks=axes_y_ticks, render_lines=render_lines,
                    line_colour=line_colour, line_style=line_style,
                    line_width=line_width, render_markers=render_markers,
                    marker_style=marker_style, marker_size=marker_size,
                    marker_face_colour=marker_face_colour,
                    marker_edge_colour=marker_edge_colour,
                    marker_edge_width=marker_edge_width, render_legend=False,
                    render_axes=render_axes, axes_font_name=axes_font_name,
                    axes_font_size=axes_font_size,
                    axes_font_style=axes_font_style,
                    axes_font_weight=axes_font_weight, figure_size=figure_size,
                    render_grid=render_grid,  grid_line_style=grid_line_style,
                    grid_line_width=grid_line_width)
        else:
            raise ValueError('costs is not well defined for the chosen ATM '
                             'algorithm')

    @property
    def _reconstruction_indices(self):
        r"""
        Returns a list with the indices of reconstructed shapes in the `shapes`
        list.

        :type: `list` of `int`
        """
        if self.initial_shape is not None:
            return [1]
        else:
            return [0]

    def view_iterations(self, figure_id=None, new_figure=False,
                        iters=None, render_image=True, subplots_enabled=False,
                        channels=None, interpolation='bilinear',
                        cmap_name=None, alpha=1., masked=True, render_lines=True,
                        line_style='-', line_width=2, line_colour=None,
                        render_markers=True, marker_edge_colour=None,
                        marker_face_colour=None, marker_style='o',
                        marker_size=4, marker_edge_width=1.,
                        render_numbering=False,
                        numbers_horizontal_align='center',
                        numbers_vertical_align='bottom',
                        numbers_font_name='sans-serif', numbers_font_size=10,
                        numbers_font_style='normal',
                        numbers_font_weight='normal',
                        numbers_font_colour='k', render_legend=True,
                        legend_title='', legend_font_name='sans-serif',
                        legend_font_style='normal', legend_font_size=10,
                        legend_font_weight='normal', legend_marker_scale=None,
                        legend_location=2, legend_bbox_to_anchor=(1.05, 1.),
                        legend_border_axes_pad=None, legend_n_columns=1,
                        legend_horizontal_spacing=None,
                        legend_vertical_spacing=None, legend_border=True,
                        legend_border_padding=None, legend_shadow=False,
                        legend_rounded_corners=False, render_axes=False,
                        axes_font_name='sans-serif', axes_font_size=10,
                        axes_font_style='normal', axes_font_weight='normal',
                        axes_x_limits=None, axes_y_limits=None,
                        axes_x_ticks=None, axes_y_ticks=None,
                        figure_size=(10, 8)):
        """
        Visualize the iterations of the fitting process.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        iters : `int` or `list` of `int` or ``None``, optional
            The iterations to be visualized. If ``None``, then all the
            iterations are rendered.

            ========= ==================================== ======================
            No.       Visualised shape                     Description
            ========= ==================================== ======================
            0           `self.initial_shape`               Initial shape
            1           `self.reconstructed_initial_shape` Reconstructed initial
            2           `self.shapes[2]`                   Iteration 1
            i           `self.shapes[i]`                   Iteration i-1
            n_iters+1 `self.final_shape`                   Final shape
            ========= ==================================== ======================

        render_image : `bool`, optional
            If ``True`` and the image exists, then it gets rendered.
        subplots_enabled : `bool`, optional
            If ``True``, then the requested final, initial and ground truth
            shapes get rendered on separate subplots.
        channels : `int` or `list` of `int` or ``all`` or ``None``
            If `int` or `list` of `int`, the specified channel(s) will be
            rendered. If ``all``, all the channels will be rendered in subplots.
            If ``None`` and the image is RGB, it will be rendered in RGB mode.
            If ``None`` and the image is not RGB, it is equivalent to ``all``.
        interpolation : `str` (See Below), optional
            The interpolation used to render the image. For example, if
            ``bilinear``, the image will be smooth and if ``nearest``, the
            image will be pixelated.
            Example options ::

                {none, nearest, bilinear, bicubic, spline16, spline36, hanning,
                 hamming, hermite, kaiser, quadric, catrom, gaussian, bessel,
                 mitchell, sinc, lanczos}

        cmap_name: `str`, optional,
            If ``None``, single channel and three channel images default
            to greyscale and rgb colormaps respectively.
        alpha : `float`, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        masked : `bool`, optional
            If ``True``, then the image is rendered as masked.
        render_lines : `bool` or `list` of `bool`, optional
            If ``True``, the lines will be rendered. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per iteration shape.
        line_style : `str` or `list` of `str` (See below), optional
            The style of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
            Example options::

                {-, --, -., :}

        line_width : `float` or `list` of `float`, optional
            The width of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
        line_colour : `colour` or `list` of `colour` (See Below), optional
            The colour of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_markers : `bool` or `list` of `bool`, optional
            If ``True``, the markers will be rendered. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per iteration shape.
        marker_style : `str or `list` of `str` (See below), optional
            The style of the markers. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
            Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int` or `list` of `int`, optional
            The size of the markers in points. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per iteration shape.
        marker_edge_colour : `colour` or `list` of `colour` (See Below), optional
            The edge colour of the markers. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per iteration shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_face_colour : `colour` or `list` of `colour` (See Below), optional
            The face (filling) colour of the markers. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per iteration shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float` or `list` of `float`, optional
            The width of the markers' edge. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per iteration shape.
        render_numbering : `bool`, optional
            If ``True``, the landmarks will be numbered.
        numbers_horizontal_align : `str` (See below), optional
            The horizontal alignment of the numbers' texts.
            Example options ::

                {center, right, left}

        numbers_vertical_align : `str` (See below), optional
            The vertical alignment of the numbers' texts.
            Example options ::

                {center, top, bottom, baseline}

        numbers_font_name : `str` (See below), optional
            The font of the numbers.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        numbers_font_size : `int`, optional
            The font size of the numbers.
        numbers_font_style : ``{normal, italic, oblique}``, optional
            The font style of the numbers.
        numbers_font_weight : `str` (See below), optional
            The font weight of the numbers.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        numbers_font_colour : See Below, optional
            The font colour of the numbers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_legend : `bool`, optional
            If ``True``, the legend will be rendered.
        legend_title : `str`, optional
            The title of the legend.
        legend_font_name : See below, optional
            The font of the legend.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        legend_font_style : `str` (See below), optional
            The font style of the legend.
            Example options ::

                {normal, italic, oblique}

        legend_font_size : `int`, optional
            The font size of the legend.
        legend_font_weight : `str` (See below), optional
            The font weight of the legend.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        legend_marker_scale : `float`, optional
            The relative size of the legend markers with respect to the original
        legend_location : `int`, optional
            The location of the legend. The predefined values are:

            =============== ==
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
            =============== ==

        legend_bbox_to_anchor : (`float`, `float`) `tuple`, optional
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
        axes_font_name : `str` (See below), optional
            The font of the axes.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{normal, italic, oblique}``, optional
            The font style of the axes.
        axes_font_weight : `str` (See below), optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the Image as a percentage of the Image's width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the Image as a percentage of the Image's height.
            If `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.

        Returns
        -------
        renderer : `class`
            The renderer object.
        """
        # Parse iters
        iters = _parse_iters(iters, len(self.shapes))
        # Create image instance
        if self.image is None:
            image = Image(np.zeros((10, 10)))
            render_image = False
        else:
            image = Image(self.image.pixels)
        # Assign pointclouds to image
        n_digits = len(str(self.n_iters))
        groups = []
        subplots_titles = {}
        iters_offset = 0
        if self.initial_shape is not None:
            iters_offset = 1
        for j in iters:
            if j == 0 and self.initial_shape is not None:
                name = 'Initial'
                image.landmarks[name] = self.initial_shape
            elif j in self._reconstruction_indices:
                name = 'Reconstruction'
                image.landmarks[name] = self.shapes[j]
            elif j == len(self.shapes) - 1:
                name = 'Final'
                image.landmarks[name] = self.final_shape
            else:
                s = _get_scale_of_iter(j, self._reconstruction_indices)
                name = "iteration {:0{}d}".format(j - s + iters_offset, n_digits)
                image.landmarks[name] = self.shapes[j]
            groups.append(name)
            subplots_titles[name] = name
        # Render
        return view_image_multiple_landmarks(
            image, groups, with_labels=None, figure_id=figure_id,
            new_figure=new_figure, subplots_enabled=subplots_enabled,
            subplots_titles=subplots_titles, render_image=render_image,
            render_landmarks=True, masked=masked,
            channels=channels, interpolation=interpolation,
            cmap_name=cmap_name, alpha=alpha, image_view=True,
            render_lines=render_lines, line_style=line_style,
            line_width=line_width, line_colour=line_colour,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_edge_width=marker_edge_width,
            marker_edge_colour=marker_edge_colour,
            marker_face_colour=marker_face_colour,
            render_numbering=render_numbering,
            numbers_horizontal_align=numbers_horizontal_align,
            numbers_vertical_align=numbers_vertical_align,
            numbers_font_name=numbers_font_name,
            numbers_font_size=numbers_font_size,
            numbers_font_style=numbers_font_style,
            numbers_font_weight=numbers_font_weight,
            numbers_font_colour=numbers_font_colour,
            render_legend=render_legend, legend_title=legend_title,
            legend_font_name=legend_font_name,
            legend_font_style=legend_font_style,
            legend_font_size=legend_font_size,
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
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)


class LucasKanadeResult(MultiScaleNonParametricIterativeResult):
    r"""
    Class for storing the multi-scale iterative fitting result of an ATM. It
    holds the shapes, shape parameters and costs per iteration.

    Parameters
    ----------
    results : `list` of :map:`ATMAlgorithmResult`
        The `list` of optimization results per scale.
    scales : `list` or `tuple`
        The `list` of scale values per scale (low to high).
    affine_transforms : `list` of `menpo.transform.Affine`
        The list of affine transforms per scale that transform the shapes into
        the original image space.
    scale_transforms : `list` of `menpo.shape.Scale`
        The list of scaling transforms per scale.
    image : `menpo.image.Image` or `subclass` or ``None``, optional
        The image on which the fitting process was applied. Note that a copy
        of the image will be assigned as an attribute. If ``None``, then no
        image is assigned.
    gt_shape : `menpo.shape.PointCloud` or ``None``, optional
        The ground truth shape associated with the image. If ``None``, then no
        ground truth shape is assigned.
    """
    def __init__(self, results, scales, affine_transforms, scale_transforms,
                 image=None, gt_shape=None):
        super(LucasKanadeResult, self).__init__(
            results=results, scales=scales, affine_transforms=affine_transforms,
            scale_transforms=scale_transforms, image=image, gt_shape=gt_shape)
        # Create parameters list
        self._homogeneous_parameters = results[0].homogeneous_parameters
        for r in results[1:]:
            if r.initial_shape is None:
                self._homogeneous_parameters += r.homogeneous_parameters
            else:
                self._homogeneous_parameters += r.homogeneous_parameters[1:]
        # Create costs list
        self._costs = None
        if results[0].costs is not None:
            self._costs = []
            for r in results:
                self._costs += r.costs

    @property
    def homogeneous_parameters(self):
        r"""
        Returns the `list` of parameters of the homogeneous transform
        obtained at each iteration of the fitting process. The `list`
        includes the parameters of the `initial_shape` (if it exists) and
        `final_shape`.

        :type: `list` of ``(n_params,)`` `ndarray`
        """
        return self._homogeneous_parameters

    @property
    def costs(self):
        r"""
        Returns a `list` with the cost per iteration.

        :type: `list` of `float`
        """
        return self._costs

    def plot_costs(self, figure_id=None, new_figure=False, render_lines=True,
                   line_colour='b', line_style='-', line_width=2,
                   render_markers=True, marker_style='o', marker_size=4,
                   marker_face_colour='b', marker_edge_colour='k',
                   marker_edge_width=1., render_axes=True,
                   axes_font_name='sans-serif', axes_font_size=10,
                   axes_font_style='normal', axes_font_weight='normal',
                   axes_x_limits=0., axes_y_limits=None, axes_x_ticks=None,
                   axes_y_ticks=None, figure_size=(10, 6),
                   render_grid=True, grid_line_style='--',
                   grid_line_width=0.5):
        r"""
        Plot of the cost function evolution at each fitting iteration.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_lines : `bool`, optional
            If ``True``, the line will be rendered.
        line_colour : `colour` or ``None``, optional
            The colour of the line. If ``None``, the colour is sampled from
            the jet colormap.
            Example `colour` options are ::

                    {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                    or
                    (3, ) ndarray

        line_style : ``{'-', '--', '-.', ':'}``, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : `marker`, optional
            The style of the markers.
            Example `marker` options ::

                    {'.', ',', 'o', 'v', '^', '<', '>', '+', 'x', 'D', 'd', 's',
                     'p', '*', 'h', 'H', '1', '2', '3', '4', '8'}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : `colour` or ``None``, optional
            The face (filling) colour of the markers. If ``None``, the colour
            is sampled from the jet colormap.
            Example `colour` options are ::

                    {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                    or
                    (3, ) ndarray

        marker_edge_colour : `colour` or ``None``, optional
            The edge colour of the markers.If ``None``, the colour
            is sampled from the jet colormap.
            Example `colour` options are ::

                    {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                    or
                    (3, ) ndarray

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See below, optional
            The font of the axes.
            Example options ::

                {'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{'normal', 'italic', 'oblique'}``, optional
            The font style of the axes.
        axes_font_weight : See below, optional
            The font weight of the axes.
            Example options ::

                {'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
                 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
                 'extra bold', 'black'}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the graph as a percentage of the curves' width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_y_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the graph as a percentage of the curves' height.
            If `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) or ``None``, optional
            The size of the figure in inches.
        render_grid : `bool`, optional
            If ``True``, the grid will be rendered.
        grid_line_style : ``{'-', '--', '-.', ':'}``, optional
            The style of the grid lines.
        grid_line_width : `float`, optional
            The width of the grid lines.

        Returns
        -------
        renderer : `menpo.visualize.GraphPlotter`
            The renderer object.
        """
        from menpo.visualize import plot_curve
        costs = self.costs
        if costs is not None:
            return plot_curve(
                    x_axis=list(range(len(costs))), y_axis=[costs],
                    figure_id=figure_id, new_figure=new_figure,
                    title='Cost per Iteration', x_label='Iteration',
                    y_label='Cost Function', axes_x_limits=axes_x_limits,
                    axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
                    axes_y_ticks=axes_y_ticks, render_lines=render_lines,
                    line_colour=line_colour, line_style=line_style,
                    line_width=line_width, render_markers=render_markers,
                    marker_style=marker_style, marker_size=marker_size,
                    marker_face_colour=marker_face_colour,
                    marker_edge_colour=marker_edge_colour,
                    marker_edge_width=marker_edge_width, render_legend=False,
                    render_axes=render_axes, axes_font_name=axes_font_name,
                    axes_font_size=axes_font_size,
                    axes_font_style=axes_font_style,
                    axes_font_weight=axes_font_weight, figure_size=figure_size,
                    render_grid=render_grid,  grid_line_style=grid_line_style,
                    grid_line_width=grid_line_width)
        else:
            raise ValueError('costs is not well defined for the chosen ATM '
                             'algorithm')

    @property
    def _reconstruction_indices(self):
        r"""
        Returns a list with the indices of reconstructed shapes in the `shapes`
        list.

        :type: `list` of `int`
        """
        initial_val = 0
        if self.initial_shape is not None:
            initial_val = 1
        ids = []
        for i in list(range(self.n_scales)):
            if i == 0:
                ids.append(initial_val)
            else:
                previous_val = ids[i - 1]
                ids.append(previous_val + self.n_iters_per_scale[i - 1] + 1)
        return ids

    def view_iterations(self, figure_id=None, new_figure=False,
                        iters=None, render_image=True, subplots_enabled=False,
                        channels=None, interpolation='bilinear',
                        cmap_name=None, alpha=1., masked=True, render_lines=True,
                        line_style='-', line_width=2, line_colour=None,
                        render_markers=True, marker_edge_colour=None,
                        marker_face_colour=None, marker_style='o',
                        marker_size=4, marker_edge_width=1.,
                        render_numbering=False,
                        numbers_horizontal_align='center',
                        numbers_vertical_align='bottom',
                        numbers_font_name='sans-serif', numbers_font_size=10,
                        numbers_font_style='normal',
                        numbers_font_weight='normal',
                        numbers_font_colour='k', render_legend=True,
                        legend_title='', legend_font_name='sans-serif',
                        legend_font_style='normal', legend_font_size=10,
                        legend_font_weight='normal', legend_marker_scale=None,
                        legend_location=2, legend_bbox_to_anchor=(1.05, 1.),
                        legend_border_axes_pad=None, legend_n_columns=1,
                        legend_horizontal_spacing=None,
                        legend_vertical_spacing=None, legend_border=True,
                        legend_border_padding=None, legend_shadow=False,
                        legend_rounded_corners=False, render_axes=False,
                        axes_font_name='sans-serif', axes_font_size=10,
                        axes_font_style='normal', axes_font_weight='normal',
                        axes_x_limits=None, axes_y_limits=None,
                        axes_x_ticks=None, axes_y_ticks=None,
                        figure_size=(10, 8)):
        """
        Visualize the iterations of the fitting process.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        iters : `int` or `list` of `int` or ``None``, optional
            The iterations to be visualized. If ``None``, then all the
            iterations are rendered.

            ========= ==================================== ======================
            No.       Visualised shape                     Description
            ========= ==================================== ======================
            0           `self.initial_shape`               Initial shape
            1           `self.reconstructed_initial_shape` Reconstructed initial
            2           `self.shapes[2]`                   Iteration 1
            i           `self.shapes[i]`                   Iteration i-1
            n_iters+1 `self.final_shape`                   Final shape
            ========= ==================================== ======================

        render_image : `bool`, optional
            If ``True`` and the image exists, then it gets rendered.
        subplots_enabled : `bool`, optional
            If ``True``, then the requested final, initial and ground truth
            shapes get rendered on separate subplots.
        channels : `int` or `list` of `int` or ``all`` or ``None``
            If `int` or `list` of `int`, the specified channel(s) will be
            rendered. If ``all``, all the channels will be rendered in subplots.
            If ``None`` and the image is RGB, it will be rendered in RGB mode.
            If ``None`` and the image is not RGB, it is equivalent to ``all``.
        interpolation : `str` (See Below), optional
            The interpolation used to render the image. For example, if
            ``bilinear``, the image will be smooth and if ``nearest``, the
            image will be pixelated.
            Example options ::

                {none, nearest, bilinear, bicubic, spline16, spline36, hanning,
                 hamming, hermite, kaiser, quadric, catrom, gaussian, bessel,
                 mitchell, sinc, lanczos}

        cmap_name: `str`, optional,
            If ``None``, single channel and three channel images default
            to greyscale and rgb colormaps respectively.
        alpha : `float`, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        masked : `bool`, optional
            If ``True``, then the image is rendered as masked.
        render_lines : `bool` or `list` of `bool`, optional
            If ``True``, the lines will be rendered. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per iteration shape.
        line_style : `str` or `list` of `str` (See below), optional
            The style of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
            Example options::

                {-, --, -., :}

        line_width : `float` or `list` of `float`, optional
            The width of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
        line_colour : `colour` or `list` of `colour` (See Below), optional
            The colour of the lines. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_markers : `bool` or `list` of `bool`, optional
            If ``True``, the markers will be rendered. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per iteration shape.
        marker_style : `str or `list` of `str` (See below), optional
            The style of the markers. You can either provide a single value that
            will be used for all shapes or a list with a different value per
            iteration shape.
            Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int` or `list` of `int`, optional
            The size of the markers in points. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per iteration shape.
        marker_edge_colour : `colour` or `list` of `colour` (See Below), optional
            The edge colour of the markers. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per iteration shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_face_colour : `colour` or `list` of `colour` (See Below), optional
            The face (filling) colour of the markers. You can either provide a
            single value that will be used for all shapes or a list with a
            different value per iteration shape.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float` or `list` of `float`, optional
            The width of the markers' edge. You can either provide a single
            value that will be used for all shapes or a list with a different
            value per iteration shape.
        render_numbering : `bool`, optional
            If ``True``, the landmarks will be numbered.
        numbers_horizontal_align : `str` (See below), optional
            The horizontal alignment of the numbers' texts.
            Example options ::

                {center, right, left}

        numbers_vertical_align : `str` (See below), optional
            The vertical alignment of the numbers' texts.
            Example options ::

                {center, top, bottom, baseline}

        numbers_font_name : `str` (See below), optional
            The font of the numbers.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        numbers_font_size : `int`, optional
            The font size of the numbers.
        numbers_font_style : ``{normal, italic, oblique}``, optional
            The font style of the numbers.
        numbers_font_weight : `str` (See below), optional
            The font weight of the numbers.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        numbers_font_colour : See Below, optional
            The font colour of the numbers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_legend : `bool`, optional
            If ``True``, the legend will be rendered.
        legend_title : `str`, optional
            The title of the legend.
        legend_font_name : See below, optional
            The font of the legend.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        legend_font_style : `str` (See below), optional
            The font style of the legend.
            Example options ::

                {normal, italic, oblique}

        legend_font_size : `int`, optional
            The font size of the legend.
        legend_font_weight : `str` (See below), optional
            The font weight of the legend.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        legend_marker_scale : `float`, optional
            The relative size of the legend markers with respect to the original
        legend_location : `int`, optional
            The location of the legend. The predefined values are:

            =============== ==
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
            =============== ==

        legend_bbox_to_anchor : (`float`, `float`) `tuple`, optional
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
        axes_font_name : `str` (See below), optional
            The font of the axes.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{normal, italic, oblique}``, optional
            The font style of the axes.
        axes_font_weight : `str` (See below), optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the Image as a percentage of the Image's width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the Image as a percentage of the Image's height.
            If `tuple` or `list`, then it defines the axis limits. If ``None``,
            then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.

        Returns
        -------
        renderer : `class`
            The renderer object.
        """
        # Parse iters
        iters = _parse_iters(iters, len(self.shapes))
        # Create image instance
        if self.image is None:
            image = Image(np.zeros((10, 10)))
            render_image = False
        else:
            image = Image(self.image.pixels)
        # Assign pointclouds to image
        n_digits = len(str(self.n_iters))
        groups = []
        subplots_titles = {}
        iters_offset = -2
        if self.initial_shape is not None:
            iters_offset = -1
        for j in iters:
            if j == 0 and self.initial_shape is not None:
                name = 'Initial'
                image.landmarks[name] = self.initial_shape
            elif j in self._reconstruction_indices:
                name = 'Reconstruction'
                image.landmarks[name] = self.shapes[j]
            elif j == len(self.shapes) - 1:
                name = 'Final'
                image.landmarks[name] = self.final_shape
            else:
                s = _get_scale_of_iter(j, self._reconstruction_indices)
                name = "iteration {:0{}d}".format(j - s + iters_offset, n_digits)
                image.landmarks[name] = self.shapes[j]
            groups.append(name)
            subplots_titles[name] = name
        # Render
        return view_image_multiple_landmarks(
            image, groups, with_labels=None, figure_id=figure_id,
            new_figure=new_figure, subplots_enabled=subplots_enabled,
            subplots_titles=subplots_titles, render_image=render_image,
            render_landmarks=True, masked=masked,
            channels=channels, interpolation=interpolation,
            cmap_name=cmap_name, alpha=alpha, image_view=True,
            render_lines=render_lines, line_style=line_style,
            line_width=line_width, line_colour=line_colour,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_edge_width=marker_edge_width,
            marker_edge_colour=marker_edge_colour,
            marker_face_colour=marker_face_colour,
            render_numbering=render_numbering,
            numbers_horizontal_align=numbers_horizontal_align,
            numbers_vertical_align=numbers_vertical_align,
            numbers_font_name=numbers_font_name,
            numbers_font_size=numbers_font_size,
            numbers_font_style=numbers_font_style,
            numbers_font_weight=numbers_font_weight,
            numbers_font_colour=numbers_font_colour,
            render_legend=render_legend, legend_title=legend_title,
            legend_font_name=legend_font_name,
            legend_font_style=legend_font_style,
            legend_font_size=legend_font_size,
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
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)
