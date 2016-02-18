import numpy as np

from menpofit.error import compute_cumulative_error


def plot_cumulative_error_distribution(
        errors, error_range=None, figure_id=None, new_figure=False,
        title='Cumulative Error Distribution',
        x_label='Normalized Point-to-Point Error', y_label='Images Proportion',
        legend_entries=None, render_lines=True, line_colour=None,
        line_style='-', line_width=2, render_markers=True, marker_style='s',
        marker_size=7, marker_face_colour='w', marker_edge_colour=None,
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
        Specifies the horizontal axis range, i.e. ::

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
    line_colour : `colour` or `list` of `colour` or ``None``, optional
        The colour of the lines. If not a `list`, this value will be
        used for all curves. If `list`, a value must be specified for each
        curve, thus it must have the same length as `y_axis`. If ``None``, the
        colours will be linearly sampled from jet colormap.
        Example `colour` options are ::

                {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                or
                (3, ) ndarray

    line_style : ``{'-', '--', '-.', ':'}`` or `list` of those, optional
        The style of the lines. If not a `list`, this value will be used for all
        curves. If `list`, a value must be specified for each curve, thus it
        must have the same length as `errors`.
    line_width : `float` or `list` of `float`, optional
        The width of the lines. If `float`, this value will be used for all
        curves. If `list`, a value must be specified for each curve, thus it
        must have the same length as `errors`.
    render_markers : `bool` or `list` of `bool`, optional
        If ``True``, the markers will be rendered. If `bool`, this value will be
        used for all curves. If `list`, a value must be specified for each
        curve, thus it must have the same length as `errors`.
    marker_style : `marker` or `list` of `markers`, optional
        The style of the markers. If not a `list`, this value will be used for
        all curves. If `list`, a value must be specified for each curve, thus it
        must have the same length as `errors`.
        Example `marker` options ::

                {'.', ',', 'o', 'v', '^', '<', '>', '+', 'x', 'D', 'd', 's',
                 'p', '*', 'h', 'H', '1', '2', '3', '4', '8'}

    marker_size : `int` or `list` of `int`, optional
        The size of the markers in points. If `int`, this value will be used
        for all curves. If `list`, a value must be specified for each curve, thus
        it must have the same length as `errors`.
    marker_face_colour : `colour` or `list` of `colour` or ``None``, optional
        The face (filling) colour of the markers. If not a `list`, this value
        will be used for all curves. If `list`, a value must be specified for
        each curve, thus it must have the same length as `errors`. If ``None``,
        the colours will be linearly sampled from jet colormap.
        Example `colour` options are ::

                {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                or
                (3, ) ndarray

    marker_edge_colour : `colour` or `list` of `colour` or ``None``, optional
        The edge colour of the markers. If not a `list`, this value will be used
        for all curves. If `list`, a value must be specified for each curve, thus
        it must have the same length as `errors`. If ``None``, the colours will
        be linearly sampled from jet colormap.
        Example `colour` options are ::

                {'r', 'g', 'b', 'c', 'm', 'k', 'w'}
                or
                (3, ) ndarray

    marker_edge_width : `float` or `list` of `float`, optional
        The width of the markers' edge. If `float`, this value will be used for
        all curves. If `list`, a value must be specified for each curve, thus it
        must have the same length as `errors`.
    render_legend : `bool`, optional
        If ``True``, the legend will be rendered.
    legend_title : `str`, optional
        The title of the legend.
    legend_font_name : See below, optional
        The font of the legend.
        Example options ::

            {'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'}

    legend_font_style : ``{'normal', 'italic', 'oblique'}``, optional
        The font style of the legend.
    legend_font_size : `int`, optional
        The font size of the legend.
    legend_font_weight : See below, optional
        The font weight of the legend.
        Example options ::

            {'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
             'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
             'extra bold', 'black'}

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
        `tuple` or `list`, then it defines the axis limits. If ``None``, then
        the limits are set to ``(0., error_range[1])``.
    axes_y_limits : `float` or (`float`, `float`) or ``None``, optional
        The limits of the y axis. If `float`, then it sets padding on the
        top and bottom of the graph as a percentage of the curves' height. If
        `tuple` or `list`, then it defines the axis limits. If ``None``, then
        the limits are set to ``(0., 1.)``.
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

    Raises
    ------
    ValueError
        legend_entries list has different length than errors list

    Returns
    -------
    viewer : `menpo.visualize.GraphPlotter`
        The viewer object.
    """
    from menpo.visualize import plot_curve

    # make sure that errors is a list even with one list member
    if not isinstance(errors[0], list):
        errors = [errors]

    # create x and y axes lists
    if error_range is None:
        error_range = [0., 0.101, 0.005]
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
            x_axis=x_axis, y_axis=ceds, figure_id=figure_id,
            new_figure=new_figure, legend_entries=legend_entries,
            title=title, x_label=x_label, y_label=y_label,
            axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
            axes_x_ticks=axes_x_ticks, axes_y_ticks=axes_y_ticks,
            render_lines=render_lines, line_colour=line_colour,
            line_style=line_style, line_width=line_width,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width, render_legend=render_legend,
            legend_title=legend_title, legend_font_name=legend_font_name,
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
            axes_font_weight=axes_font_weight, figure_size=figure_size,
            render_grid=render_grid, grid_line_style=grid_line_style,
            grid_line_width=grid_line_width)
