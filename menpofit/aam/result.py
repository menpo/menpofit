from menpofit.result import (ParametricIterativeResult,
                             MultiScaleParametricIterativeResult)


class AAMAlgorithmResult(ParametricIterativeResult):
    r"""
    Class for storing the iterative result of an AAM optimisation algorithm.

    .. note:: When using a method with a parametric shape model, the first step
              is to **reconstruct the initial shape** using the shape model. The
              generated reconstructed shape is then used as initialisation for
              the iterative optimisation. This step is not counted in the number
              of iterations.

    Parameters
    ----------
    shapes : `list` of `menpo.shape.PointCloud`
        The `list` of shapes per iteration. The first and last members
        correspond to the initial and final shapes, respectively.
    shape_parameters : `list` of ``(n_shape_parameters,)`` `ndarray`
        The `list` of shape parameters per iteration. The first and last members
        correspond to the initial and final shapes, respectively.
    appearance_parameters : `list` of ``(n_appearance_parameters,)`` `ndarray`
        The `list` of appearance parameters per iteration. The first and last
        members correspond to the initial and final shapes, respectively.
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
    def __init__(self, shapes, shape_parameters, appearance_parameters,
                 initial_shape=None,  cost_functions=None, image=None,
                 gt_shape=None):
        super(AAMAlgorithmResult, self).__init__(
            shapes=shapes, shape_parameters=shape_parameters,
            initial_shape=initial_shape, image=image, gt_shape=gt_shape)
        self._appearance_parameters = appearance_parameters
        self._cost_functions = cost_functions

    @property
    def appearance_parameters(self):
        r"""
        Returns the `list` of appearance parameters obtained at each iteration
        of the fitting process. The `list` includes the parameters of the
        `initial_shape` (if it exists) and `final_shape`.

        :type: `list` of ``(n_params,)`` `ndarray`
        """
        return self._appearance_parameters

    @property
    def cost_functions(self):
        r"""
        Returns a `list` with the cost closure per iteration.

        :type: `list` of `closure`
        """
        return self._cost_functions


class AAMResult(MultiScaleParametricIterativeResult):
    r"""
    Class for storing the multi-scale iterative fitting result of an AAM. It
    holds the shapes, shape parameters, appearance parameters and costs per
    iteration.

    .. note:: When using a method with a parametric shape model, the first step
              is to **reconstruct the initial shape** using the shape model. The
              generated reconstructed shape is then used as initialisation for
              the iterative optimisation. This step is not counted in the number
              of iterations.

    Parameters
    ----------
    results : `list` of :map:`AAMAlgorithmResult`
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
        super(AAMResult, self).__init__(
            results=results, scales=scales, affine_transforms=affine_transforms,
            scale_transforms=scale_transforms, image=image, gt_shape=gt_shape)
        # Create appearance parameters list
        self._appearance_parameters = None
        if results[0].appearance_parameters is not None:
            self._appearance_parameters = []
            for r in results:
                self._appearance_parameters += r.appearance_parameters
        # Create cost functions list
        self._cost_functions = None
        if results[0].cost_functions is not None:
            self._cost_functions = []
            for r in results:
                self._cost_functions += r.cost_functions

    @property
    def appearance_parameters(self):
        r"""
        Returns the `list` of appearance parameters obtained at each iteration
        of the fitting process. The `list` includes the parameters of the
        `initial_shape` (if it exists) and `final_shape`.

        :type: `list` of ``(n_params,)`` `ndarray`
        """
        return self._appearance_parameters

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
            raise ValueError('costs is not well defined for the chosen AAM '
                             'algorithm')
