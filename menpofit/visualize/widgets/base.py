import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import collections as mc

import ipywidgets
import IPython.display as ipydisplay

from menpo.visualize.widgets import (RendererOptionsWidget,
                                     ChannelOptionsWidget,
                                     LandmarkOptionsWidget, TextPrintWidget,
                                     AnimationOptionsWidget, GraphOptionsWidget,
                                     SaveFigureOptionsWidget)
from menpo.visualize.widgets.tools import (_format_box, LogoWidget,
                                           _map_styles_to_hex_colours)
from menpo.visualize.widgets.base import _extract_groups_labels
from menpo.visualize.widgets.base import _visualize as _visualize_menpo
from menpo.visualize.viewmatplotlib import (MatplotlibImageViewer2d,
                                            sample_colours_from_colourmap,
                                            MatplotlibSubplots)

from .options import (LinearModelParametersWidget, FittingResultOptionsWidget,
                      FittingResultIterationsOptionsWidget)

# This glyph import is called frequently during visualisation, so we ensure
# that we only import it once. The same for the sum_channels method.
glyph = None
sum_channels = None

def _check_n_parameters(n_params, n_levels, max_n_params):
    r"""
    Checks the maximum number of components per level either of the shape
    or the appearance model. It must be ``None`` or `int` or `float` or a `list`
    of those containing ``1`` or ``n_scales`` elements.
    """
    str_error = ("n_params must be None or 1 <= int <= max_n_params or "
                 "a list of those containing 1 or {} elements").format(n_levels)
    if not isinstance(n_params, list):
        n_params_list = [n_params] * n_levels
    elif len(n_params) == 1:
        n_params_list = [n_params[0]] * n_levels
    elif len(n_params) == n_levels:
        n_params_list = n_params
    else:
        raise ValueError(str_error)
    for i, comp in enumerate(n_params_list):
        if comp is None:
            n_params_list[i] = max_n_params[i]
        else:
            if isinstance(comp, int):
                if comp > max_n_params[i]:
                    n_params_list[i] = max_n_params[i]
            else:
                raise ValueError(str_error)
    return n_params_list


def visualize_shape_model(shape_model, n_parameters=5, mode='multiple',
                          parameters_bounds=(-3.0, 3.0), figure_size=(10, 8),
                          style='coloured'):
    r"""
    Widget that allows the dynamic visualization of a multilevel linear
    statistical shape model.

    Parameters
    -----------
    shape_model : `list` of :map:`PCAModel` or subclass
        The multilevel shape model to be visualized. Note that each level can
        have different number of components.
    n_parameters : `int` or `list` of `int` or ``None``, optional
        The number of principal components to be used for the parameters
        sliders. If `int`, then the number of sliders per level is the minimum
        between `n_parameters` and the number of active components per level.
        If `list` of `int`, then a number of sliders is defined per level.
        If ``None``, all the active components per level will have a slider.
    mode : {``'single'``, ``'multiple'``}, optional
        If ``'single'``, then only a single slider is constructed along with a
        drop down menu. If ``'multiple'``, then a slider is constructed for each
        parameter.
    parameters_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.
    figure_size : (`int`, `int`), optional
        The size of the plotted figures.
    style : {``'coloured'``, ``'minimal'``}, optional
        If ``'coloured'``, then the style of the widget will be coloured. If
        ``minimal``, then the style is simple using black and white colours.
    """
    print('Initializing...')

    # Make sure that shape_model is a list even with one member
    if not isinstance(shape_model, list):
        shape_model = [shape_model]

    # Get the number of levels (i.e. number of shape models)
    n_levels = len(shape_model)

    # Define the styling options
    if style == 'coloured':
        model_parameters_style = 'info'
        logo_style = 'warning'
        widget_box_style = 'warning'
        widget_border_radius = 10
        widget_border_width = 1
        info_style = 'info'
        renderer_box_style = 'info'
        renderer_box_border_colour = _map_styles_to_hex_colours('info')
        renderer_box_border_radius = 10
        renderer_style = 'danger'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'danger'
    elif style == 'minimal':
        model_parameters_style = 'minimal'
        logo_style = 'minimal'
        widget_box_style = ''
        widget_border_radius = 0
        widget_border_width = 0
        info_style = 'minimal'
        renderer_box_style = ''
        renderer_box_border_colour = 'black'
        renderer_box_border_radius = 0
        renderer_style = 'minimal'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'minimal'
    else:
        raise ValueError("style must be either coloured or minimal")

    # Get the maximum number of components per level
    max_n_params = [sp.n_active_components for sp in shape_model]

    # Check the given number of parameters (the returned n_parameters is a list
    # of len n_scales)
    n_parameters = _check_n_parameters(n_parameters, n_levels, max_n_params)

    # Initial options dictionaries
    line_options = {'render_lines': True, 'line_width': 1,
                    'line_colour': ['r'], 'line_style': '-'}
    marker_options = {'render_markers': True, 'marker_size': 20,
                      'marker_face_colour': ['r'], 'marker_edge_colour': ['k'],
                      'marker_style': 'o', 'marker_edge_width': 1}
    figure_options = {'x_scale': 1., 'y_scale': 1., 'render_axes': False,
                      'axes_font_name': 'sans-serif', 'axes_font_size': 10,
                      'axes_font_style': 'normal', 'axes_font_weight': 'normal',
                      'axes_x_limits': None, 'axes_y_limits': None}
    renderer_options = {'lines': line_options, 'markers': marker_options,
                        'figure': figure_options}

    # Define render function
    def render_function(name, value):
        # Clear current figure, but wait until the generation of the new data
        # that will be rendered
        ipydisplay.clear_output(wait=True)

        # Get selected level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # Compute weights
        parameters = model_parameters_wid.parameters
        weights = (parameters *
                   shape_model[level].eigenvalues[:len(parameters)] ** 0.5)

        # Get the mean
        mean = shape_model[level].mean()

        # Render shape instance with selected options
        tmp1 = renderer_options_wid.selected_values[0]['lines']
        tmp2 = renderer_options_wid.selected_values[0]['markers']
        tmp3 = renderer_options_wid.selected_values[0]['figure']
        new_figure_size = (tmp3['x_scale'] * figure_size[0],
                           tmp3['y_scale'] * figure_size[1])
        if mode_wid.value == 1:
            # Deformation mode
            # Compute instance
            instance = shape_model[level].instance(weights)

            # Render mean shape
            if mean_wid.value:
                mean.view(
                    figure_id=save_figure_wid.renderer.figure_id,
                    new_figure=False, image_view=axes_mode_wid.value == 1,
                    render_lines=tmp1['render_lines'],
                    line_colour='y', line_style='solid',
                    line_width=tmp1['line_width'],
                    render_markers=tmp2['render_markers'],
                    marker_style=tmp2['marker_style'],
                    marker_size=tmp2['marker_size'], marker_face_colour='y',
                    marker_edge_colour='y',
                    marker_edge_width=tmp2['marker_edge_width'],
                    render_axes=tmp3['render_axes'],
                    axes_font_name=tmp3['axes_font_name'],
                    axes_font_size=tmp3['axes_font_size'],
                    axes_font_style=tmp3['axes_font_style'],
                    axes_font_weight=tmp3['axes_font_weight'], figure_size=None)

            # Render instance
            renderer = instance.view(
                figure_id=save_figure_wid.renderer.figure_id, new_figure=False,
                image_view=axes_mode_wid.value == 1,
                render_lines=tmp1['render_lines'],
                line_colour=tmp1['line_colour'][0],
                line_style=tmp1['line_style'], line_width=tmp1['line_width'],
                render_markers=tmp2['render_markers'],
                marker_style=tmp2['marker_style'],
                marker_size=tmp2['marker_size'],
                marker_face_colour=tmp2['marker_face_colour'],
                marker_edge_colour=tmp2['marker_edge_colour'],
                marker_edge_width=tmp2['marker_edge_width'],
                render_axes=tmp3['render_axes'],
                axes_font_name=tmp3['axes_font_name'],
                axes_font_size=tmp3['axes_font_size'],
                axes_font_style=tmp3['axes_font_style'],
                axes_font_weight=tmp3['axes_font_weight'],
                axes_x_limits=tmp3['axes_x_limits'],
                axes_y_limits=tmp3['axes_y_limits'],
                figure_size=new_figure_size)

            # Invert y axis if needed
            if mean_wid.value and axes_mode_wid.value == 1:
                plt.gca().invert_yaxis()

            # Get instance range
            instance_range = instance.range()
        else:
            # Vectors mode
            # Compute instance
            instance_lower = shape_model[level].instance([-p for p in weights])
            instance_upper = shape_model[level].instance(weights)

            # Render mean shape
            renderer = mean.view(
                figure_id=save_figure_wid.renderer.figure_id, new_figure=False,
                image_view=axes_mode_wid.value == 1,
                render_lines=tmp1['render_lines'],
                line_colour=tmp1['line_colour'][0],
                line_style=tmp1['line_style'], line_width=tmp1['line_width'],
                render_markers=tmp2['render_markers'],
                marker_style=tmp2['marker_style'],
                marker_size=tmp2['marker_size'],
                marker_face_colour=tmp2['marker_face_colour'],
                marker_edge_colour=tmp2['marker_edge_colour'],
                marker_edge_width=tmp2['marker_edge_width'],
                render_axes=tmp3['render_axes'],
                axes_font_name=tmp3['axes_font_name'],
                axes_font_size=tmp3['axes_font_size'],
                axes_font_style=tmp3['axes_font_style'],
                axes_font_weight=tmp3['axes_font_weight'],
                axes_x_limits=tmp3['axes_x_limits'],
                axes_y_limits=tmp3['axes_y_limits'],
                figure_size=new_figure_size)

            # Render vectors
            ax = plt.gca()
            for p in range(mean.n_points):
                xm = mean.points[p, 0]
                ym = mean.points[p, 1]
                xl = instance_lower.points[p, 0]
                yl = instance_lower.points[p, 1]
                xu = instance_upper.points[p, 0]
                yu = instance_upper.points[p, 1]
                if axes_mode_wid.value == 1:
                    # image mode
                    lines = [[(ym, xm), (yl, xl)], [(ym, xm), (yu, xu)]]
                else:
                    # point cloud mode
                    lines = [[(xm, ym), (xl, yl)], [(xm, ym), (xu, yu)]]
                lc = mc.LineCollection(lines, colors=('g', 'b'),
                                       linestyles='solid', linewidths=2)
                ax.add_collection(lc)

            # Get instance range
            instance_range = mean.range()

        plt.show()

        # Save the current figure id
        save_figure_wid.renderer = renderer

        # Update info
        update_info(level, instance_range)

    # Define function that updates the info text
    def update_info(level, instance_range):
        text_per_line = [
            "> Level {} out of {}".format(level + 1, n_levels),
            "> {} components in total".format(shape_model[level].n_components),
            "> {} active components".format(
                shape_model[level].n_active_components),
            "> {:.1f}% variance kept".format(
                shape_model[level].variance_ratio() * 100),
            "> Instance range: {:.1f} x {:.1f}".format(instance_range[0],
                                                       instance_range[1]),
            "> {} landmark points, {} features".format(
                shape_model[level].mean().n_points,
                shape_model[level].n_features)]
        info_wid.set_widget_state(n_lines=6, text_per_line=text_per_line)

    # Plot variance function
    def plot_variance(name):
        # Clear current figure, but wait until the generation of the new data
        # that will be rendered
        ipydisplay.clear_output(wait=True)

        # Get selected level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # Render
        new_figure_size = (
            renderer_options_wid.selected_values[0]['figure']['x_scale'] * 10,
            renderer_options_wid.selected_values[0]['figure']['y_scale'] * 3)
        plt.subplot(121)
        shape_model[level].plot_eigenvalues_ratio(
            figure_id=save_figure_wid.renderer.figure_id)
        plt.subplot(122)
        renderer = shape_model[level].plot_eigenvalues_cumulative_ratio(
            figure_id=save_figure_wid.renderer.figure_id,
            figure_size=new_figure_size)
        plt.show()

        # Save the current figure id
        save_figure_wid.renderer = renderer

    # Create widgets
    mode_dict = OrderedDict()
    mode_dict['Deformation'] = 1
    mode_dict['Vectors'] = 2
    mode_wid = ipywidgets.RadioButtons(options=mode_dict,
                                       description='Mode:', value=1)
    mode_wid.on_trait_change(render_function, 'value')
    mean_wid = ipywidgets.Checkbox(value=False,
                                   description='Render mean shape')
    mean_wid.on_trait_change(render_function, 'value')

    # Function that controls mean shape checkbox visibility
    def mean_visible(name, value):
        if value == 1:
            mean_wid.disabled = False
        else:
            mean_wid.disabled = True
            mean_wid.value = False
    mode_wid.on_trait_change(mean_visible, 'value')
    model_parameters_wid = LinearModelParametersWidget(
        [0] * n_parameters[0], render_function, params_str='param ',
        mode=mode, params_bounds=parameters_bounds, params_step=0.1,
        plot_variance_visible=True, plot_variance_function=plot_variance,
        style=model_parameters_style)
    axes_mode_wid = ipywidgets.RadioButtons(
        options={'Image': 1, 'Point cloud': 2}, description='Axes mode:',
        value=2)
    axes_mode_wid.on_trait_change(render_function, 'value')
    renderer_options_wid = RendererOptionsWidget(
        renderer_options, ['markers', 'lines', 'figure_one'],
        object_selection_dropdown_visible=False,
        render_function=render_function, style=renderer_style,
        tabs_style=renderer_tabs_style)
    renderer_options_box = ipywidgets.VBox(
        children=[axes_mode_wid, renderer_options_wid], align='center',
        margin='0.1cm')
    info_wid = TextPrintWidget(n_lines=6, text_per_line=[''] * 6,
                               style=info_style)
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = SaveFigureOptionsWidget(initial_renderer,
                                              style=save_figure_style)

    # Define function that updates options' widgets state
    def update_widgets(name, value):
        model_parameters_wid.set_widget_state([0] * n_parameters[value],
                                              params_str='param ',
                                              allow_callback=True)

    # Group widgets
    if n_levels > 1:
        radio_str = OrderedDict()
        for l in range(n_levels):
            if l == 0:
                radio_str["Level {} (low)".format(l)] = l
            elif l == n_levels - 1:
                radio_str["Level {} (high)".format(l)] = l
            else:
                radio_str["Level {}".format(l)] = l
        level_wid = ipywidgets.RadioButtons(
            options=radio_str, description='Pyramid:', value=0)
        level_wid.on_trait_change(update_widgets, 'value')
        level_wid.on_trait_change(render_function, 'value')
        radio_children = [level_wid, mode_wid, mean_wid]
    else:
        radio_children = [mode_wid, mean_wid]
    radio_wids = ipywidgets.VBox(children=radio_children, margin='0.3cm')
    tmp_wid = ipywidgets.HBox(children=[radio_wids, model_parameters_wid])
    options_box = ipywidgets.Tab(children=[tmp_wid, renderer_options_box,
                                           info_wid, save_figure_wid])
    tab_titles = ['Model', 'Renderer', 'Info', 'Export']
    for (k, tl) in enumerate(tab_titles):
        options_box.set_title(k, tl)
    logo_wid = LogoWidget(style=logo_style)
    logo_wid.margin = '0.1cm'
    wid = ipywidgets.HBox(children=[logo_wid, options_box], align='start')

    # Set widget's style
    wid.box_style = widget_box_style
    wid.border_radius = widget_border_radius
    wid.border_width = widget_border_width
    wid.border_color = _map_styles_to_hex_colours(widget_box_style)
    renderer_options_wid.margin = '0.2cm'
    _format_box(renderer_options_box, renderer_box_style, True,
                renderer_box_border_colour, 'solid', 1,
                renderer_box_border_radius, '0.1cm', '0.2cm')

    # Display final widget
    ipydisplay.display(wid)

    # Reset value to trigger initial visualization
    axes_mode_wid.value = 1

def visualize_appearance_model(appearance_model, n_parameters=5,
                               mode='multiple', parameters_bounds=(-3.0, 3.0),
                               figure_size=(10, 8), style='coloured'):
    r"""
    Widget that allows the dynamic visualization of a multilevel linear
    statistical appearance model.

    Parameters
    -----------
    appearance_model : `list` of :map:`PCAModel` or subclass
        The multilevel appearance model to be visualized. Note that each level
        can have different number of components.
    n_parameters : `int` or `list` of `int` or ``None``, optional
        The number of principal components to be used for the parameters
        sliders. If `int`, then the number of sliders per level is the minimum
        between `n_parameters` and the number of active components per level.
        If `list` of `int`, then a number of sliders is defined per level.
        If ``None``, all the active components per level will have a slider.
    mode : {``'single'``, ``'multiple'``}, optional
        If ``'single'``, then only a single slider is constructed along with a
        drop down menu. If ``'multiple'``, then a slider is constructed for each
        parameter.
    parameters_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.
    figure_size : (`int`, `int`), optional
        The size of the plotted figures.
    style : {``'coloured'``, ``'minimal'``}, optional
        If ``'coloured'``, then the style of the widget will be coloured. If
        ``minimal``, then the style is simple using black and white colours.
    """
    from menpo.image import MaskedImage
    print('Initializing...')

    # Make sure that appearance_model is a list even with one member
    if not isinstance(appearance_model, list):
        appearance_model = [appearance_model]

    # Get the number of levels (i.e. number of appearance models)
    n_levels = len(appearance_model)

    # Define the styling options
    if style == 'coloured':
        model_parameters_style = 'info'
        channels_style = 'info'
        landmarks_style = 'info'
        logo_style = 'success'
        widget_box_style = 'success'
        widget_border_radius = 10
        widget_border_width = 1
        info_style = 'info'
        renderer_style = 'warning'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'danger'
    elif style == 'minimal':
        model_parameters_style = 'minimal'
        channels_style = 'minimal'
        landmarks_style = 'minimal'
        logo_style = 'minimal'
        widget_box_style = ''
        widget_border_radius = 0
        widget_border_width = 0
        info_style = 'minimal'
        renderer_style = 'minimal'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'minimal'
    else:
        raise ValueError("style must be either coloured or minimal")

    # Get the maximum number of components per level
    max_n_params = [ap.n_active_components for ap in appearance_model]

    # Check the given number of parameters (the returned n_parameters is a list
    # of len n_scales)
    n_parameters = _check_n_parameters(n_parameters, n_levels, max_n_params)

    # Find initial groups and labels that will be passed to the landmark options
    # widget creation
    mean_has_landmarks = appearance_model[0].mean().has_landmarks
    if mean_has_landmarks:
        all_groups_keys, all_labels_keys = _extract_groups_labels(
            appearance_model[0].mean())
    else:
        all_groups_keys = [' ']
        all_labels_keys = [[' ']]

    # Get initial line and marker colours for each available group
    if len(all_labels_keys[0]) == 1:
        colours = ['r']
    else:
        colours = sample_colours_from_colourmap(len(all_labels_keys[0]), 'jet')

    # Initial options dictionaries
    channels_default = 0
    if appearance_model[0].mean().n_channels == 3:
        channels_default = None
    channel_options = {'n_channels': appearance_model[0].mean().n_channels,
                       'image_is_masked': isinstance(appearance_model[0].mean(),
                                                     MaskedImage),
                       'channels': channels_default, 'glyph_enabled': False,
                       'glyph_block_size': 3, 'glyph_use_negative': False,
                       'sum_enabled': False,
                       'masked_enabled': isinstance(appearance_model[0].mean(),
                                                    MaskedImage)}
    landmark_options = {'has_landmarks': mean_has_landmarks,
                        'render_landmarks': mean_has_landmarks,
                        'group_keys': all_groups_keys,
                        'labels_keys': all_labels_keys,
                        'group': all_groups_keys[0],
                        'with_labels': all_labels_keys[0]}
    image_options = {'alpha': 1.0, 'interpolation': 'none', 'cmap_name': None}
    line_options = {'render_lines': True, 'line_width': 1,
                    'line_colour': colours, 'line_style': '-'}
    marker_options = {'render_markers': True, 'marker_size': 20,
                      'marker_face_colour': colours,
                      'marker_edge_colour': colours,
                      'marker_style': 'o', 'marker_edge_width': 1}
    figure_options = {'x_scale': 1., 'y_scale': 1., 'render_axes': True,
                      'axes_font_name': 'sans-serif', 'axes_font_size': 10,
                      'axes_font_style': 'normal', 'axes_font_weight': 'normal',
                      'axes_x_limits': None, 'axes_y_limits': None}
    renderer_options = {'lines': line_options, 'markers': marker_options,
                        'figure': figure_options, 'image': image_options}

    # Define render function
    def render_function(name, value):
        # Clear current figure, but wait until the generation of the new data
        # that will be rendered
        ipydisplay.clear_output(wait=True)

        # Get selected level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # Compute weights and instance
        parameters = model_parameters_wid.parameters
        weights = (parameters *
                   appearance_model[level].eigenvalues[:len(parameters)] ** 0.5)
        instance = appearance_model[level].instance(weights)

        # Update info
        update_info(instance, level,
                    landmark_options_wid.selected_values['group'])

        # Render instance with selected options
        tmp1 = renderer_options_wid.selected_values[0]['lines']
        tmp2 = renderer_options_wid.selected_values[0]['markers']
        tmp3 = renderer_options_wid.selected_values[0]['figure']
        tmp4 = renderer_options_wid.selected_values[0]['image']
        new_figure_size = (tmp3['x_scale'] * figure_size[0],
                           tmp3['y_scale'] * figure_size[1])

        # Find the with_labels' indices
        with_labels_idx = [
            landmark_options_wid.selected_values['labels_keys'][0].index(lbl)
            for lbl in landmark_options_wid.selected_values['with_labels']]

        # Get line and marker colours
        line_colour = [tmp1['line_colour'][lbl_idx]
                       for lbl_idx in with_labels_idx]
        marker_face_colour = [tmp2['marker_face_colour'][lbl_idx]
                              for lbl_idx in with_labels_idx]
        marker_edge_colour = [tmp2['marker_edge_colour'][lbl_idx]
                              for lbl_idx in with_labels_idx]

        renderer = _visualize_menpo(
            instance, save_figure_wid.renderer,
            landmark_options_wid.selected_values['render_landmarks'],
            channel_options_wid.selected_values['image_is_masked'],
            channel_options_wid.selected_values['masked_enabled'],
            channel_options_wid.selected_values['channels'],
            channel_options_wid.selected_values['glyph_enabled'],
            channel_options_wid.selected_values['glyph_block_size'],
            channel_options_wid.selected_values['glyph_use_negative'],
            channel_options_wid.selected_values['sum_enabled'],
            landmark_options_wid.selected_values['group'],
            landmark_options_wid.selected_values['with_labels'],
            tmp1['render_lines'], tmp1['line_style'], tmp1['line_width'],
            line_colour, tmp2['render_markers'], tmp2['marker_style'],
            tmp2['marker_size'], tmp2['marker_edge_width'], marker_edge_colour,
            marker_face_colour, False, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, False, None, None, new_figure_size,
            tmp3['render_axes'], tmp3['axes_font_name'], tmp3['axes_font_size'],
            tmp3['axes_font_style'], tmp3['axes_x_limits'],
            tmp3['axes_y_limits'], tmp3['axes_font_weight'],
            tmp4['interpolation'], tmp4['alpha'], tmp4['cmap_name'])

        # Save the current figure id
        save_figure_wid.renderer = renderer

    # Define function that updates the info text
    def update_info(image, level, group):
        lvl_app_mod = appearance_model[level]
        text_per_line = [
            "> Level: {} out of {}.".format(level + 1, n_levels),
            "> {} components in total.".format(lvl_app_mod.n_components),
            "> {} active components.".format(lvl_app_mod.n_active_components),
            "> {:.1f}% variance kept.".format(
                lvl_app_mod.variance_ratio() * 100),
            "> Reference shape of size {} with {} channel{}.".format(
                image._str_shape,
                image.n_channels, 's' * (image.n_channels > 1)),
            "> {} features.".format(lvl_app_mod.n_features),
            "> {} landmark points.".format(image.landmarks[group].lms.n_points),
            "> Instance: min={:.3f}, max={:.3f}".format(image.pixels.min(),
                                                        image.pixels.max())]
        info_wid.set_widget_state(n_lines=8, text_per_line=text_per_line)

    # Plot variance function
    def plot_variance(name):
        # Clear current figure, but wait until the generation of the new data
        # that will be rendered
        ipydisplay.clear_output(wait=True)

        # Get selected level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # Render
        new_figure_size = (
            renderer_options_wid.selected_values[0]['figure']['x_scale'] * 10,
            renderer_options_wid.selected_values[0]['figure']['y_scale'] * 3)
        plt.subplot(121)
        appearance_model[level].plot_eigenvalues_ratio(
            figure_id=save_figure_wid.renderer.figure_id)
        plt.subplot(122)
        renderer = appearance_model[level].plot_eigenvalues_cumulative_ratio(
            figure_id=save_figure_wid.renderer.figure_id,
            figure_size=new_figure_size)
        plt.show()

        # Save the current figure id
        save_figure_wid.renderer = renderer

    # Create widgets
    model_parameters_wid = LinearModelParametersWidget(
        [0] * n_parameters[0], render_function, params_str='param ',
        mode=mode, params_bounds=parameters_bounds, params_step=0.1,
        plot_variance_visible=True, plot_variance_function=plot_variance,
        style=model_parameters_style)
    channel_options_wid = ChannelOptionsWidget(
        channel_options, render_function=render_function, style=channels_style)
    landmark_options_wid = LandmarkOptionsWidget(
        landmark_options, render_function=render_function,
        style=landmarks_style)
    renderer_options_wid = RendererOptionsWidget(
        renderer_options, ['image', 'markers', 'lines', 'figure_one'],
        object_selection_dropdown_visible=False,
        render_function=render_function, style=renderer_style,
        tabs_style=renderer_tabs_style)
    info_wid = TextPrintWidget(n_lines=8, text_per_line=[''] * 8,
                               style=info_style)
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = SaveFigureOptionsWidget(initial_renderer,
                                              style=save_figure_style)

    # Define function that updates options' widgets state
    def update_widgets(name, value):
        # Update model parameters widget
        model_parameters_wid.set_widget_state([0] * n_parameters[value],
                                              params_str='param ',
                                              allow_callback=True)
        # Update channel options
        tmp_n_channels = appearance_model[value].mean().n_channels
        tmp_channels = channel_options_wid.selected_values['channels']
        tmp_glyph_enabled = channel_options_wid.selected_values['glyph_enabled']
        tmp_sum_enabled = channel_options_wid.selected_values['sum_enabled']
        if np.max(tmp_channels) > tmp_n_channels - 1:
            tmp_channels = 0
            tmp_glyph_enabled = False
            tmp_sum_enabled = False
        tmp_glyph_block_size = \
            channel_options_wid.selected_values['glyph_block_size']
        tmp_glyph_use_negative = \
            channel_options_wid.selected_values['glyph_use_negative']
        if not(tmp_n_channels == 3) and tmp_channels is None:
            tmp_channels = 0
        channel_options = {
            'n_channels': tmp_n_channels,
            'image_is_masked': isinstance(appearance_model[0].mean(),
                                          MaskedImage),
            'channels': tmp_channels, 'glyph_enabled': tmp_glyph_enabled,
            'glyph_block_size': tmp_glyph_block_size,
            'glyph_use_negative': tmp_glyph_use_negative,
            'sum_enabled': tmp_sum_enabled,
            'masked_enabled': isinstance(appearance_model[0].mean(),
                                         MaskedImage)}
        channel_options_wid.set_widget_state(channel_options, True)

    # Group widgets
    tmp_children = [model_parameters_wid]
    if n_levels > 1:
        radio_str = OrderedDict()
        for l in range(n_levels):
            if l == 0:
                radio_str["Level {} (low)".format(l)] = l
            elif l == n_levels - 1:
                radio_str["Level {} (high)".format(l)] = l
            else:
                radio_str["Level {}".format(l)] = l
        level_wid = ipywidgets.RadioButtons(
            options=radio_str, description='Pyramid:', value=0)
        level_wid.on_trait_change(update_widgets, 'value')
        level_wid.on_trait_change(render_function, 'value')
        tmp_children.insert(0, level_wid)
    tmp_wid = ipywidgets.HBox(children=tmp_children)
    options_box = ipywidgets.Tab(children=[tmp_wid, channel_options_wid,
                                           landmark_options_wid,
                                           renderer_options_wid,
                                           info_wid, save_figure_wid])
    tab_titles = ['Model', 'Channels', 'Landmarks', 'Renderer', 'Info',
                  'Export']
    for (k, tl) in enumerate(tab_titles):
        options_box.set_title(k, tl)
    logo_wid = LogoWidget(style=logo_style)
    logo_wid.margin = '0.1cm'
    wid = ipywidgets.HBox(children=[logo_wid, options_box], align='start')

    # Set widget's style
    wid.box_style = widget_box_style
    wid.border_radius = widget_border_radius
    wid.border_width = widget_border_width
    wid.border_color = _map_styles_to_hex_colours(widget_box_style)
    renderer_options_wid.margin = '0.2cm'

    # Display final widget
    ipydisplay.display(wid)

    # Reset value to trigger initial visualization
    renderer_options_wid.options_widgets[3].render_axes_checkbox.value = False

def visualize_aam(aam, n_shape_parameters=5, n_appearance_parameters=5,
                  mode='multiple', parameters_bounds=(-3.0, 3.0),
                  figure_size=(10, 8), style='coloured'):
    r"""
    Widget that allows the dynamic visualization of a multilevel Active
    Appearance Model.

    Parameters
    -----------
    aam : :map:`AAM`
        The multilevel AAM to be visualized. Note that each level can have
        different number of components.
    n_shape_parameters : `int` or `list` of `int` or ``None``, optional
        The number of principal components to be used for the shape parameters
        sliders. If `int`, then the number of sliders per level is the minimum
        between `n_parameters` and the number of active components per level.
        If `list` of `int`, then a number of sliders is defined per level.
        If ``None``, all the active components per level will have a slider.
    n_appearance_parameters : `int` or `list` of `int` or ``None``, optional
        The number of principal components to be used for the appearance
        parameters sliders. If `int`, then the number of sliders per level is
        the minimum between `n_parameters` and the number of active components
        per level. If `list` of `int`, then a number of sliders is defined per
        level. If ``None``, all the active components per level will have a
        slider.
    mode : {``'single'``, ``'multiple'``}, optional
        If ``'single'``, then only a single slider is constructed along with a
        drop down menu. If ``'multiple'``, then a slider is constructed for each
        parameter.
    parameters_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.
    figure_size : (`int`, `int`), optional
        The size of the plotted figures.
    style : {``'coloured'``, ``'minimal'``}, optional
        If ``'coloured'``, then the style of the widget will be coloured. If
        ``minimal``, then the style is simple using black and white colours.
    """
    from menpo.image import MaskedImage
    print('Initializing...')

    # Get the number of levels
    n_levels = aam.n_scales

    # Define the styling options
    if style == 'coloured':
        model_style = 'info'
        model_tab_style = 'danger'
        model_parameters_style = 'danger'
        channels_style = 'danger'
        landmarks_style = 'danger'
        logo_style = 'info'
        widget_box_style = 'info'
        widget_border_radius = 10
        widget_border_width = 1
        info_style = 'danger'
        renderer_style = 'danger'
        renderer_tabs_style = 'info'
        save_figure_style = 'danger'
    elif style == 'minimal':
        model_style = ''
        model_tab_style = ''
        model_parameters_style = 'minimal'
        channels_style = 'minimal'
        landmarks_style = 'minimal'
        logo_style = 'minimal'
        widget_box_style = ''
        widget_border_radius = 0
        widget_border_width = 0
        info_style = 'minimal'
        renderer_style = 'minimal'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'minimal'
    else:
        raise ValueError("style must be either coloured or minimal")

    # Get the maximum number of components per level
    max_n_shape = [sp.n_active_components for sp in aam.shape_models]
    max_n_appearance = [ap.n_active_components for ap in aam.appearance_models]

    # Check the given number of parameters (the returned n_parameters is a list
    # of len n_scales)
    n_shape_parameters = _check_n_parameters(n_shape_parameters, n_levels,
                                             max_n_shape)
    n_appearance_parameters = _check_n_parameters(n_appearance_parameters,
                                                  n_levels, max_n_appearance)

    # Find initial groups and labels that will be passed to the landmark options
    # widget creation
    mean_has_landmarks = aam.appearance_models[0].mean().has_landmarks
    if mean_has_landmarks:
        all_groups_keys, all_labels_keys = _extract_groups_labels(
            aam.appearance_models[0].mean())
    else:
        all_groups_keys = [' ']
        all_labels_keys = [[' ']]

    # Get initial line and marker colours for each available group
    if len(all_labels_keys[0]) == 1:
        colours = ['r']
    else:
        colours = sample_colours_from_colourmap(len(all_labels_keys[0]), 'jet')

    # Initial options dictionaries
    channels_default = 0
    if aam.appearance_models[0].mean().n_channels == 3:
        channels_default = None
    channel_options = {'n_channels': aam.appearance_models[0].mean().n_channels,
                       'image_is_masked': isinstance(
                           aam.appearance_models[0].mean(), MaskedImage),
                       'channels': channels_default, 'glyph_enabled': False,
                       'glyph_block_size': 3, 'glyph_use_negative': False,
                       'sum_enabled': False,
                       'masked_enabled': isinstance(
                           aam.appearance_models[0].mean(), MaskedImage)}
    landmark_options = {'has_landmarks': mean_has_landmarks,
                        'render_landmarks': mean_has_landmarks,
                        'group_keys': all_groups_keys,
                        'labels_keys': all_labels_keys,
                        'group': all_groups_keys[0],
                        'with_labels': all_labels_keys[0]}
    image_options = {'alpha': 1.0, 'interpolation': 'none', 'cmap_name': None}
    line_options = {'render_lines': True, 'line_width': 1,
                    'line_colour': colours, 'line_style': '-'}
    marker_options = {'render_markers': True, 'marker_size': 20,
                      'marker_face_colour': colours,
                      'marker_edge_colour': colours,
                      'marker_style': 'o', 'marker_edge_width': 1}
    figure_options = {'x_scale': 1., 'y_scale': 1., 'render_axes': True,
                      'axes_font_name': 'sans-serif', 'axes_font_size': 10,
                      'axes_font_style': 'normal', 'axes_font_weight': 'normal',
                      'axes_x_limits': None, 'axes_y_limits': None}
    renderer_options = {'lines': line_options, 'markers': marker_options,
                        'figure': figure_options, 'image': image_options}

    # Define render function
    def render_function(name, value):
        # Clear current figure, but wait until the generation of the new data
        # that will be rendered
        ipydisplay.clear_output(wait=True)

        # Get selected level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # Compute weights and instance
        shape_weights = shape_model_parameters_wid.parameters
        appearance_weights = appearance_model_parameters_wid.parameters
        instance = aam.instance(scale_index=level, shape_weights=shape_weights,
                                appearance_weights=appearance_weights)

        # Update info
        update_info(aam, instance, level,
                    landmark_options_wid.selected_values['group'])

        # Render instance with selected options
        tmp1 = renderer_options_wid.selected_values[0]['lines']
        tmp2 = renderer_options_wid.selected_values[0]['markers']
        tmp3 = renderer_options_wid.selected_values[0]['figure']
        tmp4 = renderer_options_wid.selected_values[0]['image']
        new_figure_size = (tmp3['x_scale'] * figure_size[0],
                           tmp3['y_scale'] * figure_size[1])

        # Find the with_labels' indices
        with_labels_idx = [
            landmark_options_wid.selected_values['labels_keys'][0].index(lbl)
            for lbl in landmark_options_wid.selected_values['with_labels']]

        # Get line and marker colours
        line_colour = [tmp1['line_colour'][lbl_idx]
                       for lbl_idx in with_labels_idx]
        marker_face_colour = [tmp2['marker_face_colour'][lbl_idx]
                              for lbl_idx in with_labels_idx]
        marker_edge_colour = [tmp2['marker_edge_colour'][lbl_idx]
                              for lbl_idx in with_labels_idx]

        renderer = _visualize_menpo(
            instance, save_figure_wid.renderer,
            landmark_options_wid.selected_values['render_landmarks'],
            channel_options_wid.selected_values['image_is_masked'],
            channel_options_wid.selected_values['masked_enabled'],
            channel_options_wid.selected_values['channels'],
            channel_options_wid.selected_values['glyph_enabled'],
            channel_options_wid.selected_values['glyph_block_size'],
            channel_options_wid.selected_values['glyph_use_negative'],
            channel_options_wid.selected_values['sum_enabled'],
            landmark_options_wid.selected_values['group'],
            landmark_options_wid.selected_values['with_labels'],
            tmp1['render_lines'], tmp1['line_style'], tmp1['line_width'],
            line_colour, tmp2['render_markers'], tmp2['marker_style'],
            tmp2['marker_size'], tmp2['marker_edge_width'], marker_edge_colour,
            marker_face_colour, False, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, False, None, None, new_figure_size,
            tmp3['render_axes'], tmp3['axes_font_name'], tmp3['axes_font_size'],
            tmp3['axes_font_style'], tmp3['axes_x_limits'],
            tmp3['axes_y_limits'], tmp3['axes_font_weight'],
            tmp4['interpolation'], tmp4['alpha'], tmp4['cmap_name'])

        # Save the current figure id
        save_figure_wid.renderer = renderer

    # Define function that updates the info text
    def update_info(aam, instance, level, group):
        # features info
        from menpofit.base import name_of_callable

        lvl_app_mod = aam.appearance_models[level]
        lvl_shape_mod = aam.shape_models[level]
        aam_mean = lvl_app_mod.mean()
        n_channels = aam_mean.n_channels
        tmplt_inst = lvl_app_mod.template_instance
        feat = aam.holistic_features[level]

        # Feature string
        tmp_feat = 'Feature is {} with {} channel{}'.format(
            name_of_callable(feat), n_channels, 's' * (n_channels > 1))

        # update info widgets
        text_per_line = [
            "> Warp using {} transform".format(aam.transform.__name__),
            "> Level {}/{}".format(
                level + 1, aam.n_scales),
            "> {} landmark points".format(
                instance.landmarks[group].lms.n_points),
            "> {} shape components ({:.2f}% of variance)".format(
                lvl_shape_mod.n_components,
                lvl_shape_mod.variance_ratio() * 100),
            "> {}".format(tmp_feat),
            "> Reference frame of length {} ({} x {}C, {} x {}C)".format(
                lvl_app_mod.n_features, tmplt_inst.n_true_pixels(), n_channels,
                tmplt_inst._str_shape, n_channels),
            "> {} appearance components ({:.2f}% of variance)".format(
                lvl_app_mod.n_components, lvl_app_mod.variance_ratio() * 100),
            "> Instance: min={:.3f} , max={:.3f}".format(
                instance.pixels.min(), instance.pixels.max())]
        info_wid.set_widget_state(n_lines=len(text_per_line),
                                  text_per_line=text_per_line)

    # Plot shape variance function
    def plot_shape_variance(name):
        # Clear current figure, but wait until the generation of the new data
        # that will be rendered
        ipydisplay.clear_output(wait=True)

        # Get selected level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # Render
        new_figure_size = (
            renderer_options_wid.selected_values[0]['figure']['x_scale'] * 10,
            renderer_options_wid.selected_values[0]['figure']['y_scale'] * 3)
        plt.subplot(121)
        aam.shape_models[level].plot_eigenvalues_ratio(
            figure_id=save_figure_wid.renderer.figure_id)
        plt.subplot(122)
        renderer = aam.shape_models[level].plot_eigenvalues_cumulative_ratio(
            figure_id=save_figure_wid.renderer.figure_id,
            figure_size=new_figure_size)
        plt.show()

        # Save the current figure id
        save_figure_wid.renderer = renderer

    # Plot appearance variance function
    def plot_appearance_variance(name):
        # Clear current figure, but wait until the generation of the new data
        # that will be rendered
        ipydisplay.clear_output(wait=True)

        # Get selected level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # Render
        new_figure_size = (
            renderer_options_wid.selected_values[0]['figure']['x_scale'] * 10,
            renderer_options_wid.selected_values[0]['figure']['y_scale'] * 3)
        plt.subplot(121)
        aam.appearance_models[level].plot_eigenvalues_ratio(
            figure_id=save_figure_wid.renderer.figure_id)
        plt.subplot(122)
        renderer = aam.appearance_models[level].\
            plot_eigenvalues_cumulative_ratio(
            figure_id=save_figure_wid.renderer.figure_id,
            figure_size=new_figure_size)
        plt.show()

        # Save the current figure id
        save_figure_wid.renderer = renderer

    # Create widgets
    shape_model_parameters_wid = LinearModelParametersWidget(
        [0] * n_shape_parameters[0], render_function, params_str='param ',
        mode=mode, params_bounds=parameters_bounds, params_step=0.1,
        plot_variance_visible=True, plot_variance_function=plot_shape_variance,
        style=model_parameters_style)
    appearance_model_parameters_wid = LinearModelParametersWidget(
        [0] * n_appearance_parameters[0], render_function, params_str='param ',
        mode=mode, params_bounds=parameters_bounds, params_step=0.1,
        plot_variance_visible=True, style=model_parameters_style,
        plot_variance_function=plot_appearance_variance)
    channel_options_wid = ChannelOptionsWidget(
        channel_options, render_function=render_function, style=channels_style)
    landmark_options_wid = LandmarkOptionsWidget(
        landmark_options, render_function=render_function,
        style=landmarks_style)
    renderer_options_wid = RendererOptionsWidget(
        renderer_options, ['markers', 'lines', 'image', 'figure_one'],
        object_selection_dropdown_visible=False,
        render_function=render_function, style=renderer_style,
        tabs_style=renderer_tabs_style)
    info_wid = TextPrintWidget(n_lines=11, text_per_line=[''] * 11,
                               style=info_style)
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = SaveFigureOptionsWidget(initial_renderer,
                                              style=save_figure_style)

    # Define function that updates options' widgets state
    def update_widgets(name, value):
        # Update shape model parameters
        shape_model_parameters_wid.set_widget_state(
            [0] * n_shape_parameters[value], params_str='param ',
            allow_callback=True)

        # Update shape model parameters
        appearance_model_parameters_wid.set_widget_state(
            [0] * n_appearance_parameters[value], params_str='param ',
            allow_callback=True)

        # Update channel options
        tmp_n_channels = aam.appearance_models[value].mean().n_channels
        tmp_channels = channel_options_wid.selected_values['channels']
        tmp_glyph_enabled = channel_options_wid.selected_values['glyph_enabled']
        tmp_sum_enabled = channel_options_wid.selected_values['sum_enabled']
        if np.max(tmp_channels) > tmp_n_channels - 1:
            tmp_channels = 0
            tmp_glyph_enabled = False
            tmp_sum_enabled = False
        tmp_glyph_block_size = \
            channel_options_wid.selected_values['glyph_block_size']
        tmp_glyph_use_negative = \
            channel_options_wid.selected_values['glyph_use_negative']
        if not(tmp_n_channels == 3) and tmp_channels is None:
            tmp_channels = 0
        channel_options = {
            'n_channels': tmp_n_channels,
            'image_is_masked': isinstance(aam.appearance_models[0].mean(),
                                          MaskedImage),
            'channels': tmp_channels, 'glyph_enabled': tmp_glyph_enabled,
            'glyph_block_size': tmp_glyph_block_size,
            'glyph_use_negative': tmp_glyph_use_negative,
            'sum_enabled': tmp_sum_enabled,
            'masked_enabled': isinstance(aam.appearance_models[0].mean(),
                                         MaskedImage)}
        channel_options_wid.set_widget_state(channel_options, True)

    # Group widgets
    model_parameters_wid = ipywidgets.Tab(
        children=[shape_model_parameters_wid, appearance_model_parameters_wid])
    model_parameters_wid.set_title(0, 'Shape')
    model_parameters_wid.set_title(1, 'Appearance')
    model_parameters_wid = ipywidgets.FlexBox(children=[model_parameters_wid],
                                              margin='0.2cm', padding='0.1cm',
                                              box_style=model_tab_style)
    tmp_children = [model_parameters_wid]
    if n_levels > 1:
        radio_str = OrderedDict()
        for l in range(n_levels):
            if l == 0:
                radio_str["Level {} (low)".format(l)] = l
            elif l == n_levels - 1:
                radio_str["Level {} (high)".format(l)] = l
            else:
                radio_str["Level {}".format(l)] = l
        level_wid = ipywidgets.RadioButtons(
            options=radio_str, description='Pyramid:', value=0)
        level_wid.on_trait_change(update_widgets, 'value')
        level_wid.on_trait_change(render_function, 'value')
        tmp_children.insert(0, level_wid)
    tmp_wid = ipywidgets.HBox(children=tmp_children, align='center',
                              box_style=model_style)
    options_box = ipywidgets.Tab(children=[tmp_wid, channel_options_wid,
                                           landmark_options_wid,
                                           renderer_options_wid,
                                           info_wid, save_figure_wid])
    tab_titles = ['Model', 'Channels', 'Landmarks', 'Renderer', 'Info',
                  'Export']
    for (k, tl) in enumerate(tab_titles):
        options_box.set_title(k, tl)
    logo_wid = LogoWidget(style=logo_style)
    logo_wid.margin = '0.1cm'
    wid = ipywidgets.HBox(children=[logo_wid, options_box], align='start')

    # Set widget's style
    wid.box_style = widget_box_style
    wid.border_radius = widget_border_radius
    wid.border_width = widget_border_width
    wid.border_color = _map_styles_to_hex_colours(widget_box_style)
    renderer_options_wid.margin = '0.2cm'

    # Display final widget
    ipydisplay.display(wid)

    # Reset value to trigger initial visualization
    renderer_options_wid.options_widgets[3].render_axes_checkbox.value = False

def visualize_atm(atm, n_shape_parameters=5, mode='multiple',
                  parameters_bounds=(-3.0, 3.0), figure_size=(10, 8),
                  style='coloured'):
    r"""
    Widget that allows the dynamic visualization of a multilevel Active
    Template Model.

    Parameters
    -----------
    atm : :map:`ATM`
        The multilevel ATM to be visualized. Note that each level can have
        different number of components.
    n_shape_parameters : `int` or `list` of `int` or ``None``, optional
        The number of principal components to be used for the shape parameters
        sliders. If `int`, then the number of sliders per level is the minimum
        between `n_parameters` and the number of active components per level.
        If `list` of `int`, then a number of sliders is defined per level.
        If ``None``, all the active components per level will have a slider.
    mode : {``'single'``, ``'multiple'``}, optional
        If ``'single'``, then only a single slider is constructed along with a
        drop down menu. If ``'multiple'``, then a slider is constructed for each
        parameter.
    parameters_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.
    figure_size : (`int`, `int`), optional
        The size of the plotted figures.
    style : {``'coloured'``, ``'minimal'``}, optional
        If ``'coloured'``, then the style of the widget will be coloured. If
        ``minimal``, then the style is simple using black and white colours.
    """
    from menpo.image import MaskedImage
    print('Initializing...')

    # Get the number of levels
    n_levels = atm.n_scales

    # Define the styling options
    if style == 'coloured':
        model_style = 'info'
        model_parameters_style = 'danger'
        channels_style = 'danger'
        landmarks_style = 'danger'
        logo_style = 'info'
        widget_box_style = 'info'
        widget_border_radius = 10
        widget_border_width = 1
        info_style = 'danger'
        renderer_style = 'danger'
        renderer_tabs_style = 'info'
        save_figure_style = 'danger'
    elif style == 'minimal':
        model_style = ''
        model_parameters_style = 'minimal'
        channels_style = 'minimal'
        landmarks_style = 'minimal'
        logo_style = 'minimal'
        widget_box_style = ''
        widget_border_radius = 0
        widget_border_width = 0
        info_style = 'minimal'
        renderer_style = 'minimal'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'minimal'
    else:
        raise ValueError("style must be either coloured or minimal")

    # Get the maximum number of components per level
    max_n_shape = [sp.n_active_components for sp in atm.shape_models]

    # Check the given number of parameters (the returned n_parameters is a list
    # of len n_scales)
    n_shape_parameters = _check_n_parameters(n_shape_parameters, n_levels,
                                             max_n_shape)

    # Find initial groups and labels that will be passed to the landmark options
    # widget creation
    template_has_landmarks = atm.warped_templates[0].has_landmarks
    if template_has_landmarks:
        all_groups_keys, all_labels_keys = _extract_groups_labels(
            atm.warped_templates[0])
    else:
        all_groups_keys = [' ']
        all_labels_keys = [[' ']]

    # Get initial line and marker colours for each available group
    if len(all_labels_keys[0]) == 1:
        colours = ['r']
    else:
        colours = sample_colours_from_colourmap(len(all_labels_keys[0]), 'jet')

    # Initial options dictionaries
    channels_default = 0
    if atm.warped_templates[0].n_channels == 3:
        channels_default = None
    channel_options = {'n_channels': atm.warped_templates[0].n_channels,
                       'image_is_masked': isinstance(atm.warped_templates[0],
                                                     MaskedImage),
                       'channels': channels_default, 'glyph_enabled': False,
                       'glyph_block_size': 3, 'glyph_use_negative': False,
                       'sum_enabled': False,
                       'masked_enabled': isinstance(atm.warped_templates[0],
                                                    MaskedImage)}
    landmark_options = {'has_landmarks': template_has_landmarks,
                        'render_landmarks': template_has_landmarks,
                        'group_keys': all_groups_keys,
                        'labels_keys': all_labels_keys,
                        'group': all_groups_keys[0],
                        'with_labels': all_labels_keys[0]}
    image_options = {'alpha': 1.0, 'interpolation': 'none', 'cmap_name': None}
    line_options = {'render_lines': True, 'line_width': 1,
                    'line_colour': colours, 'line_style': '-'}
    marker_options = {'render_markers': True, 'marker_size': 20,
                      'marker_face_colour': colours,
                      'marker_edge_colour': colours,
                      'marker_style': 'o', 'marker_edge_width': 1}
    figure_options = {'x_scale': 1., 'y_scale': 1., 'render_axes': True,
                      'axes_font_name': 'sans-serif', 'axes_font_size': 10,
                      'axes_font_style': 'normal', 'axes_font_weight': 'normal',
                      'axes_x_limits': None, 'axes_y_limits': None}
    renderer_options = {'lines': line_options, 'markers': marker_options,
                        'figure': figure_options, 'image': image_options}

    # Define render function
    def render_function(name, value):
        # Clear current figure, but wait until the generation of the new data
        # that will be rendered
        ipydisplay.clear_output(wait=True)

        # Get selected level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # Compute weights and instance
        shape_weights = shape_model_parameters_wid.parameters
        instance = atm.instance(scale_index=level, shape_weights=shape_weights)

        # Update info
        update_info(atm, instance, level,
                    landmark_options_wid.selected_values['group'])

        # Render instance with selected options
        tmp1 = renderer_options_wid.selected_values[0]['lines']
        tmp2 = renderer_options_wid.selected_values[0]['markers']
        tmp3 = renderer_options_wid.selected_values[0]['figure']
        tmp4 = renderer_options_wid.selected_values[0]['image']
        new_figure_size = (tmp3['x_scale'] * figure_size[0],
                           tmp3['y_scale'] * figure_size[1])

        # Find the with_labels' indices
        with_labels_idx = [
            landmark_options_wid.selected_values['labels_keys'][0].index(lbl)
            for lbl in landmark_options_wid.selected_values['with_labels']]

        # Get line and marker colours
        line_colour = [tmp1['line_colour'][lbl_idx]
                       for lbl_idx in with_labels_idx]
        marker_face_colour = [tmp2['marker_face_colour'][lbl_idx]
                              for lbl_idx in with_labels_idx]
        marker_edge_colour = [tmp2['marker_edge_colour'][lbl_idx]
                              for lbl_idx in with_labels_idx]

        renderer = _visualize_menpo(
            instance, save_figure_wid.renderer,
            landmark_options_wid.selected_values['render_landmarks'],
            channel_options_wid.selected_values['image_is_masked'],
            channel_options_wid.selected_values['masked_enabled'],
            channel_options_wid.selected_values['channels'],
            channel_options_wid.selected_values['glyph_enabled'],
            channel_options_wid.selected_values['glyph_block_size'],
            channel_options_wid.selected_values['glyph_use_negative'],
            channel_options_wid.selected_values['sum_enabled'],
            landmark_options_wid.selected_values['group'],
            landmark_options_wid.selected_values['with_labels'],
            tmp1['render_lines'], tmp1['line_style'], tmp1['line_width'],
            line_colour, tmp2['render_markers'], tmp2['marker_style'],
            tmp2['marker_size'], tmp2['marker_edge_width'], marker_edge_colour,
            marker_face_colour, False, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, False, None, None, new_figure_size,
            tmp3['render_axes'], tmp3['axes_font_name'], tmp3['axes_font_size'],
            tmp3['axes_font_style'], tmp3['axes_x_limits'],
            tmp3['axes_y_limits'], tmp3['axes_font_weight'],
            tmp4['interpolation'], tmp4['alpha'], tmp4['cmap_name'])

        # Save the current figure id
        save_figure_wid.renderer = renderer

    # Define function that updates the info text
    def update_info(atm, instance, level, group):
        from menpofit.base import name_of_callable

        lvl_shape_mod = atm.shape_models[level]
        tmplt_inst = atm.warped_templates[level]
        n_channels = tmplt_inst.n_channels
        feat = atm.holistic_features[level]

        # Feature string
        tmp_feat = 'Feature is {} with {} channel{}'.format(
            name_of_callable(feat), n_channels, 's' * (n_channels > 1))

        # update info widgets
        text_per_line = [
            "> Warp using {} transform".format(atm.transform.__name__),
            "> Level {}/{}".format(
                level + 1, atm.n_scales),
            "> {} landmark points".format(
                instance.landmarks[group].lms.n_points),
            "> {} shape components ({:.2f}% of variance)".format(
                lvl_shape_mod.n_components,
                lvl_shape_mod.variance_ratio() * 100),
            "> {}".format(tmp_feat),
            "> Reference frame of length {} ({} x {}C, {} x {}C)".format(
                tmplt_inst.n_true_pixels() * n_channels,
                tmplt_inst.n_true_pixels(), n_channels, tmplt_inst._str_shape,
                n_channels),
            "> Instance: min={:.3f} , max={:.3f}".format(
                instance.pixels.min(), instance.pixels.max())]
        info_wid.set_widget_state(n_lines=len(text_per_line),
                                  text_per_line=text_per_line)

    # Plot shape variance function
    def plot_shape_variance(name):
        # Clear current figure, but wait until the generation of the new data
        # that will be rendered
        ipydisplay.clear_output(wait=True)

        # Get selected level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # Render
        new_figure_size = (
            renderer_options_wid.selected_values[0]['figure']['x_scale'] * 10,
            renderer_options_wid.selected_values[0]['figure']['y_scale'] * 3)
        plt.subplot(121)
        atm.shape_models[level].plot_eigenvalues_ratio(
            figure_id=save_figure_wid.renderer.figure_id)
        plt.subplot(122)
        renderer = atm.shape_models[level].plot_eigenvalues_cumulative_ratio(
            figure_id=save_figure_wid.renderer.figure_id,
            figure_size=new_figure_size)
        plt.show()

        # Save the current figure id
        save_figure_wid.renderer = renderer

    # Create widgets
    shape_model_parameters_wid = LinearModelParametersWidget(
        [0] * n_shape_parameters[0], render_function, params_str='param ',
        mode=mode, params_bounds=parameters_bounds, params_step=0.1,
        plot_variance_visible=True, plot_variance_function=plot_shape_variance,
        style=model_parameters_style)
    channel_options_wid = ChannelOptionsWidget(
        channel_options, render_function=render_function, style=channels_style)
    landmark_options_wid = LandmarkOptionsWidget(
        landmark_options, render_function=render_function,
        style=landmarks_style)
    renderer_options_wid = RendererOptionsWidget(
        renderer_options, ['markers', 'lines', 'image', 'figure_one'],
        object_selection_dropdown_visible=False,
        render_function=render_function, style=renderer_style,
        tabs_style=renderer_tabs_style)
    info_wid = TextPrintWidget(n_lines=10, text_per_line=[''] * 10,
                               style=info_style)
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = SaveFigureOptionsWidget(initial_renderer,
                                              style=save_figure_style)

    # Define function that updates options' widgets state
    def update_widgets(name, value):
        # Update shape model parameters
        shape_model_parameters_wid.set_widget_state(
            [0] * n_shape_parameters[value], params_str='param ',
            allow_callback=True)

        # Update channel options
        tmp_n_channels = atm.warped_templates[value].n_channels
        tmp_channels = channel_options_wid.selected_values['channels']
        tmp_glyph_enabled = channel_options_wid.selected_values['glyph_enabled']
        tmp_sum_enabled = channel_options_wid.selected_values['sum_enabled']
        if np.max(tmp_channels) > tmp_n_channels - 1:
            tmp_channels = 0
            tmp_glyph_enabled = False
            tmp_sum_enabled = False
        tmp_glyph_block_size = \
            channel_options_wid.selected_values['glyph_block_size']
        tmp_glyph_use_negative = \
            channel_options_wid.selected_values['glyph_use_negative']
        if not(tmp_n_channels == 3) and tmp_channels is None:
            tmp_channels = 0
        channel_options = {
            'n_channels': tmp_n_channels,
            'image_is_masked': isinstance(atm.warped_templates[0],
                                          MaskedImage),
            'channels': tmp_channels, 'glyph_enabled': tmp_glyph_enabled,
            'glyph_block_size': tmp_glyph_block_size,
            'glyph_use_negative': tmp_glyph_use_negative,
            'sum_enabled': tmp_sum_enabled,
            'masked_enabled': isinstance(atm.warped_templates[0],
                                         MaskedImage)}
        channel_options_wid.set_widget_state(channel_options, True)

    # Group widgets
    tmp_children = [shape_model_parameters_wid]
    if n_levels > 1:
        radio_str = OrderedDict()
        for l in range(n_levels):
            if l == 0:
                radio_str["Level {} (low)".format(l)] = l
            elif l == n_levels - 1:
                radio_str["Level {} (high)".format(l)] = l
            else:
                radio_str["Level {}".format(l)] = l
        level_wid = ipywidgets.RadioButtons(
            options=radio_str, description='Pyramid:', value=0)
        level_wid.on_trait_change(update_widgets, 'value')
        level_wid.on_trait_change(render_function, 'value')
        tmp_children.insert(0, level_wid)
    tmp_wid = ipywidgets.HBox(children=tmp_children, align='center',
                              box_style=model_style)
    options_box = ipywidgets.Tab(children=[tmp_wid, channel_options_wid,
                                           landmark_options_wid,
                                           renderer_options_wid,
                                           info_wid, save_figure_wid])
    tab_titles = ['Model', 'Channels', 'Landmarks', 'Renderer', 'Info',
                  'Export']
    for (k, tl) in enumerate(tab_titles):
        options_box.set_title(k, tl)
    logo_wid = LogoWidget(style=logo_style)
    logo_wid.margin = '0.1cm'
    wid = ipywidgets.HBox(children=[logo_wid, options_box], align='start')

    # Set widget's style
    wid.box_style = widget_box_style
    wid.border_radius = widget_border_radius
    wid.border_width = widget_border_width
    wid.border_color = _map_styles_to_hex_colours(widget_box_style)
    renderer_options_wid.margin = '0.2cm'

    # Display final widget
    ipydisplay.display(wid)

    # Reset value to trigger initial visualization
    renderer_options_wid.options_widgets[3].render_axes_checkbox.value = False

def plot_ced(errors, legend_entries=None, error_range=None,
             error_type='me_norm', figure_size=(10, 6), style='coloured',
             return_widget=False):
    r"""
    Widget for visualizing the cumulative error curves of the provided errors.

    Parameters
    ----------
    errors : `list` of `lists` of `float`
        A `list` that stores a `list` of errors to be plotted.
    legend_entries : `list` or `str` or ``None``, optional
        The `list` of names that will appear on the legend for each curve. If
        ``None``, then the names format is ``Curve {}.format(i)``.
    error_range : `list` of `float` with length 3, optional
        Specifies the horizontal axis range, i.e. ::

            error_range[0] = min_error
            error_range[1] = max_error
            error_range[2] = error_step

        If ``None``, then ::

            error_range = [0., 0.101, 0.005] for error_type = 'me_norm'
            error_range = [0., 20., 1.] for error_type = 'me'
            error_range = [0., 20., 1.] for error_type = 'rmse'

    error_type : {``'me_norm'``, ``'me'``, ``'rmse'``}, optional
        Specifies the type of the provided errors.
    figure_size : (`int`, `int`), optional
        The initial size of the rendered figure.
    style : {``'coloured'``, ``'minimal'``}, optional
        If ``'coloured'``, then the style of the widget will be coloured. If
        ``minimal``, then the style is simple using black and white colours.
    return_widget : `bool`, optional
        If ``True``, the widget object will be returned so that it can be used
        as part of a parent widget. If ``False``, the widget object is not
        returned, it is just visualized.
    """
    from menpofit.result import plot_cumulative_error_distribution
    print('Initializing...')

    # Make sure that errors is a list even with one list member
    if not isinstance(errors[0], list):
        errors = [errors]

    # Get number of curves to be plotted
    n_curves = len(errors)

    # Define the styling options
    if style == 'coloured':
        logo_style = 'danger'
        widget_box_style = 'danger'
        tabs_style = 'warning'
        renderer_tabs_style = 'info'
        save_figure_style = 'warning'
    else:
        logo_style = 'minimal'
        widget_box_style = 'minimal'
        tabs_style = 'minimal'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'minimal'

    # Parse options
    if legend_entries is None:
        legend_entries = ["Curve {}".format(i) for i in range(n_curves)]

    # Get horizontal axis errors
    if error_range is None:
        if error_type == 'me_norm':
            error_range = [0., 0.101, 0.005]
            x_axis_limit = 0.05
            x_axis_step = 0.005
            x_label = 'Normalized Point-to-Point Error'
        elif error_type == 'me' or error_type == 'rmse':
            error_range = [0., 20., 0.5]
            x_axis_limit = 5.
            x_axis_step = 0.5
            x_label = 'Point-to-Point Error'
        else:
            raise ValueError('error_type must be me_norm or me or rmse')
    else:
        x_axis_limit = (error_range[1] + error_range[0]) / 2
        x_axis_step = error_range[2]
        x_label = 'Error'
    y_slider_options = (0., 1., 0.05)
    y_label = 'Images Proportion'
    title = 'Cumulative error distribution'

    # Get initial line and marker colours for each curve
    if n_curves == 1:
        line_colours = ['b']
        marker_edge_colours = ['b']
    else:
        colours_tmp = sample_colours_from_colourmap(n_curves, 'jet')
        line_colours = [list(i) for i in colours_tmp]
        marker_edge_colours = [list(i) for i in colours_tmp]

    # Initial options dictionaries
    graph_options = {'legend_entries': legend_entries, 'x_label': x_label,
                     'y_label': y_label, 'title': title,
                     'x_axis_limits': [0., x_axis_limit],
                     'y_axis_limits': [0., 1.],
                     'render_lines': [True] * n_curves,
                     'line_colour': line_colours,
                     'line_style': ['-'] * n_curves,
                     'line_width': [2] * n_curves,
                     'render_markers': [True] * n_curves,
                     'marker_style': ['s'] * n_curves,
                     'marker_size': [8] * n_curves,
                     'marker_face_colour': ['w'] * n_curves,
                     'marker_edge_colour': marker_edge_colours,
                     'marker_edge_width': [2] * n_curves,
                     'render_legend': n_curves > 1, 'legend_title': '',
                     'legend_font_name': 'sans-serif',
                     'legend_font_style': 'normal', 'legend_font_size': 10,
                     'legend_font_weight': 'normal', 'legend_marker_scale': 1.,
                     'legend_location': 2, 'legend_bbox_to_anchor': (1.05, 1.),
                     'legend_border_axes_pad': 1., 'legend_n_columns': 1,
                     'legend_horizontal_spacing': 1.,
                     'legend_vertical_spacing': 1., 'legend_border': True,
                     'legend_border_padding': 0.5, 'legend_shadow': False,
                     'legend_rounded_corners': False, 'render_axes': False,
                     'axes_font_name': 'sans-serif', 'axes_font_size': 10,
                     'axes_font_style': 'normal', 'axes_font_weight': 'normal',
                     'figure_size': figure_size, 'render_grid': True,
                     'grid_line_style': '--', 'grid_line_width': 1}

    # Define render function
    def render_function(name, value):
        # Clear current figure, but wait until the generation of the new data
        # that will be rendered
        ipydisplay.clear_output(wait=True)

        # plot with selected options
        opts = wid.selected_values
        renderer = plot_cumulative_error_distribution(
            errors, error_range=[opts['x_axis_limits'][0],
                                 1.0001 * opts['x_axis_limits'][1],
                                 x_axis_step],
            figure_id=save_figure_wid.renderer.figure_id, new_figure=False,
            title=opts['title'], x_label=opts['x_label'],
            y_label=opts['y_label'], legend_entries=opts['legend_entries'],
            render_lines=opts['render_lines'], line_colour=opts['line_colour'],
            line_style=opts['line_style'], line_width=opts['line_width'],
            render_markers=opts['render_markers'],
            marker_style=opts['marker_style'], marker_size=opts['marker_size'],
            marker_face_colour=opts['marker_face_colour'],
            marker_edge_colour=opts['marker_edge_colour'],
            marker_edge_width=opts['marker_edge_width'],
            render_legend=opts['render_legend'],
            legend_title=opts['legend_title'],
            legend_font_name=opts['legend_font_name'],
            legend_font_style=opts['legend_font_style'],
            legend_font_size=opts['legend_font_size'],
            legend_font_weight=opts['legend_font_weight'],
            legend_marker_scale=opts['legend_marker_scale'],
            legend_location=opts['legend_location'],
            legend_bbox_to_anchor=opts['legend_bbox_to_anchor'],
            legend_border_axes_pad=opts['legend_border_axes_pad'],
            legend_n_columns=opts['legend_n_columns'],
            legend_horizontal_spacing=opts['legend_horizontal_spacing'],
            legend_vertical_spacing=opts['legend_vertical_spacing'],
            legend_border=opts['legend_border'],
            legend_border_padding=opts['legend_border_padding'],
            legend_shadow=opts['legend_shadow'],
            legend_rounded_corners=opts['legend_rounded_corners'],
            render_axes=opts['render_axes'],
            axes_font_name=opts['axes_font_name'],
            axes_font_size=opts['axes_font_size'],
            axes_font_style=opts['axes_font_style'],
            axes_font_weight=opts['axes_font_weight'],
            axes_x_limits=opts['x_axis_limits'],
            axes_y_limits=opts['y_axis_limits'],
            figure_size=opts['figure_size'], render_grid=opts['render_grid'],
            grid_line_style=opts['grid_line_style'],
            grid_line_width=opts['grid_line_width'])

        # show plot
        plt.show()

        # Save the current figure id
        save_figure_wid.renderer = renderer

    # Create widgets
    wid = GraphOptionsWidget(graph_options, error_range, y_slider_options,
                             render_function=render_function,
                             style=widget_box_style, tabs_style=tabs_style,
                             renderer_tabs_style=renderer_tabs_style)
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = SaveFigureOptionsWidget(initial_renderer,
                                              style=save_figure_style)

    # Group widgets
    logo = LogoWidget(style=logo_style)
    logo.margin = '0.1cm'
    wid.options_tab.children = [wid.graph_related_options, wid.renderer_widget,
                                save_figure_wid]
    wid.options_tab.set_title(0, 'Graph')
    wid.options_tab.set_title(1, 'Renderer')
    wid.options_tab.set_title(2, 'Export')
    wid.children = [logo, wid.options_tab]
    wid.align = 'start'

    # Display final widget
    ipydisplay.display(wid)

    # Reset value to trigger initial visualization
    wid.renderer_widget.options_widgets[3].render_axes_checkbox.value = True

    # return widget object if asked
    if return_widget:
        return wid

def _visualize(image, renderer, render_image, render_landmarks, image_is_masked,
               masked_enabled, channels, glyph_enabled, glyph_block_size,
               glyph_use_negative, sum_enabled, groups, with_labels,
               subplots_enabled, subplots_titles, image_axes_mode,
               render_lines, line_style, line_width, line_colour,
               render_markers, marker_style, marker_size, marker_edge_width,
               marker_edge_colour, marker_face_colour, render_numbering,
               numbers_horizontal_align, numbers_vertical_align,
               numbers_font_name, numbers_font_size, numbers_font_style,
               numbers_font_weight, numbers_font_colour, render_legend,
               legend_title, legend_font_name, legend_font_style,
               legend_font_size, legend_font_weight, legend_marker_scale,
               legend_location, legend_bbox_to_anchor, legend_border_axes_pad,
               legend_n_columns, legend_horizontal_spacing,
               legend_vertical_spacing, legend_border, legend_border_padding,
               legend_shadow, legend_rounded_corners, render_axes,
               axes_font_name, axes_font_size, axes_font_style,
               axes_font_weight, axes_x_limits, axes_y_limits, interpolation,
               alpha, figure_size):
    import matplotlib.pyplot as plt

    global glyph
    if glyph is None:
        from menpo.feature.visualize import glyph
    global sum_channels
    if sum_channels is None:
        from menpo.feature.visualize import sum_channels

    # This makes the code shorter for dealing with masked images vs non-masked
    # images
    mask_arguments = ({'masked': masked_enabled}
                      if image_is_masked else {})

    # plot
    if render_image:
        # image will be displayed
        if render_landmarks and len(groups) > 0:
            # there are selected landmark groups and they will be displayed
            if subplots_enabled:
                # calculate subplots structure
                subplots = MatplotlibSubplots()._subplot_layout(len(groups))
            # show image with landmarks
            for k, group in enumerate(groups):
                if subplots_enabled:
                    # create subplot
                    plt.subplot(subplots[0], subplots[1], k + 1)
                    if render_legend:
                        # set subplot's title
                        plt.title(subplots_titles[group],
                                  fontname=legend_font_name,
                                  fontstyle=legend_font_style,
                                  fontweight=legend_font_weight,
                                  fontsize=legend_font_size)
                if glyph_enabled:
                    # image, landmarks, masked, glyph
                    renderer = glyph(image, vectors_block_size=glyph_block_size,
                                     use_negative=glyph_use_negative,
                                     channels=channels).view_landmarks(
                        group=group, with_labels=with_labels[k],
                        without_labels=None, figure_id=renderer.figure_id,
                        new_figure=False, render_lines=render_lines[k],
                        line_style=line_style[k], line_width=line_width[k],
                        line_colour=line_colour[k],
                        render_markers=render_markers[k],
                        marker_style=marker_style[k],
                        marker_size=marker_size[k],
                        marker_edge_width=marker_edge_width[k],
                        marker_edge_colour=marker_edge_colour[k],
                        marker_face_colour=marker_face_colour[k],
                        render_numbering=render_numbering,
                        numbers_horizontal_align=numbers_horizontal_align,
                        numbers_vertical_align=numbers_vertical_align,
                        numbers_font_name=numbers_font_name,
                        numbers_font_size=numbers_font_size,
                        numbers_font_style=numbers_font_style,
                        numbers_font_weight=numbers_font_weight,
                        numbers_font_colour=numbers_font_colour,
                        render_legend=render_legend and not subplots_enabled,
                        legend_title=legend_title,
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
                        axes_font_size=axes_font_size,
                        axes_font_style=axes_font_style,
                        axes_font_weight=axes_font_weight,
                        axes_x_limits=axes_x_limits,
                        axes_y_limits=axes_y_limits,
                        interpolation=interpolation, alpha=alpha,
                        figure_size=figure_size, **mask_arguments)
                elif sum_enabled:
                    # image, landmarks, masked, sum
                    renderer = sum_channels(image,
                                            channels=channels).view_landmarks(
                        group=group, with_labels=with_labels[k],
                        without_labels=None, figure_id=renderer.figure_id,
                        new_figure=False, render_lines=render_lines[k],
                        line_style=line_style[k], line_width=line_width[k],
                        line_colour=line_colour[k],
                        render_markers=render_markers[k],
                        marker_style=marker_style[k],
                        marker_size=marker_size[k],
                        marker_edge_width=marker_edge_width[k],
                        marker_edge_colour=marker_edge_colour[k],
                        marker_face_colour=marker_face_colour[k],
                        render_numbering=render_numbering,
                        numbers_horizontal_align=numbers_horizontal_align,
                        numbers_vertical_align=numbers_vertical_align,
                        numbers_font_name=numbers_font_name,
                        numbers_font_size=numbers_font_size,
                        numbers_font_style=numbers_font_style,
                        numbers_font_weight=numbers_font_weight,
                        numbers_font_colour=numbers_font_colour,
                        render_legend=render_legend and not subplots_enabled,
                        legend_title=legend_title,
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
                        axes_font_size=axes_font_size,
                        axes_font_style=axes_font_style,
                        axes_font_weight=axes_font_weight,
                        axes_x_limits=axes_x_limits,
                        axes_y_limits=axes_y_limits,
                        interpolation=interpolation, alpha=alpha,
                        figure_size=figure_size, **mask_arguments)
                else:
                    # image, landmarks, masked, not glyph/sum
                    renderer = image.view_landmarks(
                        channels=channels, group=group,
                        with_labels=with_labels[k], without_labels=None,
                        figure_id=renderer.figure_id, new_figure=False,
                        render_lines=render_lines[k], line_style=line_style[k],
                        line_width=line_width[k], line_colour=line_colour[k],
                        render_markers=render_markers[k],
                        marker_style=marker_style[k],
                        marker_size=marker_size[k],
                        marker_edge_width=marker_edge_width[k],
                        marker_edge_colour=marker_edge_colour[k],
                        marker_face_colour=marker_face_colour[k],
                        render_numbering=render_numbering,
                        numbers_horizontal_align=numbers_horizontal_align,
                        numbers_vertical_align=numbers_vertical_align,
                        numbers_font_name=numbers_font_name,
                        numbers_font_size=numbers_font_size,
                        numbers_font_style=numbers_font_style,
                        numbers_font_weight=numbers_font_weight,
                        numbers_font_colour=numbers_font_colour,
                        render_legend=render_legend and not subplots_enabled,
                        legend_title=legend_title,
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
                        axes_font_size=axes_font_size,
                        axes_font_style=axes_font_style,
                        axes_font_weight=axes_font_weight,
                        axes_x_limits=axes_x_limits,
                        axes_y_limits=axes_y_limits,
                        interpolation=interpolation, alpha=alpha,
                        figure_size=figure_size, **mask_arguments)
        else:
            # either there are not any landmark groups selected or they won't
            # be displayed
            if glyph_enabled:
                # image, not landmarks, masked, glyph
                renderer = glyph(image, vectors_block_size=glyph_block_size,
                                 use_negative=glyph_use_negative,
                                 channels=channels).view(
                    render_axes=render_axes, axes_font_name=axes_font_name,
                    axes_font_size=axes_font_size,
                    axes_font_style=axes_font_style,
                    axes_font_weight=axes_font_weight,
                    axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
                    figure_size=figure_size, interpolation=interpolation,
                    alpha=alpha, **mask_arguments)
            elif sum_enabled:
                # image, not landmarks, masked, sum
                renderer = sum_channels(image, channels=channels).view(
                    render_axes=render_axes, axes_font_name=axes_font_name,
                    axes_font_size=axes_font_size,
                    axes_font_style=axes_font_style,
                    axes_font_weight=axes_font_weight,
                    axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
                    figure_size=figure_size, interpolation=interpolation,
                    alpha=alpha, **mask_arguments)
            else:
                # image, not landmarks, masked, not glyph/sum
                renderer = image.view(
                    channels=channels, render_axes=render_axes,
                    axes_font_name=axes_font_name,
                    axes_font_size=axes_font_size,
                    axes_font_style=axes_font_style,
                    axes_font_weight=axes_font_weight,
                    axes_x_limits=axes_x_limits,
                    axes_y_limits=axes_y_limits, figure_size=figure_size,
                    interpolation=interpolation, alpha=alpha, **mask_arguments)
    else:
        # image won't be displayed
        if render_landmarks and len(groups) > 0:
            # there are selected landmark groups and they will be displayed
            if subplots_enabled:
                # calculate subplots structure
                subplots = MatplotlibSubplots()._subplot_layout(len(groups))
            # not image, landmarks
            for k, group in enumerate(groups):
                if subplots_enabled:
                    # create subplot
                    plt.subplot(subplots[0], subplots[1], k + 1)
                    if render_legend:
                        # set subplot's title
                        plt.title(subplots_titles[group],
                                  fontname=legend_font_name,
                                  fontstyle=legend_font_style,
                                  fontweight=legend_font_weight,
                                  fontsize=legend_font_size)
                image.landmarks[group].lms.view(
                    image_view=image_axes_mode, render_lines=render_lines[k],
                    line_style=line_style[k], line_width=line_width[k],
                    line_colour=line_colour[k],
                    render_markers=render_markers[k],
                    marker_style=marker_style[k], marker_size=marker_size[k],
                    marker_edge_width=marker_edge_width[k],
                    marker_edge_colour=marker_edge_colour[k],
                    marker_face_colour=marker_face_colour[k],
                    render_axes=render_axes, axes_font_name=axes_font_name,
                    axes_font_size=axes_font_size,
                    axes_font_style=axes_font_style,
                    axes_font_weight=axes_font_weight,
                    axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
                    figure_size=figure_size)
            if not subplots_enabled:
                if len(groups) % 2 == 0:
                    plt.gca().invert_yaxis()
                if render_legend:
                    # Options related to legend's font
                    prop = {'family': legend_font_name,
                            'size': legend_font_size,
                            'style': legend_font_style,
                            'weight': legend_font_weight}

                    # display legend on side
                    plt.gca().legend(groups, title=legend_title, prop=prop,
                                     loc=legend_location,
                                     bbox_to_anchor=legend_bbox_to_anchor,
                                     borderaxespad=legend_border_axes_pad,
                                     ncol=legend_n_columns,
                                     columnspacing=legend_horizontal_spacing,
                                     labelspacing=legend_vertical_spacing,
                                     frameon=legend_border,
                                     borderpad=legend_border_padding,
                                     shadow=legend_shadow,
                                     fancybox=legend_rounded_corners,
                                     markerscale=legend_marker_scale)

    # show plot
    plt.show()

    return renderer

def visualize_fitting_result(fitting_results, figure_size=(10, 8),
                             style='coloured', browser_style='buttons'):
    r"""
    Widget that allows browsing through a `list` of fitting results.

    Parameters
    -----------
    fitting_results : `list` of :map:`FittingResult` or subclass
        The `list` of fitting results to be displayed. Note that the fitting
        results can have different attributes between them, i.e. different
        number of iterations, number of channels etc.
    figure_size : (`int`, `int`), optional
        The initial size of the plotted figures.
    style : {``'coloured'``, ``'minimal'``}, optional
        If ``'coloured'``, then the style of the widget will be coloured. If
        ``minimal``, then the style is simple using black and white colours.
    browser_style : {``'buttons'``, ``'slider'``}, optional
        It defines whether the selector of the objects will have the form of
        plus/minus buttons or a slider.
    """
    from menpo.image import MaskedImage
    print('Initializing...')

    # Make sure that fitting_results is a list even with one fitting_result
    if not isinstance(fitting_results, list):
        fitting_results = [fitting_results]

    # Get the number of fitting_results
    n_fitting_results = len(fitting_results)

    # Define the styling options
    if style == 'coloured':
        logo_style = 'info'
        widget_box_style = 'info'
        widget_border_radius = 10
        widget_border_width = 1
        animation_style = 'info'
        fitting_result_style = 'danger'
        fitting_result_iterations_style = 'danger'
        fitting_result_iterations_sliders_style = 'warning'
        channels_style = 'danger'
        info_style = 'danger'
        renderer_style = 'danger'
        renderer_tabs_style = 'info'
        save_figure_style = 'danger'
        plot_ced_but_style = 'primary'
    else:
        logo_style = 'minimal'
        widget_box_style = ''
        widget_border_radius = 0
        widget_border_width = 0
        fitting_result_style = 'minimal'
        fitting_result_iterations_style = 'minimal'
        fitting_result_iterations_sliders_style = 'minimal'
        channels_style = 'minimal'
        animation_style = 'minimal'
        info_style = 'minimal'
        renderer_style = 'minimal'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'minimal'
        plot_ced_but_style = ''

    # Check if all fitting_results have gt_shape in order to show the ced button
    show_ced = all(f.gt_shape is not None for f in fitting_results)

    # Create dictionaries
    all_groups = ['final', 'initial', 'ground', 'iterations']
    groups_final_dict = dict()
    colour_final_dict = dict()
    groups_final_dict['initial'] = 'Initial shape'
    colour_final_dict['initial'] = 'b'
    groups_final_dict['final'] = 'Final shape'
    colour_final_dict['final'] = 'r'
    groups_final_dict['ground'] = 'Ground-truth shape'
    colour_final_dict['ground'] = 'y'
    groups_final_dict['iterations'] = 'Iterations'
    colour_final_dict['iterations'] = 'r'

    # Initial options dictionaries
    channels_default = 0
    if fitting_results[0].fitted_image.n_channels == 3:
        channels_default = None
    channel_options = \
        {'n_channels': fitting_results[0].fitted_image.n_channels,
         'image_is_masked': isinstance(fitting_results[0].fitted_image,
                                       MaskedImage),
         'channels': channels_default,
         'glyph_enabled': False,
         'glyph_block_size': 3,
         'glyph_use_negative': False,
         'sum_enabled': False,
         'masked_enabled': False}
    all_groups_keys, _ = _extract_groups_labels(fitting_results[0].fitted_image)
    fitting_result_options = {'all_groups': all_groups_keys,
                              'render_image': True,
                              'selected_groups': ['final'],
                              'subplots_enabled': True}
    fitting_result_iterations_options = \
        {'n_iters': fitting_results[0].iter_image.landmarks.n_groups,
         'image_has_gt_shape': not fitting_results[0].gt_shape is None,
         'n_points': fitting_results[0].fitted_image.landmarks['final'].lms.n_points,
         'iter_str': 'iter_',
         'selected_groups': ['iter_0'],
         'render_image': True,
         'subplots_enabled': True,
         'displacement_type': 'mean'}
    index = {'min': 0, 'max': n_fitting_results - 1, 'step': 1, 'index': 0}
    figure_options = {'x_scale': 1., 'y_scale': 1., 'render_axes': True,
                      'axes_font_name': 'sans-serif', 'axes_font_size': 10,
                      'axes_font_style': 'normal', 'axes_font_weight': 'normal',
                      'axes_x_limits': None, 'axes_y_limits': None}
    numbering_options = {'render_numbering': False,
                         'numbers_font_name': 'serif',
                         'numbers_font_size': 10,
                         'numbers_font_style': 'normal',
                         'numbers_font_weight': 'normal',
                         'numbers_font_colour': ['k'],
                         'numbers_horizontal_align': 'center',
                         'numbers_vertical_align': 'bottom'}
    legend_options = {'render_legend': False, 'legend_title': '',
                      'legend_font_name': 'sans-serif',
                      'legend_font_style': 'normal', 'legend_font_size': 11,
                      'legend_font_weight': 'normal', 'legend_marker_scale': 1.,
                      'legend_location': 2, 'legend_bbox_to_anchor': (1.05, 1.),
                      'legend_border_axes_pad': 1., 'legend_n_columns': 1,
                      'legend_horizontal_spacing': 1,
                      'legend_vertical_spacing': 1., 'legend_border': True,
                      'legend_border_padding': 0.5, 'legend_shadow': False,
                      'legend_rounded_corners': True}
    image_options = {'interpolation': 'bilinear', 'cmap_name': None,
                     'alpha': 1.0}
    renderer_options = []
    for group in all_groups:
        lines_options = {'render_lines': True, 'line_width': 1,
                         'line_colour': [colour_final_dict[group]],
                         'line_style': '-'}
        marker_options = {'render_markers': True, 'marker_size': 20,
                          'marker_face_colour': [colour_final_dict[group]],
                          'marker_edge_colour': [colour_final_dict[group]],
                          'marker_style': 'o', 'marker_edge_width': 1}
        tmp = {'lines': lines_options, 'markers': marker_options,
               'numbering': numbering_options, 'legend': legend_options,
               'figure': figure_options, 'image': image_options}
        renderer_options.append(tmp)

    # Define function that plots errors curve
    def plot_errors_function(name):
        # Clear current figure, but wait until the new data to be displayed are
        # generated
        ipydisplay.clear_output(wait=True)

        # Get selected index
        im = 0
        if n_fitting_results > 1:
            im = image_number_wid.selected_values['index']

        # Render
        new_figure_size = (
            renderer_options_wid.selected_values[0]['figure']['x_scale'] * 10,
            renderer_options_wid.selected_values[0]['figure']['y_scale'] * 3)
        renderer = fitting_results[im].plot_errors(
            error_type=_error_type_key_to_func(error_type_wid.value),
            figure_id=save_figure_wid.renderer.figure_id,
            figure_size=new_figure_size)

        # Show figure
        plt.show()

        # Save the current figure id
        save_figure_wid.renderer = renderer

    # Define function that plots displacements curve
    def plot_displacements_function(name, value):
        # Clear current figure, but wait until the new data to be displayed are
        # generated
        ipydisplay.clear_output(wait=True)

        # Get selected index
        im = 0
        if n_fitting_results > 1:
            im = image_number_wid.selected_values['index']

        # Render
        new_figure_size = (
            renderer_options_wid.selected_values[0]['figure']['x_scale'] * 10,
            renderer_options_wid.selected_values[0]['figure']['y_scale'] * 3)
        if (value == 'max' or value == 'min' or value == 'mean' or
                value == 'median'):
            renderer = fitting_results[im].plot_displacements(
                figure_id=save_figure_wid.renderer.figure_id,
                figure_size=new_figure_size, stat_type=value)
        else:
            all_displacements = fitting_results[im].displacements()
            d_curve = [iteration_displacements[value]
                       for iteration_displacements in all_displacements]
            from menpo.visualize import GraphPlotter
            ylabel = "Displacement of Point {}".format(value)
            title = "Point {} displacement per " \
                    "iteration of Image {}".format(value, im)
            renderer = GraphPlotter(
                figure_id=save_figure_wid.renderer.figure_id,
                new_figure=False, x_axis=range(len(d_curve)), y_axis=[d_curve],
                title=title, x_label='Iteration', y_label=ylabel,
                x_axis_limits=(0, len(d_curve)-1), y_axis_limits=None).render(
                    render_lines=True, line_colour='b', line_style='-',
                    line_width=2, render_markers=True, marker_style='o',
                    marker_size=4, marker_face_colour='b',
                    marker_edge_colour='k', marker_edge_width=1.,
                    render_legend=False, render_axes=True,
                    axes_font_name='sans-serif', axes_font_size=10,
                    axes_font_style='normal', axes_font_weight='normal',
                    render_grid=True, grid_line_style='--', grid_line_width=0.5,
                    figure_size=new_figure_size)

        # Show figure
        plt.show()

        # Save the current figure id
        save_figure_wid.renderer = renderer

    # Define render function
    def render_function(name, value):
        # Clear current figure, but wait until the generation of the new data
        # that will be rendered
        ipydisplay.clear_output(wait=True)

        # get selected image
        im = 0
        if n_fitting_results > 1:
            im = image_number_wid.selected_values['index']

        # selected mode: final or iterations
        final_enabled = False
        if result_wid.selected_index == 0:
            final_enabled = True

        # update info text widget
        update_info('', error_type_wid.value)

        # get selected options
        if final_enabled:
            # image object
            image = fitting_results[im].fitted_image
            render_image = fitting_result_wid.selected_values['render_image']
            # groups
            groups = fitting_result_wid.selected_values['selected_groups']
            # subplots
            subplots_enabled = fitting_result_wid.selected_values[
                'subplots_enabled']
            subplots_titles = groups_final_dict
            # lines and markers options
            render_lines = []
            line_colour = []
            line_style = []
            line_width = []
            render_markers = []
            marker_style = []
            marker_size = []
            marker_face_colour = []
            marker_edge_colour = []
            marker_edge_width = []
            for g in groups:
                group_idx = all_groups.index(g)
                tmp1 = renderer_options_wid.selected_values[group_idx]['lines']
                tmp2 = renderer_options_wid.selected_values[group_idx]['markers']
                render_lines.append(tmp1['render_lines'])
                line_colour.append(tmp1['line_colour'])
                line_style.append(tmp1['line_style'])
                line_width.append(tmp1['line_width'])
                render_markers.append(tmp2['render_markers'])
                marker_style.append(tmp2['marker_style'])
                marker_size.append(tmp2['marker_size'])
                marker_face_colour.append(tmp2['marker_face_colour'])
                marker_edge_colour.append(tmp2['marker_edge_colour'])
                marker_edge_width.append(tmp2['marker_edge_width'])
        else:
            # image object
            image = fitting_results[im].iter_image
            render_image = fitting_result_iterations_wid.selected_values[
                'render_image']
            # groups
            groups = fitting_result_iterations_wid.selected_values[
                'selected_groups']
            # subplots
            subplots_enabled = fitting_result_iterations_wid.selected_values[
                'subplots_enabled']
            subplots_titles = dict()
            iter_str = fitting_result_iterations_wid.selected_values['iter_str']
            for i, g in enumerate(groups):
                iter_num = g[len(iter_str)::]
                subplots_titles[iter_str + iter_num] = "Iteration " + iter_num
            # lines and markers options
            group_idx = all_groups.index('iterations')
            tmp1 = renderer_options_wid.selected_values[group_idx]['lines']
            tmp2 = renderer_options_wid.selected_values[group_idx]['markers']
            render_lines = [tmp1['render_lines']] * len(groups)
            line_style = [tmp1['line_style']] * len(groups)
            line_width = [tmp1['line_width']] * len(groups)
            render_markers = [tmp2['render_markers']] * len(groups)
            marker_style = [tmp2['marker_style']] * len(groups)
            marker_size = [tmp2['marker_size']] * len(groups)
            marker_edge_colour = [tmp2['marker_edge_colour']] * len(groups)
            marker_edge_width = [tmp2['marker_edge_width']] * len(groups)
            if (subplots_enabled or
                    fitting_result_iterations_wid.iterations_mode.value ==
                    'animation'):
                line_colour = [tmp1['line_colour']] * len(groups)
                marker_face_colour = [tmp2['marker_face_colour']] * len(groups)
            else:
                cols = sample_colours_from_colourmap(len(groups), 'jet')
                line_colour = cols
                marker_face_colour = cols

        tmp1 = renderer_options_wid.selected_values[0]['numbering']
        tmp2 = renderer_options_wid.selected_values[0]['legend']
        tmp3 = renderer_options_wid.selected_values[0]['figure']
        tmp4 = renderer_options_wid.selected_values[0]['image']
        new_figure_size = (tmp3['x_scale'] * figure_size[0],
                           tmp3['y_scale'] * figure_size[1])

        # call helper _visualize
        renderer = _visualize(
            image=image, renderer=save_figure_wid.renderer,
            render_image=render_image, render_landmarks=True,
            image_is_masked=False, masked_enabled=False,
            channels=channel_options_wid.selected_values['channels'],
            glyph_enabled=channel_options_wid.selected_values['glyph_enabled'],
            glyph_block_size=channel_options_wid.selected_values['glyph_block_size'],
            glyph_use_negative=channel_options_wid.selected_values['glyph_use_negative'],
            sum_enabled=channel_options_wid.selected_values['sum_enabled'],
            groups=groups, with_labels=[None] * len(groups),
            subplots_enabled=subplots_enabled, subplots_titles=subplots_titles,
            image_axes_mode=True, render_lines=render_lines,
            line_style=line_style, line_width=line_width,
            line_colour=line_colour, render_markers=render_markers,
            marker_style=marker_style, marker_size=marker_size,
            marker_edge_width=marker_edge_width,
            marker_edge_colour=marker_edge_colour,
            marker_face_colour=marker_face_colour,
            render_numbering=tmp1['render_numbering'],
            numbers_horizontal_align=tmp1['numbers_horizontal_align'],
            numbers_vertical_align=tmp1['numbers_vertical_align'],
            numbers_font_name=tmp1['numbers_font_name'],
            numbers_font_size=tmp1['numbers_font_size'],
            numbers_font_style=tmp1['numbers_font_style'],
            numbers_font_weight=tmp1['numbers_font_weight'],
            numbers_font_colour=tmp1['numbers_font_colour'],
            render_legend=tmp2['render_legend'],
            legend_title=tmp2['legend_title'],
            legend_font_name=tmp2['legend_font_name'],
            legend_font_style=tmp2['legend_font_style'],
            legend_font_size=tmp2['legend_font_size'],
            legend_font_weight=tmp2['legend_font_weight'],
            legend_marker_scale=tmp2['legend_marker_scale'],
            legend_location=tmp2['legend_location'],
            legend_bbox_to_anchor=tmp2['legend_bbox_to_anchor'],
            legend_border_axes_pad=tmp2['legend_border_axes_pad'],
            legend_n_columns=tmp2['legend_n_columns'],
            legend_horizontal_spacing=tmp2['legend_horizontal_spacing'],
            legend_vertical_spacing=tmp2['legend_vertical_spacing'],
            legend_border=tmp2['legend_border'],
            legend_border_padding=tmp2['legend_border_padding'],
            legend_shadow=tmp2['legend_shadow'],
            legend_rounded_corners=tmp2['legend_rounded_corners'],
            render_axes=tmp3['render_axes'],
            axes_font_name=tmp3['axes_font_name'],
            axes_font_size=tmp3['axes_font_size'],
            axes_font_style=tmp3['axes_font_style'],
            axes_font_weight=tmp3['axes_font_weight'],
            axes_x_limits=tmp3['axes_x_limits'],
            axes_y_limits=tmp3['axes_y_limits'],
            interpolation=tmp4['interpolation'],
            alpha=tmp4['alpha'], figure_size=new_figure_size)

        # Save the current figure id
        save_figure_wid.renderer = renderer

    # Define function that updates info text
    def update_info(name, value):
        # Get selected image
        im = 0
        if n_fitting_results > 1:
            im = image_number_wid.selected_values['index']

        # Create output str
        if fitting_results[im].gt_shape is not None:
            from menpofit.result import (
                compute_root_mean_square_error, compute_point_to_point_error,
                compute_normalise_point_to_point_error)
            if value is 'me_norm':
                func = compute_normalise_point_to_point_error
            elif value is 'me':
                func = compute_point_to_point_error
            elif value is 'rmse':
                func = compute_root_mean_square_error
            text_per_line = [
                "> Initial error: {:.4f}".format(
                    fitting_results[im].initial_error(compute_error=func)),
                "> Final error: {:.4f}".format(
                    fitting_results[im].final_error(compute_error=func)),
                "> {} iterations".format(fitting_results[im].n_iters)]
        else:
            text_per_line = [
                "> {} iterations".format(fitting_results[im].n_iters)]
        if hasattr(fitting_results[im], 'n_scales'):  # Multilevel result
            text_per_line.append("> {} scales".format(
                fitting_results[im].n_scales))
        info_wid.set_widget_state(n_lines=len(text_per_line),
                                  text_per_line=text_per_line)

    # Create options widgets
    fitting_result_wid = FittingResultOptionsWidget(
        fitting_result_options, render_function=render_function,
        style=fitting_result_style)
    fitting_result_iterations_wid = FittingResultIterationsOptionsWidget(
        fitting_result_iterations_options, render_function=render_function,
        plot_errors_function=plot_errors_function,
        plot_displacements_function=plot_displacements_function,
        style=fitting_result_iterations_style,
        sliders_style=fitting_result_iterations_sliders_style)
    channel_options_wid = ChannelOptionsWidget(
        channel_options, render_function=render_function, style=channels_style)
    renderer_options_wid = RendererOptionsWidget(
        renderer_options,
        ['markers', 'lines', 'figure_one', 'legend', 'numbering', 'image'],
        objects_names=all_groups,
        object_selection_dropdown_visible=True,
        render_function=render_function, selected_object=0,
        style=renderer_style, tabs_style=renderer_tabs_style)
    info_wid = TextPrintWidget(n_lines=4, text_per_line=[''] * 4,
                               style=info_style)
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = SaveFigureOptionsWidget(initial_renderer,
                                              style=save_figure_style)

    # Create error type radio buttons
    error_type_values = OrderedDict()
    error_type_values['Point-to-point Normalized Mean Error'] = 'me_norm'
    error_type_values['Point-to-point Mean Error'] = 'me'
    error_type_values['RMS Error'] = 'rmse'
    error_type_wid = ipywidgets.RadioButtons(
        options=error_type_values, value='me_norm', description='Error type')
    error_type_wid.on_trait_change(update_info, 'value')
    plot_ced_but = ipywidgets.Button(description='Plot CED', visible=show_ced,
                                     button_style=plot_ced_but_style)
    error_wid = ipywidgets.VBox(children=[error_type_wid, plot_ced_but],
                                align='center')

    # Invoke plot_ced widget
    def plot_ced_fun(name):
        # Make button invisible, so that it cannot be pressed again until
        # widget closes
        plot_ced_but.visible = False

        error_type = error_type_wid.value
        func = _error_type_key_to_func(error_type)

        # Create errors list
        fit_errors = [f.final_error(compute_error=func)
                      for f in fitting_results]
        initial_errors = [f.initial_error(compute_error=func)
                          for f in fitting_results]
        errors = [fit_errors, initial_errors]

        # Call plot_ced
        plot_ced_widget = plot_ced(
            errors, figure_size=(9, 5), error_type=error_type,
            error_range=None, legend_entries=['Final Fitting',
                                              'Initialization'],
            style=style, return_widget=True)

        # If another tab is selected, then close the widget.
        def close_plot_ced_fun(name, value):
            if value != 3:
                plot_ced_widget.close()
                plot_ced_but.visible = True
        options_box.on_trait_change(close_plot_ced_fun, 'selected_index')

        # If another error type, then close the widget
        def close_plot_ced_fun_2(name, value):
            plot_ced_widget.close()
            plot_ced_but.visible = True
        error_type_wid.on_trait_change(close_plot_ced_fun_2, 'value')
    plot_ced_but.on_click(plot_ced_fun)

    # Define function that updates options' widgets state
    def update_widgets(name, value):
        # Get new groups and labels
        group_keys, labels_keys = _extract_groups_labels(
            fitting_results[value].fitted_image)
        # Update channel options
        tmp_channels = channel_options_wid.selected_values['channels']
        tmp_glyph_enabled = channel_options_wid.selected_values['glyph_enabled']
        tmp_sum_enabled = channel_options_wid.selected_values['sum_enabled']
        if (np.max(tmp_channels) >
                fitting_results[value].fitted_image.n_channels - 1):
            tmp_channels = 0
            tmp_glyph_enabled = False
            tmp_sum_enabled = False
        tmp_glyph_block_size = \
            channel_options_wid.selected_values['glyph_block_size']
        tmp_glyph_use_negative = \
            channel_options_wid.selected_values['glyph_use_negative']
        if (not(fitting_results[value].fitted_image.n_channels == 3) and
                tmp_channels is None):
            tmp_channels = 0
        channel_options = {
            'n_channels': fitting_results[value].fitted_image.n_channels,
            'image_is_masked': isinstance(fitting_results[value].fitted_image,
                                          MaskedImage),
            'channels': tmp_channels, 'glyph_enabled': tmp_glyph_enabled,
            'glyph_block_size': tmp_glyph_block_size,
            'glyph_use_negative': tmp_glyph_use_negative,
            'sum_enabled': tmp_sum_enabled,
            'masked_enabled': isinstance(fitting_results[value].fitted_image,
                                         MaskedImage)}
        channel_options_wid.set_widget_state(channel_options, False)

        # Update final result's options
        tmp_groups = []
        for g in fitting_result_wid.selected_values['selected_groups']:
            if g in group_keys:
                tmp_groups.append(g)
        fitting_result_options = {
            'all_groups': group_keys, 'selected_groups': tmp_groups,
            'render_image': fitting_result_wid.selected_values['render_image'],
            'subplots_enabled':
                fitting_result_wid.selected_values['subplots_enabled']}
        fitting_result_wid.set_widget_state(fitting_result_options,
                                            allow_callback=False)

        # Update iterations result's options
        fitting_result_iterations_options = {
            'n_iters': fitting_results[value].iter_image.landmarks.n_groups,
            'image_has_gt_shape': fitting_results[value].gt_shape is not None,
            'n_points':
            fitting_results[value].fitted_image.landmarks['final'].lms.n_points,
            'render_image': None,
            'iter_str': 'iter_',
            'selected_groups': ['iter_0'],
            'subplots_enabled': None,
            'displacement_type': None}
        fitting_result_iterations_wid.set_widget_state(
            fitting_result_iterations_options, allow_callback=True)

    # Group widgets
    options_wid = ipywidgets.Tab(children=[channel_options_wid,
                                           renderer_options_wid])
    options_wid.set_title(0, 'Channels')
    options_wid.set_title(1, 'Renderer')
    result_wid = ipywidgets.Tab(children=[fitting_result_wid,
                                          fitting_result_iterations_wid])
    result_wid.set_title(0, 'Final')
    result_wid.set_title(1, 'Iterations')
    result_wid.on_trait_change(render_function, 'selected_index')
    if n_fitting_results > 1:
        # Image selection slider
        image_number_wid = AnimationOptionsWidget(
            index, render_function=render_function,
            update_function=update_widgets, index_style=browser_style,
            interval=0.3, description='Image', minus_description='<',
            plus_description='>', loop_enabled=True, text_editable=True,
            style=animation_style)

        # Header widget
        header_wid = ipywidgets.HBox(
            children=[LogoWidget(style=logo_style), image_number_wid],
            align='start')

        # Define function that combines the results' tab widget with the
        # animation. Specifically, if the animation is activated and the user
        # selects the iterations tab, then the animation stops.
        def results_tab_fun(name, value):
            if value == 1 and image_number_wid.play_options_toggle.value:
                image_number_wid.stop_options_toggle.value = True
        result_wid.on_trait_change(results_tab_fun, 'selected_index')

        # Widget titles
        if show_ced:
            tab_titles = ['Info', 'Result', 'Options', 'CED', 'Export']
        else:
            tab_titles = ['Info', 'Result', 'Options', 'Error type', 'Export']
    else:
        # Do not show the plot ced button
        plot_ced_but.visible = False

        # Header widget
        header_wid = LogoWidget(style=logo_style)
        tab_titles = ['Info', 'Result', 'Options', 'Error type', 'Export']
    header_wid.margin = '0.2cm'
    options_box = ipywidgets.Tab(
        children=[info_wid, result_wid, options_wid, error_wid,
                  save_figure_wid], margin='0.2cm')
    for (k, tl) in enumerate(tab_titles):
        options_box.set_title(k, tl)
    if n_fitting_results > 1:
        wid = ipywidgets.VBox(children=[header_wid, options_box], align='start')
    else:
        wid = ipywidgets.HBox(children=[header_wid, options_box], align='start')
    if n_fitting_results > 1:
        # If animation is activated and the user selects the save figure tab,
        # then the animation stops.
        def save_fig_tab_fun(name, value):
            if value == 3 and image_number_wid.play_options_toggle.value:
                image_number_wid.stop_options_toggle.value = True
        options_box.on_trait_change(save_fig_tab_fun, 'selected_index')

    # Set widget's style
    wid.box_style = widget_box_style
    wid.border_radius = widget_border_radius
    wid.border_width = widget_border_width
    wid.border_color = _map_styles_to_hex_colours(widget_box_style)

    # Display final widget
    ipydisplay.display(wid)

    # Reset value to trigger initial visualization
    renderer_options_wid.options_widgets[3].render_legend_checkbox.value = True


def _error_type_key_to_func(error_type):
    from menpofit.result import (
        compute_root_mean_square_error, compute_point_to_point_error,
        compute_normalise_point_to_point_error)
    if error_type is 'me_norm':
        func = compute_normalise_point_to_point_error
    elif error_type is 'me':
        func = compute_point_to_point_error
    elif error_type is 'rmse':
        func = compute_root_mean_square_error
    return func
