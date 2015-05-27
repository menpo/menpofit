import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import collections as mc

import IPython.html.widgets as ipywidgets
import IPython.display as ipydisplay

from menpo.visualize.widgets import (RendererOptionsWidget,
                                     ChannelOptionsWidget,
                                     LandmarkOptionsWidget, TextPrintWidget,
                                     AnimationOptionsWidget,
                                     SaveFigureOptionsWidget)
from menpo.visualize.widgets.tools import (_format_box, LogoWidget,
                                           _map_styles_to_hex_colours)
from menpo.visualize.widgets.base import _extract_groups_labels, _visualize
from menpo.visualize.viewmatplotlib import (MatplotlibImageViewer2d,
                                            sample_colours_from_colourmap,
                                            MatplotlibSubplots)

from .options import (LinearModelParametersWidget, FittingResultOptionsWidget,
                      FittingResultIterationsOptionsWidget)

# This glyph import is called frequently during visualisation, so we ensure
# that we only import it once
glyph = None

def _check_n_parameters(n_params, n_levels, max_n_params):
    r"""
    Checks the maximum number of components per level either of the shape
    or the appearance model. It must be ``None`` or `int` or `float` or a `list`
    of those containing ``1`` or ``n_levels`` elements.
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
    # of len n_levels)
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
                figure_size=new_figure_size, label=None)

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
                figure_size=new_figure_size, label=None)

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
    mode_wid = ipywidgets.RadioButtonsWidget(options=mode_dict,
                                             description='Mode:', value=1)
    mode_wid.on_trait_change(render_function, 'value')
    mean_wid = ipywidgets.CheckboxWidget(value=False,
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
    axes_mode_wid = ipywidgets.RadioButtonsWidget(
        options={'Image': 1, 'Point cloud': 2}, description='Axes mode:',
        value=2)
    axes_mode_wid.on_trait_change(render_function, 'value')
    renderer_options_wid = RendererOptionsWidget(
        renderer_options, ['lines', 'markers', 'figure_one'],
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
        level_wid = ipywidgets.RadioButtonsWidget(
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
    # of len n_levels)
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

        # Get n_labels
        n_labels = len(landmark_options_wid.selected_values['with_labels'])

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

        renderer = _visualize(
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
        renderer_options, ['lines', 'markers', 'figure_one', 'image'],
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
        level_wid = ipywidgets.RadioButtonsWidget(
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
    renderer_options_wid.options_widgets[2].render_axes_checkbox.value = False

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
    n_levels = aam.n_levels

    # Define the styling options
    if style == 'coloured':
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
    # of len n_levels)
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
        instance = aam.instance(level=level, shape_weights=shape_weights,
                                appearance_weights=appearance_weights)

        # Update info
        update_info(aam, instance, level,
                    landmark_options_wid.selected_values['group'])

        # Get n_labels
        n_labels = len(landmark_options_wid.selected_values['with_labels'])

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

        renderer = _visualize(
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
        feat = (aam.features if aam.pyramid_on_features
                else aam.features[level])

        # Feature string
        tmp_feat = 'Feature is {} with {} channel{}'.format(
            name_of_callable(feat), n_channels, 's' * (n_channels > 1))

        # create info str
        if n_levels == 1:
            tmp_shape_models = ''
            tmp_pyramid = ''
        else:  # n_levels > 1
            # shape models info
            if aam.scaled_shape_models:
                tmp_shape_models = "Each level has a scaled shape model " \
                                   "(reference frame)"
            else:
                tmp_shape_models = "Shape models (reference frames) are " \
                                   "not scaled"
            # pyramid info
            if aam.pyramid_on_features:
                tmp_pyramid = "Pyramid was applied on feature space"
            else:
                tmp_pyramid = "Features were extracted at each pyramid level"

        # update info widgets
        text_per_line = [
            "> AAM:",
            "   > {} training images".format(aam.n_training_images),
            "   > {}".format(tmp_shape_models),
            "   > Warp using {} transform".format(aam.transform.__name__),
            "   > {}".format(tmp_pyramid),
            "   > Level {}/{} (downscale={:.1f})".format(
                level + 1, aam.n_levels, aam.downscale),
            "> Shape model:",
            "   > {} landmark points".format(
                instance.landmarks[group].lms.n_points),
            "   > {} components ({:.2f}% of variance)".format(
                lvl_shape_mod.n_components,
                lvl_shape_mod.variance_ratio() * 100),
            "> Appearance model:",
            "   > {}".format(tmp_feat),
            "   > Reference frame of length {} ({} x {}C, {} x {}C)".format(
                lvl_app_mod.n_features, tmplt_inst.n_true_pixels(), n_channels,
                tmplt_inst._str_shape, n_channels),
            "   > {} components ({:.2f}% of variance)".format(
                lvl_app_mod.n_components, lvl_app_mod.variance_ratio() * 100),
            "> Instance: min={:.3f} , max={:.3f}".format(
                instance.pixels.min(), instance.pixels.max())]
        info_wid.set_widget_state(n_lines=14, text_per_line=text_per_line)

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
        renderer_options, ['lines', 'markers', 'figure_one', 'image'],
        object_selection_dropdown_visible=False,
        render_function=render_function, style=renderer_style,
        tabs_style=renderer_tabs_style)
    info_wid = TextPrintWidget(n_lines=14, text_per_line=[''] * 14,
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
    model_parameters_wid = ipywidgets.Accordion(
        children=[shape_model_parameters_wid, appearance_model_parameters_wid])
    model_parameters_wid.set_title(0, 'Shape')
    model_parameters_wid.set_title(1, 'Appearance')
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
        level_wid = ipywidgets.RadioButtonsWidget(
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
    renderer_options_wid.options_widgets[2].render_axes_checkbox.value = False
