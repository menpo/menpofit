import numpy as np
from collections import OrderedDict

from menpo.visualize.widgets.options import (viewer_options,
                                             format_viewer_options,
                                             channel_options,
                                             format_channel_options,
                                             update_channel_options,
                                             landmark_options,
                                             format_landmark_options,
                                             info_print, format_info_print,
                                             animation_options,
                                             format_animation_options,
                                             save_figure_options,
                                             format_save_figure_options)
from menpo.visualize.widgets.tools import logo
from menpo.visualize.widgets.base import _visualize as _visualize_menpo
from menpo.visualize.widgets.base import _extract_groups_labels
from menpo.visualize.widgets.compatibility import add_class, remove_class
from menpo.visualize.viewmatplotlib import (MatplotlibImageViewer2d,
                                            sample_colours_from_colourmap,
                                            MatplotlibSubplots)

from .options import (model_parameters, format_model_parameters,
                      update_model_parameters, final_result_options,
                      format_final_result_options, update_final_result_options,
                      iterations_result_options,
                      format_iterations_result_options,
                      update_iterations_result_options)

# This glyph import is called frequently during visualisation, so we ensure
# that we only import it once
glyph = None


def visualize_shape_model(shape_models, n_parameters=5,
                          parameters_bounds=(-3.0, 3.0), figure_size=(10, 8),
                          mode='multiple'):
    r"""
    Allows the dynamic visualization of a multilevel shape model.

    Parameters
    -----------
    shape_models : `list` of :map:`PCAModel` or subclass
        The multilevel shape model to be displayed. Note that each level can
        have different number of components.
    n_parameters : `int` or `list` of `int` or None, optional
        The number of principal components to be used for the parameters
        sliders.
        If `int`, then the number of sliders per level is the minimum between
        `n_parameters` and the number of active components per level.
        If `list` of `int`, then a number of sliders is defined per level.
        If ``None``, all the active components per level will have a slider.
    parameters_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.
    figure_size : (`int`, `int`), optional
        The size of the plotted figures.
    mode : {``single``, ``multiple``}, optional
        If ``single``, only a single slider is constructed along with a drop
        down menu. If ``multiple``, a slider is constructed for each parameter.
    """
    import IPython.html.widgets as ipywidgets
    import IPython.display as ipydisplay
    import matplotlib.pyplot as plt
    from matplotlib import collections as mc

    # make sure that shape_models is a list even with one member
    if not isinstance(shape_models, list):
        shape_models = [shape_models]

    # find number of levels (i.e. number of shape models)
    n_levels = len(shape_models)

    # find maximum number of components per level
    max_n_params = [sp.n_active_components for sp in shape_models]

    # check given n_parameters
    # the returned n_parameters is a list of len n_levels
    n_parameters = _check_n_parameters(n_parameters, n_levels, max_n_params)

    # initial options dictionaries
    lines_options = {'render_lines': True,
                     'line_width': 1,
                     'line_colour': ['r'],
                     'line_style': '-'}
    markers_options = {'render_markers': True,
                       'marker_size': 20,
                       'marker_face_colour': ['r'],
                       'marker_edge_colour': ['k'],
                       'marker_style': 'o',
                       'marker_edge_width': 1}
    figure_options = {'x_scale': 1.,
                      'y_scale': 1.,
                      'render_axes': False,
                      'axes_font_name': 'sans-serif',
                      'axes_font_size': 10,
                      'axes_font_style': 'normal',
                      'axes_font_weight': 'normal',
                      'axes_x_limits': None,
                      'axes_y_limits': None}
    viewer_options_default = {'lines': lines_options,
                              'markers': markers_options,
                              'figure': figure_options}

    # Define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        ipydisplay.clear_output(wait=True)

        # get selected level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # compute weights
        parameters_values = model_parameters_wid.parameters_values
        weights = (parameters_values *
                   shape_models[level].eigenvalues[:len(parameters_values)] **
                   0.5)

        # compute the mean
        mean = shape_models[level].mean()

        tmp1 = viewer_options_wid.selected_values[0]['lines']
        tmp2 = viewer_options_wid.selected_values[0]['markers']
        tmp3 = viewer_options_wid.selected_values[0]['figure']
        new_figure_size = (tmp3['x_scale'] * figure_size[0],
                           tmp3['y_scale'] * figure_size[1])

        # compute and show instance
        if mode_wid.value == 1:
            # Deformation mode
            # compute instance
            instance = shape_models[level].instance(weights)

            # plot
            if mean_wid.value:
                mean.view(
                    figure_id=save_figure_wid.renderer[0].figure_id,
                    new_figure=False, image_view=axes_mode_wid.value == 1,
                    render_lines=tmp1['render_lines'],
                    line_colour='y',
                    line_style='solid', line_width=tmp1['line_width'],
                    render_markers=tmp2['render_markers'],
                    marker_style=tmp2['marker_style'],
                    marker_size=tmp2['marker_size'], marker_face_colour='y',
                    marker_edge_colour='y',
                    marker_edge_width=tmp2['marker_edge_width'],
                    render_axes=False, figure_size=None)

            renderer = instance.view(
                figure_id=save_figure_wid.renderer[0].figure_id,
                new_figure=False, image_view=axes_mode_wid.value==1,
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
                figure_size=new_figure_size,
                label=None)

            if mean_wid.value and axes_mode_wid.value == 1:
                plt.gca().invert_yaxis()

            # instance range
            instance_range = instance.range()
        else:
            # Vectors mode
            # compute instance
            instance_lower = shape_models[level].instance([-p for p in weights])
            instance_upper = shape_models[level].instance(weights)

            # plot
            renderer = mean.view(
                figure_id=save_figure_wid.renderer[0].figure_id,
                new_figure=False, image_view=axes_mode_wid.value == 1,
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

            # instance range
            instance_range = mean.range()

        plt.show()

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

        # update info text widget
        update_info(level, instance_range)

    # define function that updates info text
    def update_info(level, instance_range):
        lvl_sha_mod = shape_models[level]
        info_wid.children[1].children[0].value = "> Level: {} out of {}.".\
            format(level + 1, n_levels)
        info_wid.children[1].children[1].value = "> {} components in total.".\
            format(lvl_sha_mod.n_components)
        info_wid.children[1].children[2].value = "> {} active components.".\
            format(lvl_sha_mod.n_active_components)
        info_wid.children[1].children[3].value = "> {:.1f}% variance kept.".\
            format(lvl_sha_mod.variance_ratio() * 100)
        info_wid.children[1].children[4].value = "> Instance range: {:.1f} " \
                                                 "x {:.1f}.".\
            format(instance_range[0], instance_range[1])
        info_wid.children[1].children[5].value = "> {} landmark points, " \
                                                 "{} features.".\
            format(lvl_sha_mod.mean().n_points, lvl_sha_mod.n_features)

    # Plot eigenvalues function
    def plot_eigenvalues(name):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        ipydisplay.clear_output(wait=True)

        # get level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # get the current figure id and plot the eigenvalues
        new_figure_size = (viewer_options_wid.selected_values[0]['figure']['x_scale'] * 10,
                           viewer_options_wid.selected_values[0]['figure']['y_scale'] * 3)
        plt.subplot(121)
        shape_models[level].plot_eigenvalues_ratio(
            figure_id=save_figure_wid.renderer[0].figure_id)
        plt.subplot(122)
        renderer = shape_models[level].plot_eigenvalues_cumulative_ratio(
            figure_id=save_figure_wid.renderer[0].figure_id,
            figure_size=new_figure_size)
        plt.show()

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

    # create options widgets
    mode_dict = OrderedDict()
    mode_dict['Deformation'] = 1
    mode_dict['Vectors'] = 2
    mode_wid = ipywidgets.RadioButtonsWidget(options=mode_dict,
                                             description='Mode:', value=1)
    mode_wid.on_trait_change(plot_function, 'value')
    mean_wid = ipywidgets.CheckboxWidget(value=False,
                                         description='Show mean shape')
    mean_wid.on_trait_change(plot_function, 'value')

    # controls mean shape checkbox visibility
    def mean_visible(name, value):
        if value == 1:
            mean_wid.disabled = False
        else:
            mean_wid.disabled = True
            mean_wid.value = False
    mode_wid.on_trait_change(mean_visible, 'value')
    model_parameters_wid = model_parameters(n_parameters[0], plot_function,
                                            params_str='param ', mode=mode,
                                            params_bounds=parameters_bounds,
                                            toggle_show_default=True,
                                            toggle_show_visible=False,
                                            plot_eig_visible=True,
                                            plot_eig_function=plot_eigenvalues)

    # viewer options widget
    axes_mode_wid = ipywidgets.RadioButtonsWidget(
        options={'Image': 1, 'Point cloud': 2}, description='Axes mode:',
        value=2)
    axes_mode_wid.on_trait_change(plot_function, 'value')
    viewer_options_wid = viewer_options(viewer_options_default,
                                        ['lines', 'markers', 'figure_one'],
                                        objects_names=None,
                                        plot_function=plot_function,
                                        toggle_show_visible=False,
                                        toggle_show_default=True)
    viewer_options_all = ipywidgets.ContainerWidget(children=[axes_mode_wid,
                                                    viewer_options_wid])
    info_wid = info_print(n_bullets=6, toggle_show_default=True,
                          toggle_show_visible=False)

    # save figure widget
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = save_figure_options(initial_renderer,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)

    # define function that updates options' widgets state
    def update_widgets(name, value):
        update_model_parameters(model_parameters_wid, n_parameters[value],
                                plot_function, params_str='param ')

    # create final widget
    if n_levels > 1:
        radio_str = OrderedDict()
        for l in range(n_levels):
            if l == 0:
                radio_str["Level {} (low)".format(l)] = l
            elif l == n_levels - 1:
                radio_str["Level {} (high)".format(l)] = l
            else:
                radio_str["Level {}".format(l)] = l
        level_wid = ipywidgets.RadioButtonsWidget(options=radio_str,
                                                  description='Pyramid:',
                                                  value=0)
        level_wid.on_trait_change(update_widgets, 'value')
        level_wid.on_trait_change(plot_function, 'value')
        radio_children = [level_wid, mode_wid, mean_wid]
    else:
        radio_children = [mode_wid, mean_wid]
    radio_wids = ipywidgets.ContainerWidget(children=radio_children)
    tmp_wid = ipywidgets.ContainerWidget(children=[radio_wids,
                                                   model_parameters_wid])
    tab_wid = ipywidgets.TabWidget(children=[tmp_wid, viewer_options_all,
                                             info_wid, save_figure_wid])
    logo_wid = logo()
    wid = ipywidgets.ContainerWidget(children=[logo_wid, tab_wid])

    # display final widget
    ipydisplay.display(wid)

    # set final tab titles
    tab_titles = ['Shape parameters', 'Viewer options', 'Info', 'Save figure']
    for (k, tl) in enumerate(tab_titles):
        tab_wid.set_title(k, tl)

    # align widgets
    remove_class(tmp_wid, 'vbox')
    add_class(tmp_wid, 'hbox')
    format_model_parameters(model_parameters_wid, container_padding='6px',
                            container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold',
                            border_visible=True)
    format_viewer_options(viewer_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False,
                          suboptions_border_visible=True)
    format_info_print(info_wid, font_size_in_pt='10pt', container_padding='6px',
                      container_margin='6px',
                      container_border='1px solid black',
                      toggle_button_font_weight='bold', border_visible=False)
    format_save_figure_options(save_figure_wid, container_padding='6px',
                               container_margin='6px',
                               container_border='1px solid black',
                               toggle_button_font_weight='bold',
                               tab_top_margin='0cm', border_visible=False)

    # update widgets' state for image number 0
    update_widgets('', 0)

    # Reset value to enable initial visualization
    axes_mode_wid.value = 1


def visualize_appearance_model(appearance_models, n_parameters=5,
                               parameters_bounds=(-3.0, 3.0), figure_size=(10, 8),
                               mode='multiple'):
    r"""
    Allows the dynamic visualization of a multilevel appearance model.

    Parameters
    -----------
    appearance_models : `list` of :map:`PCAModel` or subclass
        The multilevel appearance model to be displayed. Note that each level can
        have different number of components.
    n_parameters : `int` or `list` of `int` or None, optional
        The number of principal components to be used for the parameters
        sliders.
        If `int`, then the number of sliders per level is the minimum between
        `n_parameters` and the number of active components per level.
        If `list` of `int`, then a number of sliders is defined per level.
        If ``None``, all the active components per level will have a slider.
    parameters_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.
    figure_size : (`int`, `int`), optional
        The size of the plotted figures.
    mode : {``single``, ``multiple``}, optional
        If ``single``, only a single slider is constructed along with a drop
        down menu. If ``multiple``, a slider is constructed for each parameter.
    """
    import IPython.html.widgets as ipywidgets
    import IPython.display as ipydisplay
    import matplotlib.pyplot as plt
    from menpo.image import MaskedImage

    # make sure that appearance_models is a list even with one member
    if not isinstance(appearance_models, list):
        appearance_models = [appearance_models]

    # find number of levels (i.e. number of appearance models)
    n_levels = len(appearance_models)

    # find maximum number of components per level
    max_n_params = [ap.n_active_components for ap in appearance_models]

    # check given n_parameters
    # the returned n_parameters is a list of len n_levels
    n_parameters = _check_n_parameters(n_parameters, n_levels, max_n_params)

    # find initial groups and labels that will be passed to the landmark options
    # widget creation
    mean_has_landmarks = appearance_models[0].mean().landmarks.n_groups != 0
    if mean_has_landmarks:
        all_groups_keys, all_labels_keys = _extract_groups_labels(
            appearance_models[0].mean())
    else:
        all_groups_keys = [' ']
        all_labels_keys = [[' ']]

    # get initial line colours for each available label
    if len(all_labels_keys[0]) == 1:
        line_colours = ['r']
    else:
        line_colours = sample_colours_from_colourmap(len(all_labels_keys[0]),
                                                     'jet')

    # initial options dictionaries
    channels_default = 0
    if appearance_models[0].mean().n_channels == 3:
        channels_default = None
    channels_options_default = \
        {'n_channels': appearance_models[0].mean().n_channels,
         'image_is_masked': isinstance(appearance_models[0].mean(),
                                       MaskedImage),
         'channels': channels_default,
         'glyph_enabled': False,
         'glyph_block_size': 3,
         'glyph_use_negative': False,
         'sum_enabled': False,
         'masked_enabled': isinstance(appearance_models[0].mean(), MaskedImage)}
    landmark_options_default = {'render_landmarks': mean_has_landmarks,
                                'group_keys': all_groups_keys,
                                'labels_keys': all_labels_keys,
                                'group': None,
                                'with_labels': None}
    lines_options = {'render_lines': True,
                     'line_width': 1,
                     'line_colour': line_colours,
                     'line_style': '-'}
    markers_options = {'render_markers': True,
                       'marker_size': 20,
                       'marker_face_colour': ['r'],
                       'marker_edge_colour': ['k'],
                       'marker_style': 'o',
                       'marker_edge_width': 1}
    figure_options = {'x_scale': 1.,
                      'y_scale': 1.,
                      'render_axes': True,
                      'axes_font_name': 'sans-serif',
                      'axes_font_size': 10,
                      'axes_font_style': 'normal',
                      'axes_font_weight': 'normal',
                      'axes_x_limits': None,
                      'axes_y_limits': None}
    image_options = {'interpolation': 'none',
                     'alpha': 1.}
    viewer_options_default = {'lines': lines_options,
                              'markers': markers_options,
                              'figure': figure_options,
                              'image': image_options}

    # Define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        ipydisplay.clear_output(wait=True)

        # get selected level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # compute weights and instance
        parameters_values = model_parameters_wid.parameters_values
        weights = (parameters_values *
                   appearance_models[level].eigenvalues[:len(parameters_values)] **
                   0.5)
        instance = appearance_models[level].instance(weights)

        # update info text widget
        update_info(instance, level,
                    landmark_options_wid.selected_values['group'])
        n_labels = len(landmark_options_wid.selected_values['with_labels'])

        # compute the mean
        tmp1 = viewer_options_wid.selected_values[0]['lines']
        tmp2 = viewer_options_wid.selected_values[0]['markers']
        tmp3 = viewer_options_wid.selected_values[0]['figure']
        tmp4 = viewer_options_wid.selected_values[0]['image']
        new_figure_size = (tmp3['x_scale'] * figure_size[0],
                           tmp3['y_scale'] * figure_size[1])
        renderer = _visualize_menpo(
            instance, save_figure_wid.renderer[0],
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
            tmp1['line_colour'][:n_labels], tmp2['render_markers'],
            tmp2['marker_style'], tmp2['marker_size'], tmp2['marker_edge_width'],
            tmp2['marker_edge_colour'], tmp2['marker_face_colour'],
            False, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None,
            False, None, None, new_figure_size, tmp3['render_axes'],
            tmp3['axes_font_name'], tmp3['axes_font_size'],
            tmp3['axes_font_style'], tmp3['axes_x_limits'],
            tmp3['axes_y_limits'], tmp3['axes_font_weight'],
            tmp4['interpolation'], tmp4['alpha'])

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

    # define function that updates info text
    def update_info(image, level, group):
        lvl_app_mod = appearance_models[level]
        info_wid.children[1].children[0].value = "> Level: {} out of {}.".\
            format(level + 1, n_levels)
        info_wid.children[1].children[1].value = "> {} components in total.".\
            format(lvl_app_mod.n_components)
        info_wid.children[1].children[2].value = "> {} active components.".\
            format(lvl_app_mod.n_active_components)
        info_wid.children[1].children[3].value = "> {:.1f}% variance kept.".\
            format(lvl_app_mod.variance_ratio() * 100)
        info_wid.children[1].children[4].value = "> Reference shape of size " \
                                                 "{} with {} channel{}.".\
            format(image._str_shape,
                   image.n_channels, 's' * (image.n_channels > 1))
        info_wid.children[1].children[5].value = "> {} features.".\
            format(lvl_app_mod.n_features)
        info_wid.children[1].children[6].value = "> {} landmark points.".\
            format(image.landmarks[group].lms.n_points)
        info_wid.children[1].children[7].value = "> Instance: min={:.3f}, " \
                                                 "max={:.3f}".\
            format(image.pixels.min(), image.pixels.max())

    # Plot eigenvalues function
    def plot_eigenvalues(name):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        ipydisplay.clear_output(wait=True)

        # get level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # get the current figure id and plot the eigenvalues
        new_figure_size = (viewer_options_wid.selected_values[0]['figure']['x_scale'] * 10,
                           viewer_options_wid.selected_values[0]['figure']['y_scale'] * 3)
        plt.subplot(121)
        appearance_models[level].plot_eigenvalues_ratio(
            figure_id=save_figure_wid.renderer[0].figure_id)
        plt.subplot(122)
        renderer = appearance_models[level].plot_eigenvalues_cumulative_ratio(
            figure_id=save_figure_wid.renderer[0].figure_id,
            figure_size=new_figure_size)
        plt.show()

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

    # create parameters, channels nad landmarks options widgets
    model_parameters_wid = model_parameters(n_parameters[0], plot_function,
                                            params_str='param ', mode=mode,
                                            params_bounds=parameters_bounds,
                                            toggle_show_default=True,
                                            toggle_show_visible=False,
                                            plot_eig_visible=True,
                                            plot_eig_function=plot_eigenvalues)
    channel_options_wid = channel_options(channels_options_default,
                                          plot_function=plot_function,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)
    landmark_options_wid = landmark_options(landmark_options_default,
                                            plot_function=plot_function,
                                            toggle_show_default=True,
                                            toggle_show_visible=False)
    # if the mean doesn't have landmarks, then landmarks checkbox should be
    # disabled
    landmark_options_wid.children[1].disabled = not mean_has_landmarks

    # viewer options widget
    viewer_options_wid = viewer_options(viewer_options_default,
                                        ['lines', 'markers', 'figure_one',
                                         'image'],
                                        objects_names=None,
                                        plot_function=plot_function,
                                        toggle_show_visible=False,
                                        toggle_show_default=True)
    info_wid = info_print(n_bullets=8, toggle_show_default=True,
                          toggle_show_visible=False)

    # save figure widget
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = save_figure_options(initial_renderer,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)

    # define function that updates options' widgets state
    def update_widgets(name, value):
        # update model parameters
        update_model_parameters(model_parameters_wid, n_parameters[value],
                                plot_function, params_str='param ')

        # update channel options
        update_channel_options(channel_options_wid,
                               appearance_models[value].mean().n_channels,
                               isinstance(appearance_models[0].mean(),
                                          MaskedImage))

    # create final widget
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
        level_wid = ipywidgets.RadioButtonsWidget(options=radio_str,
                                                  description='Pyramid:',
                                                  value=0)
        level_wid.on_trait_change(update_widgets, 'value')
        level_wid.on_trait_change(plot_function, 'value')
        tmp_children.insert(0, level_wid)
    tmp_wid = ipywidgets.ContainerWidget(children=tmp_children)
    tab_wid = ipywidgets.TabWidget(children=[tmp_wid, channel_options_wid,
                                             landmark_options_wid,
                                             viewer_options_wid,
                                             info_wid, save_figure_wid])
    logo_wid = logo()
    wid = ipywidgets.ContainerWidget(children=[logo_wid, tab_wid])

    # display final widget
    ipydisplay.display(wid)

    # set final tab titles
    tab_titles = ['Appearance parameters', 'Channels options',
                  'Landmarks options', 'Viewer options', 'Info', 'Save figure']
    for (k, tl) in enumerate(tab_titles):
        tab_wid.set_title(k, tl)

    # align widgets
    remove_class(tmp_wid, 'vbox')
    add_class(tmp_wid, 'hbox')
    format_model_parameters(model_parameters_wid, container_padding='6px',
                            container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold',
                            border_visible=True)
    format_channel_options(channel_options_wid, container_padding='6px',
                           container_margin='6px',
                           container_border='1px solid black',
                           toggle_button_font_weight='bold',
                           border_visible=False)
    format_landmark_options(landmark_options_wid, container_padding='6px',
                            container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold',
                            border_visible=False)
    format_viewer_options(viewer_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False,
                          suboptions_border_visible=True)
    format_info_print(info_wid, font_size_in_pt='10pt', container_padding='6px',
                      container_margin='6px',
                      container_border='1px solid black',
                      toggle_button_font_weight='bold', border_visible=False)
    format_save_figure_options(save_figure_wid, container_padding='6px',
                               container_margin='6px',
                               container_border='1px solid black',
                               toggle_button_font_weight='bold',
                               tab_top_margin='0cm', border_visible=False)

    # update widgets' state for image number 0
    update_widgets('', 0)

    # Reset value to enable initial visualization
    viewer_options_wid.children[1].children[1].children[2].children[2].value = \
        False


def visualize_aam(aam, n_shape_parameters=5, n_appearance_parameters=5,
                  parameters_bounds=(-3.0, 3.0), figure_size=(10, 8),
                  mode='multiple'):
    r"""
    Allows the dynamic visualization of a multilevel AAM.

    Parameters
    -----------
    aam : :map:`AAM` or subclass
        The multilevel AAM to be displayed. Note that each level can have
        different attributes, e.g. number of active components, feature type,
        number of channels.
    n_shape_parameters : `int` or `list` of `int` or None, optional
        The number of shape components to be used for the parameters
        sliders.
        If `int`, then the number of sliders per level is the minimum between
        `n_parameters` and the number of active components per level.
        If `list` of `int`, then a number of sliders is defined per level.
        If ``None``, all the active components per level will have a slider.
    n_appearance_parameters : `int` or `list` of `int` or None, optional
        The number of appearance components to be used for the parameters
        sliders.
        If `int`, then the number of sliders per level is the minimum between
        `n_parameters` and the number of active components per level.
        If `list` of `int`, then a number of sliders is defined per level.
        If ``None``, all the active components per level will have a slider.
    parameters_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.
    figure_size : (`int`, `int`), optional
        The size of the plotted figures.
    mode : {``single``, ``multiple``}, optional
        If ``single``, only a single slider is constructed along with a drop
        down menu. If ``multiple``, a slider is constructed for each parameter.
    """
    import IPython.html.widgets as ipywidgets
    import IPython.display as ipydisplay
    import matplotlib.pyplot as plt
    from menpo.image import MaskedImage

    # find number of levels
    n_levels = aam.n_levels

    # find maximum number of components per level
    max_n_shape = [sp.n_active_components for sp in aam.shape_models]
    max_n_appearance = [ap.n_active_components for ap in aam.appearance_models]

    # check given n_parameters
    # the returned n_parameters is a list of len n_levels
    n_shape_parameters = _check_n_parameters(n_shape_parameters, n_levels,
                                             max_n_shape)
    n_appearance_parameters = _check_n_parameters(n_appearance_parameters,
                                                  n_levels, max_n_appearance)

    # find initial groups and labels that will be passed to the landmark options
    # widget creation
    mean_has_landmarks = aam.appearance_models[0].mean().landmarks.n_groups != 0
    if mean_has_landmarks:
        all_groups_keys, all_labels_keys = _extract_groups_labels(
            aam.appearance_models[0].mean())
    else:
        all_groups_keys = [' ']
        all_labels_keys = [[' ']]

    # get initial line colours for each available label
    if len(all_labels_keys[0]) == 1:
        line_colours = ['r']
    else:
        line_colours = sample_colours_from_colourmap(len(all_labels_keys[0]),
                                                     'jet')

    # initial options dictionaries
    channels_default = 0
    if aam.appearance_models[0].mean().n_channels == 3:
        channels_default = None
    channels_options_default = \
        {'n_channels': aam.appearance_models[0].mean().n_channels,
         'image_is_masked': isinstance(aam.appearance_models[0].mean(),
                                       MaskedImage),
         'channels': channels_default,
         'glyph_enabled': False,
         'glyph_block_size': 3,
         'glyph_use_negative': False,
         'sum_enabled': False,
         'masked_enabled': isinstance(aam.appearance_models[0].mean(),
                                      MaskedImage)}
    landmark_options_default = {'render_landmarks': mean_has_landmarks,
                                'group_keys': all_groups_keys,
                                'labels_keys': all_labels_keys,
                                'group': None,
                                'with_labels': None}
    lines_options = {'render_lines': True,
                     'line_width': 1,
                     'line_colour': line_colours,
                     'line_style': '-'}
    markers_options = {'render_markers': True,
                       'marker_size': 20,
                       'marker_face_colour': ['r'],
                       'marker_edge_colour': ['k'],
                       'marker_style': 'o',
                       'marker_edge_width': 1}
    figure_options = {'x_scale': 1.,
                      'y_scale': 1.,
                      'render_axes': True,
                      'axes_font_name': 'sans-serif',
                      'axes_font_size': 10,
                      'axes_font_style': 'normal',
                      'axes_font_weight': 'normal',
                      'axes_x_limits': None,
                      'axes_y_limits': None}
    image_options = {'interpolation': 'none',
                     'alpha': 1.0}
    viewer_options_default = {'lines': lines_options,
                              'markers': markers_options,
                              'figure': figure_options,
                              'image': image_options}

    # Define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        ipydisplay.clear_output(wait=True)

        # get selected level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # compute weights and instance
        shape_weights = shape_model_parameters_wid.parameters_values
        appearance_weights = appearance_model_parameters_wid.parameters_values
        instance = aam.instance(level=level, shape_weights=shape_weights,
                                appearance_weights=appearance_weights)

        # update info text widget
        update_info(aam, instance, level,
                    landmark_options_wid.selected_values['group'])
        n_labels = len(landmark_options_wid.selected_values['with_labels'])

        # plot
        tmp1 = viewer_options_wid.selected_values[0]['lines']
        tmp2 = viewer_options_wid.selected_values[0]['markers']
        tmp3 = viewer_options_wid.selected_values[0]['figure']
        tmp4 = viewer_options_wid.selected_values[0]['image']
        new_figure_size = (tmp3['x_scale'] * figure_size[0],
                           tmp3['y_scale'] * figure_size[1])
        renderer = _visualize_menpo(
            instance, save_figure_wid.renderer[0],
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
            tmp1['line_colour'][:n_labels], tmp2['render_markers'],
            tmp2['marker_style'], tmp2['marker_size'], tmp2['marker_edge_width'],
            tmp2['marker_edge_colour'], tmp2['marker_face_colour'],
            False, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None,
            False, None, None, new_figure_size, tmp3['render_axes'],
            tmp3['axes_font_name'], tmp3['axes_font_size'],
            tmp3['axes_font_style'], tmp3['axes_x_limits'],
            tmp3['axes_y_limits'], tmp3['axes_font_weight'],
            tmp4['interpolation'], tmp4['alpha'])

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

    # define function that updates info text
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
        tmp_feat = 'Feature is {} with {} channel{}.'.format(
            name_of_callable(feat), n_channels, 's' * (n_channels > 1))

        # create info str
        if n_levels == 1:
            tmp_shape_models = ''
            tmp_pyramid = ''
        else:  # n_levels > 1
            # shape models info
            if aam.scaled_shape_models:
                tmp_shape_models = "Each level has a scaled shape model " \
                                   "(reference frame)."
            else:
                tmp_shape_models = "Shape models (reference frames) are " \
                                   "not scaled."
            # pyramid info
            if aam.pyramid_on_features:
                tmp_pyramid = "Pyramid was applied on feature space."
            else:
                tmp_pyramid = "Features were extracted at each pyramid level."

        # update info widgets
        info_wid.children[1].children[0].value = "> {} training images.".\
            format(aam.n_training_images)
        info_wid.children[1].children[1].value = "> Warp using {} transform.".\
            format(aam.transform.__name__)
        info_wid.children[1].children[2].value = "> Level {}/{}  " \
                                                 "(downscale={:.1f}).".\
            format(level + 1, aam.n_levels, aam.downscale)
        info_wid.children[1].children[3].value = "> {}".format(tmp_shape_models)
        info_wid.children[1].children[4].value = "> {}".format(tmp_pyramid)
        info_wid.children[1].children[5].value = "> {}".format(tmp_feat)
        info_wid.children[1].children[6].value = "> Reference frame of " \
                                                 "length {} ({} x {}C, {} x " \
                                                 "{}C).".\
            format(lvl_app_mod.n_features, tmplt_inst.n_true_pixels(),
                   n_channels, tmplt_inst._str_shape, n_channels)
        info_wid.children[1].children[7].value = "> {} shape components " \
                                                 "({:.2f}% of variance).".\
            format(lvl_shape_mod.n_components,
                   lvl_shape_mod.variance_ratio() * 100)
        info_wid.children[1].children[8].value = "> {} appearance components " \
                                                 "({:.2f}% of variance).".\
            format(lvl_app_mod.n_components, lvl_app_mod.variance_ratio() * 100)
        info_wid.children[1].children[9].value = "> {} landmark points.".\
            format(instance.landmarks[group].lms.n_points)
        info_wid.children[1].children[10].value = "> Instance: min={:.3f} , " \
                                                  "max={:.3f}.".\
            format(instance.pixels.min(), instance.pixels.max())

    # Plot shape eigenvalues function
    def plot_shape_eigenvalues(name):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        ipydisplay.clear_output(wait=True)

        # get level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # get the current figure id and plot the eigenvalues
        new_figure_size = (viewer_options_wid.selected_values[0]['figure']['x_scale'] * 10,
                           viewer_options_wid.selected_values[0]['figure']['y_scale'] * 3)
        plt.subplot(121)
        aam.shape_models[level].plot_eigenvalues_ratio(
            figure_id=save_figure_wid.renderer[0].figure_id)
        plt.subplot(122)
        renderer = aam.shape_models[level].plot_eigenvalues_cumulative_ratio(
            figure_id=save_figure_wid.renderer[0].figure_id,
            figure_size=new_figure_size)
        plt.show()

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

    # Plot appearance eigenvalues function
    def plot_appearance_eigenvalues(name):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        ipydisplay.clear_output(wait=True)

        # get level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # get the current figure id and plot the eigenvalues
        new_figure_size = (viewer_options_wid.selected_values[0]['figure']['x_scale'] * 10,
                           viewer_options_wid.selected_values[0]['figure']['y_scale'] * 3)
        plt.subplot(121)
        aam.appearance_models[level].plot_eigenvalues_ratio(
            figure_id=save_figure_wid.renderer[0].figure_id)
        plt.subplot(122)
        renderer = aam.appearance_models[level].plot_eigenvalues_cumulative_ratio(
            figure_id=save_figure_wid.renderer[0].figure_id,
            figure_size=new_figure_size)
        plt.show()

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

    # create parameters, channels nad landmarks options widgets
    shape_model_parameters_wid = model_parameters(
        n_shape_parameters[0], plot_function, params_str='param ', mode=mode,
        params_bounds=parameters_bounds, toggle_show_default=True,
        toggle_show_visible=False, toggle_show_name='Shape Parameters',
        plot_eig_visible=True, plot_eig_function=plot_shape_eigenvalues)
    appearance_model_parameters_wid = model_parameters(
        n_appearance_parameters[0], plot_function, params_str='param ',
        mode=mode, params_bounds=parameters_bounds, toggle_show_default=True,
        toggle_show_visible=False, toggle_show_name='Appearance Parameters',
        plot_eig_visible=True, plot_eig_function=plot_appearance_eigenvalues)
    channel_options_wid = channel_options(channels_options_default,
                                          plot_function=plot_function,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)
    landmark_options_wid = landmark_options(landmark_options_default,
                                            plot_function=plot_function,
                                            toggle_show_default=True,
                                            toggle_show_visible=False)
    # if the mean doesn't have landmarks, then landmarks checkbox should be
    # disabled
    landmark_options_wid.children[1].disabled = not mean_has_landmarks

    # viewer options widget
    viewer_options_wid = viewer_options(viewer_options_default,
                                        ['lines', 'markers', 'figure_one',
                                         'image'],
                                        objects_names=None,
                                        plot_function=plot_function,
                                        toggle_show_visible=False,
                                        toggle_show_default=True)
    info_wid = info_print(n_bullets=11, toggle_show_default=True,
                          toggle_show_visible=False)

    # save figure widget
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = save_figure_options(initial_renderer,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)

    # define function that updates options' widgets state
    def update_widgets(name, value):
        # update shape model parameters
        update_model_parameters(shape_model_parameters_wid,
                                n_shape_parameters[value],
                                plot_function, params_str='param ')
        # update appearance model parameters
        update_model_parameters(appearance_model_parameters_wid,
                                n_appearance_parameters[value],
                                plot_function, params_str='param ')

        # update channel options
        update_channel_options(channel_options_wid,
                               aam.appearance_models[value].mean().n_channels,
                               isinstance(aam.appearance_models[0].mean(),
                                          MaskedImage))

    # create final widget
    model_parameters_wid = ipywidgets.AccordionWidget(
        children=[shape_model_parameters_wid, appearance_model_parameters_wid])
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
        level_wid = ipywidgets.RadioButtonsWidget(options=radio_str,
                                                  description='Pyramid:',
                                                  value=0)
        level_wid.on_trait_change(update_widgets, 'value')
        level_wid.on_trait_change(plot_function, 'value')
        tmp_children.insert(0, level_wid)
    tmp_wid = ipywidgets.ContainerWidget(children=tmp_children)
    tab_wid = ipywidgets.TabWidget(children=[tmp_wid, channel_options_wid,
                                             landmark_options_wid,
                                             viewer_options_wid,
                                             info_wid, save_figure_wid])
    logo_wid = logo()
    wid = ipywidgets.ContainerWidget(children=[logo_wid, tab_wid])

    # display final widget
    ipydisplay.display(wid)

    # set final tab titles
    tab_titles = ['AAM parameters', 'Channels options',
                  'Landmarks options', 'Viewer options', 'Info', 'Save figure']
    for (k, tl) in enumerate(tab_titles):
        tab_wid.set_title(k, tl)
    tab_titles = ['Shape parameters', 'Appearance parameters']
    for (k, tl) in enumerate(tab_titles):
        model_parameters_wid.set_title(k, tl)

    # align widgets
    remove_class(tmp_wid, 'vbox')
    add_class(tmp_wid, 'hbox')
    format_model_parameters(shape_model_parameters_wid,
                            container_padding='6px', container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold',
                            border_visible=False)
    format_model_parameters(appearance_model_parameters_wid,
                            container_padding='6px', container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold',
                            border_visible=False)
    format_channel_options(channel_options_wid, container_padding='6px',
                           container_margin='6px',
                           container_border='1px solid black',
                           toggle_button_font_weight='bold',
                           border_visible=False)
    format_landmark_options(landmark_options_wid, container_padding='6px',
                            container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold',
                            border_visible=False)
    format_viewer_options(viewer_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False,
                          suboptions_border_visible=True)
    format_info_print(info_wid, font_size_in_pt='10pt', container_padding='6px',
                      container_margin='6px',
                      container_border='1px solid black',
                      toggle_button_font_weight='bold', border_visible=False)
    format_save_figure_options(save_figure_wid, container_padding='6px',
                               container_margin='6px',
                               container_border='1px solid black',
                               toggle_button_font_weight='bold',
                               tab_top_margin='0cm', border_visible=False)

    # update widgets' state for image number 0
    update_widgets('', 0)

    # Reset value to enable initial visualization
    viewer_options_wid.children[1].children[1].children[2].children[2].value = \
        False


def visualize_atm(atm, n_shape_parameters=5, parameters_bounds=(-3.0, 3.0),
                  figure_size=(10, 8), mode='multiple'):
    r"""
    Allows the dynamic visualization of a multilevel ATM.

    Parameters
    -----------
    atm : :map:`ATM` or subclass
        The multilevel ATM to be displayed. Note that each level can have
        different attributes, e.g. number of active components, feature type,
        number of channels.
    n_shape_parameters : `int` or `list` of `int` or None, optional
        The number of shape components to be used for the parameters
        sliders.
        If `int`, then the number of sliders per level is the minimum between
        `n_parameters` and the number of active components per level.
        If `list` of `int`, then a number of sliders is defined per level.
        If ``None``, all the active components per level will have a slider.
    parameters_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.
    figure_size : (`int`, `int`), optional
        The size of the plotted figures.
    mode : {``single``, ``multiple``}, optional
        If ``single``, only a single slider is constructed along with a drop
        down menu. If ``multiple``, a slider is constructed for each parameter.
    """
    import IPython.html.widgets as ipywidgets
    import IPython.display as ipydisplay
    import matplotlib.pyplot as plt
    from menpo.image import MaskedImage

    # find number of levels
    n_levels = atm.n_levels

    # find maximum number of components per level
    max_n_shape = [sp.n_active_components for sp in atm.shape_models]

    # check given n_parameters
    # the returned n_parameters is a list of len n_levels
    n_shape_parameters = _check_n_parameters(n_shape_parameters, n_levels,
                                             max_n_shape)

    # find initial groups and labels that will be passed to the landmark options
    # widget creation
    template_has_landmarks = atm.warped_templates[0].landmarks.n_groups != 0
    if template_has_landmarks:
        all_groups_keys, all_labels_keys = _extract_groups_labels(
            atm.warped_templates[0])
    else:
        all_groups_keys = [' ']
        all_labels_keys = [[' ']]

    # get initial line colours for each available label
    if len(all_labels_keys[0]) == 1:
        line_colours = ['r']
    else:
        line_colours = sample_colours_from_colourmap(len(all_labels_keys[0]),
                                                     'jet')

    # initial options dictionaries
    channels_default = 0
    if atm.warped_templates[0].n_channels == 3:
        channels_default = None
    channels_options_default = \
        {'n_channels': atm.warped_templates[0].n_channels,
         'image_is_masked': isinstance(atm.warped_templates[0],
                                       MaskedImage),
         'channels': channels_default,
         'glyph_enabled': False,
         'glyph_block_size': 3,
         'glyph_use_negative': False,
         'sum_enabled': False,
         'masked_enabled': isinstance(atm.warped_templates[0],
                                      MaskedImage)}
    landmark_options_default = {'render_landmarks': template_has_landmarks,
                                'group_keys': all_groups_keys,
                                'labels_keys': all_labels_keys,
                                'group': None,
                                'with_labels': None}
    lines_options = {'render_lines': True,
                     'line_width': 1,
                     'line_colour': line_colours,
                     'line_style': '-'}
    markers_options = {'render_markers': True,
                       'marker_size': 20,
                       'marker_face_colour': ['r'],
                       'marker_edge_colour': ['k'],
                       'marker_style': 'o',
                       'marker_edge_width': 1}
    figure_options = {'x_scale': 1.,
                      'y_scale': 1.,
                      'render_axes': True,
                      'axes_font_name': 'sans-serif',
                      'axes_font_size': 10,
                      'axes_font_style': 'normal',
                      'axes_font_weight': 'normal',
                      'axes_x_limits': None,
                      'axes_y_limits': None}
    image_options = {'interpolation': 'none',
                     'alpha': 1.0}
    viewer_options_default = {'lines': lines_options,
                              'markers': markers_options,
                              'figure': figure_options,
                              'image': image_options}

    # Define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        ipydisplay.clear_output(wait=True)

        # get selected level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # compute weights and instance
        shape_weights = shape_model_parameters_wid.parameters_values
        instance = atm.instance(level=level, shape_weights=shape_weights)

        # update info text widget
        update_info(atm, instance, level,
                    landmark_options_wid.selected_values['group'])
        n_labels = len(landmark_options_wid.selected_values['with_labels'])

        # plot
        tmp1 = viewer_options_wid.selected_values[0]['lines']
        tmp2 = viewer_options_wid.selected_values[0]['markers']
        tmp3 = viewer_options_wid.selected_values[0]['figure']
        tmp4 = viewer_options_wid.selected_values[0]['image']
        new_figure_size = (tmp3['x_scale'] * figure_size[0],
                           tmp3['y_scale'] * figure_size[1])
        renderer = _visualize_menpo(
            instance, save_figure_wid.renderer[0],
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
            tmp1['line_colour'][:n_labels], tmp2['render_markers'],
            tmp2['marker_style'], tmp2['marker_size'], tmp2['marker_edge_width'],
            tmp2['marker_edge_colour'], tmp2['marker_face_colour'],
            False, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None,
            False, None, None, new_figure_size, tmp3['render_axes'],
            tmp3['axes_font_name'], tmp3['axes_font_size'],
            tmp3['axes_font_style'], tmp3['axes_x_limits'],
            tmp3['axes_y_limits'], tmp3['axes_font_weight'],
            tmp4['interpolation'], tmp4['alpha'])

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

    # define function that updates info text
    def update_info(atm, instance, level, group):
        # features info
        from menpofit.base import name_of_callable

        lvl_shape_mod = atm.shape_models[level]
        tmplt_inst = atm.warped_templates[level]
        n_channels = tmplt_inst.n_channels
        feat = (atm.features if atm.pyramid_on_features
                else atm.features[level])

        # Feature string
        tmp_feat = 'Feature is {} with {} channel{}.'.format(
            name_of_callable(feat), n_channels, 's' * (n_channels > 1))

        # create info str
        if n_levels == 1:
            tmp_shape_models = ''
            tmp_pyramid = ''
        else:  # n_levels > 1
            # shape models info
            if atm.scaled_shape_models:
                tmp_shape_models = "Each level has a scaled shape model " \
                                   "(reference frame)."
            else:
                tmp_shape_models = "Shape models (reference frames) are " \
                                   "not scaled."
            # pyramid info
            if atm.pyramid_on_features:
                tmp_pyramid = "Pyramid was applied on feature space."
            else:
                tmp_pyramid = "Features were extracted at each pyramid level."

        # update info widgets
        info_wid.children[1].children[0].value = "> {} training shapes.".\
            format(atm.n_training_shapes)
        info_wid.children[1].children[1].value = "> Warp using {} transform.".\
            format(atm.transform.__name__)
        info_wid.children[1].children[2].value = "> Level {}/{}  " \
                                                 "(downscale={:.1f}).".\
            format(level + 1, atm.n_levels, atm.downscale)
        info_wid.children[1].children[3].value = "> {}".format(tmp_shape_models)
        info_wid.children[1].children[4].value = "> {}".format(tmp_pyramid)
        info_wid.children[1].children[5].value = "> {}".format(tmp_feat)
        info_wid.children[1].children[6].value = "> Reference frame of " \
                                                 "length {} ({} x {}C, {} x " \
                                                 "{}C).".\
            format(tmplt_inst.n_true_pixels() * n_channels,
                   tmplt_inst.n_true_pixels(),
                   n_channels, tmplt_inst._str_shape, n_channels)
        info_wid.children[1].children[7].value = "> {} shape components " \
                                                 "({:.2f}% of variance).".\
            format(lvl_shape_mod.n_components,
                   lvl_shape_mod.variance_ratio() * 100)
        info_wid.children[1].children[8].value = "> {} landmark points.".\
            format(instance.landmarks[group].lms.n_points)
        info_wid.children[1].children[9].value = "> Instance: min={:.3f} , " \
                                                  "max={:.3f}.".\
            format(instance.pixels.min(), instance.pixels.max())

    # Plot shape eigenvalues function
    def plot_shape_eigenvalues(name):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        ipydisplay.clear_output(wait=True)

        # get level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # get the current figure id and plot the eigenvalues
        new_figure_size = (viewer_options_wid.selected_values[0]['figure']['x_scale'] * 10,
                           viewer_options_wid.selected_values[0]['figure']['y_scale'] * 3)
        plt.subplot(121)
        atm.shape_models[level].plot_eigenvalues_ratio(
            figure_id=save_figure_wid.renderer[0].figure_id)
        plt.subplot(122)
        renderer = atm.shape_models[level].plot_eigenvalues_cumulative_ratio(
            figure_id=save_figure_wid.renderer[0].figure_id,
            figure_size=new_figure_size)
        plt.show()

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

    # create parameters, channels nad landmarks options widgets
    shape_model_parameters_wid = model_parameters(
        n_shape_parameters[0], plot_function, params_str='param ', mode=mode,
        params_bounds=parameters_bounds, toggle_show_default=True,
        toggle_show_visible=False, toggle_show_name='Shape Parameters',
        plot_eig_visible=True, plot_eig_function=plot_shape_eigenvalues)
    channel_options_wid = channel_options(channels_options_default,
                                          plot_function=plot_function,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)
    landmark_options_wid = landmark_options(landmark_options_default,
                                            plot_function=plot_function,
                                            toggle_show_default=True,
                                            toggle_show_visible=False)
    # if the mean doesn't have landmarks, then landmarks checkbox should be
    # disabled
    landmark_options_wid.children[1].disabled = not template_has_landmarks

    # viewer options widget
    viewer_options_wid = viewer_options(viewer_options_default,
                                        ['lines', 'markers', 'figure_one',
                                         'image'],
                                        objects_names=None,
                                        plot_function=plot_function,
                                        toggle_show_visible=False,
                                        toggle_show_default=True)
    info_wid = info_print(n_bullets=10, toggle_show_default=True,
                          toggle_show_visible=False)

    # save figure widget
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = save_figure_options(initial_renderer,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)

    # define function that updates options' widgets state
    def update_widgets(name, value):
        # update shape model parameters
        update_model_parameters(shape_model_parameters_wid,
                                n_shape_parameters[value],
                                plot_function, params_str='param ')

        # update channel options
        update_channel_options(channel_options_wid,
                               atm.warped_templates[value].n_channels,
                               isinstance(atm.warped_templates[value],
                                          MaskedImage))

    # create final widget
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
        level_wid = ipywidgets.RadioButtonsWidget(options=radio_str,
                                                  description='Pyramid:',
                                                  value=0)
        level_wid.on_trait_change(update_widgets, 'value')
        level_wid.on_trait_change(plot_function, 'value')
        tmp_children.insert(0, level_wid)
    tmp_wid = ipywidgets.ContainerWidget(children=tmp_children)
    tab_wid = ipywidgets.TabWidget(children=[tmp_wid, channel_options_wid,
                                             landmark_options_wid,
                                             viewer_options_wid,
                                             info_wid, save_figure_wid])
    logo_wid = logo()
    wid = ipywidgets.ContainerWidget(children=[logo_wid, tab_wid])

    # display final widget
    ipydisplay.display(wid)

    # set final tab titles
    tab_titles = ['Shape parameters', 'Channels options',
                  'Landmarks options', 'Viewer options', 'Info', 'Save figure']
    for (k, tl) in enumerate(tab_titles):
        tab_wid.set_title(k, tl)

    # align widgets
    remove_class(tmp_wid, 'vbox')
    add_class(tmp_wid, 'hbox')
    format_model_parameters(shape_model_parameters_wid,
                            container_padding='6px', container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold',
                            border_visible=False)
    format_channel_options(channel_options_wid, container_padding='6px',
                           container_margin='6px',
                           container_border='1px solid black',
                           toggle_button_font_weight='bold',
                           border_visible=False)
    format_landmark_options(landmark_options_wid, container_padding='6px',
                            container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold',
                            border_visible=False)
    format_viewer_options(viewer_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False,
                          suboptions_border_visible=True)
    format_info_print(info_wid, font_size_in_pt='10pt', container_padding='6px',
                      container_margin='6px',
                      container_border='1px solid black',
                      toggle_button_font_weight='bold', border_visible=False)
    format_save_figure_options(save_figure_wid, container_padding='6px',
                               container_margin='6px',
                               container_border='1px solid black',
                               toggle_button_font_weight='bold',
                               tab_top_margin='0cm', border_visible=False)

    # update widgets' state for image number 0
    update_widgets('', 0)

    # Reset value to enable initial visualization
    viewer_options_wid.children[1].children[1].children[2].children[2].value = \
        False


def visualize_fitting_results(fitting_results, figure_size=(10, 8),
                              browser_style='buttons'):
    r"""
    Widget that allows browsing through a list of fitting results.

    Parameters
    -----------
    fitting_results : `list` of :map:`FittingResult` or subclass
        The `list` of fitting results to be displayed. Note that the fitting
        results can have different attributes between them, i.e. different
        number of iterations, number of channels etc.
    figure_size : (`int`, `int`), optional
        The initial size of the plotted figures.
    browser_style : {``buttons``, ``slider``}, optional
        It defines whether the selector of the fitting results will have the form of
        plus/minus buttons or a slider.
    """
    import IPython.html.widgets as ipywidgets
    import IPython.display as ipydisplay
    import matplotlib.pyplot as plt
    from menpo.image import MaskedImage
    from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
    print 'Initializing...'

    # make sure that fitting_results is a list even with one fitting_result
    if not isinstance(fitting_results, list):
        fitting_results = [fitting_results]

    # check if all fitting_results have gt_shape in order to show the ced button
    show_ced = all(not f.gt_shape is None for f in fitting_results)

    # find number of fitting_results
    n_fitting_results = len(fitting_results)

    # create dictionaries
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

    # initial options dictionaries
    channels_default = 0
    if fitting_results[0].fitted_image.n_channels == 3:
        channels_default = None
    channels_options_default = \
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
    final_result_options_default = {'all_groups': all_groups_keys,
                                    'render_image': True,
                                    'selected_groups': ['final'],
                                    'subplots_enabled': True}
    iterations_result_options_default = \
        {'n_iters': fitting_results[0].n_iters,
         'image_has_gt_shape': not fitting_results[0].gt_shape is None,
         'n_points': fitting_results[0].fitted_image.landmarks['final'].lms.n_points,
         'iter_str': 'iter_',
         'selected_groups': ['iter_0'],
         'render_image': True,
         'subplots_enabled': True,
         'displacement_type': 'mean'}
    markers_options = {'render_markers': True,
                       'marker_size': 20,
                       'marker_face_colour': ['r'],
                       'marker_edge_colour': ['k'],
                       'marker_style': 'o',
                       'marker_edge_width': 1}
    lines_options_default = {'render_lines': True,
                             'line_width': 1,
                             'line_colour': ['r'],
                             'line_style': '-'}
    figure_options = {'x_scale': 1.,
                      'y_scale': 1.,
                      'render_axes': True,
                      'axes_font_name': 'sans-serif',
                      'axes_font_size': 10,
                      'axes_font_style': 'normal',
                      'axes_font_weight': 'normal',
                      'axes_x_limits': None,
                      'axes_y_limits': None}
    numbering_options = {'render_numbering': False,
                         'numbers_font_name': 'serif',
                         'numbers_font_size': 10,
                         'numbers_font_style': 'normal',
                         'numbers_font_weight': 'normal',
                         'numbers_font_colour': ['k'],
                         'numbers_horizontal_align': 'center',
                         'numbers_vertical_align': 'bottom'}
    legend_options = {'render_legend': True,
                      'legend_title': '',
                      'legend_font_name': 'sans-serif',
                      'legend_font_style': 'normal',
                      'legend_font_size': 11,
                      'legend_font_weight': 'normal',
                      'legend_marker_scale': 1.,
                      'legend_location': 2,
                      'legend_bbox_to_anchor': (1.05, 1.),
                      'legend_border_axes_pad': 1.,
                      'legend_n_columns': 1,
                      'legend_horizontal_spacing': 1.,
                      'legend_vertical_spacing': 1.,
                      'legend_border': True,
                      'legend_border_padding': 0.5,
                      'legend_shadow': False,
                      'legend_rounded_corners': True}
    image_options = {'interpolation': 'bilinear',
                     'alpha': 1.0}
    viewer_options_default = []
    for group in all_groups:
        tmp_lines = lines_options_default.copy()
        tmp_lines['line_colour'] = [colour_final_dict[group]]
        tmp_markers = markers_options.copy()
        tmp_markers['marker_face_colour'] = [colour_final_dict[group]]
        tmp = {'markers': tmp_markers,
               'lines': tmp_lines,
               'figure': figure_options,
               'legend': legend_options,
               'numbering': numbering_options,
               'image': image_options}
        viewer_options_default.append(tmp)
    index_selection_default = {'min': 0,
                               'max': n_fitting_results - 1,
                               'step': 1,
                               'index': 0}

    # define function that plots errors curve
    def plot_errors_function(name):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        ipydisplay.clear_output(wait=True)

        # get selected image
        im = 0
        if n_fitting_results > 1:
            im = image_number_wid.selected_values['index']

        # get figure size
        new_figure_size = (viewer_options_wid.selected_values[0]['figure']['x_scale'] * figure_size[0],
                           viewer_options_wid.selected_values[0]['figure']['y_scale'] * figure_size[1])

        # plot errors curve
        renderer = fitting_results[im].plot_errors(
            error_type=error_type_wid.value,
            figure_id=save_figure_wid.renderer[0].figure_id,
            figure_size=new_figure_size)

        # show figure
        plt.show()

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

    # define function that plots displacements curve
    def plot_displacements_function(name):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        ipydisplay.clear_output(wait=True)

        # get selected image
        im = 0
        if n_fitting_results > 1:
            im = image_number_wid.selected_values['index']

        # get figure size
        new_figure_size = (viewer_options_wid.selected_values[0]['figure'][
                                                    'x_scale'] * figure_size[0],
                           viewer_options_wid.selected_values[0]['figure'][
                                                    'y_scale'] * figure_size[1])

        # plot errors curve
        d_type = iterations_wid.selected_values['displacement_type']
        if (d_type == 'max' or d_type == 'min' or d_type == 'mean' or
                d_type == 'median'):
            renderer = fitting_results[im].plot_displacements(
                figure_id=save_figure_wid.renderer[0].figure_id,
                figure_size=new_figure_size, stat_type=d_type)
        else:
            all_displacements = fitting_results[im].displacements()
            d_curve = [iteration_displacements[d_type]
                       for iteration_displacements in all_displacements]
            from menpo.visualize import GraphPlotter
            ylabel = "Displacement of Point {}".format(d_type)
            title = "Point {} displacement per " \
                    "iteration of Image {}".format(d_type, im)
            renderer = GraphPlotter(
                figure_id=save_figure_wid.renderer[0].figure_id,
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

        # show figure
        plt.show()

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

    # define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
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
            render_image = final_result_wid.selected_values['render_image']
            # groups
            groups = final_result_wid.selected_values['selected_groups']
            # subplots
            subplots_enabled = final_result_wid.selected_values[
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
                tmp1 = viewer_options_wid.selected_values[group_idx]['lines']
                tmp2 = viewer_options_wid.selected_values[group_idx]['markers']
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
            render_image = iterations_wid.selected_values['render_image']
            # groups
            groups = iterations_wid.selected_values['selected_groups']
            # subplots
            subplots_enabled = iterations_wid.selected_values[
                'subplots_enabled']
            subplots_titles = dict()
            iter_str = iterations_wid.selected_values['iter_str']
            for i, g in enumerate(groups):
                iter_num = g[len(iter_str)::]
                subplots_titles[iter_str + iter_num] = "Iteration " + iter_num
            # lines and markers options
            group_idx = all_groups.index('iterations')
            tmp1 = viewer_options_wid.selected_values[group_idx]['lines']
            tmp2 = viewer_options_wid.selected_values[group_idx]['markers']
            render_lines = [tmp1['render_lines']] * len(groups)
            line_style = [tmp1['line_style']] * len(groups)
            line_width = [tmp1['line_width']] * len(groups)
            render_markers = [tmp2['render_markers']] * len(groups)
            marker_style = [tmp2['marker_style']] * len(groups)
            marker_size = [tmp2['marker_size']] * len(groups)
            marker_edge_colour = [tmp2['marker_edge_colour']] * len(groups)
            marker_edge_width = [tmp2['marker_edge_width']] * len(groups)
            if (subplots_enabled or
                    iterations_wid.children[1].children[0].children[0].value ==
                    'animation'):
                line_colour = [tmp1['line_colour']] * len(groups)
                marker_face_colour = [tmp2['marker_face_colour']] * len(groups)
            else:
                cols = sample_colours_from_colourmap(len(groups), 'jet')
                line_colour = cols
                marker_face_colour = cols

        tmp1 = viewer_options_wid.selected_values[0]['numbering']
        tmp2 = viewer_options_wid.selected_values[0]['legend']
        tmp3 = viewer_options_wid.selected_values[0]['figure']
        tmp4 = viewer_options_wid.selected_values[0]['image']
        new_figure_size = (tmp3['x_scale'] * figure_size[0],
                           tmp3['y_scale'] * figure_size[1])

        # call helper _visualize
        renderer = _visualize(
            image=image, renderer=save_figure_wid.renderer[0],
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

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

    # define function that updates info text
    def update_info(name, value):
        # get selected image
        im = 0
        if n_fitting_results > 1:
            im = image_number_wid.selected_values['index']

        # create output str
        if fitting_results[im].gt_shape is not None:
            info_wid.children[1].children[0].value = \
                "> Initial error: {:.4f}".format(
                    fitting_results[im].initial_error(error_type=value))
            info_wid.children[1].children[0].visible = True
            info_wid.children[1].children[1].value = \
                "> Final error: {:.4f}".format(
                    fitting_results[im].final_error(error_type=value))
            info_wid.children[1].children[1].visible = True
            info_wid.children[1].children[2].value = \
                "> {} iterations".format(fitting_results[im].n_iters)
        else:
            info_wid.children[1].children[0].value = ''
            info_wid.children[1].children[0].visible = False
            info_wid.children[1].children[1].value = ''
            info_wid.children[1].children[1].visible = False
            info_wid.children[1].children[2].value = "> {} iterations".format(
                fitting_results[im].n_iters)
        if hasattr(fitting_results[im], 'n_levels'):  # Multilevel result
            info_wid.children[1].children[3].value = \
                "> {} levels with downscale of {:.1f}".format(
                    fitting_results[im].n_levels, fitting_results[im].downscale)
            info_wid.children[1].children[1].visible = True
        else:
            info_wid.children[1].children[1].value = ''
            info_wid.children[1].children[1].visible = True

    # Create options widgets
    channel_options_wid = channel_options(
        channels_options_default, plot_function=plot_function,
        toggle_show_default=True, toggle_show_visible=False)

    # viewer options widget
    viewer_options_wid = viewer_options(
        viewer_options_default,
        ['markers', 'lines', 'figure_one', 'legend', 'numbering', 'image'],
        objects_names=all_groups, plot_function=plot_function,
        toggle_show_visible=False, toggle_show_default=True)
    info_wid = info_print(n_bullets=4, toggle_show_default=True,
                          toggle_show_visible=False)

    # save figure widget
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = save_figure_options(initial_renderer,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)

    # final result and iterations options
    final_result_wid = final_result_options(
        final_result_options_default, plot_function=plot_function,
        title='Final', toggle_show_default=True, toggle_show_visible=False)
    iterations_wid = iterations_result_options(
        iterations_result_options_default, plot_function=plot_function,
        plot_errors_function=plot_errors_function,
        plot_displacements_function=plot_displacements_function,
        title='Iterations', toggle_show_default=True, toggle_show_visible=False)

    # Create error type radio buttons
    error_type_values = OrderedDict()
    error_type_values['Point-to-point Normalized Mean Error'] = 'me_norm'
    error_type_values['Point-to-point Mean Error'] = 'me'
    error_type_values['RMS Error'] = 'rmse'
    error_type_wid = ipywidgets.RadioButtonsWidget(
        options=error_type_values, value='me_norm', description='Error type')
    error_type_wid.on_trait_change(update_info, 'value')
    plot_ced_but = ipywidgets.ButtonWidget(description='Plot CED',
                                           visible=show_ced)
    error_wid = ipywidgets.ContainerWidget(children=[error_type_wid,
                                                     plot_ced_but])

    # define function that updates options' widgets state
    def update_widgets(name, value):
        # get new groups and labels, update landmark options and format them
        group_keys, labels_keys = _extract_groups_labels(
            fitting_results[value].fitted_image)
        # update channel options
        update_channel_options(
            channel_options_wid,
            fitting_results[value].fitted_image.n_channels,
            isinstance(fitting_results[value].fitted_image,
                       MaskedImage))
        # update final result's options
        update_final_result_options(final_result_wid, group_keys, plot_function)
        # update iterations result's options
        iterations_result_options_default = \
            {'n_iters': fitting_results[value].n_iters,
             'image_has_gt_shape': not fitting_results[value].gt_shape is None,
             'n_points': fitting_results[value].fitted_image.landmarks['final'].lms.n_points}
        update_iterations_result_options(iterations_wid,
                                         iterations_result_options_default)

    # Create final widget
    options_wid = ipywidgets.TabWidget(children=[channel_options_wid,
                                                 viewer_options_wid])
    result_wid = ipywidgets.TabWidget(children=[final_result_wid,
                                                iterations_wid])
    result_wid.on_trait_change(plot_function, 'selected_index')
    if n_fitting_results > 1:
        # image selection slider
        image_number_wid = animation_options(
            index_selection_default, plot_function=plot_function,
            update_function=update_widgets, index_description='Image Number',
            index_minus_description='<', index_plus_description='>',
            index_style=browser_style, index_text_editable=True,
            loop_default=True, interval_default=0.3,
            toggle_show_title='Image Options', toggle_show_default=True,
            toggle_show_visible=False)

        # final widget
        logo_wid = ipywidgets.ContainerWidget(children=[logo(),
                                                        image_number_wid])

        # define function that combines the results' tab widget with the
        # animation
        # If animation is activated and the user selects the iterations tab,
        # then the animation stops.
        def results_tab_fun(name, value):
            if (value == 1 and
                    image_number_wid.children[1].children[1].children[0].children[0].value):
                image_number_wid.children[1].children[1].children[0].children[1].value = True
        result_wid.on_trait_change(results_tab_fun, 'selected_index')

        # If animation is activated and the user selects the save figure tab,
        # then the animation stops.
        def save_fig_tab_fun(name, value):
            if (value == 3 and
                    image_number_wid.children[1].children[1].children[0].children[0].value):
                image_number_wid.children[1].children[1].children[0].children[1].value = True

        # final widget
        if show_ced:
            tab_titles = ['Info', 'Result', 'Options', 'CED', 'Save figure']
        else:
            tab_titles = ['Info', 'Result', 'Options', 'Error type',
                          'Save figure']
        button_title = 'Fitting Results Menu'
    else:
        # do not show the plot ced button
        plot_ced_but.visible = False

        # final widget
        logo_wid = logo()
        tab_titles = ['Info', 'Result', 'Options', 'Error type', 'Save figure']
        button_title = 'Fitting Result Menu'

    # final widget
    cont_wid = ipywidgets.TabWidget(children=[info_wid, result_wid, options_wid,
                                              error_wid, save_figure_wid])
    if n_fitting_results > 1:
        cont_wid.on_trait_change(save_fig_tab_fun, 'selected_index')

    wid = ipywidgets.ContainerWidget(children=[logo_wid, cont_wid])

    # invoke plot_ced widget
    def plot_ced_fun(name):
        # Make button invisible, so that it cannot be pressed again until
        # widget closes
        plot_ced_but.visible = False

        # get error type
        error_type = error_type_wid.value

        # create errors list
        fit_errors = [f.final_error(error_type=error_type)
                      for f in fitting_results]
        initial_errors = [f.initial_error(error_type=error_type)
                          for f in fitting_results]
        errors = [fit_errors, initial_errors]

        # call plot_ced
        plot_ced_widget = plot_ced(
            errors, figure_size=(9, 5), error_type=error_type,
            error_range=None, legend_entries=['Final Fitting',
                                              'Initialization'],
            return_widget=True)

        # If another tab is selected, then close the widget.
        def close_plot_ced_fun(name, value):
            if value != 3:
                plot_ced_widget.close()
                plot_ced_but.visible = True
        cont_wid.on_trait_change(close_plot_ced_fun, 'selected_index')

        # If another error type, then close the widget
        def close_plot_ced_fun_2(name, value):
            plot_ced_widget.close()
            plot_ced_but.visible = True
        error_type_wid.on_trait_change(close_plot_ced_fun_2, 'value')
    plot_ced_but.on_click(plot_ced_fun)

    # display final widget
    ipydisplay.display(wid)

    # set final tab titles
    for (k, tl) in enumerate(tab_titles):
        wid.children[1].set_title(k, tl)

    result_wid.set_title(0, 'Final Fitting')
    result_wid.set_title(1, 'Iterations')
    options_wid.set_title(0, 'Channels')
    options_wid.set_title(1, 'Viewer')

    # format options' widgets
    if n_fitting_results > 1:
        remove_class(wid.children[0], 'vbox')
        add_class(wid.children[0], 'hbox')
        format_animation_options(image_number_wid, index_text_width='1.0cm',
                                 container_padding='6px',
                                 container_margin='6px',
                                 container_border='1px solid black',
                                 toggle_button_font_weight='bold',
                                 border_visible=False)
    format_channel_options(channel_options_wid, container_padding='6px',
                           container_margin='6px',
                           container_border='1px solid black',
                           toggle_button_font_weight='bold',
                           border_visible=False)
    format_viewer_options(viewer_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False,
                          suboptions_border_visible=True)
    format_info_print(info_wid, font_size_in_pt='9pt', container_padding='6px',
                      container_margin='6px',
                      container_border='1px solid black',
                      toggle_button_font_weight='bold', border_visible=False)
    format_final_result_options(final_result_wid, container_padding='6px',
                                container_margin='6px',
                                container_border='1px solid black',
                                toggle_button_font_weight='bold',
                                border_visible=False)
    format_iterations_result_options(iterations_wid, container_padding='6px',
                                     container_margin='6px',
                                     container_border='1px solid black',
                                     toggle_button_font_weight='bold',
                                     border_visible=False)
    format_save_figure_options(save_figure_wid, container_padding='6px',
                               container_margin='6px',
                               container_border='1px solid black',
                               toggle_button_font_weight='bold',
                               tab_top_margin='0cm', border_visible=False)

    # Reset value to enable initial visualization
    viewer_options_wid.children[1].children[1].children[2].children[2].value = \
        False


def plot_ced(errors, figure_size=(10, 8), error_type='me_norm',
             error_range=None, legend_entries=None, return_widget=False):
    r"""
    Widget for visualizing the cumulative error curves of the provided errors.
    The generated figures can be saved to files.

    Parameters
    -----------
    errors : `list` of `list` of `float`
        The list of errors to be used.
    figure_size : (`int`, `int`), optional
        The initial size of the plotted figures.
    error_type : {``me_norm``, ``me``, ``rmse``}, optional
        Specifies the type of the provided errors.
    error_range : `list` of `float` with length 3, optional
        Specifies the horizontal axis range, i.e.

        ::

        error_range[0] = min_error
        error_range[1] = max_error
        error_range[2] = error_step

        If ``None``, then

        ::

        error_range = [0., 0.101, 0.005] for error_type = 'me_norm'
        error_range = [0., 20., 1.] for error_type = 'me'
        error_range = [0., 20., 1.] for error_type = 'rmse'

    legend_entries : `list` of `str`
        The entries of the legend. The list must have the same length as errors.
        If ``None``, the entries will have the form ``'Curve %d'``.
    return_widget : `bool`, optional
        If ``True``, the widget object will be returned so that it can be used
        as part of a bigger widget. If ``False``, the widget object is not
        returned, it is just visualized.
    """
    import IPython.html.widgets as ipywidgets
    import IPython.display as ipydisplay
    from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
    from menpofit.fittingresult import plot_cumulative_error_distribution

    # make sure that errors is a list even with one list member
    if not isinstance(errors[0], list):
        errors = [errors]

    # find number of curves
    n_curves = len(errors)

    # fix legend_entries
    if legend_entries is None:
        legend_entries = ["Curve {}".format(k) for k in range(n_curves)]

    # get horizontal axis errors
    x_label_initial_value = 'Error'
    x_axis_limit_initial_value = 0
    x_axis_step_initial_value = 0
    if error_range is None:
        if error_type == 'me_norm':
            error_range = [0., 0.101, 0.005]
            x_axis_limit_initial_value = 0.05
            x_axis_step_initial_value = 0.005
            x_label_initial_value = 'Normalized Point-to-Point Error'
        elif error_type == 'me' or error_type == 'rmse':
            error_range = [0., 20., 0.5]
            x_axis_limit_initial_value = 5.
            x_axis_step_initial_value = 0.5
            x_label_initial_value = 'Point-to-Point Error'
    else:
        x_axis_limit_initial_value = (error_range[1] + error_range[0]) / 2

    # initial options dictionaries
    figure_options = {'x_scale': 1.,
                      'y_scale': 1.,
                      'render_axes': True,
                      'axes_font_name': 'sans-serif',
                      'axes_font_size': 10,
                      'axes_font_style': 'normal',
                      'axes_font_weight': 'normal',
                      'axes_x_limits': None,
                      'axes_y_limits': (0., 1.)}
    legend_options = {'render_legend': True,
                      'legend_title': '',
                      'legend_font_name': 'sans-serif',
                      'legend_font_style': 'normal',
                      'legend_font_size': 10,
                      'legend_font_weight': 'normal',
                      'legend_marker_scale': 1.,
                      'legend_location': 2,
                      'legend_bbox_to_anchor': (1.05, 1.),
                      'legend_border_axes_pad': 1.,
                      'legend_n_columns': 1,
                      'legend_horizontal_spacing': 1.,
                      'legend_vertical_spacing': 1.,
                      'legend_border': True,
                      'legend_border_padding': 0.5,
                      'legend_shadow': False,
                      'legend_rounded_corners': False}
    grid_options = {'render_grid': True,
                    'grid_line_style': '--',
                    'grid_line_width': 0.5}

    colours = sample_colours_from_colourmap(n_curves, 'jet')
    viewer_options_default = []
    for i in range(n_curves):
        lines_options_default = {'render_lines': True,
                                 'line_width': 2,
                                 'line_colour': [colours[i]],
                                 'line_style': '-'}
        markers_options = {'render_markers': True,
                           'marker_size': 10,
                           'marker_face_colour': ['w'],
                           'marker_edge_colour': [colours[i]],
                           'marker_style': 's',
                           'marker_edge_width': 1}
        tmp = {'lines': lines_options_default,
               'markers': markers_options,
               'legend': legend_options,
               'figure': figure_options,
               'grid': grid_options}
        viewer_options_default.append(tmp)

    # define plot function
    def plot_function(name, value):
        import matplotlib.pyplot as plt
        # clear current figure, but wait until the new data to be displayed are
        # generated
        ipydisplay.clear_output(wait=True)

        # get options that need to be a list
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
        for idx in range(n_curves):
            tmp1 = viewer_options_wid.selected_values[idx]['lines']
            tmp2 = viewer_options_wid.selected_values[idx]['markers']
            render_lines.append(tmp1['render_lines'])
            line_colour.append(tmp1['line_colour'][0])
            line_style.append(tmp1['line_style'])
            line_width.append(tmp1['line_width'])
            render_markers.append(tmp2['render_markers'])
            marker_style.append(tmp2['marker_style'])
            marker_size.append(tmp2['marker_size'])
            marker_face_colour.append(tmp2['marker_face_colour'][0])
            marker_edge_colour.append(tmp2['marker_edge_colour'][0])
            marker_edge_width.append(tmp2['marker_edge_width'])

        # rest of options
        tmp3 = viewer_options_wid.selected_values[0]['legend']
        tmp4 = viewer_options_wid.selected_values[0]['figure']
        tmp5 = viewer_options_wid.selected_values[0]['grid']
        new_figure_size = (tmp4['x_scale'] * figure_size[0],
                           tmp4['y_scale'] * figure_size[1])

        # horizontal axis limits
        x_axis_limits = (0,
                         np.arange(0, errors_max.value, errors_step.value)[-1])

        # render
        renderer = plot_cumulative_error_distribution(
            errors, error_range=[0., errors_max.value, errors_step.value],
            figure_id=save_figure_wid.renderer[0].figure_id, new_figure=False,
            title=title.value, x_label=x_label.value, y_label=y_label.value,
            legend_entries=str(legend_entries_wid.value).split('\n')[:n_curves],
            render_lines=render_lines, line_colour=line_colour,
            line_style=line_style, line_width=line_width,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width,
            render_legend=tmp3['render_legend'],
            legend_title=tmp3['legend_title'],
            legend_font_name=tmp3['legend_font_name'],
            legend_font_style=tmp3['legend_font_style'],
            legend_font_size=tmp3['legend_font_size'],
            legend_font_weight=tmp3['legend_font_weight'],
            legend_marker_scale=tmp3['legend_marker_scale'],
            legend_location=tmp3['legend_location'],
            legend_bbox_to_anchor=tmp3['legend_bbox_to_anchor'],
            legend_border_axes_pad=tmp3['legend_border_axes_pad'],
            legend_n_columns=tmp3['legend_n_columns'],
            legend_horizontal_spacing=tmp3['legend_horizontal_spacing'],
            legend_vertical_spacing=tmp3['legend_vertical_spacing'],
            legend_border=tmp3['legend_border'],
            legend_border_padding=tmp3['legend_border_padding'],
            legend_shadow=tmp3['legend_shadow'],
            legend_rounded_corners=tmp3['legend_rounded_corners'],
            render_axes=tmp4['render_axes'],
            axes_font_name=tmp4['axes_font_name'],
            axes_font_size=tmp4['axes_font_size'],
            axes_font_style=tmp4['axes_font_style'],
            axes_font_weight=tmp4['axes_font_weight'],
            axes_x_limits=x_axis_limits,
            axes_y_limits=viewer_options_wid.selected_values[0]['figure']['axes_y_limits'],
            figure_size=new_figure_size, render_grid=tmp5['render_grid'],
            grid_line_style=tmp5['grid_line_style'],
            grid_line_width=tmp5['grid_line_width'])

        plt.show()

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

    # create options widgets
    # error_range
    errors_max = ipywidgets.FloatSliderWidget(
        min=error_range[0] + error_range[2], max=error_range[1],
        step=error_range[2], description='Error axis max',
        value=x_axis_limit_initial_value)
    if error_type == 'me_norm':
        errors_step = ipywidgets.FloatSliderWidget(
            min=0., max=0.05, step=0.001, description='Error axis step',
            value=x_axis_step_initial_value)
    else:
        errors_step = ipywidgets.FloatSliderWidget(
            min=0., max=error_range[1], step=error_range[2] / 10.,
            description='Error axis step', value=x_axis_step_initial_value)
    error_range_wid = ipywidgets.ContainerWidget(children=[errors_max,
                                                           errors_step])

    # legend_entries, x label, y label, title container
    legend_entries_wid = ipywidgets.TextareaWidget(
        description='Legend entries', value="\n".join(legend_entries))
    x_label = ipywidgets.TextWidget(description='Horizontal axis label',
                                    value=x_label_initial_value)
    y_label = ipywidgets.TextWidget(description='Vertical axis label',
                                    value='Images Proportion')
    title = ipywidgets.TextWidget(description='Figure title',
                                  value=' ')
    labels_wid = ipywidgets.ContainerWidget(children=[legend_entries_wid,
                                                      x_label, y_label, title])

    # viewer options widget
    viewer_options_wid = viewer_options(viewer_options_default,
                                        ['lines', 'markers', 'legend',
                                         'figure_two', 'grid'],
                                        objects_names=legend_entries,
                                        plot_function=plot_function,
                                        toggle_show_visible=False,
                                        toggle_show_default=True,
                                        labels=[legend_entries])

    # save figure widget
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = save_figure_options(initial_renderer,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)

    # assign plot function
    x_label.on_trait_change(plot_function, 'value')
    y_label.on_trait_change(plot_function, 'value')
    title.on_trait_change(plot_function, 'value')
    legend_entries_wid.on_trait_change(plot_function, 'value')
    errors_max.on_trait_change(plot_function, 'value')
    errors_step.on_trait_change(plot_function, 'value')

    # create final widget
    tab_wid = ipywidgets.TabWidget(children=[error_range_wid, labels_wid,
                                             viewer_options_wid,
                                             save_figure_wid])

    wid = ipywidgets.ContainerWidget(children=[logo(), tab_wid],
                                     button_text='CED Menu')

    # display final widget
    ipydisplay.display(wid)

    # set final tab titles
    tab_titles = ['Error axis options', 'Labels options', 'Viewer options',
                  'Save figure']
    for (k, tl) in enumerate(tab_titles):
        tab_wid.set_title(k, tl)

    # format options' widgets
    add_class(labels_wid, 'align-end')
    legend_entries_wid.width = '6cm'
    legend_entries_wid.height = '2cm'
    x_label.width = '6cm'
    y_label.width = '6cm'
    title.width = '6cm'
    errors_max.width = '6cm'
    errors_step.width = '6cm'
    format_viewer_options(viewer_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False,
                          suboptions_border_visible=True)
    format_save_figure_options(save_figure_wid, container_padding='6px',
                               container_margin='6px',
                               container_border='1px solid black',
                               toggle_button_font_weight='bold',
                               tab_top_margin='0cm', border_visible=False)

    # Reset value to trigger initial visualization
    title.value = 'Cumulative error distribution'

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
        from menpo.visualize.image import glyph

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
                if glyph_enabled or sum_enabled:
                    # image, landmarks, masked, glyph
                    renderer = glyph(image, vectors_block_size=glyph_block_size,
                                     use_negative=glyph_use_negative,
                                     channels=channels).\
                        view_landmarks(
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
                            render_axes=render_axes,
                            axes_font_name=axes_font_name,
                            axes_font_size=axes_font_size,
                            axes_font_style=axes_font_style,
                            axes_font_weight=axes_font_weight,
                            axes_x_limits=axes_x_limits,
                            axes_y_limits=axes_y_limits,
                            interpolation=interpolation, alpha=alpha,
                            figure_size=figure_size, **mask_arguments)
                else:
                    # image, landmarks, masked, not glyph
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
            if glyph_enabled or sum_enabled:
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
            else:
                # image, not landmarks, masked, not glyph
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


def _check_n_parameters(n_params, n_levels, max_n_params):
    r"""
    Checks the maximum number of components per level either of the shape
    or the appearance model. It must be None or int or float or a list of
    those containing 1 or {n_levels} elements.
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
