from menpo.visualize.widgets.helpers import (figure_options,
                                             format_figure_options,
                                             figure_options_two_scales,
                                             format_figure_options_two_scales,
                                             channel_options,
                                             format_channel_options,
                                             update_channel_options,
                                             landmark_options,
                                             format_landmark_options,
                                             info_print, format_info_print,
                                             model_parameters,
                                             format_model_parameters,
                                             update_model_parameters,
                                             final_result_options,
                                             format_final_result_options,
                                             update_final_result_options,
                                             iterations_result_options,
                                             format_iterations_result_options,
                                             update_iterations_result_options,
                                             animation_options,
                                             format_animation_options,
                                             plot_options,
                                             format_plot_options,
                                             save_figure_options,
                                             format_save_figure_options)
from menpo.visualize.widgets.base import (_plot_figure, _plot_graph,
                                          _plot_eigenvalues,
                                          _check_n_parameters,
                                          _raw_info_string_to_latex,
                                          _extract_groups_labels)
from IPython.html.widgets import (FloatTextWidget, TextWidget, PopupWidget,
                                  ContainerWidget, TabWidget, FloatSliderWidget,
                                  RadioButtonsWidget, CheckboxWidget,
                                  DropdownWidget, AccordionWidget, ButtonWidget)
from IPython.display import display, clear_output
import matplotlib.pylab as plt
import numpy as np
from collections import OrderedDict

# This glyph import is called frequently during visualisation, so we ensure
# that we only import it once
glyph = None


def visualize_shape_model(shape_models, n_parameters=5,
                          parameters_bounds=(-3.0, 3.0), figure_size=(7, 7),
                          mode='multiple', popup=False, **kwargs):
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
        If int, then the number of sliders per level is the minimum between
        n_parameters and the number of active components per level.
        If list of int, then a number of sliders is defined per level.
        If None, all the active components per level will have a slider.

    parameters_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.

    figure_size : (`int`, `int`), optional
        The size of the plotted figures.

    mode : 'single' or 'multiple', optional
        If single, only a single slider is constructed along with a drop down
        menu.
        If multiple, a slider is constructed for each parameter.

    popup : `boolean`, optional
        If enabled, the widget will appear as a popup window.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
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

    # Define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get params
        level = 0
        if n_levels > 1:
            level = level_wid.value
        def_mode = mode_wid.value
        axis_mode = axes_mode_wid.value
        parameters_values = model_parameters_wid.parameters_values
        x_scale = figure_options_wid.x_scale
        y_scale = figure_options_wid.y_scale
        axes_visible = figure_options_wid.axes_visible

        # compute weights
        weights = (parameters_values *
                   shape_models[level].eigenvalues[:len(parameters_values)] ** 0.5)

        # compute the mean
        mean = shape_models[level].mean()

        # select figure
        figure_id = plt.figure(save_figure_wid.figure_id.number)

        # invert axis if image mode is enabled
        if axis_mode == 1:
            plt.gca().invert_yaxis()

        # compute and show instance
        if def_mode == 1:
            # Deformation mode
            # compute instance
            instance = shape_models[level].instance(weights)

            # plot
            if mean_wid.value:
                mean.view(image_view=axis_mode == 1, colour_array='y',
                          **kwargs)
                plt.hold(True)
            instance.view(image_view=axis_mode == 1, **kwargs)

            # instance range
            tmp_range = instance.range()
        else:
            # Vectors mode
            # compute instance
            instance_lower = shape_models[level].instance([-p for p in weights])
            instance_upper = shape_models[level].instance(weights)

            # plot
            mean.view(image_view=axis_mode == 1, **kwargs)
            plt.hold(True)
            for p in range(mean.n_points):
                xm = mean.points[p, 0]
                ym = mean.points[p, 1]
                xl = instance_lower.points[p, 0]
                yl = instance_lower.points[p, 1]
                xu = instance_upper.points[p, 0]
                yu = instance_upper.points[p, 1]
                if axis_mode == 1:
                    # image mode
                    plt.plot([ym, yl], [xm, xl], 'r-', lw=2)
                    plt.plot([ym, yu], [xm, xu], 'g-', lw=2)
                else:
                    # point cloud mode
                    plt.plot([xm, xl], [ym, yl], 'r-', lw=2)
                    plt.plot([xm, xu], [ym, yu], 'g-', lw=2)

            # instance range
            tmp_range = mean.range()

        plt.hold(False)
        plt.gca().axis('equal')
        # set figure size
        plt.gcf().set_size_inches([x_scale, y_scale] * np.asarray(figure_size))
        # turn axis on/off
        if not axes_visible:
            plt.axis('off')
        plt.show()

        # save the current figure id
        save_figure_wid.figure_id = figure_id

        # info_wid string
        info_txt = r"""
            Level: {} out of   {}.
            {} components in total.
            {} active components.
            {:.1f} % variance kept.
            Instance range: {:.1f} x {:.1f}.
            {} landmark points, {} features.
        """.format(level + 1, n_levels, shape_models[level].n_components,
                   shape_models[level].n_active_components,
                   shape_models[level].variance_ratio() * 100, tmp_range[0],
                   tmp_range[1], mean.n_points,
                   shape_models[level].n_features)

        info_wid.children[1].value = _raw_info_string_to_latex(info_txt)

    # Plot eigenvalues function
    def plot_eigenvalues(name):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get parameters
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # get the current figure id
        figure_id = save_figure_wid.figure_id

        # show eigenvalues plots
        new_figure_id = _plot_eigenvalues(figure_id, shape_models[level],
                                          figure_size,
                                          figure_options_wid.x_scale,
                                          figure_options_wid.y_scale)

        # save the current figure id
        save_figure_wid.figure_id = new_figure_id

    # create options widgets
    mode_dict = OrderedDict()
    mode_dict['Deformation'] = 1
    mode_dict['Vectors'] = 2
    mode_wid = RadioButtonsWidget(values=mode_dict, description='Mode:',
                                  value=1)
    mode_wid.on_trait_change(plot_function, 'value')
    mean_wid = CheckboxWidget(value=False, description='Show mean shape')
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
    figure_options_wid = figure_options(plot_function, scale_default=1.,
                                        show_axes_default=True,
                                        toggle_show_default=True,
                                        toggle_show_visible=False)
    axes_mode_wid = RadioButtonsWidget(values={'Image': 1, 'Point cloud': 2},
                                       description='Axes mode:', value=1)
    axes_mode_wid.on_trait_change(plot_function, 'value')
    ch = list(figure_options_wid.children)
    ch.insert(3, axes_mode_wid)
    figure_options_wid.children = ch
    info_wid = info_print(toggle_show_default=True, toggle_show_visible=False)
    initial_figure_id = plt.figure()
    save_figure_wid = save_figure_options(initial_figure_id,
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
        level_wid = RadioButtonsWidget(values=radio_str,
                                       description='Pyramid:', value=0)
        level_wid.on_trait_change(update_widgets, 'value')
        level_wid.on_trait_change(plot_function, 'value')
        radio_children = [level_wid, mode_wid, mean_wid]
    else:
        radio_children = [mode_wid, mean_wid]
    radio_wids = ContainerWidget(children=radio_children)
    tmp_wid = ContainerWidget(children=[radio_wids, model_parameters_wid])
    wid = TabWidget(children=[tmp_wid, figure_options_wid, info_wid,
                              save_figure_wid])
    if popup:
        wid = PopupWidget(children=[wid], button_text='Shape Model Menu')

    # display final widget
    display(wid)

    # set final tab titles
    tab_titles = ['Shape parameters', 'Figure options', 'Model info',
                  'Save figure']
    if popup:
        for (k, tl) in enumerate(tab_titles):
            wid.children[0].set_title(k, tl)
    else:
        for (k, tl) in enumerate(tab_titles):
            wid.set_title(k, tl)

    # align widgets
    tmp_wid.remove_class('vbox')
    tmp_wid.add_class('hbox')
    format_model_parameters(model_parameters_wid, container_padding='6px',
                            container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold',
                            border_visible=True)
    format_figure_options(figure_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False)
    format_info_print(info_wid, font_size_in_pt='9pt', container_padding='6px',
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
    figure_options_wid.children[2].value = False


def visualize_appearance_model(appearance_models, n_parameters=5,
                               parameters_bounds=(-3.0, 3.0),
                               figure_size=(7, 7), mode='multiple',
                               popup=False, **kwargs):
    r"""
    Allows the dynamic visualization of a multilevel appearance model.

    Parameters
    -----------
    appearance_models : `list` of :map:`PCAModel` or subclass
        The multilevel appearance model to be displayed. Note that each level
        can have different attributes, e.g. number of parameters, feature type,
        number of channels.

    n_parameters : `int` or `list` of `int` or None, optional
        The number of principal components to be used for the parameters
        sliders.
        If int, then the number of sliders per level is the minimum between
        n_parameters and the number of active components per level.
        If list of int, then a number of sliders is defined per level.
        If None, all the active components per level will have a slider.

    parameters_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.

    figure_size : (`int`, `int`), optional
        The size of the plotted figures.

    mode : 'single' or 'multiple', optional
        If single, only a single slider is constructed along with a drop down
        menu.
        If multiple, a slider is constructed for each parameter.

    popup : `boolean`, optional
        If enabled, the widget will appear as a popup window.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
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

    # define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get selected level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # get parameters values
        parameters_values = model_parameters_wid.parameters_values

        # compute instance
        weights = parameters_values * appearance_models[level].eigenvalues[:len(parameters_values)] ** 0.5
        instance = appearance_models[level].instance(weights)

        # get the current figure id
        figure_id = save_figure_wid.figure_id

        # show image with selected options
        new_figure_id = _plot_figure(
            image=instance, figure_id=figure_id, image_enabled=True,
            landmarks_enabled=landmark_options_wid.landmarks_enabled,
            image_is_masked=channel_options_wid.image_is_masked,
            masked_enabled=channel_options_wid.masked_enabled,
            channels=channel_options_wid.channels,
            glyph_enabled=channel_options_wid.glyph_enabled,
            glyph_block_size=channel_options_wid.glyph_block_size,
            glyph_use_negative=channel_options_wid.glyph_use_negative,
            sum_enabled=channel_options_wid.sum_enabled,
            groups=[landmark_options_wid.group],
            with_labels=[landmark_options_wid.with_labels],
            groups_colours=dict(), subplots_enabled=False,
            subplots_titles=dict(), image_axes_mode=True,
            legend_enabled=landmark_options_wid.legend_enabled,
            numbering_enabled=landmark_options_wid.numbering_enabled,
            x_scale=figure_options_wid.x_scale,
            y_scale=figure_options_wid.y_scale,
            axes_visible=figure_options_wid.axes_visible,
            figure_size=figure_size, **kwargs)

        # save the current figure id
        save_figure_wid.figure_id = new_figure_id

        # update info text widget
        update_info(instance, level, landmark_options_wid.group)

    # define function that updates info text
    def update_info(image, level, group):
        lvl_app_mod = appearance_models[level]

        info_txt = r"""
            Level: {} out of {}.
            {} components in total.
            {} active components.
            {:.1f}% variance kept.
            Reference shape of size {} with {} channel{}.
            {} features.
            {} landmark points.
            Instance: min={:.3f}, max={:.3f}
        """.format(level + 1, n_levels, lvl_app_mod.n_components,
                   lvl_app_mod.n_active_components,
                   lvl_app_mod.variance_ratio() * 100, image._str_shape,
                   image.n_channels, 's' * (image.n_channels > 1),
                   lvl_app_mod.n_features, image.landmarks[group].lms.n_points,
                   image.pixels.min(), image.pixels.max())

        # update info widget text
        info_wid.children[1].value = _raw_info_string_to_latex(info_txt)

    # Plot eigenvalues function
    def plot_eigenvalues(name):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get parameters
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # get the current figure id
        figure_id = save_figure_wid.figure_id

        # show eigenvalues plots
        new_figure_id = _plot_eigenvalues(figure_id, appearance_models[level],
                                          figure_size,
                                          figure_options_wid.x_scale,
                                          figure_options_wid.y_scale)

        # save the current figure id
        save_figure_wid.figure_id = new_figure_id

    # create options widgets
    model_parameters_wid = model_parameters(n_parameters[0], plot_function,
                                            params_str='param ', mode=mode,
                                            params_bounds=parameters_bounds,
                                            toggle_show_default=True,
                                            toggle_show_visible=False,
                                            plot_eig_visible=True,
                                            plot_eig_function=plot_eigenvalues)
    channel_options_wid = channel_options(
        appearance_models[0].mean().n_channels,
        isinstance(appearance_models[0].mean(), MaskedImage), plot_function,
        masked_default=True, toggle_show_default=True,
        toggle_show_visible=False)

    # find initial groups and labels that will be passed to the landmark options
    # widget creation
    mean_has_landmarks = appearance_models[0].mean().landmarks.n_groups != 0
    if mean_has_landmarks:
        all_groups_keys, all_labels_keys = _extract_groups_labels(
            appearance_models[0].mean())
    else:
        all_groups_keys = [' ']
        all_labels_keys = [[' ']]
    landmark_options_wid = landmark_options(
        all_groups_keys, all_labels_keys, plot_function,
        toggle_show_default=True, landmarks_default=mean_has_landmarks,
        legend_default=False, numbering_default=False,
        toggle_show_visible=False)
    # if the mean doesn't have landmarks, then landmarks checkbox should be
    # disabled
    landmark_options_wid.children[1].children[0].disabled = \
        not mean_has_landmarks
    figure_options_wid = figure_options(plot_function, scale_default=1.,
                                        show_axes_default=True,
                                        toggle_show_default=True,
                                        toggle_show_visible=False)
    info_wid = info_print(toggle_show_default=True, toggle_show_visible=False)
    initial_figure_id = plt.figure()
    save_figure_wid = save_figure_options(initial_figure_id,
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
        level_wid = RadioButtonsWidget(values=radio_str,
                                       description='Pyramid:', value=0)
        level_wid.on_trait_change(update_widgets, 'value')
        level_wid.on_trait_change(plot_function, 'value')
        tmp_children.insert(0, level_wid)
    tmp_wid = ContainerWidget(children=tmp_children)
    wid = TabWidget(children=[tmp_wid, channel_options_wid,
                              landmark_options_wid, figure_options_wid,
                              info_wid, save_figure_wid])
    if popup:
        wid = PopupWidget(children=[wid], button_text='Appearance Model Menu')

    # display final widget
    display(wid)

    # set final tab titles
    tab_titles = ['Appearance parameters', 'Channels options',
                  'Landmarks options', 'Figure options', 'Model info',
                  'Save figure']
    if popup:
        for (k, tl) in enumerate(tab_titles):
            wid.children[0].set_title(k, tl)
    else:
        for (k, tl) in enumerate(tab_titles):
            wid.set_title(k, tl)

    # align widgets
    tmp_wid.remove_class('vbox')
    tmp_wid.add_class('hbox')
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
    format_figure_options(figure_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False)
    format_info_print(info_wid, font_size_in_pt='9pt', container_padding='6px',
                      container_margin='6px',
                      container_border='1px solid black',
                      toggle_button_font_weight='bold', border_visible=False)
    format_save_figure_options(save_figure_wid, container_padding='6px',
                               container_margin='6px',
                               container_border='1px solid black',
                               toggle_button_font_weight='bold',
                               tab_top_margin='0cm', border_visible=False)

    # update widgets' state for level 0
    update_widgets('', 0)

    # Reset value to enable initial visualization
    figure_options_wid.children[2].value = False


def visualize_aam(aam, n_shape_parameters=5, n_appearance_parameters=5,
                  parameters_bounds=(-3.0, 3.0), figure_size=(7, 7),
                  mode='multiple', popup=False, **kwargs):
    r"""
    Allows the dynamic visualization of a multilevel AAM.

    Parameters
    -----------
    aam : :map:`AAM` or subclass
        The multilevel AAM to be displayed. Note that each level can have
        different attributes, e.g. number of active components, feature type,
        number of channels.

    n_shape_parameters : `int` or `list` of `int` or None, optional
        The number of shape principal components to be used for the parameters
        sliders.
        If int, then the number of sliders per level is the minimum between
        n_parameters and the number of active components per level.
        If list of int, then a number of sliders is defined per level.
        If None, all the active components per level will have a slider.

    n_appearance_parameters : `int` or `list` of `int` or None, optional
        The number of appearance principal components to be used for the
        parameters sliders.
        If int, then the number of sliders per level is the minimum between
        n_parameters and the number of active components per level.
        If list of int, then a number of sliders is defined per level.
        If None, all the active components per level will have a slider.

    parameters_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.

    figure_size : (`int`, `int`), optional
        The size of the plotted figures.

    mode : 'single' or 'multiple', optional
        If single, only a single slider is constructed along with a drop down
        menu.
        If multiple, a slider is constructed for each parameter.

    popup : `boolean`, optional
        If enabled, the widget will appear as a popup window.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
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

    # define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get selected level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # get weights and compute instance
        shape_weights = shape_model_parameters_wid.parameters_values
        appearance_weights = appearance_model_parameters_wid.parameters_values
        instance = aam.instance(level=level, shape_weights=shape_weights,
                                appearance_weights=appearance_weights)

        # get the current figure id
        figure_id = save_figure_wid.figure_id

        # show image with selected options
        new_figure_id = _plot_figure(
            image=instance, figure_id=figure_id, image_enabled=True,
            landmarks_enabled=landmark_options_wid.landmarks_enabled,
            image_is_masked=channel_options_wid.image_is_masked,
            masked_enabled=channel_options_wid.masked_enabled,
            channels=channel_options_wid.channels,
            glyph_enabled=channel_options_wid.glyph_enabled,
            glyph_block_size=channel_options_wid.glyph_block_size,
            glyph_use_negative=channel_options_wid.glyph_use_negative,
            sum_enabled=channel_options_wid.sum_enabled,
            groups=[landmark_options_wid.group],
            with_labels=[landmark_options_wid.with_labels],
            groups_colours=dict(), subplots_enabled=False,
            subplots_titles=dict(), image_axes_mode=True,
            legend_enabled=landmark_options_wid.legend_enabled,
            numbering_enabled=landmark_options_wid.numbering_enabled,
            x_scale=figure_options_wid.x_scale,
            y_scale=figure_options_wid.y_scale,
            axes_visible=figure_options_wid.axes_visible,
            figure_size=figure_size, **kwargs)

        # save the current figure id
        save_figure_wid.figure_id = new_figure_id

        # update info text widget
        update_info(aam, instance, level, landmark_options_wid.group)

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

        # Formatting is a bit ugly but this is MUCH easier to read.
        info_txt = r"""
            {} training images.
            Warp using {} transform.
            Level {}/{}  (downscale={:.1f}).
            {}
            {}
            {}
            Reference frame of length {} ({} x {}C, {} x {}C).
            {} shape components ({:.2f}% of variance)
            {} appearance components ({:.2f}% of variance)
            {} landmark points.
            Instance: min={:.3f} , max={:.3f}
            """.format(aam.n_training_images, aam.transform.__name__,
                       level + 1,
                       aam.n_levels, aam.downscale, tmp_shape_models,
                       tmp_pyramid, tmp_feat, lvl_app_mod.n_features,
                       tmplt_inst.n_true_pixels(), n_channels,
                       tmplt_inst._str_shape, n_channels,
                       lvl_shape_mod.n_components,
                       lvl_shape_mod.variance_ratio() * 100,
                       lvl_app_mod.n_components,
                       lvl_app_mod.variance_ratio() * 100,
                       instance.landmarks[group].lms.n_points,
                       instance.pixels.min(), instance.pixels.max())

        info_wid.children[1].value = _raw_info_string_to_latex(info_txt)

    # Plot shape eigenvalues function
    def plot_shape_eigenvalues(name):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get parameters
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # get the current figure id
        figure_id = save_figure_wid.figure_id

        # show eigenvalues plots
        new_figure_id = _plot_eigenvalues(figure_id, aam.shape_models[level],
                                          figure_size,
                                          figure_options_wid.x_scale,
                                          figure_options_wid.y_scale)

        # save the current figure id
        save_figure_wid.figure_id = new_figure_id

    # Plot appearance eigenvalues function
    def plot_appearance_eigenvalues(name):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get parameters
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # get the current figure id
        figure_id = save_figure_wid.figure_id

        # show eigenvalues plots
        new_figure_id = _plot_eigenvalues(figure_id,
                                          aam.appearance_models[level],
                                          figure_size,
                                          figure_options_wid.x_scale,
                                          figure_options_wid.y_scale)

        # save the current figure id
        save_figure_wid.figure_id = new_figure_id

    # create options widgets
    shape_model_parameters_wid = model_parameters(
        n_shape_parameters[0], plot_function, params_str='param ', mode=mode,
        params_bounds=parameters_bounds, toggle_show_default=False,
        toggle_show_visible=True, toggle_show_name='Shape Parameters',
        plot_eig_visible=True, plot_eig_function=plot_shape_eigenvalues)
    appearance_model_parameters_wid = model_parameters(
        n_appearance_parameters[0], plot_function, params_str='param ',
        mode=mode, params_bounds=parameters_bounds, toggle_show_default=False,
        toggle_show_visible=True, toggle_show_name='Appearance Parameters',
        plot_eig_visible=True, plot_eig_function=plot_appearance_eigenvalues)
    channel_options_wid = channel_options(
        aam.appearance_models[0].mean().n_channels,
        isinstance(aam.appearance_models[0].mean(), MaskedImage), plot_function,
        masked_default=True, toggle_show_default=True,
        toggle_show_visible=False)
    all_groups_keys, all_labels_keys = \
        _extract_groups_labels(aam.appearance_models[0].mean())
    landmark_options_wid = landmark_options(all_groups_keys, all_labels_keys,
                                            plot_function,
                                            toggle_show_default=True,
                                            landmarks_default=True,
                                            legend_default=False,
                                            numbering_default=False,
                                            toggle_show_visible=False)
    figure_options_wid = figure_options(plot_function, scale_default=1.,
                                        show_axes_default=True,
                                        toggle_show_default=True,
                                        toggle_show_visible=False)
    info_wid = info_print(toggle_show_default=True, toggle_show_visible=False)
    initial_figure_id = plt.figure()
    save_figure_wid = save_figure_options(initial_figure_id,
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
    model_parameters_wid = ContainerWidget(
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
        level_wid = RadioButtonsWidget(values=radio_str,
                                       description='Pyramid:', value=0)
        level_wid.on_trait_change(update_widgets, 'value')
        level_wid.on_trait_change(plot_function, 'value')
        tmp_children.insert(0, level_wid)
    tmp_wid = ContainerWidget(children=tmp_children)
    wid = TabWidget(children=[tmp_wid, channel_options_wid,
                              landmark_options_wid, figure_options_wid,
                              info_wid, save_figure_wid])
    if popup:
        wid = PopupWidget(children=[wid], button_text='AAM Menu')

    # display final widget
    display(wid)

    # set final tab titles
    tab_titles = ['AAM parameters', 'Channels options', 'Landmarks options',
                  'Figure options', 'Model info', 'Save figure']
    if popup:
        for (k, tl) in enumerate(tab_titles):
            wid.children[0].set_title(k, tl)
    else:
        for (k, tl) in enumerate(tab_titles):
            wid.set_title(k, tl)

    # align widgets
    if n_levels > 1:
        tmp_wid.remove_class('vbox')
        tmp_wid.add_class('hbox')
    format_model_parameters(shape_model_parameters_wid,
                            container_padding='6px', container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold',
                            border_visible=True)
    format_model_parameters(appearance_model_parameters_wid,
                            container_padding='6px', container_margin='6px',
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
    format_figure_options(figure_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False)
    format_info_print(info_wid, font_size_in_pt='9pt', container_padding='6px',
                      container_margin='6px',
                      container_border='1px solid black',
                      toggle_button_font_weight='bold', border_visible=False)
    format_save_figure_options(save_figure_wid, container_padding='6px',
                               container_margin='6px',
                               container_border='1px solid black',
                               toggle_button_font_weight='bold',
                               tab_top_margin='0cm', border_visible=False)

    # update widgets' state for level 0
    update_widgets('', 0)

    # Reset value to enable initial visualization
    figure_options_wid.children[2].value = False


def visualize_atm(atm, n_shape_parameters=5, parameters_bounds=(-3.0, 3.0),
                  figure_size=(7, 7), mode='multiple', popup=False, **kwargs):
    r"""
    Allows the dynamic visualization of a multilevel ATM.

    Parameters
    -----------
    atm : :map:`ATM` or subclass
        The multilevel ATM to be displayed. Note that each level can have
        different attributes, e.g. number of active components, feature type,
        number of channels.

    n_shape_parameters : `int` or `list` of `int` or None, optional
        The number of shape principal components to be used for the parameters
        sliders.
        If int, then the number of sliders per level is the minimum between
        n_parameters and the number of active components per level.
        If list of int, then a number of sliders is defined per level.
        If None, all the active components per level will have a slider.

    parameters_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.

    figure_size : (`int`, `int`), optional
        The size of the plotted figures.

    mode : 'single' or 'multiple', optional
        If single, only a single slider is constructed along with a drop down
        menu.
        If multiple, a slider is constructed for each parameter.

    popup : `boolean`, optional
        If enabled, the widget will appear as a popup window.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
    from menpo.image import MaskedImage

    # find number of levels
    n_levels = atm.n_levels

    # find maximum number of components per level
    max_n_shape = [sp.n_active_components for sp in atm.shape_models]

    # check given n_parameters
    # the returned n_parameters is a list of len n_levels
    n_shape_parameters = _check_n_parameters(n_shape_parameters, n_levels,
                                             max_n_shape)

    # define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get selected level
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # get weights and compute instance
        shape_weights = shape_model_parameters_wid.parameters_values
        instance = atm.instance(level=level, shape_weights=shape_weights)

        # get the current figure id
        figure_id = save_figure_wid.figure_id

        # show image with selected options
        new_figure_id = _plot_figure(
            image=instance, figure_id=figure_id, image_enabled=True,
            landmarks_enabled=landmark_options_wid.landmarks_enabled,
            image_is_masked=channel_options_wid.image_is_masked,
            masked_enabled=channel_options_wid.masked_enabled,
            channels=channel_options_wid.channels,
            glyph_enabled=channel_options_wid.glyph_enabled,
            glyph_block_size=channel_options_wid.glyph_block_size,
            glyph_use_negative=channel_options_wid.glyph_use_negative,
            sum_enabled=channel_options_wid.sum_enabled,
            groups=[landmark_options_wid.group],
            with_labels=[landmark_options_wid.with_labels],
            groups_colours=dict(), subplots_enabled=False,
            subplots_titles=dict(), image_axes_mode=True,
            legend_enabled=landmark_options_wid.legend_enabled,
            numbering_enabled=landmark_options_wid.numbering_enabled,
            x_scale=figure_options_wid.x_scale,
            y_scale=figure_options_wid.y_scale,
            axes_visible=figure_options_wid.axes_visible,
            figure_size=figure_size, **kwargs)

        # save the current figure id
        save_figure_wid.figure_id = new_figure_id

        # update info text widget
        update_info(atm, instance, level, landmark_options_wid.group)

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

        # Formatting is a bit ugly but this is MUCH easier to read.
        info_txt = r"""
            {} training shapes.
            Warp using {} transform.
            Level {}/{}  (downscale={:.1f}).
            {}
            {}
            {}
            Reference frame of length {} ({} x {}C, {} x {}C).
            {} shape components ({:.2f}% of variance)
            {} landmark points.
            Instance: min={:.3f} , max={:.3f}
            """.format(atm.n_training_shapes, atm.transform.__name__,
                       level + 1,
                       atm.n_levels, atm.downscale, tmp_shape_models,
                       tmp_pyramid, tmp_feat,
                       tmplt_inst.n_true_pixels() * n_channels,
                       tmplt_inst.n_true_pixels(), n_channels,
                       tmplt_inst._str_shape, n_channels,
                       lvl_shape_mod.n_components,
                       lvl_shape_mod.variance_ratio() * 100,
                       instance.landmarks[group].lms.n_points,
                       instance.pixels.min(), instance.pixels.max())

        info_wid.children[1].value = _raw_info_string_to_latex(info_txt)

    # Plot shape eigenvalues function
    def plot_shape_eigenvalues(name):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get parameters
        level = 0
        if n_levels > 1:
            level = level_wid.value

        # get the current figure id
        figure_id = save_figure_wid.figure_id

        # show eigenvalues plots
        new_figure_id = _plot_eigenvalues(figure_id, atm.shape_models[level],
                                          figure_size,
                                          figure_options_wid.x_scale,
                                          figure_options_wid.y_scale)

        # save the current figure id
        save_figure_wid.figure_id = new_figure_id

    # create options widgets
    shape_model_parameters_wid = model_parameters(
        n_shape_parameters[0], plot_function, params_str='param ', mode=mode,
        params_bounds=parameters_bounds, toggle_show_default=True,
        toggle_show_visible=False, toggle_show_name='Shape Parameters',
        plot_eig_visible=True, plot_eig_function=plot_shape_eigenvalues)
    channel_options_wid = channel_options(
        atm.warped_templates[0].n_channels,
        isinstance(atm.warped_templates[0], MaskedImage),
        plot_function, masked_default=True, toggle_show_default=True,
        toggle_show_visible=False)
    all_groups_keys, all_labels_keys = \
        _extract_groups_labels(atm.warped_templates[0])
    landmark_options_wid = landmark_options(all_groups_keys, all_labels_keys,
                                            plot_function,
                                            toggle_show_default=True,
                                            landmarks_default=True,
                                            legend_default=False,
                                            numbering_default=False,
                                            toggle_show_visible=False)
    figure_options_wid = figure_options(plot_function, scale_default=1.,
                                        show_axes_default=True,
                                        toggle_show_default=True,
                                        toggle_show_visible=False)
    info_wid = info_print(toggle_show_default=True, toggle_show_visible=False)
    initial_figure_id = plt.figure()
    save_figure_wid = save_figure_options(initial_figure_id,
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
        level_wid = RadioButtonsWidget(values=radio_str,
                                       description='Pyramid:', value=0)
        level_wid.on_trait_change(update_widgets, 'value')
        level_wid.on_trait_change(plot_function, 'value')
        tmp_children.insert(0, level_wid)
    tmp_wid = ContainerWidget(children=tmp_children)
    wid = TabWidget(children=[tmp_wid, channel_options_wid,
                              landmark_options_wid, figure_options_wid,
                              info_wid, save_figure_wid])
    if popup:
        wid = PopupWidget(children=[wid], button_text='ATM Menu')

    # display final widget
    display(wid)

    # set final tab titles
    tab_titles = ['Shape parameters', 'Channels options', 'Landmarks options',
                  'Figure options', 'Model info', 'Save figure']
    if popup:
        for (k, tl) in enumerate(tab_titles):
            wid.children[0].set_title(k, tl)
    else:
        for (k, tl) in enumerate(tab_titles):
            wid.set_title(k, tl)

    # align widgets
    if n_levels > 1:
        tmp_wid.remove_class('vbox')
        tmp_wid.add_class('hbox')
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
    format_figure_options(figure_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False)
    format_info_print(info_wid, font_size_in_pt='9pt', container_padding='6px',
                      container_margin='6px',
                      container_border='1px solid black',
                      toggle_button_font_weight='bold', border_visible=False)
    format_save_figure_options(save_figure_wid, container_padding='6px',
                               container_margin='6px',
                               container_border='1px solid black',
                               toggle_button_font_weight='bold',
                               tab_top_margin='0cm', border_visible=False)

    # update widgets' state for level 0
    update_widgets('', 0)

    # Reset value to enable initial visualization
    figure_options_wid.children[2].value = False


def visualize_fitting_results(fitting_results, figure_size=(7, 7), popup=False,
                              **kwargs):
    r"""
    Widget that allows browsing through a list of fitting results.

    Parameters
    -----------
    fitting_results : `list` of :map:`FittingResult` or subclass
        The list of fitting results to be displayed. Note that the fitting
        results can have different attributes between them, i.e. different
        number of iterations, number of channels etc.

    figure_size : (`int`, `int`), optional
        The initial size of the plotted figures.

    popup : `boolean`, optional
        If enabled, the widget will appear as a popup window.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
    from menpo.image import MaskedImage

    # make sure that fitting_results is a list even with one fitting_result
    if not isinstance(fitting_results, list):
        fitting_results = [fitting_results]

    # check if all fitting_results have gt_shape in order to show the ced button
    show_ced = all(not f.gt_shape is None for f in fitting_results)

    # find number of fitting_results
    n_fitting_results = len(fitting_results)

    # create dictionaries
    iter_str = 'iter_'
    groups_final_dict = dict()
    colour_final_dict = dict()
    groups_final_dict['initial'] = 'Initial shape'
    colour_final_dict['initial'] = 'r'
    groups_final_dict['final'] = 'Final shape'
    colour_final_dict['final'] = 'b'
    groups_final_dict['ground'] = 'Ground-truth shape'
    colour_final_dict['ground'] = 'y'

    # define function that plots errors curve
    def plot_errors_function(name):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get selected image
        im = 0
        if n_fitting_results > 1:
            im = image_number_wid.selected_index

        # select figure
        figure_id = plt.figure(save_figure_wid.figure_id.number)

        # plot errors curve
        plt.plot(range(len(fitting_results[im].errors())),
                 fitting_results[im].errors(), '-bo')
        plt.gca().set_xlim(0, len(fitting_results[im].errors())-1)
        plt.xlabel('Iteration')
        plt.ylabel('Fitting Error')
        plt.title("Fitting error evolution of Image {}".format(im))
        plt.grid("on")

        # set figure size
        x_scale = figure_options_wid.x_scale
        y_scale = figure_options_wid.y_scale
        plt.gcf().set_size_inches([x_scale, y_scale] * np.asarray(figure_size))

        # show figure
        plt.show()

        # save the current figure id
        save_figure_wid.figure_id = figure_id

    # define function that plots displacements curve
    def plot_displacements_function(name):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get selected image
        im = 0
        if n_fitting_results > 1:
            im = image_number_wid.selected_index

        # select figure
        figure_id = plt.figure(save_figure_wid.figure_id.number)

        # plot displacements curve
        d_type = iterations_wid.displacement_type
        if (d_type == 'max' or d_type == 'min' or d_type == 'mean' or
                d_type == 'median'):
            d_curve = fitting_results[im].displacements_stats(stat_type=d_type)
        else:
            all_displacements = fitting_results[im].displacements()
            d_curve = [iteration_displacements[d_type]
                       for iteration_displacements in all_displacements]
        plt.plot(range(len(d_curve)), d_curve, '-bo')
        plt.gca().set_xlim(0, len(d_curve)-1)
        plt.grid("on")
        plt.xlabel('Iteration')

        # set labels
        if d_type == 'max':
            plt.ylabel('Maximum Displacement')
            plt.title("Maximum displacement evolution of Image {}".format(im))
        elif d_type == 'min':
            plt.ylabel('Minimum Displacement')
            plt.title("Minimum displacement evolution of Image {}".format(im))
        elif d_type == 'mean':
            plt.ylabel('Mean Displacement')
            plt.title("Mean displacement evolution of Image {}".format(im))
        elif d_type == 'median':
            plt.ylabel('Median Displacement')
            plt.title("Median displacement evolution of Image {}".format(im))
        else:
            plt.ylabel("Displacement of Point {}".format(d_type))
            plt.title("Point {} displacement evolution of Image {}".format(
                d_type, im))

        # set figure size
        x_scale = figure_options_wid.x_scale
        y_scale = figure_options_wid.y_scale
        plt.gcf().set_size_inches([x_scale, y_scale] * np.asarray(figure_size))

        # show figure
        plt.show()

        # save the current figure id
        save_figure_wid.figure_id = figure_id

    # define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get selected image
        im = 0
        if n_fitting_results > 1:
            im = image_number_wid.selected_index

        # selected mode: final or iterations
        final_enabled = False
        if result_wid.selected_index == 0:
            final_enabled = True

        # update info text widget
        update_info('', error_type_wid.value)

        # get the current figure id
        figure_id = save_figure_wid.figure_id

        # call helper _plot_figure
        if final_enabled:
            new_figure_id = _plot_figure(
                image=fitting_results[im].fitted_image, figure_id=figure_id,
                image_enabled=final_result_wid.show_image,
                landmarks_enabled=True, image_is_masked=False,
                masked_enabled=False, channels=channel_options_wid.channels,
                glyph_enabled=channel_options_wid.glyph_enabled,
                glyph_block_size=channel_options_wid.glyph_block_size,
                glyph_use_negative=channel_options_wid.glyph_use_negative,
                sum_enabled=channel_options_wid.sum_enabled,
                groups=final_result_wid.groups,
                with_labels=[None] * len(final_result_wid.groups),
                groups_colours=colour_final_dict,
                subplots_enabled=final_result_wid.subplots_enabled,
                subplots_titles=groups_final_dict, image_axes_mode=True,
                legend_enabled=final_result_wid.legend_enabled,
                numbering_enabled=final_result_wid.numbering_enabled,
                x_scale=figure_options_wid.x_scale,
                y_scale=figure_options_wid.y_scale,
                axes_visible=figure_options_wid.axes_visible,
                figure_size=figure_size, **kwargs)
        else:
            # create subplot titles dict and colours dict
            groups_dict = dict()
            colour_dict = dict()
            cols = np.random.random([3, len(iterations_wid.groups)])
            for i, group in enumerate(iterations_wid.groups):
                iter_num = group[len(iter_str)::]
                groups_dict[iter_str + iter_num] = "Iteration " + iter_num
                colour_dict[iter_str + iter_num] = cols[:, i]

            # plot
            new_figure_id = _plot_figure(
                image=fitting_results[im].iter_image, figure_id=figure_id,
                image_enabled=iterations_wid.show_image, landmarks_enabled=True,
                image_is_masked=False, masked_enabled=False,
                channels=channel_options_wid.channels,
                glyph_enabled=channel_options_wid.glyph_enabled,
                glyph_block_size=channel_options_wid.glyph_block_size,
                glyph_use_negative=channel_options_wid.glyph_use_negative,
                sum_enabled=channel_options_wid.sum_enabled,
                groups=iterations_wid.groups,
                with_labels=[None] * len(iterations_wid.groups),
                groups_colours=colour_dict,
                subplots_enabled=iterations_wid.subplots_enabled,
                subplots_titles=groups_dict, image_axes_mode=True,
                legend_enabled=iterations_wid.legend_enabled,
                numbering_enabled=iterations_wid.numbering_enabled,
                x_scale=figure_options_wid.x_scale,
                y_scale=figure_options_wid.y_scale,
                axes_visible=figure_options_wid.axes_visible,
                figure_size=figure_size, **kwargs)

        # save the current figure id
        save_figure_wid.figure_id = new_figure_id

    # define function that updates info text
    def update_info(name, value):
        # get selected image
        im = 0
        if n_fitting_results > 1:
            im = image_number_wid.selected_index

        # create output str
        if fitting_results[im].gt_shape is not None:
            info_txt = r"""
                Initial error: {:.4f}
                Final error: {:.4f}
                {} iterations
            """.format(fitting_results[im].initial_error(error_type=value),
                       fitting_results[im].final_error(error_type=value),
                       fitting_results[im].n_iters)
        else:
            info_txt = r"""
                {} iterations
            """.format(fitting_results[im].n_iters)
        if hasattr(fitting_results[im], 'n_levels'):  # Multilevel result
            info_txt += r"""
                {} levels with downscale of {:.1f}
            """.format(fitting_results[im].n_levels,
                       fitting_results[im].downscale)

        info_wid.children[1].value = _raw_info_string_to_latex(info_txt)

    # Create options widgets
    channel_options_wid = channel_options(
        fitting_results[0].fitted_image.n_channels,
        isinstance(fitting_results[0].fitted_image, MaskedImage), plot_function,
        masked_default=False, toggle_show_default=True,
        toggle_show_visible=False)
    figure_options_wid = figure_options(plot_function, scale_default=1.,
                                        show_axes_default=True,
                                        toggle_show_default=True,
                                        toggle_show_visible=False)
    info_wid = info_print(toggle_show_default=True, toggle_show_visible=False)
    initial_figure_id = plt.figure()
    save_figure_wid = save_figure_options(initial_figure_id,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)

    # Create landmark groups checkboxes
    all_groups_keys, all_labels_keys = _extract_groups_labels(
        fitting_results[0].fitted_image)
    final_result_wid = final_result_options(all_groups_keys, plot_function,
                                            title='Final',
                                            show_image_default=True,
                                            subplots_enabled_default=True,
                                            legend_default=True,
                                            numbering_default=False,
                                            toggle_show_default=True,
                                            toggle_show_visible=False)
    iterations_wid = iterations_result_options(
        fitting_results[0].n_iters, not fitting_results[0].gt_shape is None,
        fitting_results[0].fitted_image.landmarks['final'].lms.n_points,
        plot_function, plot_errors_function, plot_displacements_function,
        iter_str=iter_str, title='Iterations', show_image_default=True,
        subplots_enabled_default=False, legend_default=True,
        numbering_default=False, toggle_show_default=True,
        toggle_show_visible=False)
    iterations_wid.children[2].children[4].on_click(plot_errors_function)
    iterations_wid.children[2].children[5].children[0].on_click(
        plot_displacements_function)

    # Create error type radio buttons
    error_type_values = OrderedDict()
    error_type_values['Point-to-point Normalized Mean Error'] = 'me_norm'
    error_type_values['Point-to-point Mean Error'] = 'me'
    error_type_values['RMS Error'] = 'rmse'
    error_type_wid = RadioButtonsWidget(values=error_type_values,
                                        value='me_norm',
                                        description='Error type')
    error_type_wid.on_trait_change(update_info, 'value')
    plot_ced_but = ButtonWidget(description='Plot CED', visible=show_ced)
    error_wid = ContainerWidget(children=[error_type_wid, plot_ced_but])

    # define function that updates options' widgets state
    def update_widgets(name, value):
        # get new groups and labels, update landmark options and format them
        group_keys, labels_keys = _extract_groups_labels(
            fitting_results[value].fitted_image)
        # update channel options
        update_channel_options(
            channel_options_wid,
            n_channels=fitting_results[value].fitted_image.n_channels,
            image_is_masked=isinstance(fitting_results[value].fitted_image,
                                       MaskedImage))
        # update final result's options
        update_final_result_options(final_result_wid, group_keys, plot_function)
        # update iterations result's options
        update_iterations_result_options(
            iterations_wid, fitting_results[value].n_iters,
            not fitting_results[value].gt_shape is None,
            fitting_results[value].fitted_image.landmarks['final'].lms.n_points,
            iter_str=iter_str)

    # Create final widget
    options_wid = TabWidget(children=[channel_options_wid, figure_options_wid])
    result_wid = TabWidget(children=[final_result_wid, iterations_wid])
    result_wid.on_trait_change(plot_function, 'selected_index')
    if n_fitting_results > 1:
        # image selection slider
        image_number_wid = animation_options(
            index_min_val=0, index_max_val=n_fitting_results-1,
            plot_function=plot_function, update_function=update_widgets,
            index_step=1, index_default=0,
            index_description='Image Number', index_minus_description='<',
            index_plus_description='>', index_style='buttons',
            index_text_editable=True, loop_default=True, interval_default=0.3,
            toggle_show_title='Image Options', toggle_show_default=True,
            toggle_show_visible=False)

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
        tab_wid = TabWidget(children=[info_wid, result_wid, options_wid,
                                      error_wid, save_figure_wid])
        tab_wid.on_trait_change(save_fig_tab_fun, 'selected_index')
        wid = ContainerWidget(children=[image_number_wid, tab_wid])
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
        wid = TabWidget(children=[info_wid, result_wid, options_wid, error_wid,
                                  save_figure_wid])
        tab_titles = ['Image info', 'Result', 'Options', 'Error type',
                      'Save figure']
        button_title = 'Fitting Result Menu'
    # create popup widget if asked
    if popup:
        wid = PopupWidget(children=[wid], button_text=button_title)

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
        plot_ced_widget = plot_ced(errors, figure_size=(9, 5), popup=True,
                                   error_type=error_type, error_range=None,
                                   legend_entries=['Final Fitting',
                                                   'Initialization'],
                                   return_widget=True)

        # If another tab is selected, then close the widget.
        def close_plot_ced_fun(name, value):
            if value != 3:
                plot_ced_widget.close()
                plot_ced_but.visible = True
        if n_fitting_results > 1:
            tab_wid.on_trait_change(close_plot_ced_fun, 'selected_index')
        else:
            if popup:
                wid.children[0].on_trait_change(close_plot_ced_fun,
                                                'selected_index')
            else:
                wid.on_trait_change(close_plot_ced_fun, 'selected_index')

        # If another error type, then close the widget
        def close_plot_ced_fun_2(name, value):
            plot_ced_widget.close()
            plot_ced_but.visible = True
        error_type_wid.on_trait_change(close_plot_ced_fun_2, 'value')
    plot_ced_but.on_click(plot_ced_fun)

    # display final widget
    display(wid)

    # set final tab titles
    if popup:
        if n_fitting_results > 1:
            for (k, tl) in enumerate(tab_titles):
                wid.children[0].children[1].set_title(k, tl)
        else:
            for (k, tl) in enumerate(tab_titles):
                wid.children[0].set_title(k, tl)
    else:
        if n_fitting_results > 1:
            for (k, tl) in enumerate(tab_titles):
                wid.children[1].set_title(k, tl)
        else:
            for (k, tl) in enumerate(tab_titles):
                wid.set_title(k, tl)
    result_wid.set_title(0, 'Final Fitting')
    result_wid.set_title(1, 'Iterations')
    options_wid.set_title(0, 'Channels')
    options_wid.set_title(1, 'Figure')

    # format options' widgets
    if n_fitting_results > 1:
        format_animation_options(image_number_wid, index_text_width='0.5cm',
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
    format_figure_options(figure_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False)
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
    figure_options_wid.children[2].value = False


def plot_ced(errors, figure_size=(9, 5), popup=False, error_type='me_norm',
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

    popup : `boolean`, optional
        If enabled, the widget will appear as a popup window.

    error_type : `str` ``{'me_norm', 'me', 'rmse'}``, optional
        Specifies the type of the provided errors.

    error_range : `list` of `float` with length 3, optional
        Specifies the horizontal axis range, i.e.
        error_range[0] = min_error
        error_range[1] = max_error
        error_range[2] = error_step
        If None, then
        error_range = [0., 0.101, 0.005] for error_type = 'me_norm'
        error_range = [0., 20., 1.] for error_type = 'me'
        error_range = [0., 20., 1.] for error_type = 'rmse'

    legend_entries : `list` of `str`
        The entries of the legend. The list must have the same length as errors.
        If None, the entries will have the form 'Curve %d'.

    return_widget : `boolean`, optional
        If True, the widget object will be returned so that it can be used as a
        part of a bigger widget. If False, the widget object is not returned, it
        is just visualized.
    """
    from menpofit.fittingresult import compute_cumulative_error

    # make sure that images is a list even with one image member
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
    if error_range is None:
        if error_type == 'me_norm':
            error_range = [0., 0.101, 0.005]
            x_axis_limit_initial_value = 0.05
            x_label_initial_value = 'Normalized Point-to-Point Error'
        elif error_type == 'me' or error_type == 'rmse':
            error_range = [0., 20., 0.5]
            x_axis_limit_initial_value = 5.
            x_label_initial_value = 'Point-to-Point Error'
    else:
        x_axis_limit_initial_value = (error_range[1] + error_range[0]) / 2
    x_axis = np.arange(error_range[0], error_range[1], error_range[2])

    # compute cumulative error curves
    ceds = [compute_cumulative_error(e, x_axis) for e in errors]
    x_axis = [x_axis] * len(ceds)

    # initialize plot options dictionaries and legend entries
    colors = [np.random.random((3,)) for _ in range(n_curves)]
    plot_options_list = []
    for k in range(n_curves):
        plot_options_list.append({'show_line':True,
                                  'linewidth':2,
                                  'linecolor':colors[k],
                                  'linestyle':'-',
                                  'show_marker':True,
                                  'markersize':10,
                                  'markerfacecolor':'w',
                                  'markeredgecolor':colors[k],
                                  'markerstyle':'s',
                                  'markeredgewidth':1,
                                  'legend_entry':legend_entries[k]})

    # define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get the current figure id
        figure_id = save_figure_wid.figure_id

        # plot the graph with the selected options
        new_figure_id = _plot_graph(
            figure_id, horizontal_axis_values=x_axis, vertical_axis_values=ceds,
            plot_options_list=plot_options_wid.selected_options,
            legend_visible=legend_visible.value,
            grid_visible=grid_visible.value, gridlinestyle=gridlinestyle.value,
            x_limit=x_axis_limit.value, y_limit=y_axis_limit.value,
            title=title.value, x_label=x_label.value, y_label=y_label.value,
            x_scale=fig.x_scale, y_scale=fig.y_scale, figure_size=figure_size,
            axes_fontsize=axes_fontsize.value,
            labels_fontsize=labels_fontsize.value)

        # save the current figure id
        save_figure_wid.figure_id = new_figure_id

    # create options widgets
    # x label, y label, title container
    x_label = TextWidget(description='Horizontal axis label',
                         value=x_label_initial_value)
    y_label = TextWidget(description='Vertical axis label',
                         value='Images Proportion')
    title = TextWidget(description='Figure title',
                       value='Cumulative error ditribution')
    labels_wid = ContainerWidget(children=[x_label, y_label, title])

    # figure size
    fig = figure_options_two_scales(plot_function, x_scale_default=1.,
                                    y_scale_default=1., coupled_default=False,
                                    show_axes_default=True,
                                    toggle_show_default=True,
                                    figure_scales_bounds=(0.1, 2),
                                    figure_scales_step=0.1,
                                    figure_scales_visible=True,
                                    show_axes_visible=False,
                                    toggle_show_visible=False)
    # fontsizes
    labels_fontsize = FloatTextWidget(description='Labels fontsize', value=12.)
    axes_fontsize = FloatTextWidget(description='Axes fontsize', value=12.)
    fontsize_wid = ContainerWidget(children=[labels_fontsize, axes_fontsize])

    # checkboxes
    grid_visible = CheckboxWidget(description='Grid visible', value=False)
    gridlinestyle_dict = OrderedDict()
    gridlinestyle_dict['solid'] = '-'
    gridlinestyle_dict['dashed'] = '--'
    gridlinestyle_dict['dash-dot'] = '-.'
    gridlinestyle_dict['dotted'] = ':'
    gridlinestyle = DropdownWidget(values=gridlinestyle_dict,
                                   value=':',
                                   description='Grid style', disabled=False)

    def gridlinestyle_visibility(name, value):
        gridlinestyle.disabled = not value
    grid_visible.on_trait_change(gridlinestyle_visibility, 'value')
    legend_visible = CheckboxWidget(description='Legend visible', value=True)
    checkbox_wid = ContainerWidget(children=[grid_visible, gridlinestyle,
                                             legend_visible])

    # container of various options
    tmp_various_wid = ContainerWidget(children=[fontsize_wid, checkbox_wid])
    various_wid = ContainerWidget(children=[fig, tmp_various_wid])

    # axis limits
    y_axis_limit = FloatSliderWidget(min=0., max=1.1, step=0.1,
                                     description='Y axis limit', value=1.)
    x_axis_limit = FloatSliderWidget(min=error_range[0] + error_range[2],
                                     max=error_range[1],
                                     step=error_range[2],
                                     description='X axis limit',
                                     value=x_axis_limit_initial_value)
    axis_limits_wid = ContainerWidget(children=[x_axis_limit, y_axis_limit])

    # accordion widget
    figure_wid = AccordionWidget(children=[axis_limits_wid, labels_wid,
                                           various_wid])
    figure_wid.set_title(0, 'Axes Limits')
    figure_wid.set_title(1, 'Labels and Title')
    figure_wid.set_title(2, 'Figure Size, Grid and Legend')

    # per curve options
    plot_options_wid = plot_options(plot_options_list,
                                    plot_function=plot_function,
                                    toggle_show_visible=False,
                                    toggle_show_default=True)

    # save figure options widget
    # create figure and store its id
    initial_figure_id = plt.figure()
    save_figure_wid = save_figure_options(initial_figure_id,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)

    # assign plot function
    x_label.on_trait_change(plot_function, 'value')
    y_label.on_trait_change(plot_function, 'value')
    title.on_trait_change(plot_function, 'value')
    grid_visible.on_trait_change(plot_function, 'value')
    gridlinestyle.on_trait_change(plot_function, 'value')
    legend_visible.on_trait_change(plot_function, 'value')
    y_axis_limit.on_trait_change(plot_function, 'value')
    x_axis_limit.on_trait_change(plot_function, 'value')
    labels_fontsize.on_trait_change(plot_function, 'value')
    axes_fontsize.on_trait_change(plot_function, 'value')

    # create final widget
    wid = TabWidget(children=[figure_wid, plot_options_wid,
                              save_figure_wid])

    # create popup widget if asked
    if popup:
        wid = PopupWidget(children=[wid], button_text='CED Menu')

    # display final widget
    display(wid)

    # set final tab titles
    tab_titles = ['Figure options', 'Per Curve options', 'Save figure']
    if n_curves == 1:
        tab_titles[1] = 'Curve options'
    if popup:
        for (k, tl) in enumerate(tab_titles):
            wid.children[0].set_title(k, tl)
    else:
        for (k, tl) in enumerate(tab_titles):
            wid.set_title(k, tl)

    # format options' widgets
    labels_wid.add_class('align-end')
    axis_limits_wid.add_class('align-start')
    fontsize_wid.add_class('align-end')
    fontsize_wid.set_css('margin-right', '1cm')
    checkbox_wid.add_class('align-end')
    tmp_various_wid.remove_class('vbox')
    tmp_various_wid.add_class('hbox')
    format_plot_options(plot_options_wid, container_padding='1px',
                        container_margin='1px',
                        container_border='1px solid black',
                        toggle_button_font_weight='bold', border_visible=False,
                        suboptions_border_visible=True)
    format_figure_options_two_scales(fig, container_padding='6px',
                                     container_margin='6px',
                                     container_border='1px solid black',
                                     toggle_button_font_weight='bold',
                                     border_visible=False)
    format_save_figure_options(save_figure_wid, container_padding='6px',
                               container_margin='6px',
                               container_border='1px solid black',
                               toggle_button_font_weight='bold',
                               tab_top_margin='0cm',
                               border_visible=False)

    # Reset value to trigger initial visualization
    grid_visible.value = True

    # return widget object if asked
    if return_widget:
        return wid
