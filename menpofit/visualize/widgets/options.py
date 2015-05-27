from collections import OrderedDict

from menpo.visualize.widgets.compatibility import add_class, remove_class
from menpo.visualize.widgets.tools import (colour_selection,
                                           format_colour_selection)
from menpo.visualize.widgets.options import (animation_options,
                                             format_animation_options,
                                             _compare_groups_and_labels)


def model_parameters(n_params, plot_function=None, params_str='',
                     mode='multiple', params_bounds=(-3., 3.),
                     plot_eig_visible=True, plot_eig_function=None,
                     toggle_show_default=True, toggle_show_visible=True,
                     toggle_show_name='Parameters'):
    r"""
    Creates a widget with Model Parameters. Specifically, it has:
        1) A slider for each parameter if mode is 'multiple'.
        2) A single slider and a drop down menu selection if mode is 'single'.
        3) A reset button.
        4) A button and two radio buttons for plotting the eigenvalues variance
           ratio.

    The structure of the widgets is the following:
        model_parameters_wid.children = [toggle_button, parameters_and_reset]
        parameters_and_reset.children = [parameters_widgets, reset]
        If plot_eig_visible is True:
            reset = [plot_eigenvalues, reset_button]
        Else:
            reset = reset_button
        If mode is single:
            parameters_widgets.children = [drop_down_menu, slider]
        If mode is multiple:
            parameters_widgets.children = [all_sliders]

    The returned widget saves the selected values in the following fields:
        model_parameters_wid.parameters_values
        model_parameters_wid.mode
        model_parameters_wid.plot_eig_visible

    To fix the alignment within this widget please refer to
    `format_model_parameters()` function.

    To update the state of this widget, please refer to
    `update_model_parameters()` function.

    Parameters
    ----------
    n_params : `int`
        The number of principal components to use for the sliders.

    plot_function : `function` or None, optional
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.

    params_str : `str`, optional
        The string that will be used for each parameters name.

    mode : 'single' or 'multiple', optional
        If single, only a single slider is constructed along with a drop down
        menu.
        If multiple, a slider is constructed for each parameter.

    params_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.

    plot_eig_visible : `boolean`, optional
        Defines whether the options for plotting the eigenvalues variance ratio
        will be visible upon construction.

    plot_eig_function : `function` or None, optional
        The plot function that is executed when the plot eigenvalues button is
        clicked. If None, then nothing is assigned.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.

    toggle_show_name : `str`, optional
        The name of the toggle button.
    """
    import IPython.html.widgets as ipywidgets
    # If only one slider requested, then set mode to multiple
    if n_params == 1:
        mode = 'multiple'

    # Create all necessary widgets
    but = ipywidgets.ToggleButton(description=toggle_show_name,
                                  value=toggle_show_default,
                                  visible=toggle_show_visible)
    reset_button = ipywidgets.Button(description='Reset')
    if mode == 'multiple':
        sliders = [ipywidgets.FloatSlider(
            description="{}{}".format(params_str, p),
            min=params_bounds[0], max=params_bounds[1],
            value=0.)
                   for p in range(n_params)]
        parameters_wid = ipywidgets.Box(children=sliders)
    else:
        vals = OrderedDict()
        for p in range(n_params):
            vals["{}{}".format(params_str, p)] = p
        slider = ipywidgets.FloatSlider(description='',
                                        min=params_bounds[0],
                                        max=params_bounds[1], value=0.)
        dropdown_params = ipywidgets.Dropdown(options=vals)
        parameters_wid = ipywidgets.Box(
            children=[dropdown_params, slider])

    # Group widgets
    if plot_eig_visible:
        plot_button = ipywidgets.Button(description='Plot eigenvalues')
        if plot_eig_function is not None:
            plot_button.on_click(plot_eig_function)
        plot_and_reset = ipywidgets.Box(
            children=[plot_button, reset_button])
        params_and_reset = ipywidgets.Box(children=[parameters_wid,
                                                    plot_and_reset])
    else:
        params_and_reset = ipywidgets.Box(children=[parameters_wid,
                                                    reset_button])

    # Widget container
    model_parameters_wid = ipywidgets.Box(
        children=[but, params_and_reset])

    # Save mode and parameters values
    model_parameters_wid.parameters_values = [0.0] * n_params
    model_parameters_wid.mode = mode
    model_parameters_wid.plot_eig_visible = plot_eig_visible

    # set up functions
    if mode == 'single':
        # assign slider value to parameters values list
        def save_slider_value(name, value):
            model_parameters_wid.parameters_values[dropdown_params.value] = \
                value
        slider.on_trait_change(save_slider_value, 'value')

        # set correct value to slider when drop down menu value changes
        def set_slider_value(name, value):
            slider.value = model_parameters_wid.parameters_values[value]
        dropdown_params.on_trait_change(set_slider_value, 'value')

        # assign main plotting function when slider value changes
        if plot_function is not None:
            slider.on_trait_change(plot_function, 'value')
    else:
        # assign slider value to parameters values list
        def save_slider_value_from_id(description, name, value):
            i = int(description[len(params_str)::])
            model_parameters_wid.parameters_values[i] = value

        # partial function that helps get the widget's description str
        def partial_widget(description):
            return lambda name, value: save_slider_value_from_id(description,
                                                                 name, value)

        # assign saving values and main plotting function to all sliders
        for w in parameters_wid.children:
            # The widget (w) is lexically scoped and so we need a way of
            # ensuring that we don't just receive the final value of w at every
            # iteration. Therefore we create another lambda function that
            # creates a new lexical scoping so that we can ensure the value of w
            # is maintained (as x) at each iteration.
            # In JavaScript, we would just use the 'let' keyword...
            w.on_trait_change(partial_widget(w.description), 'value')
            if plot_function is not None:
                w.on_trait_change(plot_function, 'value')

    # reset function
    def reset_params(name):
        model_parameters_wid.parameters_values = \
            [0.0] * len(model_parameters_wid.parameters_values)
        if mode == 'multiple':
            for ww in parameters_wid.children:
                ww.value = 0.
        else:
            parameters_wid.children[0].value = 0
            parameters_wid.children[1].value = 0.
    reset_button.on_click(reset_params)

    # Toggle button function
    def show_options(name, value):
        params_and_reset.visible = value
    show_options('', toggle_show_default)
    but.on_trait_change(show_options, 'value')

    return model_parameters_wid


def format_model_parameters(model_parameters_wid, container_padding='6px',
                            container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold',
                            border_visible=True):
    r"""
    Function that corrects the align (style format) of a given model_parameters
    widget. Usage example:
        model_parameters_wid = model_parameters()
        display(model_parameters_wid)
        format_model_parameters(model_parameters_wid)

    Parameters
    ----------
    model_parameters_wid :
        The widget object generated by the `model_parameters()` function.

    container_padding : `str`, optional
        The padding around the widget, e.g. '6px'

    container_margin : `str`, optional
        The margin around the widget, e.g. '6px'

    container_border : `str`, optional
        The border around the widget, e.g. '1px solid black'

    toggle_button_font_weight : `str`
        The font weight of the toggle button, e.g. 'bold'

    border_visible : `boolean`, optional
        Defines whether to draw the border line around the widget.
    """
    if model_parameters_wid.mode == 'single':
        # align drop down menu and slider
        remove_class(model_parameters_wid.children[1].children[0], 'vbox')
        add_class(model_parameters_wid.children[1].children[0], 'hbox')
    else:
        # align sliders
        add_class(model_parameters_wid.children[1].children[0], 'start')

    # align reset button to right
    if model_parameters_wid.plot_eig_visible:
        remove_class(model_parameters_wid.children[1].children[1], 'vbox')
        add_class(model_parameters_wid.children[1].children[1], 'hbox')
    add_class(model_parameters_wid.children[1], 'align-end')

    # set toggle button font bold
    model_parameters_wid.children[0].font_weight = toggle_button_font_weight

    # margin and border around plot_eigenvalues widget
    if model_parameters_wid.plot_eig_visible:
        model_parameters_wid.children[1].children[1].children[0].margin_right = container_margin

    # margin and border around container widget
    model_parameters_wid.padding = container_padding
    model_parameters_wid.margin = container_margin
    if border_visible:
        model_parameters_wid.border = container_border


def update_model_parameters(model_parameters_wid, n_params, plot_function=None,
                            params_str=''):
    r"""
    Function that updates the state of a given model_parameters widget if the
    requested number of parameters has changed. Usage example:
        model_parameters_wid = model_parameters(n_params=5)
        display(model_parameters_wid)
        format_model_parameters(model_parameters_wid)
        update_model_parameters(model_parameters_wid, 3)

    Parameters
    ----------
    model_parameters_wid :
        The widget object generated by the `model_parameters()` function.

    n_params : `int`
        The requested number of parameters.

    plot_function : `function` or None, optional
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.

    params_str : `str`, optional
        The string that will be used for each parameters name.
    """
    import IPython.html.widgets as ipywidgets

    if model_parameters_wid.mode == 'multiple':
        # get the number of enabled parameters (number of sliders)
        enabled_params = len(model_parameters_wid.children[1].children[0].children)
        if n_params != enabled_params:
            # reset all parameters values
            model_parameters_wid.parameters_values = [0.0] * n_params
            # get params_bounds
            pb = [model_parameters_wid.children[1].children[0].children[0].min,
                  model_parameters_wid.children[1].children[0].children[0].max]
            # create sliders widgets
            sliders = [ipywidgets.FloatSlider(
                            description="{}{}".format(params_str,
                                                      p),
                            min=pb[0], max=pb[1], value=0.)
                       for p in range(n_params)]
            # assign sliders to container
            model_parameters_wid.children[1].children[0].children = sliders

            # assign slider value to parameters values list
            def save_slider_value_from_id(description, name, value):
                i = int(description[len(params_str)::])
                model_parameters_wid.parameters_values[i] = value

            # partial function that helps get the widget's description str
            def partial_widget(description):
                return lambda name, value: save_slider_value_from_id(
                    description,
                    name, value)

            # assign saving values and main plotting function to all sliders
            for w in model_parameters_wid.children[1].children[0].children:
                # The widget (w) is lexically scoped and so we need a way of
                # ensuring that we don't just receive the final value of w at
                # every iteration. Therefore we create another lambda function
                # that creates a new lexical scoping so that we can ensure the
                # value of w is maintained (as x) at each iteration
                # In JavaScript, we would just use the 'let' keyword...
                w.on_trait_change(partial_widget(w.description), 'value')
                if plot_function is not None:
                    w.on_trait_change(plot_function, 'value')
    else:
        # get the number of enabled parameters (len of list of drop down menu)
        enabled_params = len(
            model_parameters_wid.children[1].children[0].children[0].values)
        if n_params != enabled_params:
            # reset all parameters values
            model_parameters_wid.parameters_values = [0.0] * n_params
            # change drop down menu values
            vals = OrderedDict()
            for p in range(n_params):
                vals["{}{}".format(params_str, p)] = p
            model_parameters_wid.children[1].children[0].children[0].options = \
                vals
            # set initial value to the first and slider value to zero
            model_parameters_wid.children[1].children[0].children[0].value = \
                vals["{}{}".format(params_str, 0)]
            model_parameters_wid.children[1].children[0].children[1].value = 0.


def final_result_options(final_result_options_default, plot_function=None,
                         title='Final Result', toggle_show_default=True,
                         toggle_show_visible=True):
    r"""
    Creates a widget with Final Result Options. Specifically, it has:
        1) A set of toggle buttons representing usually the initial, final and
           ground truth shapes.
        2) A checkbox that controls the rendering of the image.
        3) A set of radio buttons that define whether subplots are enabled.
        4) A toggle button that controls the visibility of all the above, i.e.
           the final result options.

    The structure of the widgets is the following:
        final_result_wid.children = [toggle_button, shapes_toggle_buttons,
                                     options]
        options.children = [plot_mode_radio_buttons, show_image_checkbox]

    The returned widget saves the selected values in the following fields:
        final_result_wid.selected_values

    To fix the alignment within this widget please refer to
    `format_final_result_options()` function.

    To update the state of this widget, please refer to
    `update_final_result_options()` function.

    Parameters
    ----------
    final_result_options_default : `dict`
        The default options. For example:
            final_result_options_default = {'all_groups': ['initial', 'final',
                                                           'ground'],
                                            'selected_groups': ['final'],
                                            'render_image': True,
                                            'subplots_enabled': True}
    plot_function : `function` or None, optional
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.
    title : `str`, optional
        The title of the widget printed at the toggle button.
    toggle_show_default : `bool`, optional
        Defines whether the options will be visible upon construction.
    toggle_show_visible : `bool`, optional
        The visibility of the toggle button.
    """
    import IPython.html.widgets as ipywidgets
    # Toggle button that controls options' visibility
    but = ipywidgets.ToggleButton(description=title,
                                        value=toggle_show_default,
                                        visible=toggle_show_visible)

    # Create widgets
    shapes_checkboxes = [ipywidgets.Latex(value='Select shape:')]
    for group in final_result_options_default['all_groups']:
        t = ipywidgets.ToggleButton(
            description=group,
            value=group in final_result_options_default['selected_groups'])
        shapes_checkboxes.append(t)
    render_image = ipywidgets.Checkbox(
        description='Render image',
        value=final_result_options_default['render_image'])
    mode = ipywidgets.RadioButtons(
        description='Plot mode:', options={'Single': False, 'Multiple': True},
        value=final_result_options_default['subplots_enabled'])

    # Group widgets
    shapes_wid = ipywidgets.Box(children=shapes_checkboxes)
    opts = ipywidgets.Box(children=[mode, render_image])

    # Widget container
    final_result_wid = ipywidgets.Box(children=[but, shapes_wid,
                                                            opts])

    # Initialize variables
    final_result_wid.selected_values = final_result_options_default

    # Groups function
    def groups_fun(name, value):
        final_result_wid.selected_values['selected_groups'] = []
        for i in shapes_wid.children[1::]:
            if i.value:
                final_result_wid.selected_values['selected_groups'].\
                    append(str(i.description))
    for w in shapes_wid.children[1::]:
        w.on_trait_change(groups_fun, 'value')

    # Render image function
    def render_image_fun(name, value):
        final_result_wid.selected_values['render_image'] = value
    render_image.on_trait_change(render_image_fun, 'value')

    # Plot mode function
    def plot_mode_fun(name, value):
        final_result_wid.selected_values['subplots_enabled'] = value
    mode.on_trait_change(plot_mode_fun, 'value')

    # Toggle button function
    def show_options(name, value):
        shapes_wid.visible = value
        opts.visible = value
    show_options('', toggle_show_default)
    but.on_trait_change(show_options, 'value')

    # assign plot_function
    if plot_function is not None:
        render_image.on_trait_change(plot_function, 'value')
        mode.on_trait_change(plot_function, 'value')
        for w in shapes_wid.children[1::]:
            w.on_trait_change(plot_function, 'value')

    return final_result_wid


def format_final_result_options(final_result_wid, container_padding='6px',
                                container_margin='6px',
                                container_border='1px solid black',
                                toggle_button_font_weight='bold',
                                border_visible=True):
    r"""
    Function that corrects the align (style format) of a given
    final_result_options widget. Usage example:
        final_result_options_default = {'all_groups': ['initial', 'final',
                                                       'ground'],
                                        'selected_groups': ['final'],
                                        'render_image': True,
                                        'subplots_enabled': True}
        final_result_wid = final_result_options(final_result_options_default)
        display(final_result_wid)
        format_final_result_options(final_result_wid)

    Parameters
    ----------
    final_result_wid :
        The widget object generated by the `final_result_options()` function.
    container_padding : `str`, optional
        The padding around the widget, e.g. '6px'
    container_margin : `str`, optional
        The margin around the widget, e.g. '6px'
    container_border : `str`, optional
        The border around the widget, e.g. '1px solid black'
    toggle_button_font_weight : `str`
        The font weight of the toggle button, e.g. 'bold'
    border_visible : `boolean`, optional
        Defines whether to draw the border line around the widget.
    """
    # align shapes toggle buttons
    remove_class(final_result_wid.children[1], 'vbox')
    add_class(final_result_wid.children[1], 'hbox')
    add_class(final_result_wid.children[1], 'align-center')
    final_result_wid.children[1].children[0].margin_right = container_margin

    # align mode and legend options
    remove_class(final_result_wid.children[2], 'vbox')
    add_class(final_result_wid.children[2], 'hbox')
    final_result_wid.children[2].children[0].margin_right = '20px'

    # set toggle button font bold
    final_result_wid.children[0].font_weight = toggle_button_font_weight
    final_result_wid.children[1].margin_top = container_margin

    # margin and border around container widget
    final_result_wid.padding = container_padding
    final_result_wid.margin = container_margin
    if border_visible:
        final_result_wid.border = container_border


def update_final_result_options(final_result_wid, group_keys, plot_function):
    r"""
    Function that updates the state of a given final_result_options widget if
    the group keys of an image has changed. Usage example:
        final_result_options_default = {'all_groups': ['group1', 'group2'],
                                        'selected_groups': ['group1'],
                                        'render_image': True,
                                        'subplots_enabled': True}
        final_result_wid = final_result_options(final_result_options_default)
        display(final_result_wid)
        format_final_result_options(final_result_wid)
        update_final_result_options(final_result_wid, group_keys=['group3'])
        format_final_result_options(final_result_wid)

    Note that the `format_final_result_options()` function needs to be called
    again after the `update_final_result_options()` function.

    Parameters
    ----------
    final_result_wid :
        The widget object generated by the `final_result_options()` function.
    group_keys : `list` of `str`
        A list of the available landmark groups.
    plot_function : `function` or None
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.
    """
    import IPython.html.widgets as ipywidgets
    # check if the new group_keys are the same as the old ones
    if not _compare_groups_and_labels(
            group_keys, [], final_result_wid.selected_values['all_groups'], []):
        # Create all necessary widgets
        shapes_checkboxes = [ipywidgets.Latex(value='Select shape:')]
        for group in group_keys:
            t = ipywidgets.ToggleButton(description=group, value=True)
            shapes_checkboxes.append(t)

        # Group widgets
        final_result_wid.children[1].children = shapes_checkboxes

        # Initialize output variables
        final_result_wid.selected_values['all_groups'] = group_keys
        final_result_wid.selected_values['selected_groups'] = group_keys

        # Groups function
        def groups_fun(name, value):
            final_result_wid.selected_values['selected_groups'] = []
            for i in final_result_wid.children[1].children[1::]:
                if i.value:
                    final_result_wid.selected_values['selected_groups'].append(str(i.description))
        for w in final_result_wid.children[1].children[1::]:
            w.on_trait_change(groups_fun, 'value')

        # Toggle button function
        def show_options(name, value):
            final_result_wid.children[1].visible = value
            final_result_wid.children[2].visible = value
        show_options('', final_result_wid.children[0].value)
        final_result_wid.children[0].on_trait_change(show_options, 'value')

        # assign plot_function
        if plot_function is not None:
            final_result_wid.children[2].children[0].on_trait_change(
                plot_function, 'value')
            final_result_wid.children[2].children[1].on_trait_change(
                plot_function, 'value')
            for w in final_result_wid.children[1].children[1::]:
                w.on_trait_change(plot_function, 'value')


def iterations_result_options(iterations_result_options_default,
                              plot_function=None, plot_errors_function=None,
                              plot_displacements_function=None,
                              title='Iterations Result',
                              toggle_show_default=True,
                              toggle_show_visible=True):
    r"""
    Creates a widget with Iterations Result Options. Specifically, it has:
        1) Two radio buttons that select an options mode, depending on whether
           the user wants to visualize iterations in ``Animation`` or ``Static``
           mode.
        2) If mode is ``Animation``, an animation options widget appears.
           If mode is ``Static``, the iterations range is selected by two
           sliders and there is an update plot button.
        3) A checkbox that controls the visibility of the image.
        4) A set of radio buttons that define whether subplots are enabled.
        5) A button to plot the error evolution.
        6) A button to plot the landmark points' displacement.
        7) A drop down menu to select which displacement to plot.
        8) A toggle button that controls the visibility of all the above, i.e.
           the final result options.

    The structure of the widgets is the following:
        iterations_result_wid.children = [toggle_button, all_options]
        all_options.children = [iterations_mode_and_sliders, options]
        iterations_mode_and_sliders.children = [iterations_mode_radio_buttons,
                                                all_sliders]
        all_sliders.children = [animation_slider, first_slider, second_slider,
                                update_and_axes]
        update_and_axes.children = [same_axes_checkbox, update_button]
        options.children = [render_image_checkbox, plot_errors_button,
                            plot_displacements]
        plot_displacements.children = [plot_displacements_button,
                                       plot_displacements_drop_down_menu]

    The returned widget saves the selected values in the following fields:
        iterations_result_wid.selected_values

    To fix the alignment within this widget please refer to
    `format_iterations_result_options()` function.

    To update the state of this widget, please refer to
    `update_iterations_result_options()` function.

    Parameters
    ----------
    iterations_result_options_default : `dict`
        The default options. For example:
            iterations_result_options_default = {'n_iters': 10,
                                                 'image_has_gt_shape': True,
                                                 'n_points': 68,
                                                 'iter_str': 'iter_',
                                                 'selected_groups': [0],
                                                 'render_image': True,
                                                 'subplots_enabled': True,
                                                 'displacement_type': 'mean'}
    plot_function : `function` or None, optional
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.
    plot_errors_function : `function` or None, optional
        The plot function that is executed when the 'Plot Errors' button is
        pressed.
        If None, then nothing is assigned.
    plot_displacements_function : `function` or None, optional
        The plot function that is executed when the 'Plot Displacements' button
        is pressed.
        If None, then nothing is assigned.
    title : `str`, optional
        The title of the widget printed at the toggle button.
    toggle_show_default : `bool`, optional
        Defines whether the options will be visible upon construction.
    toggle_show_visible : `bool`, optional
        The visibility of the toggle button.
    """
    import IPython.html.widgets as ipywidgets
    # Create all necessary widgets
    but = ipywidgets.ToggleButton(description=title,
                                        value=toggle_show_default,
                                        visible=toggle_show_visible)
    iterations_mode = ipywidgets.RadioButtons(
        options={'Animation': 'animation', 'Static': 'static'},
        value='animation', description='Mode:', visible=toggle_show_default)
    # Don't assign the plot function to the animation_wid at this point. We
    # first need to assign the get_groups function and then the plot_function()
    # for synchronization reasons.
    index_selection_default = {
        'min': 0, 'max': iterations_result_options_default['n_iters'] - 1,
        'step': 1, 'index': 0}
    animation_wid = animation_options(
        index_selection_default, plot_function=None, update_function=None,
        index_description='Iteration', index_style='slider', loop_default=False,
        interval_default=0.2, toggle_show_default=toggle_show_default,
        toggle_show_visible=False)
    first_slider_wid = ipywidgets.IntSlider(
        min=0, max=iterations_result_options_default['n_iters'] - 1, step=1,
        value=0, description='From', visible=False)
    second_slider_wid = ipywidgets.IntSlider(
        min=0, max=iterations_result_options_default['n_iters'] - 1, step=1,
        value=iterations_result_options_default['n_iters'] - 1,
        description='To', visible=False)
    same_axes = ipywidgets.Checkbox(
        description='Same axes',
        value=not iterations_result_options_default['subplots_enabled'],
        visible=False)
    update_but = ipywidgets.Button(description='Update Plot',
                                         visible=False)
    render_image = ipywidgets.Checkbox(
        description='Render image',
        value=iterations_result_options_default['render_image'])
    plot_errors_button = ipywidgets.Button(description='Plot Errors')
    plot_displacements_button = ipywidgets.Button(
        description='Plot Displacements')
    dropdown_menu = OrderedDict()
    dropdown_menu['mean'] = 'mean'
    dropdown_menu['median'] = 'median'
    dropdown_menu['max'] = 'max'
    dropdown_menu['min'] = 'min'
    for p in range(iterations_result_options_default['n_points']):
        dropdown_menu["point {}".format(p)] = p
    plot_displacements_menu = ipywidgets.Dropdown(
        options=dropdown_menu,
        value=iterations_result_options_default['displacement_type'])

    # if just one iteration, disable multiple options
    if iterations_result_options_default['n_iters'] == 1:
        iterations_mode.value = 'animation'
        iterations_mode.disabled = True
        first_slider_wid.disabled = True
        animation_wid.children[1].children[0].disabled = True
        animation_wid.children[1].children[1].children[0].children[0].\
            disabled = True
        animation_wid.children[1].children[1].children[0].children[1].\
            disabled = True
        animation_wid.children[1].children[1].children[0].children[2].\
            disabled = True
        second_slider_wid.disabled = True
        plot_errors_button.disabled = True
        plot_displacements_button.disabled = True
        plot_displacements_menu.disabled = True

    # Group widgets
    update_and_subplots = ipywidgets.Box(
        children=[same_axes, update_but])
    sliders = ipywidgets.Box(
        children=[animation_wid, first_slider_wid, second_slider_wid,
                  update_and_subplots])
    iterations_mode_and_sliders = ipywidgets.Box(
        children=[iterations_mode, sliders])
    plot_displacements = ipywidgets.Box(
        children=[plot_displacements_button, plot_displacements_menu])
    opts = ipywidgets.Box(
        children=[render_image, plot_errors_button, plot_displacements])
    all_options = ipywidgets.Box(
        children=[iterations_mode_and_sliders, opts])

    # Widget container
    iterations_result_wid = ipywidgets.Box(children=[but,
                                                                 all_options])

    # Initialize variables
    iterations_result_options_default['selected_groups'] = \
        _convert_iterations_to_groups(
            0, 0, iterations_result_options_default['iter_str'])
    iterations_result_wid.selected_values = iterations_result_options_default

    # Define iterations mode visibility
    def iterations_mode_selection(name, value):
        if value == 'animation':
            # get val that needs to be assigned
            val = first_slider_wid.value
            # update visibility
            animation_wid.visible = True
            first_slider_wid.visible = False
            second_slider_wid.visible = False
            same_axes.visible = False
            update_but.visible = False
            # set correct values
            animation_wid.children[1].children[0].value = val
            animation_wid.selected_values['index'] = val
            first_slider_wid.value = 0
            second_slider_wid.value = \
                iterations_result_wid.selected_values['n_iters'] - 1
        else:
            # get val that needs to be assigned
            val = animation_wid.selected_values['index']
            # update visibility
            animation_wid.visible = False
            first_slider_wid.visible = True
            second_slider_wid.visible = True
            same_axes.visible = True
            update_but.visible = True
            # set correct values
            second_slider_wid.value = val
            first_slider_wid.value = val
            animation_wid.children[1].children[0].value = 0
            animation_wid.selected_values['index'] = 0
    iterations_mode.on_trait_change(iterations_mode_selection, 'value')

    # Check first slider's value
    def first_slider_val(name, value):
        if value > second_slider_wid.value:
            first_slider_wid.value = second_slider_wid.value
    first_slider_wid.on_trait_change(first_slider_val, 'value')

    # Check second slider's value
    def second_slider_val(name, value):
        if value < first_slider_wid.value:
            second_slider_wid.value = first_slider_wid.value
    second_slider_wid.on_trait_change(second_slider_val, 'value')

    # Convert slider values to groups
    def get_groups(name, value):
        if iterations_mode.value == 'animation':
            iterations_result_wid.selected_values['selected_groups'] = \
                _convert_iterations_to_groups(
                    animation_wid.selected_values['index'],
                    animation_wid.selected_values['index'],
                    iterations_result_wid.selected_values['iter_str'])
        else:
            iterations_result_wid.selected_values['selected_groups'] = \
                _convert_iterations_to_groups(
                    first_slider_wid.value, second_slider_wid.value,
                    iterations_result_wid.selected_values['iter_str'])
    first_slider_wid.on_trait_change(get_groups, 'value')
    second_slider_wid.on_trait_change(get_groups, 'value')

    # assign get_groups() to the slider of animation_wid
    animation_wid.children[1].children[0].on_trait_change(get_groups, 'value')

    # Render image function
    def render_image_fun(name, value):
        iterations_result_wid.selected_values['render_image'] = value
    render_image.on_trait_change(render_image_fun, 'value')

    # Same axes function
    def same_axes_fun(name, value):
        iterations_result_wid.selected_values['subplots_enabled'] = not value
    same_axes.on_trait_change(same_axes_fun, 'value')

    # Displacement type function
    def displacement_type_fun(name, value):
        iterations_result_wid.selected_values['displacement_type'] = value
    plot_displacements_menu.on_trait_change(displacement_type_fun, 'value')

    # Toggle button function
    def show_options(name, value):
        iterations_mode.visible = value
        render_image.visible = value
        plot_errors_button.visible = \
            iterations_result_wid.selected_values['image_has_gt_shape'] and value
        plot_displacements.visible = value
        if value:
            if iterations_mode.value == 'animation':
                animation_wid.visible = True
            else:
                first_slider_wid.visible = True
                second_slider_wid.visible = True
                same_axes.visible = True
                update_but.visible = True
        else:
            animation_wid.visible = False
            first_slider_wid.visible = False
            second_slider_wid.visible = False
            same_axes.visible = False
            update_but.visible = False
    show_options('', toggle_show_default)
    but.on_trait_change(show_options, 'value')

    # assign general plot_function
    if plot_function is not None:
        def plot_function_but(name):
            plot_function(name, 0)

        update_but.on_click(plot_function_but)
        # Here we assign plot_function() to the slider of animation_wid, as
        # we didn't do it at its creation.
        animation_wid.children[1].children[0].on_trait_change(plot_function,
                                                              'value')
        render_image.on_trait_change(plot_function, 'value')
        iterations_mode.on_trait_change(plot_function, 'value')

    # assign plot function of errors button
    if plot_errors_function is not None:
        plot_errors_button.on_click(plot_errors_function)

    # assign plot function of displacements button
    if plot_displacements_function is not None:
        plot_displacements_button.on_click(plot_displacements_function)

    return iterations_result_wid


def format_iterations_result_options(iterations_result_wid,
                                     container_padding='6px',
                                     container_margin='6px',
                                     container_border='1px solid black',
                                     toggle_button_font_weight='bold',
                                     border_visible=True):
    r"""
    Function that corrects the align (style format) of a given
    iterations_result_options widget. Usage example:
        iterations_result_wid = iterations_result_options()
        display(iterations_result_wid)
        format_iterations_result_options(iterations_result_wid)

    Parameters
    ----------
    iterations_result_wid :
        The widget object generated by the `iterations_result_options()`
        function.
    container_padding : `str`, optional
        The padding around the widget, e.g. '6px'
    container_margin : `str`, optional
        The margin around the widget, e.g. '6px'
    container_border : `str`, optional
        The border around the widget, e.g. '1px solid black'
    toggle_button_font_weight : `str`
        The font weight of the toggle button, e.g. 'bold'
    border_visible : `bool`, optional
        Defines whether to draw the border line around the widget.
    """
    # format animations options
    format_animation_options(
        iterations_result_wid.children[1].children[0].children[1].children[0],
        index_text_width='0.5cm', container_padding=container_padding,
        container_margin=container_margin, container_border=container_border,
        toggle_button_font_weight=toggle_button_font_weight,
        border_visible=False)

    # align displacement button and drop down menu
    remove_class(iterations_result_wid.children[1].children[1].children[2], 'vbox')
    add_class(iterations_result_wid.children[1].children[1].children[2], 'hbox')
    add_class(iterations_result_wid.children[1].children[1].children[2], 'align-center')
    iterations_result_wid.children[1].children[1].children[2].children[0].\
        margin_right = '0px'
    iterations_result_wid.children[1].children[1].children[2].children[1].\
        margin_left = '0px'

    # align options
    remove_class(iterations_result_wid.children[1].children[1], 'vbox')
    add_class(iterations_result_wid.children[1].children[1], 'hbox')
    add_class(iterations_result_wid.children[1].children[1], 'align-center')
    iterations_result_wid.children[1].children[1].children[0].\
        margin_right = '30px'
    iterations_result_wid.children[1].children[1].children[1].\
        margin_right = '30px'

    # align update button and same axes checkbox
    remove_class(iterations_result_wid.children[1].children[0].children[1].children[3], 'vbox')
    add_class(iterations_result_wid.children[1].children[0].children[1].children[3], 'hbox')
    iterations_result_wid.children[1].children[0].children[1].children[3].children[0].\
        margin_right = '20px'

    # align sliders
    add_class(iterations_result_wid.children[1].children[0].children[1], 'align-end')
    iterations_result_wid.children[1].children[0].children[1].\
        margin_bottom = '20px'

    # align sliders and iterations_mode
    remove_class(iterations_result_wid.children[1].children[0], 'vbox')
    add_class(iterations_result_wid.children[1].children[0], 'hbox')
    add_class(iterations_result_wid.children[1].children[0], 'align-start')

    # align sliders and options
    add_class(iterations_result_wid.children[1], 'align-end')

    # set toggle button font bold
    iterations_result_wid.children[0].font_weight = toggle_button_font_weight

    # margin and border around container widget
    iterations_result_wid.padding = container_padding
    iterations_result_wid.margin = container_margin
    if border_visible:
        iterations_result_wid.border = container_border


def update_iterations_result_options(iterations_result_wid,
                                     iterations_result_default):
    r"""
    Function that updates the state of a given iterations_result_options widget. Usage example:
        iterations_result_options_default = {'n_iters': 10,
                                             'image_has_gt_shape': True,
                                             'n_points': 68,
                                             'iter_str': 'iter_',
                                             'selected_groups': [0],
                                             'render_image': True,
                                             'subplots_enabled': True,
                                             'displacement_type': 'mean'}
        iterations_result_wid = iterations_result_options(iterations_result_options_default)
        display(iterations_result_wid)
        format_iterations_result_options(iterations_result_wid)
        iterations_result_options_default = {'n_iters': 100,
                                             'image_has_gt_shape': False,
                                             'n_points': 15,
                                             'iter_str': 'iter_',
                                             'selected_groups': [0],
                                             'render_image': False,
                                             'subplots_enabled': False,
                                             'displacement_type': 'median'}
        update_iterations_result_options(iterations_result_wid, iterations_result_options_default)

    Parameters
    ----------
    iterations_result_wid :
        The widget generated by `iterations_result_options()` function.
    iterations_result_options_default : `dict`
        The default options. For example:
            iterations_result_options_default = {'n_iters': 10,
                                                 'image_has_gt_shape': True,
                                                 'n_points': 68,
                                                 'iter_str': 'iter_',
                                                 'selected_groups': [0],
                                                 'render_image': True,
                                                 'subplots_enabled': True,
                                                 'displacement_type': 'mean'}
    """
    # if image_has_gt_shape flag has actually changed from the previous value
    if ('image_has_gt_shape' in iterations_result_default and
        iterations_result_default['image_has_gt_shape'] !=
                iterations_result_wid.selected_values['image_has_gt_shape']):
        # set the plot errors visibility
        iterations_result_wid.children[1].children[1].children[1].visible = \
            (iterations_result_wid.children[0].value and
             iterations_result_default['image_has_gt_shape'])
        # store the flag
        iterations_result_wid.selected_values['image_has_gt_shape'] = \
            iterations_result_default['image_has_gt_shape']

    # if n_points has actually changed from the previous value
    if ('n_points' in iterations_result_default and
        iterations_result_default['n_points'] !=
                iterations_result_wid.selected_values['n_points']):
        # change the contents of the displacement types
        select_menu = OrderedDict()
        select_menu['mean'] = 'mean'
        select_menu['median'] = 'median'
        select_menu['max'] = 'max'
        select_menu['min'] = 'min'
        for p in range(iterations_result_default['n_points']):
            select_menu["point {}".format(p + 1)] = p
        iterations_result_wid.children[1].children[1].children[2].children[1].\
            values = select_menu
        # store the number of points
        iterations_result_wid.selected_values['n_points'] = \
            iterations_result_default['n_points']

    # if displacement_type has actually changed from the previous value
    if ('displacement_type' in iterations_result_default and
        iterations_result_default['displacement_type'] !=
                iterations_result_wid.selected_values['displacement_type']):
        iterations_result_wid.children[1].children[1].children[2].children[1].\
            value = iterations_result_default['displacement_type']

    # if iter_str are actually different from the previous value
    if ('iter_str' in iterations_result_default and
        iterations_result_default['iter_str'] !=
                iterations_result_wid.selected_values['iter_str']):
        iterations_result_wid.selected_values['iter_str'] = \
            iterations_result_default['iter_str']

    # if render_image are actually different from the previous value
    if ('render_image' in iterations_result_default and
        iterations_result_default['render_image'] !=
                iterations_result_wid.selected_values['render_image']):
        iterations_result_wid.children[1].children[1].children[0].value = \
            iterations_result_default['render_image']

    # if subplots_enabled are actually different from the previous value
    if ('subplots_enabled' in iterations_result_default and
        iterations_result_default['subplots_enabled'] !=
                iterations_result_wid.selected_values['subplots_enabled']):
        iterations_result_wid.children[1].children[0].children[1].children[3].children[0].value = \
            not iterations_result_default['subplots_enabled']

    # if n_iters are actually different from the previous value
    if ('n_iters' in iterations_result_default and
        iterations_result_default['n_iters'] !=
                iterations_result_wid.selected_values['n_iters']):
        # change the iterations_result_wid output
        iterations_result_wid.selected_values['n_iters'] = \
            iterations_result_default['n_iters']
        iterations_result_wid.selected_values['selected_groups'] = \
            _convert_iterations_to_groups(
                0, 0, iterations_result_wid.selected_values['iter_str'])

        animation_options_wid = iterations_result_wid.children[1].children[0].children[1].children[0]
        # set the iterations options state
        if iterations_result_default['n_iters'] == 1:
            # set sliders values and visibility
            for t in range(4):
                if t == 0:
                    # first slider
                    iterations_result_wid.children[1].children[0].children[1].children[1].value = 0
                    iterations_result_wid.children[1].children[0].children[1].children[1].max = 0
                    iterations_result_wid.children[1].children[0].children[1].children[1].visible = False
                elif t == 1:
                    # second slider
                    iterations_result_wid.children[1].children[0].children[1].children[2].value = 0
                    iterations_result_wid.children[1].children[0].children[1].children[2].max = 0
                    iterations_result_wid.children[1].children[0].children[1].children[2].visible = False
                elif t == 2:
                    # animation slider
                    animation_options_wid.selected_values['index'] = 0
                    animation_options_wid.selected_values['max'] = 0
                    animation_options_wid.children[1].children[0].value = 0
                    animation_options_wid.children[1].children[0]. max = 0
                    animation_options_wid.children[1].children[0].disabled = True
                    animation_options_wid.children[1].children[1].children[0].children[0].disabled = True
                    animation_options_wid.children[1].children[1].children[0].children[1].disabled = True
                    animation_options_wid.children[1].children[1].children[0].children[2].disabled = True
                else:
                    # iterations mode
                    iterations_result_wid.children[1].children[0].children[0].value = 'animation'
                    #iterations_result_wid.groups = [iter_str + "0"]
                    iterations_result_wid.children[1].children[0].children[0].disabled = True
        else:
            # set sliders max and min values
            for t in range(4):
                if t == 0:
                    # first slider
                    iterations_result_wid.children[1].children[0].children[1].children[1].value = 0
                    iterations_result_wid.children[1].children[0].children[1].children[1].max = \
                        iterations_result_default['n_iters'] - 1
                    iterations_result_wid.children[1].children[0].children[1].children[1].visible = False
                elif t == 1:
                    # second slider
                    iterations_result_wid.children[1].children[0].children[1].children[2].value = \
                        iterations_result_default['n_iters'] - 1
                    iterations_result_wid.children[1].children[0].children[1].children[2].max = \
                        iterations_result_default['n_iters'] - 1
                    iterations_result_wid.children[1].children[0].children[1].children[2].visible = False
                elif t == 2:
                    # animation slider
                    animation_options_wid.children[1].children[0].value = 0
                    animation_options_wid.children[1].children[0].max = \
                        iterations_result_default['n_iters'] - 1
                    animation_options_wid.selected_values['index'] = 0
                    animation_options_wid.selected_values['max'] = \
                        iterations_result_default['n_iters'] - 1
                    animation_options_wid.children[1].children[0].disabled = \
                        False
                    animation_options_wid.children[1].children[1].children[0].children[0].disabled = False
                    animation_options_wid.children[1].children[1].children[0].children[1].disabled = True
                    animation_options_wid.children[1].children[1].children[0].children[2].disabled = False
                else:
                    # iterations mode
                    iterations_result_wid.children[1].children[0].children[0].\
                        value = 'animation'
                    #iterations_result_wid.groups = [iter_str + "0"]
                    iterations_result_wid.children[1].children[0].children[0].\
                        disabled = False


def plot_options(plot_options_default, plot_function=None,
                 toggle_show_visible=True, toggle_show_default=True):
    r"""
    Creates a widget with Plot Options. Specifically, it has:
        1) A drop down menu for curve selection.
        2) A text area for the legend entry.
        3) A checkbox that controls line's visibility.
        4) A checkbox that controls markers' visibility.
        5) Options for line colour, style and width.
        6) Options for markers face colour, edge colour, size, edge width and
           style.
        7) A toggle button that controls the visibility of all the above, i.e.
           the plot options.

    The structure of the widgets is the following:
        plot_options_wid.children = [toggle_button, options]
        options.children = [curve_menu, per_curve_options_wid]
        per_curve_options_wid = ipywidgets.Box(children=[legend_entry,
                                                          line_marker_wid])
        line_marker_wid = ipywidgets.Box(children=[line_widget, marker_widget])
        line_widget.children = [show_line_checkbox, line_options]
        marker_widget.children = [show_marker_checkbox, marker_options]
        line_options.children = [linestyle, linewidth, linecolour]
        marker_options.children = [markerstyle, markersize, markeredgewidth,
                                   markerfacecolour, markeredgecolour]

    The returned widget saves the selected values in the following dictionary:
        plot_options_wid.selected_options

    To fix the alignment within this widget please refer to
    `format_plot_options()` function.

    Parameters
    ----------
    plot_options_default : list of `dict`
        A list of dictionaries with the initial selected plot options per curve.
        Example:
            plot_options_1={'show_line':True,
                            'linewidth':2,
                            'linecolour':'r',
                            'linestyle':'-',
                            'show_marker':True,
                            'markersize':20,
                            'markerfacecolour':'r',
                            'markeredgecolour':'b',
                            'markerstyle':'o',
                            'markeredgewidth':1,
                            'legend_entry':'final errors'}
            plot_options_2={'show_line':False,
                            'linewidth':3,
                            'linecolour':'r',
                            'linestyle':'-',
                            'show_marker':True,
                            'markersize':60,
                            'markerfacecolour':[0.1, 0.2, 0.3],
                            'markeredgecolour':'k',
                            'markerstyle':'x',
                            'markeredgewidth':1,
                            'legend_entry':'initial errors'}
            plot_options_default = [plot_options_1, plot_options_2]

    plot_function : `function` or None, optional
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.
    """
    import IPython.html.widgets as ipywidgets
    # make sure that plot_options_default is a list even with one member
    if not isinstance(plot_options_default, list):
        plot_options_default = [plot_options_default]

    # find number of curves
    n_curves = len(plot_options_default)

    # Create widgets
    # toggle button
    but = ipywidgets.ToggleButton(description='Plot Options',
                                        value=toggle_show_default,
                                        visible=toggle_show_visible)

    # select curve drop down menu
    curves_dict = OrderedDict()
    for k in range(n_curves):
        curves_dict['Curve ' + str(k)] = k
    curve_selection = ipywidgets.Dropdown(options=curves_dict,
                                                value=0,
                                                description='Select curve',
                                                visible=n_curves > 1)

    # legend entry
    legend_entry = ipywidgets.Text(description='Legend entry',
                                         value=plot_options_default[0][
                                             'legend_entry'])

    # show line, show markers checkboxes
    show_line = ipywidgets.Checkbox(description='Show line',
                                          value=plot_options_default[0][
                                              'show_line'])
    show_marker = ipywidgets.Checkbox(description='Show markers',
                                            value=plot_options_default[0][
                                                'show_marker'])

    # linewidth, markersize
    linewidth = ipywidgets.FloatText(description='Width',
                                           value=plot_options_default[0][
                                               'linewidth'])
    markersize = ipywidgets.IntText(description='Size',
                                          value=plot_options_default[0][
                                              'markersize'])
    markeredgewidth = ipywidgets.FloatText(
        description='Edge width',
        value=plot_options_default[0]['markeredgewidth'])

    # markerstyle
    markerstyle_dict = OrderedDict()
    markerstyle_dict['point'] = '.'
    markerstyle_dict['pixel'] = ','
    markerstyle_dict['circle'] = 'o'
    markerstyle_dict['triangle down'] = 'v'
    markerstyle_dict['triangle up'] = '^'
    markerstyle_dict['triangle left'] = '<'
    markerstyle_dict['triangle right'] = '>'
    markerstyle_dict['tri down'] = '1'
    markerstyle_dict['tri up'] = '2'
    markerstyle_dict['tri left'] = '3'
    markerstyle_dict['tri right'] = '4'
    markerstyle_dict['octagon'] = '8'
    markerstyle_dict['square'] = 's'
    markerstyle_dict['pentagon'] = 'p'
    markerstyle_dict['star'] = '*'
    markerstyle_dict['hexagon 1'] = 'h'
    markerstyle_dict['hexagon 2'] = 'H'
    markerstyle_dict['plus'] = '+'
    markerstyle_dict['x'] = 'x'
    markerstyle_dict['diamond'] = 'D'
    markerstyle_dict['thin diamond'] = 'd'
    markerstyle = ipywidgets.Dropdown(options=markerstyle_dict,
                                            value=plot_options_default[0][
                                                'markerstyle'],
                                            description='Style')

    # linestyle
    linestyle_dict = OrderedDict()
    linestyle_dict['solid'] = '-'
    linestyle_dict['dashed'] = '--'
    linestyle_dict['dash-dot'] = '-.'
    linestyle_dict['dotted'] = ':'
    linestyle = ipywidgets.Dropdown(options=linestyle_dict,
                                          value=plot_options_default[0][
                                              'linestyle'],
                                          description='Style')

    # colours
    # do not assign the plot_function here
    linecolour = colour_selection(plot_options_default[0]['linecolour'],
                                  title='Colour')
    markerfacecolour = colour_selection(
        plot_options_default[0]['markerfacecolour'],
        title='Face Colour')
    markeredgecolour = colour_selection(
        plot_options_default[0]['markeredgecolour'],
        title='Edge Colour')

    # Group widgets
    line_options = ipywidgets.Box(
        children=[linestyle, linewidth, linecolour])
    marker_options = ipywidgets.Box(
        children=[markerstyle, markersize,
                  markeredgewidth,
                  markerfacecolour,
                  markeredgecolour])
    line_wid = ipywidgets.Box(children=[show_line, line_options])
    marker_wid = ipywidgets.Box(
        children=[show_marker, marker_options])
    line_options_options_wid = ipywidgets.Box(
        children=[line_wid, marker_wid])
    options_wid = ipywidgets.Box(children=[legend_entry,
                                                       line_options_options_wid])
    options_and_curve_wid = ipywidgets.Box(
        children=[curve_selection,
                  options_wid])
    plot_options_wid = ipywidgets.Box(
        children=[but, options_and_curve_wid])

    # initialize output
    plot_options_wid.selected_options = plot_options_default

    # line options visibility
    def line_options_visible(name, value):
        linestyle.disabled = not value
        linewidth.disabled = not value
        linecolour.children[0].disabled = not value
        linecolour.children[1].children[0].disabled = not value
        linecolour.children[1].children[1].disabled = not value
        linecolour.children[1].children[2].disabled = not value
    show_line.on_trait_change(line_options_visible, 'value')

    # marker options visibility
    def marker_options_visible(name, value):
        markerstyle.disabled = not value
        markersize.disabled = not value
        markeredgewidth.disabled = not value
        markerfacecolour.children[0].disabled = not value
        markerfacecolour.children[1].children[0].disabled = not value
        markerfacecolour.children[1].children[1].disabled = not value
        markerfacecolour.children[1].children[2].disabled = not value
        markeredgecolour.children[0].disabled = not value
        markeredgecolour.children[1].children[0].disabled = not value
        markeredgecolour.children[1].children[1].disabled = not value
        markeredgecolour.children[1].children[2].disabled = not value
    show_marker.on_trait_change(marker_options_visible, 'value')

    # function that gets colour selection
    def get_colour(colour_wid):
        if colour_wid.children[0].value == 'custom':
            return [float(colour_wid.children[1].children[0].value),
                    float(colour_wid.children[1].children[1].value),
                    float(colour_wid.children[1].children[2].value)]
        else:
            return colour_wid.children[0].value

    # assign options
    def save_legend_entry(name, value):
        plot_options_wid.selected_options[curve_selection.value][
            'legend_entry'] = str(value)

    legend_entry.on_trait_change(save_legend_entry, 'value')

    def save_show_line(name, value):
        plot_options_wid.selected_options[curve_selection.value][
            'show_line'] = value

    show_line.on_trait_change(save_show_line, 'value')

    def save_show_marker(name, value):
        plot_options_wid.selected_options[curve_selection.value][
            'show_marker'] = value

    show_marker.on_trait_change(save_show_marker, 'value')

    def save_linewidth(name, value):
        plot_options_wid.selected_options[curve_selection.value][
            'linewidth'] = float(value)

    linewidth.on_trait_change(save_linewidth, 'value')

    def save_linestyle(name, value):
        plot_options_wid.selected_options[curve_selection.value][
            'linestyle'] = value

    linestyle.on_trait_change(save_linestyle, 'value')

    def save_markersize(name, value):
        plot_options_wid.selected_options[curve_selection.value][
            'markersize'] = int(value)

    markersize.on_trait_change(save_markersize, 'value')

    def save_markeredgewidth(name, value):
        plot_options_wid.selected_options[curve_selection.value][
            'markeredgewidth'] = float(value)

    markeredgewidth.on_trait_change(save_markeredgewidth, 'value')

    def save_markerstyle(name, value):
        plot_options_wid.selected_options[curve_selection.value][
            'markerstyle'] = value

    markerstyle.on_trait_change(save_markerstyle, 'value')

    def save_linecolour(name, value):
        plot_options_wid.selected_options[curve_selection.value][
            'linecolour'] = get_colour(linecolour)

    linecolour.children[0].on_trait_change(save_linecolour, 'value')
    linecolour.children[1].children[0].on_trait_change(save_linecolour, 'value')
    linecolour.children[1].children[1].on_trait_change(save_linecolour, 'value')
    linecolour.children[1].children[2].on_trait_change(save_linecolour, 'value')

    def save_markerfacecolour(name, value):
        plot_options_wid.selected_options[curve_selection.value][
            'markerfacecolour'] = get_colour(markerfacecolour)

    markerfacecolour.children[0].on_trait_change(save_markerfacecolour, 'value')
    markerfacecolour.children[1].children[0].on_trait_change(
        save_markerfacecolour, 'value')
    markerfacecolour.children[1].children[1].on_trait_change(
        save_markerfacecolour, 'value')
    markerfacecolour.children[1].children[2].on_trait_change(
        save_markerfacecolour, 'value')

    def save_markeredgecolour(name, value):
        plot_options_wid.selected_options[curve_selection.value][
            'markeredgecolour'] = get_colour(markeredgecolour)

    markeredgecolour.children[0].on_trait_change(save_markeredgecolour, 'value')
    markeredgecolour.children[1].children[0].on_trait_change(
        save_markeredgecolour, 'value')
    markeredgecolour.children[1].children[1].on_trait_change(
        save_markeredgecolour, 'value')
    markeredgecolour.children[1].children[2].on_trait_change(
        save_markeredgecolour, 'value')

    # set correct value to slider when drop down menu value changes
    def set_options(name, value):
        legend_entry.value = plot_options_wid.selected_options[value][
            'legend_entry']
        show_line.value = plot_options_wid.selected_options[value]['show_line']
        show_marker.value = plot_options_wid.selected_options[value][
            'show_marker']
        linewidth.value = plot_options_wid.selected_options[value]['linewidth']
        linestyle.value = plot_options_wid.selected_options[value]['linestyle']
        markersize.value = plot_options_wid.selected_options[value][
            'markersize']
        markerstyle.value = plot_options_wid.selected_options[value][
            'markerstyle']
        markeredgewidth.value = plot_options_wid.selected_options[value][
            'markeredgewidth']
        default_colour = plot_options_wid.selected_options[value]['linecolour']
        if not isinstance(default_colour, str):
            r_val = default_colour[0]
            g_val = default_colour[1]
            b_val = default_colour[2]
            default_colour = 'custom'
            linecolour.children[1].children[0].value = r_val
            linecolour.children[1].children[1].value = g_val
            linecolour.children[1].children[2].value = b_val
        linecolour.children[0].value = default_colour
        default_colour = plot_options_wid.selected_options[value][
            'markerfacecolour']
        if not isinstance(default_colour, str):
            r_val = default_colour[0]
            g_val = default_colour[1]
            b_val = default_colour[2]
            default_colour = 'custom'
            markerfacecolour.children[1].children[0].value = r_val
            markerfacecolour.children[1].children[1].value = g_val
            markerfacecolour.children[1].children[2].value = b_val
        markerfacecolour.children[0].value = default_colour
        default_colour = plot_options_wid.selected_options[value][
            'markeredgecolour']
        if not isinstance(default_colour, str):
            r_val = default_colour[0]
            g_val = default_colour[1]
            b_val = default_colour[2]
            default_colour = 'custom'
            markeredgecolour.children[1].children[0].value = r_val
            markeredgecolour.children[1].children[1].value = g_val
            markeredgecolour.children[1].children[2].value = b_val
        markeredgecolour.children[0].value = default_colour
    curve_selection.on_trait_change(set_options, 'value')

    # Toggle button function
    def toggle_fun(name, value):
        options_and_curve_wid.visible = value
    toggle_fun('', toggle_show_default)
    but.on_trait_change(toggle_fun, 'value')

    # assign plot_function
    if plot_function is not None:
        legend_entry.on_trait_change(plot_function, 'value')
        show_line.on_trait_change(plot_function, 'value')
        linestyle.on_trait_change(plot_function, 'value')
        linewidth.on_trait_change(plot_function, 'value')
        show_marker.on_trait_change(plot_function, 'value')
        markerstyle.on_trait_change(plot_function, 'value')
        markeredgewidth.on_trait_change(plot_function, 'value')
        markersize.on_trait_change(plot_function, 'value')
        linecolour.children[0].on_trait_change(plot_function, 'value')
        linecolour.children[1].children[0].on_trait_change(plot_function,
                                                           'value')
        linecolour.children[1].children[1].on_trait_change(plot_function,
                                                           'value')
        linecolour.children[1].children[2].on_trait_change(plot_function,
                                                           'value')
        markerfacecolour.children[0].on_trait_change(plot_function, 'value')
        markerfacecolour.children[1].children[0].on_trait_change(plot_function,
                                                                 'value')
        markerfacecolour.children[1].children[1].on_trait_change(plot_function,
                                                                 'value')
        markerfacecolour.children[1].children[2].on_trait_change(plot_function,
                                                                 'value')
        markeredgecolour.children[0].on_trait_change(plot_function, 'value')
        markeredgecolour.children[1].children[0].on_trait_change(plot_function,
                                                                 'value')
        markeredgecolour.children[1].children[1].on_trait_change(plot_function,
                                                                 'value')
        markeredgecolour.children[1].children[2].on_trait_change(plot_function,
                                                                 'value')

    return plot_options_wid


def format_plot_options(plot_options_wid, container_padding='6px',
                        container_margin='6px',
                        container_border='1px solid black',
                        toggle_button_font_weight='bold', border_visible=True,
                        suboptions_border_visible=True):
    r"""
    Function that corrects the align (style format) of a given figure_options
    widget. Usage example:
        plot_options_wid = plot_options()
        display(plot_options_wid)
        format_plot_options(figure_options_wid)

    Parameters
    ----------
    plot_options_wid :
        The widget object generated by the `figure_options()` function.

    container_padding : `str`, optional
        The padding around the widget, e.g. '6px'

    container_margin : `str`, optional
        The margin around the widget, e.g. '6px'

    container_border : `str`, optional
        The border around the widget, e.g. '1px solid black'

    toggle_button_font_weight : `str`
        The font weight of the toggle button, e.g. 'bold'

    border_visible : `boolean`, optional
        Defines whether to draw the border line around the widget.

    suboptions_border_visible : `boolean`, optional
        Defines whether to draw the border line around the per curve options.
    """
    # align line options with checkbox
    add_class(plot_options_wid.children[1].children[1].children[1].children[0], 'align-end')

    # align marker options with checkbox
    add_class(plot_options_wid.children[1].children[1].children[1].children[1], 'align-end')

    # set text boxes width
    plot_options_wid.children[1].children[1].children[1].children[0].children[
        1].children[1]. \
        width = '1cm'
    plot_options_wid.children[1].children[1].children[1].children[1].children[
        1].children[1]. \
        width = '1cm'
    plot_options_wid.children[1].children[1].children[1].children[1].children[
        1].children[2]. \
        width = '1cm'

    # align line and marker options
    remove_class(plot_options_wid.children[1].children[1].children[1], 'vbox')
    add_class(plot_options_wid.children[1].children[1].children[1], 'hbox')
    if suboptions_border_visible:
        plot_options_wid.children[1].children[1].margin = container_margin
        plot_options_wid.children[1].children[1].border = container_border

    # align curve selection with line and marker options
    add_class(plot_options_wid.children[1], 'align-start')

    # format colour options
    format_colour_selection(
        plot_options_wid.children[1].children[1].children[1].children[
            0].children[1].children[2])
    format_colour_selection(
        plot_options_wid.children[1].children[1].children[1].children[
            1].children[1].children[3])
    format_colour_selection(
        plot_options_wid.children[1].children[1].children[1].children[
            1].children[1].children[4])

    # set toggle button font bold
    plot_options_wid.children[0].font_weight = toggle_button_font_weight

    # margin and border around container widget
    plot_options_wid.padding = container_padding
    plot_options_wid.margin = container_margin
    if border_visible:
        plot_options_wid.border = container_border


def _convert_iterations_to_groups(from_iter, to_iter, iter_str):
    r"""
    Function that generates a list of group labels given the range bounds and
    the str to be used.
    """
    return ["{}{}".format(iter_str, i) for i in range(from_iter, to_iter + 1)]
