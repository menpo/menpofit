from collections import OrderedDict

import IPython.html.widgets as ipywidgets

from menpo.visualize.widgets.tools import (_format_box, _format_font,
                                           _map_styles_to_hex_colours)
from menpo.visualize.widgets import AnimationOptionsWidget


class LinearModelParametersWidget(ipywidgets.FlexBox):
    r"""
    Creates a widget for selecting parameters values when visualizing a linear
    model (e.g. PCA model). The widget consists of the following parts from
    `IPython.html.widgets`:

    == =========== ================== ==========================
    No Object      Variable (`self.`) Description
    == =========== ================== ==========================
    1  Button      `plot_button`      The plot variance button
    2  Button      `reset_button`     The reset button
    3  HBox        `plot_and_reset`   Contains 1, 2
                         If mode is 'single'
    ------------------------------------------------------------
    4  FloatSlider `slider`           The parameter value slider
    5  Dropdown    `dropdown_params`  The parameter selector
    6  HBox        `parameters_wid`   Contains 4, 5
                         If mode is 'multiple'
    ------------------------------------------------------------
    7  FloatSlider `sliders`          `list` of all sliders
    8  VBox        `parameters_wid`   Contains all 7
    == =========== ================== ==========================

    Note that:

    * The selected parameters are stored in the ``self.parameters`` `list`.
    * To set the styling please refer to the ``style()`` and
      ``predefined_style()`` methods.
    * To update the state of the widget, please refer to the
      ``set_widget_state()`` method.
    * To update the callback function please refer to the
      ``replace_render_function()`` and ``replace_variance_function()``
      methods.

    Parameters
    ----------
    parameters : `list`
        The `list` of initial parameters values.
    render_function : `function` or ``None``, optional
        The render function that is executed when a widgets' value changes.
        If ``None``, then nothing is assigned.
    mode : {``'single'``, ``'multiple'``}, optional
        If ``'single'``, only a single slider is constructed along with a
        dropdown menu that allows the parameter selection.
        If ``'multiple'``, a slider is constructed for each parameter.
    params_str : `str`, optional
        The string that will be used as description of the slider(s). The final
        description has the form `"{}{}".format(params_str, p)`, where `p` is
        the parameter number.
    params_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.
    params_step : `float`, optional
        The step, in std units, of the sliders.
    plot_variance_visible : `bool`, optional
        Defines whether the button for plotting the variance will be visible
        upon construction.
    plot_variance_function : `function` or ``None``, optional
        The plot function that is executed when the plot variance button is
        clicked. If ``None``, then nothing is assigned.
    style : See Below, optional
        Sets a predefined style at the widget. Possible options are

            ========= ============================
            Style     Description
            ========= ============================
            'minimal' Simple black and white style
            'success' Green-based style
            'info'    Blue-based style
            'warning' Yellow-based style
            'danger'  Red-based style
            ''        No style
            ========= ============================

    Example
    -------
    Let's create a linear model parameters values widget and then update its
    state. Firstly, we need to import it:

        >>> from menpofit.visualize.widgets import LinearModelParametersWidget
        >>> from IPython.display import display

    Now let's define a render function that will get called on every widget
    change and will dynamically print the selected parameters:

        >>> from menpo.visualize import print_dynamic
        >>> def render_function(name, value):
        >>>     s = "Selected parameters: {}".format(wid.parameters)
        >>>     print_dynamic(s)

    Create the widget with some initial options and display it:

        >>> parameters = [-3., -2., -1., 0., 1., 2., 3.]
        >>> wid = LinearModelParametersWidget(parameters,
        >>>                                   render_function=render_function,
        >>>                                   params_str='Parameter ',
        >>>                                   mode='multiple',
        >>>                                   params_bounds=(-3., 3.),
        >>>                                   plot_variance_visible=True,
        >>>                                   style='info')
        >>> display(wid)

    By moving the sliders, the printed message gets updated. Finally, let's
    change the widget status with a new set of options:

        >>> wid.set_widget_state(parameters=[-7.] * 3, params_str='',
        >>>                      params_step=0.1, params_bounds=(-10, 10),
        >>>                      plot_variance_visible=False,
        >>>                      allow_callback=True)
    """
    def __init__(self, parameters, render_function=None, mode='multiple',
                 params_str='', params_bounds=(-3., 3.), params_step=0.1,
                 plot_variance_visible=True, plot_variance_function=None,
                 style='minimal'):
        # Check given parameters
        n_params = len(parameters)
        self._check_parameters(parameters, params_bounds)

        # If only one slider requested, then set mode to multiple
        if n_params == 1:
            mode = 'multiple'

        # Create widgets
        if mode == 'multiple':
            self.sliders = [
                ipywidgets.FloatSlider(
                    description="{}{}".format(params_str, p),
                    min=params_bounds[0], max=params_bounds[1],
                    step=params_step, value=parameters[p])
                for p in range(n_params)]
            self.parameters_wid = ipywidgets.VBox(children=self.sliders,
                                                  margin='0.2cm')
        else:
            vals = OrderedDict()
            for p in range(n_params):
                vals["{}{}".format(params_str, p)] = p
            self.slider = ipywidgets.FloatSlider(
                description='', min=params_bounds[0], max=params_bounds[1],
                step=params_step, value=parameters[0], margin='0.2cm')
            self.dropdown_params = ipywidgets.Dropdown(options=vals,
                                                       margin='0.2cm')
            self.parameters_wid = ipywidgets.HBox(
                children=[self.dropdown_params, self.slider])
        self.plot_button = ipywidgets.Button(
            description='Variance', margin='0.05cm',
            visible=plot_variance_visible)
        self.reset_button = ipywidgets.Button(description='Reset',
                                              margin='0.05cm')
        self.plot_and_reset = ipywidgets.HBox(children=[self.plot_button,
                                                        self.reset_button])

        # Widget container
        super(LinearModelParametersWidget, self).__init__(
            children=[self.parameters_wid, self.plot_and_reset])
        self.align = 'end'

        # Assign output
        self.parameters = parameters
        self.mode = mode
        self.params_str = params_str
        self.params_bounds = params_bounds
        self.params_step = params_step
        self.plot_variance_visible = plot_variance_visible

        # Set style
        self.predefined_style(style)

        # Set functionality
        if mode == 'single':
            # Assign slider value to parameters values list
            def save_slider_value(name, value):
                self.parameters[self.dropdown_params.value] = value
            self.slider.on_trait_change(save_slider_value, 'value')

            # Set correct value to slider when drop down menu value changes
            def set_slider_value(name, value):
                # Temporarily remove render callback
                render_function = self._render_function
                self.remove_render_function()
                # Set slider value
                self.slider.value = self.parameters[value]
                # Re-assign render callback
                self.add_render_function(render_function)
            self.dropdown_params.on_trait_change(set_slider_value, 'value')
        else:
            # Assign slider value to parameters values list
            def save_slider_value_from_id(description, name, value):
                i = int(description[len(params_str)::])
                self.parameters[i] = value

            # Partial function that helps get the widget's description str
            def partial_widget(description):
                return lambda name, value: save_slider_value_from_id(
                    description, name, value)

            # Assign saving values and main plotting function to all sliders
            for w in self.sliders:
                # The widget (w) is lexically scoped and so we need a way of
                # ensuring that we don't just receive the final value of w at
                # every iteration. Therefore we create another lambda function
                # that creates a new lexical scoping so that we can ensure the
                # value of w is maintained (as x) at each iteration.
                # In JavaScript, we would just use the 'let' keyword...
                w.on_trait_change(partial_widget(w.description), 'value')

        def reset_parameters(name):
            # Temporarily remove render callback
            render_function = self._render_function
            self.remove_render_function()

            # Set parameters to 0
            self.parameters = [0.0] * len(self.parameters)
            if mode == 'multiple':
                for ww in self.parameters_wid.children:
                    ww.value = 0.
            else:
                self.parameters_wid.children[0].value = 0
                self.parameters_wid.children[1].value = 0.

            # Re-assign render callback and trigger it
            self.add_render_function(render_function)
            if self._render_function is not None:
                self._render_function('', True)
        self.reset_button.on_click(reset_parameters)

        # Set plot variance function
        self._variance_function = None
        self.add_variance_function(plot_variance_function)

        # Set render function
        self._render_function = None
        self.add_render_function(render_function)

    def _check_parameters(self, parameters, bounds):
        if parameters is not None:
            for p in range(len(parameters)):
                if parameters[p] < bounds[0]:
                    parameters[p] = bounds[0]
                if parameters[p] > bounds[1]:
                    parameters[p] = bounds[1]

    def style(self, box_style=None, border_visible=False, border_color='black',
              border_style='solid', border_width=1, border_radius=0, padding=0,
              margin=0, font_family='', font_size=None, font_style='',
              font_weight='', slider_width='', slider_handle_colour='',
              slider_background_colour='', buttons_style=''):
        r"""
        Function that defines the styling of the widget.

        Parameters
        ----------
        box_style : See Below, optional
            Style options

                ========= ============================
                Style     Description
                ========= ============================
                'success' Green-based style
                'info'    Blue-based style
                'warning' Yellow-based style
                'danger'  Red-based style
                ''        Default style
                None      No style
                ========= ============================

        border_visible : `bool`, optional
            Defines whether to draw the border line around the widget.
        border_color : `str`, optional
            The color of the border around the widget.
        border_style : `str`, optional
            The line style of the border around the widget.
        border_width : `float`, optional
            The line width of the border around the widget.
        border_radius : `float`, optional
            The radius of the corners of the box.
        padding : `float`, optional
            The padding around the widget.
        margin : `float`, optional
            The margin around the widget.
        font_family : See Below, optional
            The font family to be used.
            Example options ::

                {'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace',
                 'helvetica'}

        font_size : `int`, optional
            The font size.
        font_style : {``'normal'``, ``'italic'``, ``'oblique'``}, optional
            The font style.
        font_weight : See Below, optional
            The font weight.
            Example options ::

                {'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
                 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
                 'extra bold', 'black'}

        slider_width : `str`, optional
            The width of the slider(s).
        slider_handle_colour : `str`, optional
            The colour of the handle(s) of the slider(s).
        slider_background_colour : `str`, optional
            The background colour of the slider(s).
        buttons_style : See Below, optional
            Style options

                ========= ============================
                Style     Description
                ========= ============================
                'primary' Blue-based style
                'success' Green-based style
                'info'    Blue-based style
                'warning' Yellow-based style
                'danger'  Red-based style
                ''        Default style
                None      No style
                ========= ============================
        """
        _format_box(self, box_style, border_visible, border_color, border_style,
                    border_width, border_radius, padding, margin)
        _format_font(self, font_family, font_size, font_style, font_weight)
        _format_font(self.reset_button, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.plot_button, font_family, font_size, font_style,
                     font_weight)
        if self.mode == 'single':
            self.slider.width = slider_width
            self.slider.slider_color = slider_handle_colour
            self.slider.background_color = slider_background_colour
            _format_font(self.slider, font_family, font_size, font_style,
                         font_weight)
            _format_font(self.dropdown_params, font_family, font_size,
                         font_style, font_weight)
        else:
            for sl in self.sliders:
                sl.width = slider_width
                sl.slider_color = slider_handle_colour
                sl.background_color = slider_background_colour
                _format_font(sl, font_family, font_size, font_style,
                             font_weight)
        self.reset_button.button_style = buttons_style
        self.plot_button.button_style = buttons_style

    def predefined_style(self, style):
        r"""
        Function that sets a predefined style on the widget.

        Parameters
        ----------
        style : `str` (see below)
            Style options

                ========= ============================
                Style     Description
                ========= ============================
                'minimal' Simple black and white style
                'success' Green-based style
                'info'    Blue-based style
                'warning' Yellow-based style
                'danger'  Red-based style
                ''        No style
                ========= ============================
        """
        if style == 'minimal':
            self.style(box_style=None, border_visible=True,
                       border_color='black', border_style='solid',
                       border_width=1, border_radius=0, padding='0.2cm',
                       margin='0.3cm', font_family='', font_size=None,
                       font_style='', font_weight='', slider_width='',
                       slider_handle_colour='', slider_background_colour='',
                       buttons_style='')
        elif (style == 'info' or style == 'success' or style == 'danger' or
              style == 'warning'):
            self.style(box_style=style, border_visible=True,
                       border_color=_map_styles_to_hex_colours(style),
                       border_style='solid', border_width=1, border_radius=10,
                       padding='0.2cm', margin='0.3cm', font_family='',
                       font_size=None, font_style='', font_weight='',
                       slider_width='',
                       slider_handle_colour=_map_styles_to_hex_colours(style),
                       slider_background_colour='', buttons_style='primary')
        else:
            raise ValueError('style must be minimal or info or success or '
                             'danger or warning')

    def add_render_function(self, render_function):
        r"""
        Method that adds a `render_function()` to the widget. The signature of
        the given function is also stored in `self._render_function`.

        Parameters
        ----------
        render_function : `function` or ``None``, optional
            The render function that behaves as a callback. If ``None``, then
            nothing is added.
        """
        self._render_function = render_function
        if self._render_function is not None:
            if self.mode == 'single':
                self.slider.on_trait_change(self._render_function, 'value')
            else:
                for sl in self.sliders:
                    sl.on_trait_change(self._render_function, 'value')

    def remove_render_function(self):
        r"""
        Method that removes the current `self._render_function()` from the
        widget and sets ``self._render_function = None``.
        """
        if self.mode == 'single':
            self.slider.on_trait_change(self._render_function, 'value',
                                        remove=True)
        else:
            for sl in self.sliders:
                sl.on_trait_change(self._render_function, 'value', remove=True)
        self._render_function = None

    def replace_render_function(self, render_function):
        r"""
        Method that replaces the current `self._render_function()` of the widget
        with the given `render_function()`.

        Parameters
        ----------
        render_function : `function` or ``None``, optional
            The render function that behaves as a callback. If ``None``, then
            nothing happens.
        """
        # remove old function
        self.remove_render_function()

        # add new function
        self.add_render_function(render_function)

    def add_variance_function(self, variance_function):
        r"""
        Method that adds a `variance_function()` to the `Variance` button of the
        widget. The signature of the given function is also stored in
        `self._variance_function`.

        Parameters
        ----------
        variance_function : `function` or ``None``, optional
            The variance function that behaves as a callback. If ``None``,
            then nothing is added.
        """
        self._variance_function = variance_function
        if self._variance_function is not None:
            self.plot_button.on_click(self._variance_function)

    def remove_variance_function(self):
        r"""
        Method that removes the current `self._variance_function()` from
        the `Variance` button of the widget and sets
        ``self._variance_function = None``.
        """
        self.plot_button.on_click(self._variance_function, remove=True)
        self._variance_function = None

    def replace_variance_function(self, variance_function):
        r"""
        Method that replaces the current `self._variance_function()` of the
        `Variance` button of the widget with the given `variance_function()`.

        Parameters
        ----------
        variance_function : `function` or ``None``, optional
            The variance function that behaves as a callback. If ``None``,
            then nothing happens.
        """
        # remove old function
        self.remove_variance_function()

        # add new function
        self.add_variance_function(variance_function)

    def set_widget_state(self, parameters=None, params_str=None,
                         params_bounds=None, params_step=None,
                         plot_variance_visible=True, allow_callback=True):
        r"""
        Method that updates the state of the widget with a new set of options.

        Parameters
        ----------
        parameters : `list` or ``None``, optional
            The `list` of new parameters' values. If ``None``, then nothing
            changes.
        params_str : `str` or ``None``, optional
            The string that will be used as description of the slider(s). The
            final description has the form `"{}{}".format(params_str, p)`, where
            `p` is the parameter number. If ``None``, then nothing changes.
        params_bounds : (`float`, `float`) or ``None``, optional
            The minimum and maximum bounds, in std units, for the sliders. If
            ``None``, then nothing changes.
        params_step : `float` or ``None``, optional
            The step, in std units, of the sliders. If ``None``, then nothing
            changes.
        plot_variance_visible : `bool`, optional
            Defines whether the button for plotting the variance will be
            visible.
        allow_callback : `bool`, optional
            If ``True``, it allows triggering of any callback functions.
        """
        # Temporarily remove render callback
        render_function = self._render_function
        self.remove_render_function()

        # Parse given options
        if parameters is None:
            parameters = self.parameters
        if params_str is None:
            params_str = ''
        if params_bounds is None:
            params_bounds = self.params_bounds
        if params_step is None:
            params_step = self.params_step

        # Check given parameters
        self._check_parameters(parameters, params_bounds)

        # Set plot variance visibility
        self.plot_button.visible = plot_variance_visible

        # Update widget
        if len(parameters) == len(self.parameters):
            # The number of parameters hasn't changed
            if self.mode == 'multiple':
                for p, sl in enumerate(self.sliders):
                    sl.value = parameters[p]
                    sl.description = "{}{}".format(params_str, p)
                    sl.min = params_bounds[0]
                    sl.max = params_bounds[1]
                    sl.step = params_step
            else:
                self.slider.min = params_bounds[0]
                self.slider.max = params_bounds[1]
                self.slider.step = params_step
                if not params_str == '':
                    vals = OrderedDict()
                    for p in range(len(parameters)):
                        vals["{}{}".format(params_str, p)] = p
                    self.dropdown_params.options = vals
                self.slider.value = parameters[self.dropdown_params.value]
        else:
            # The number of parameters has changed
            if self.mode == 'multiple':
                # Create new sliders
                self.sliders = [
                    ipywidgets.FloatSlider(
                        description="{}{}".format(params_str, p),
                        min=params_bounds[0], max=params_bounds[1],
                        step=params_step, value=parameters[p])
                    for p in range(len(parameters))]
                # Set sliders as the children of the container
                self.parameters_wid.children = self.sliders

                # Assign slider value to parameters values list
                def save_slider_value_from_id(description, name, value):
                    i = int(description[len(params_str)::])
                    self.parameters[i] = value

                # Partial function that helps get the widget's description str
                def partial_widget(description):
                    return lambda name, value: save_slider_value_from_id(
                        description, name, value)

                # Assign saving values and main plotting function to all sliders
                for w in self.sliders:
                    # The widget (w) is lexically scoped and so we need a way of
                    # ensuring that we don't just receive the final value of w
                    # at every iteration. Therefore we create another lambda
                    # function that creates a new lexical scoping so that we can
                    # ensure the value of w is maintained (as x) at each
                    # iteration. In JavaScript, we would just use the 'let'
                    # keyword...
                    w.on_trait_change(partial_widget(w.description), 'value')

                # Set style
                if self.box_style is None:
                    self.predefined_style('minimal')
                else:
                    self.predefined_style(self.box_style)
            else:
                self.slider.min = params_bounds[0]
                self.slider.max = params_bounds[1]
                self.slider.step = params_step
                vals = OrderedDict()
                for p in range(len(parameters)):
                    vals["{}{}".format(params_str, p)] = p
                if self.dropdown_params.value == 0:
                    self.dropdown_params.value = 1
                    self.dropdown_params.value = 0
                else:
                    self.dropdown_params.value = 0
                self.dropdown_params.options = vals
                self.slider.value = parameters[0]

        # Re-assign render callback
        self.add_render_function(render_function)

        # Assign new selected options
        self.parameters = parameters
        self.params_str = params_str
        self.params_bounds = params_bounds
        self.params_step = params_step
        self.plot_variance_visible = plot_variance_visible

        # trigger render function if allowed
        if allow_callback:
            self._render_function('', True)


class FittingResultOptionsWidget(ipywidgets.FlexBox):
    r"""
    Creates a widget for selecting options when visualizing a fitting result.
    The widget consists of the following parts from `IPython.html.widgets`:

    == =================== ========================= ========================
    No Object              Variable (`self.`)        Description
    == =================== ========================= ========================
    1  Latex, ToggleButton `shape_selection`         The shape selectors
    2  Checkbox            `render_image`            Controls image rendering
    3  RadioButtons        `mode`                    The figure mode
    4  HBox                `shapes_wid`              Contains all 1
    5  VBox                `shapes_and_render_image` Contains 4, 2
    == =================== ========================= ========================

    Note that:

    * The selected options are stored in the ``self.selected_options`` `dict`.
    * To set the styling please refer to the ``style()`` and
      ``predefined_style()`` methods.
    * To update the state of the widget, please refer to the
      ``set_widget_state()`` method.
    * To update the callback function please refer to the
      ``replace_render_function()`` method.

    Parameters
    ----------
    fitting_result_options : `dict`
        The dictionary with the initial options. For example
        ::

            fitting_result_options = {'all_groups': ['initial', 'final',
                                                     'ground'],
                                      'selected_groups': ['final'],
                                      'render_image': True,
                                      'subplots_enabled': True}

    render_function : `function` or ``None``, optional
        The render function that is executed when a widgets' value changes.
        If ``None``, then nothing is assigned.
    style : See Below, optional
        Sets a predefined style at the widget. Possible options are

            ========= ============================
            Style     Description
            ========= ============================
            'minimal' Simple black and white style
            'success' Green-based style
            'info'    Blue-based style
            'warning' Yellow-based style
            'danger'  Red-based style
            ''        No style
            ========= ============================

    Example
    -------
    Let's create a fitting result options widget and then update its state.
    Firstly, we need to import it:

        >>> from menpofit.visualize.widgets import FittingResultOptionsWidget
        >>> from IPython.display import display

    Now let's define a render function that will get called on every widget
    change and will dynamically print the selected options:

        >>> from menpo.visualize import print_dynamic
        >>> def render_function(name, value):
        >>>     s = "Selected groups: {}, Render image: {}, Subplots enabled: {}".format(
        >>>         wid.selected_values['selected_groups'],
        >>>         wid.selected_values['render_image'],
        >>>         wid.selected_values['subplots_enabled'])
        >>>     print_dynamic(s)

    Create the widget with some initial options and display it:

        >>> fitting_result_options = {'all_groups': ['initial', 'final',
        >>>                                          'ground'],
        >>>                           'selected_groups': ['final'],
        >>>                           'render_image': True,
        >>>                           'subplots_enabled': True}
        >>> wid = FittingResultOptionsWidget(fitting_result_options,
        >>>                                  render_function=render_function,
        >>>                                  style='info')
        >>> display(wid)

    By changing the various widgets, the printed message gets updated. Finally,
    let's change the widget status with a new set of options:

        >>> fitting_result_options = {'all_groups': ['initial', 'final'],
        >>>                           'selected_groups': ['final'],
        >>>                           'render_image': True,
        >>>                           'subplots_enabled': True}
        >>> wid.set_widget_state(fitting_result_options, allow_callback=True)
    """
    def __init__(self, fitting_result_options, render_function=None,
                 style='minimal'):
        # Create widgets
        self.shape_selection = [ipywidgets.Latex(value='Shape:',
                                                 margin='0.2cm')]
        for group in fitting_result_options['all_groups']:
            t = ipywidgets.ToggleButton(
                description=group,
                value=group in fitting_result_options['selected_groups'])
            self.shape_selection.append(t)
        self.render_image = ipywidgets.Checkbox(
            description='Render image',
            value=fitting_result_options['render_image'])
        self.mode = ipywidgets.RadioButtons(
            description='Figure mode:',
            options={'Single': False, 'Multiple': True},
            value=fitting_result_options['subplots_enabled'])

        # Group widgets
        self.shapes_wid = ipywidgets.HBox(children=self.shape_selection,
                                          align='center')
        self.shapes_and_render_image = ipywidgets.VBox(
            children=[self.shapes_wid, self.render_image], align='end')
        super(FittingResultOptionsWidget, self).__init__(
            children=[self.mode, self.shapes_and_render_image])

        # Assign output
        self.selected_values = fitting_result_options

        # Set style
        self.predefined_style(style)

        # Set functionality
        def groups_selection_function(name, value):
            self.selected_values['selected_groups'] = []
            for i in self.shapes_wid.children[1::]:
                if i.value:
                    self.selected_values['selected_groups'].append(
                        str(i.description))
        for w in self.shapes_wid.children[1::]:
            w.on_trait_change(groups_selection_function, 'value')

        def render_image_function(name, value):
            self.selected_values['render_image'] = value
        self.render_image.on_trait_change(render_image_function, 'value')

        def figure_mode_function(name, value):
            self.selected_values['subplots_enabled'] = value
        self.mode.on_trait_change(figure_mode_function, 'value')

        # Set render function
        self._render_function = None
        self.add_render_function(render_function)

    def style(self, box_style=None, border_visible=False, border_color='black',
              border_style='solid', border_width=1, border_radius=0, padding=0,
              margin=0, font_family='', font_size=None, font_style='',
              font_weight='', shapes_buttons_style=''):
        r"""
        Function that defines the styling of the widget.

        Parameters
        ----------
        box_style : See Below, optional
            Style options

                ========= ============================
                Style     Description
                ========= ============================
                'success' Green-based style
                'info'    Blue-based style
                'warning' Yellow-based style
                'danger'  Red-based style
                ''        Default style
                None      No style
                ========= ============================

        border_visible : `bool`, optional
            Defines whether to draw the border line around the widget.
        border_color : `str`, optional
            The color of the border around the widget.
        border_style : `str`, optional
            The line style of the border around the widget.
        border_width : `float`, optional
            The line width of the border around the widget.
        border_radius : `float`, optional
            The radius of the corners of the box.
        padding : `float`, optional
            The padding around the widget.
        margin : `float`, optional
            The margin around the widget.
        font_family : See Below, optional
            The font family to be used.
            Example options ::

                {'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace',
                 'helvetica'}

        font_size : `int`, optional
            The font size.
        font_style : {``'normal'``, ``'italic'``, ``'oblique'``}, optional
            The font style.
        font_weight : See Below, optional
            The font weight.
            Example options ::

                {'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
                 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
                 'extra bold', 'black'}

        shapes_buttons_style : See Below, optional
            Style options

                ========= ============================
                Style     Description
                ========= ============================
                'primary' Blue-based style
                'success' Green-based style
                'info'    Blue-based style
                'warning' Yellow-based style
                'danger'  Red-based style
                ''        Default style
                None      No style
                ========= ============================
        """
        _format_box(self, box_style, border_visible, border_color, border_style,
                    border_width, border_radius, padding, margin)
        _format_font(self, font_family, font_size, font_style, font_weight)
        _format_font(self.render_image, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.mode, font_family, font_size, font_style, font_weight)
        _format_font(self.shapes_wid.children[0], font_family, font_size,
                     font_style, font_weight)
        for w in self.shapes_wid.children[1::]:
            _format_font(w, font_family, font_size, font_style, font_weight)
            w.button_style = shapes_buttons_style

    def predefined_style(self, style):
        r"""
        Function that sets a predefined style on the widget.

        Parameters
        ----------
        style : `str` (see below)
            Style options

                ========= ============================
                Style     Description
                ========= ============================
                'minimal' Simple black and white style
                'success' Green-based style
                'info'    Blue-based style
                'warning' Yellow-based style
                'danger'  Red-based style
                ''        No style
                ========= ============================
        """
        if style == 'minimal':
            self.style(box_style=None, border_visible=True,
                       border_color='black', border_style='solid',
                       border_width=1, border_radius=0, padding='0.2cm',
                       margin='0.3cm', font_family='', font_size=None,
                       font_style='', font_weight='', shapes_buttons_style='')
        elif (style == 'info' or style == 'success' or style == 'danger' or
              style == 'warning'):
            self.style(box_style=style, border_visible=True,
                       border_color=_map_styles_to_hex_colours(style),
                       border_style='solid', border_width=1, border_radius=10,
                       padding='0.2cm', margin='0.3cm', font_family='',
                       font_size=None, font_style='', font_weight='',
                       shapes_buttons_style='primary')
        else:
            raise ValueError('style must be minimal or info or success or '
                             'danger or warning')

    def add_render_function(self, render_function):
        r"""
        Method that adds a `render_function()` to the widget. The signature of
        the given function is also stored in `self._render_function`.

        Parameters
        ----------
        render_function : `function` or ``None``, optional
            The render function that behaves as a callback. If ``None``, then
            nothing is added.
        """
        self._render_function = render_function
        if self._render_function is not None:
            self.render_image.on_trait_change(self._render_function, 'value')
            self.mode.on_trait_change(self._render_function, 'value')
            for w in self.shapes_wid.children[1::]:
                w.on_trait_change(self._render_function, 'value')

    def remove_render_function(self):
        r"""
        Method that removes the current `self._render_function()` from the
        widget and sets ``self._render_function = None``.
        """
        self.render_image.on_trait_change(self._render_function, 'value',
                                          remove=True)
        self.mode.on_trait_change(self._render_function, 'value', remove=True)
        for w in self.shapes_wid.children[1::]:
            w.on_trait_change(self._render_function, 'value', remove=True)
        self._render_function = None

    def replace_render_function(self, render_function):
        r"""
        Method that replaces the current `self._render_function()` of the widget
        with the given `render_function()`.

        Parameters
        ----------
        render_function : `function` or ``None``, optional
            The render function that behaves as a callback. If ``None``, then
            nothing happens.
        """
        # remove old function
        self.remove_render_function()

        # add new function
        self.add_render_function(render_function)

    def set_widget_state(self, fitting_result_options, allow_callback=True):
        r"""
        Method that updates the state of the widget with a new set of values.

        Parameters
        ----------
        fitting_result_options : `dict`
            The dictionary with the initial options. For example
            ::

                fitting_result_options = {'all_groups': ['initial', 'final',
                                                         'ground'],
                                          'selected_groups': ['final'],
                                          'render_image': True,
                                          'subplots_enabled': True}

        allow_callback : `bool`, optional
            If ``True``, it allows triggering of any callback functions.
        """
        # Temporarily remove render callback
        render_function = self._render_function
        self.remove_render_function()

        # Update render image checkbox
        self.render_image.value = fitting_result_options['render_image']

        # Update figure mode
        self.mode.value = fitting_result_options['subplots_enabled']

        # Update shapes toggles
        if (set(fitting_result_options['all_groups']) ==
                set(self.selected_values['all_groups'])):
            # groups haven't changed so simply update the toggles' values
            for w in self.shapes_wid.children[1::]:
                w.value = (str(w.description) in
                           fitting_result_options['selected_groups'])
        else:
            # groups changed
            # Get previous buttons style
            buttons_style = self.shapes_wid.children[1].button_style
            # Create new toggles
            self.shape_selection = [ipywidgets.Latex(value='Shape:',
                                                     margin='0.2cm')]
            for group in fitting_result_options['all_groups']:
                t = ipywidgets.ToggleButton(
                    description=group, button_style=buttons_style,
                    value=group in fitting_result_options['selected_groups'])
                self.shape_selection.append(t)
            self.shapes_wid.children = self.shape_selection

            # Assign them the correct functionality
            def groups_selection_function(name, value):
                self.selected_values['selected_groups'] = []
                for i in self.shapes_wid.children[1::]:
                    if i.value:
                        self.selected_values['selected_groups'].append(
                            str(i.description))
            for w in self.shapes_wid.children[1::]:
                w.on_trait_change(groups_selection_function, 'value')

        # Assign new options dict to selected_values
        self.selected_values = fitting_result_options

        # Re-assign render callback
        self.add_render_function(render_function)

        # trigger render function if allowed
        if allow_callback:
            self._render_function('', True)


class FittingResultIterationsOptionsWidget(ipywidgets.FlexBox):
    r"""
    Creates a widget for selecting options when visualizing the iterations of a
    fitting result. The widget consists of the following parts from
    `IPython.html.widgets` and `menpo.visualize.widgets.options`:

    == ====================== ================================ =================
    No Object                 Variable (`self.`)               Description
    == ====================== ================================ =================
    1  RadioButtons           `iterations_mode`                Animation or

                                                               static mode
    2  AnimationOptionsWidget `index_animation`                The animation

                                                               mode slider
    3  IntRangeSlider         `index_slider`                   The static mode

                                                               slider
    4  ToggleButton           `common_figure`                  Enables subplots

                                                               for static mode
    5  Checkbox               `render_image`                   Controls the

                                                               image rendering
    6  Button                 `plot_errors_button`             Plots the errors

                                                               curve
    7  Button                 `plot_displacements_button`      Plots the

                                                               displacements
    8  Dropdown               `plot_displacements_menu`        The displacements

                                                               type menu
    9  HBox                   `index_slider_and_common_figure` Contains 3, 4
    10  VBox                  `index_box`                      Contains 2, 9
    11  HBox                  `iterations_mode_and_sliders`    Contains 1, 10
    12  HBox                  `plot_displacements`             Contains 7, 8
    13  HBox                  `render_image_and_plot`          Contains 5, 6, 12
    == ====================== ================================ =================

    Note that:

    * The selected options are stored in the ``self.selected_options`` `dict`.
    * To set the styling please refer to the ``style()`` and
      ``predefined_style()`` methods.
    * To update the state of the widget, please refer to the
      ``set_widget_state()`` method.
    * To update the callback functions please refer to the
      ``replace_render_function()``, ``replace_plot_displacements_function()``
       and ``replace_plot_errors_function`` methods.

    Parameters
    ----------
    fitting_result_iterations_options : `dict`
        The dictionary with the initial options. For example
        ::

            fitting_result_iterations_options = {'n_iters': 10,
                                                 'image_has_gt_shape': True,
                                                 'n_points': 68,
                                                 'iter_str': 'iter_',
                                                 'selected_groups': [0],
                                                 'render_image': True,
                                                 'subplots_enabled': True,
                                                 'displacement_type': 'mean'}

    render_function : `function` or ``None``, optional
        The render function that is executed when a widgets' value changes.
        If ``None``, then nothing is assigned.
    plot_errors_function : `function` or ``None``, optional
        The function that is executed when the `Errors` button is clicked. If
        ``None``, then nothing is assigned.
    plot_displacements_function : `function` or ``None``, optional
        The function that is executed when the `Displacements` button is
        clicked. If ``None``, then nothing is assigned.
    style : See Below, optional
        Sets a predefined style at the widget. Possible options are

            ========= ============================
            Style     Description
            ========= ============================
            'minimal' Simple black and white style
            'success' Green-based style
            'info'    Blue-based style
            'warning' Yellow-based style
            'danger'  Red-based style
            ''        No style
            ========= ============================

    sliders_style : See Below, optional
        Sets a predefined style at the sliders of the widget. Possible options
        are

            ========= ============================
            Style     Description
            ========= ============================
            'minimal' Simple black and white style
            'success' Green-based style
            'info'    Blue-based style
            'warning' Yellow-based style
            'danger'  Red-based style
            ''        No style
            ========= ============================

    Example
    -------
    Let's create a fitting result iterations options widget and then update its
    state. Firstly, we need to import it:

        >>> from menpofit.visualize.widgets import FittingResultIterationsOptionsWidget
        >>> from IPython.display import display

    Now let's define a render function that will get called on every widget
    change and will dynamically print the selected options:

        >>> from menpo.visualize import print_dynamic
        >>> def render_function(name, value):
        >>>     s = wid.selected_values['selected_groups']
        >>>     print_dynamic(s)

    Create the widget with some initial options and display it:

        >>> fitting_result_iterations_options = {'n_iters': 10,
        >>>                                      'image_has_gt_shape': True,
        >>>                                      'n_points': 68,
        >>>                                      'iter_str': 'iter_',
        >>>                                      'selected_groups': [0],
        >>>                                      'render_image': True,
        >>>                                      'subplots_enabled': True,
        >>>                                      'displacement_type': 'mean'}
        >>> wid = FittingResultIterationsOptionsWidget(
        >>>     fitting_result_iterations_options,
        >>>     render_function=render_function, style='info',
        >>>     sliders_style='danger')
        >>> display(wid)

    By changing the various widgets, the printed message gets updated. Finally,
    let's change the widget status with a new set of options:

        >>> fitting_result_iterations_options = {'n_iters': 1,
        >>>                                      'image_has_gt_shape': True,
        >>>                                      'n_points': 5,
        >>>                                      'iter_str': 'iter_',
        >>>                                      'selected_groups': [0],
        >>>                                      'render_image': False,
        >>>                                      'subplots_enabled': False,
        >>>                                      'displacement_type': 'max'}
        >>> wid.set_widget_state(fitting_result_iterations_options,
        >>>                      allow_callback=True)
    """
    def __init__(self, fitting_result_iterations_options, render_function=None,
                 plot_errors_function=None, plot_displacements_function=None,
                 style='minimal', sliders_style='minimal'):
        # Create widgets
        self.iterations_mode = ipywidgets.RadioButtons(
            options={'Animation': 'animation', 'Static': 'static'},
            value='animation', description='Iterations:', margin='0.15cm')
        self.index_selection = {
            'min': 0, 'max': fitting_result_iterations_options['n_iters'] - 1,
            'step': 1, 'index': 0}
        self.index_animation = AnimationOptionsWidget(
            self.index_selection, description='',
            index_style='slider', loop_enabled=False, interval=0.2)
        self.index_slider = ipywidgets.IntRangeSlider(
            min=0, max=fitting_result_iterations_options['n_iters'] - 1, step=1,
            value=(0, 0), description='', margin='0.15cm', width='6cm')
        self.common_figure = ipywidgets.ToggleButton(
            description='Common figure', margin='0.15cm',
            value=not fitting_result_iterations_options['subplots_enabled'])
        self.render_image = ipywidgets.Checkbox(
            description='Render image', margin='0.1cm',
            value=fitting_result_iterations_options['render_image'])
        self.plot_errors_button = ipywidgets.Button(
            description='Errors', margin='0.1cm',
            visible=fitting_result_iterations_options['image_has_gt_shape'])
        self.plot_displacements_button = ipywidgets.Button(
            description='Displacements')
        dropdown_menu = OrderedDict()
        dropdown_menu['mean'] = 'mean'
        dropdown_menu['median'] = 'median'
        dropdown_menu['max'] = 'max'
        dropdown_menu['min'] = 'min'
        for p in range(fitting_result_iterations_options['n_points']):
            dropdown_menu["point {}".format(p)] = p
        self.plot_displacements_menu = ipywidgets.Dropdown(
            options=dropdown_menu, visible=False,
            value=fitting_result_iterations_options['displacement_type'])

        # Group widgets
        self.index_slider_and_common_figure = ipywidgets.HBox(
            children=[self.index_slider, self.common_figure], align='start',
            visible=False)
        self.index_box = ipywidgets.VBox(
            children=[self.index_animation,
                      self.index_slider_and_common_figure])
        self.iterations_mode_and_sliders = ipywidgets.HBox(
            children=[self.iterations_mode, self.index_box], align='center',
            margin='0.2cm')
        self.plot_displacements = ipywidgets.HBox(
            children=[self.plot_displacements_button,
                      self.plot_displacements_menu], align='center',
            margin='0.1cm')
        self.render_image_and_plot = ipywidgets.HBox(
            children=[self.render_image, self.plot_errors_button,
                      self.plot_displacements], margin='0.2cm')
        self._widget_state_based_on_n_iters(
            fitting_result_iterations_options['n_iters'])
        super(FittingResultIterationsOptionsWidget, self).__init__(
            children=[self.iterations_mode_and_sliders,
                      self.render_image_and_plot])

        # Assign output
        self.selected_values = fitting_result_iterations_options

        # Set style
        self.predefined_style(style, sliders_style)

        # Set functionality
        def displacements_button_function(name):
            self.plot_displacements_menu.visible = True
            if self._plot_displacements_function is not None:
                self._plot_displacements_function(
                    '', self.plot_displacements_menu.value)
        self.plot_displacements_button.on_click(displacements_button_function)

        def displacements_menu_function(name, value):
            self.selected_values['displacement_type'] = \
                self.plot_displacements_menu.value
        self.plot_displacements_menu.on_trait_change(
            displacements_menu_function, 'value')

        def errors_function(name):
            self.plot_displacements_menu.visible = False
        self.plot_errors_button.on_click(errors_function)

        def iterations_mode_function(name, value):
            if value == 'animation':
                # Get value that needs to be assigned
                val = self.index_slider.value[0]
                # Update visibility
                self.index_animation.visible = True
                self.index_slider_and_common_figure.visible = False
                # Set correct values
                self.index_animation.index_wid.slider.value = val
            else:
                # Stop the animation
                self.index_animation.stop_toggle.value = True
                # Get value that needs to be assigned
                val = self.index_animation.selected_values['index']
                # Update visibility
                self.index_animation.visible = False
                self.index_slider_and_common_figure.visible = True
                # Set correct values
                self.index_slider.value = (val, val)
        self.iterations_mode.on_trait_change(iterations_mode_function, 'value')

        # Set render, plot displacements and plot errors functions
        self._render_function_tmp = None
        self._render_function = None
        self.add_render_function(render_function)
        self._plot_displacements_function = None
        self.add_plot_displacements_function(plot_displacements_function)
        self._plot_errors_function = None
        self.add_plot_errors_function(plot_errors_function)

    def style(self, box_style=None, border_visible=False, border_color='black',
              border_style='solid', border_width=1, border_radius=0, padding=0,
              margin=0, sliders_box_style=None, sliders_border_visible=False,
              sliders_border_color='black', sliders_border_style='solid',
              sliders_border_width=1, sliders_border_radius=0,
              font_family='', font_size=None, font_style='',
              font_weight='', plot_buttons_style=''):
        r"""
        Function that defines the styling of the widget.

        Parameters
        ----------
        box_style : See Below, optional
            Style options

                ========= ============================
                Style     Description
                ========= ============================
                'success' Green-based style
                'info'    Blue-based style
                'warning' Yellow-based style
                'danger'  Red-based style
                ''        Default style
                None      No style
                ========= ============================

        border_visible : `bool`, optional
            Defines whether to draw the border line around the widget.
        border_color : `str`, optional
            The color of the border around the widget.
        border_style : `str`, optional
            The line style of the border around the widget.
        border_width : `float`, optional
            The line width of the border around the widget.
        border_radius : `float`, optional
            The radius of the corners of the box.
        padding : `float`, optional
            The padding around the widget.
        margin : `float`, optional
            The margin around the widget.
        sliders_box_style : See Below, optional
            Style options

                ========= ============================
                Style     Description
                ========= ============================
                'success' Green-based style
                'info'    Blue-based style
                'warning' Yellow-based style
                'danger'  Red-based style
                ''        Default style
                None      No style
                ========= ============================

        sliders_border_visible : `bool`, optional
            Defines whether to draw the border line around the sliders' box.
        sliders_border_color : `str`, optional
            The color of the border around the sliders' box.
        sliders_border_style : `str`, optional
            The line style of the border around the sliders' box.
        sliders_border_width : `float`, optional
            The line width of the border around the sliders' box.
        sliders_border_radius : `float`, optional
            The radius of the corners of the sliders' box.
        font_family : See Below, optional
            The font family to be used.
            Example options ::

                {'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace',
                 'helvetica'}

        font_size : `int`, optional
            The font size.
        font_style : {``'normal'``, ``'italic'``, ``'oblique'``}, optional
            The font style.
        font_weight : See Below, optional
            The font weight.
            Example options ::

                {'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
                 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
                 'extra bold', 'black'}

        plot_buttons_style : See Below, optional
            Style options

                ========= ============================
                Style     Description
                ========= ============================
                'primary' Blue-based style
                'success' Green-based style
                'info'    Blue-based style
                'warning' Yellow-based style
                'danger'  Red-based style
                ''        Default style
                None      No style
                ========= ============================
        """
        _format_box(self, box_style, border_visible, border_color, border_style,
                    border_width, border_radius, padding, margin)
        _format_box(self.iterations_mode_and_sliders, sliders_box_style,
                    sliders_border_visible, sliders_border_color,
                    sliders_border_style, sliders_border_width,
                    sliders_border_radius, 0, '0.2cm')
        if sliders_box_style == '' or sliders_box_style is None:
            self.index_animation.play_toggle.button_style = ''
            self.index_animation.play_toggle.font_weight = 'normal'
            self.index_animation.stop_toggle.button_style = ''
            self.index_animation.stop_toggle.font_weight = 'normal'
            self.index_animation.play_options_toggle.button_style = ''
            _format_box(self.index_animation.loop_interval_box, '', False,
                        'black', 'solid', 1, 10, '0.1cm', '0.1cm')
            self.index_animation.index_wid.slider.slider_color = ''
            self.index_animation.index_wid.slider.background_color = ''
            self.index_slider.slider_color = ''
            self.index_slider.background_color = ''
            self.common_figure.button_style = ''
        else:
            self.index_animation.play_toggle.button_style = 'success'
            self.index_animation.play_toggle.font_weight = 'bold'
            self.index_animation.stop_toggle.button_style = 'danger'
            self.index_animation.stop_toggle.font_weight = 'bold'
            self.index_animation.play_options_toggle.button_style = 'info'
            _format_box(self.index_animation.loop_interval_box, 'info', True,
                        _map_styles_to_hex_colours('info'), 'solid', 1, 10,
                        '0.1cm', '0.1cm')
            self.index_animation.index_wid.slider.slider_color = \
                _map_styles_to_hex_colours(sliders_box_style)
            self.index_animation.index_wid.slider.background_color = \
                _map_styles_to_hex_colours(sliders_box_style)
            self.index_slider.slider_color = \
                _map_styles_to_hex_colours(sliders_box_style)
            self.index_slider.background_color = \
                _map_styles_to_hex_colours(sliders_box_style)
            self.common_figure.button_style = 'info'
        _format_font(self, font_family, font_size, font_style, font_weight)
        _format_font(self.iterations_mode, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.index_slider, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.common_figure, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.render_image, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.plot_errors_button, font_family, font_size,
                     font_style, font_weight)
        _format_font(self.plot_displacements_button, font_family, font_size,
                     font_style, font_weight)
        _format_font(self.plot_displacements_menu, font_family, font_size,
                     font_style, font_weight)
        self.plot_errors_button.button_style = plot_buttons_style
        self.plot_displacements_button.button_style = plot_buttons_style
        self.plot_displacements_menu.button_style = plot_buttons_style

    def predefined_style(self, style, sliders_style):
        r"""
        Function that sets a predefined style on the widget.

        Parameters
        ----------
        style : `str` (see below)
            The style of the widget. Possible options are

                ========= ============================
                Style     Description
                ========= ============================
                'minimal' Simple black and white style
                'success' Green-based style
                'info'    Blue-based style
                'warning' Yellow-based style
                'danger'  Red-based style
                ''        No style
                ========= ============================

        sliders_style : `str` (see below)
            The style of the sliders' box. Possible options are

                ========= ============================
                Style     Description
                ========= ============================
                'minimal' Simple black and white style
                'success' Green-based style
                'info'    Blue-based style
                'warning' Yellow-based style
                'danger'  Red-based style
                ''        No style
                ========= ============================
        """
        if style == 'minimal':
            self.style(box_style=None, border_visible=True,
                       border_color='black', border_style='solid',
                       border_width=1, border_radius=0, padding='0.2cm',
                       margin='0.3cm', sliders_box_style=None,
                       sliders_border_visible=True,
                       sliders_border_color='black',
                       sliders_border_style='solid', sliders_border_width=1,
                       sliders_border_radius=0, font_family='', font_size=None,
                       font_style='', font_weight='', plot_buttons_style='')
        elif (style == 'info' or style == 'success' or style == 'danger' or
              style == 'warning'):
            sliders_border_visible = True
            if style == sliders_style:
                sliders_border_visible = False
            self.style(
                box_style=style, border_visible=True,
                border_color=_map_styles_to_hex_colours(style),
                border_style='solid', border_width=1, border_radius=10,
                padding='0.2cm', margin='0.3cm',
                sliders_box_style=sliders_style,
                sliders_border_visible=sliders_border_visible,
                sliders_border_color=_map_styles_to_hex_colours(sliders_style),
                sliders_border_style='solid', sliders_border_width=1,
                sliders_border_radius=10, font_family='', font_size=None,
                font_style='', font_weight='', plot_buttons_style='primary')
        else:
            raise ValueError('style must be minimal or info or success or '
                             'danger or warning')

    def _convert_iterations_to_groups(self, from_iter, to_iter, iter_str):
        r"""
        Function that generates a `list` of group labels given the range bounds
        and the `str` to be used.
        """
        return ["{}{}".format(iter_str, i)
                for i in range(from_iter, to_iter + 1)]

    def _get_selected_options(self):
        self.selected_values['render_image'] = self.render_image.value
        self.selected_values['subplots_enabled'] = not self.common_figure.value
        if self.iterations_mode.value == 'animation':
            self.selected_values['selected_groups'] = \
                self._convert_iterations_to_groups(
                    self.index_animation.selected_values['index'],
                    self.index_animation.selected_values['index'],
                    self.selected_values['iter_str'])
        else:
            self.selected_values['selected_groups'] = \
                self._convert_iterations_to_groups(
                    self.index_slider.value[0], self.index_slider.value[1],
                    self.selected_values['iter_str'])

    def _widget_state_based_on_n_iters(self, n_iters):
        if n_iters == 1:
            self.iterations_mode.value = 'animation'
            self.index_animation.index_wid.slider.description = 'Iterations:'
        self.iterations_mode.visible = not n_iters == 1
        self.plot_errors_button.visible = not n_iters == 1
        self.plot_displacements.visible = not n_iters == 1

    def _widget_state_based_on_image_has_gt_shape(self, image_has_gt_shape):
        self.plot_errors_button.visible = image_has_gt_shape

    def add_plot_displacements_function(self, plot_displacements_function):
        r"""
        Method that adds a `plot_displacements_function()` to the widget. The
        signature of the given function is also stored in
        `self._plot_displacements_function`.

        Parameters
        ----------
        plot_displacements_function : `function` or ``None``, optional
            The plot displacements function that behaves as a callback.
            If ``None``, then nothing is added.
        """
        self._plot_displacements_function = plot_displacements_function
        if self._plot_displacements_function is not None:
            self.plot_displacements_menu.on_trait_change(
                self._plot_displacements_function, 'value')

    def remove_plot_displacements_function(self):
        r"""
        Method that removes the current `self._plot_displacements_function()`
        from the widget and sets ``self._plot_displacements_function = None``.
        """
        self.plot_displacements_menu.on_trait_change(
            self._plot_displacements_function, 'value', remove=True)
        self._plot_displacements_function = None

    def replace_plot_displacements_function(self, plot_displacements_function):
        r"""
        Method that replaces the current `self._plot_displacements_function()`
        of the widget with the given `plot_displacements_function()`.

        Parameters
        ----------
        plot_displacements_function : `function` or ``None``, optional
            The plot displacements function that behaves as a callback. If
            ``None``, then nothing happens.
        """
        # remove old function
        self.remove_plot_displacements_function()

        # add new function
        self.add_plot_displacements_function(plot_displacements_function)

    def add_plot_errors_function(self, plot_errors_function):
        r"""
        Method that adds a `plot_errors_function()` to the widget. The
        signature of the given function is also stored in
        `self._plot_errors_function`.

        Parameters
        ----------
        plot_errors_function : `function` or ``None``, optional
            The plot errors function that behaves as a callback.
            If ``None``, then nothing is added.
        """
        self._plot_errors_function = plot_errors_function
        if self._plot_errors_function is not None:

            self.plot_errors_button.on_click(self._plot_errors_function)

    def remove_plot_errors_function(self):
        r"""
        Method that removes the current `self._plot_errors_function()` from the
        widget and sets ``self._plot_errors_function = None``.
        """
        self.plot_errors_button.on_click(self._plot_errors_function,
                                         remove=True)
        self._plot_errors_function = None

    def replace_plot_errors_function(self, plot_errors_function):
        r"""
        Method that replaces the current `self._plot_errors_function()` of the
        widget with the given `plot_errors_function()`.

        Parameters
        ----------
        plot_errors_function : `function` or ``None``, optional
            The plot errors function that behaves as a callback. If ``None``,
            then nothing happens.
        """
        # remove old function
        self.remove_plot_errors_function()

        # add new function
        self.add_plot_errors_function(plot_errors_function)

    def add_render_function(self, render_function):
        r"""
        Method that adds a `render_function()` to the widget. The signature of
        the given function is also stored in `self._render_function`.

        Parameters
        ----------
        render_function : `function` or ``None``, optional
            The render function that behaves as a callback. If ``None``, then
            nothing is added.
        """
        self._render_function_tmp = render_function
        if self._render_function_tmp is None:
            def render_function_with_get_options(name, value):
                # Get all the selected values
                self._get_selected_options()
                # Make displacements menu invisible
                self.plot_displacements_menu.visible = False
        else:
            def render_function_with_get_options(name, value):
                # Get all the selected values
                self._get_selected_options()
                # Make displacements menu invisible
                self.plot_displacements_menu.visible = False
                # Call render function
                self._render_function_tmp(name, value)
        self._render_function = render_function_with_get_options
        self.index_animation.add_render_function(self._render_function)
        self.index_slider.on_trait_change(self._render_function, 'value')
        self.iterations_mode.on_trait_change(self._render_function, 'value')
        self.common_figure.on_trait_change(self._render_function, 'value')
        self.render_image.on_trait_change(self._render_function, 'value')

    def remove_render_function(self):
        r"""
        Method that removes the current `self._render_function()` from the
        widget and sets ``self._render_function = None``.
        """
        self.index_animation.remove_render_function()
        self.index_slider.on_trait_change(self._render_function, 'value',
                                          remove=True)
        self.iterations_mode.on_trait_change(self._render_function, 'value',
                                             remove=True)
        self.common_figure.on_trait_change(self._render_function, 'value',
                                           remove=True)
        self.render_image.on_trait_change(self._render_function, 'value',
                                          remove=True)
        self._render_function = None

    def replace_render_function(self, render_function):
        r"""
        Method that replaces the current `self._render_function()` of the widget
        with the given `render_function()`.

        Parameters
        ----------
        render_function : `function` or ``None``, optional
            The render function that behaves as a callback. If ``None``, then
            nothing is happening.
        """
        # remove old function
        self.remove_render_function()

        # add new function
        self.add_render_function(render_function)

    def set_widget_state(self, fitting_result_iterations_options,
                         allow_callback=True):
        r"""
        Method that updates the state of the widget with a new set of values.

        Parameters
        ----------
        fitting_result_iterations_options : `dict`
            The dictionary with the initial options. For example
            ::

                fitting_result_iterations_options = {'n_iters': 10,
                                                     'image_has_gt_shape': True,
                                                     'n_points': 68,
                                                     'iter_str': 'iter_',
                                                     'selected_groups': [0],
                                                     'render_image': True,
                                                     'subplots_enabled': True,
                                                     'displacement_type': 'max'}

        allow_callback : `bool`, optional
            If ``True``, it allows triggering of any callback functions.
        """
        # temporarily remove render callback
        render_function = self._render_function_tmp
        self.remove_render_function()

        # Set displacements menu invisible
        self.plot_displacements_menu.visible = False

        # Update render image checkbox
        if fitting_result_iterations_options['render_image'] is not None:
            self.render_image.value = \
                fitting_result_iterations_options['render_image']

        # Update index
        if fitting_result_iterations_options['n_iters'] is not None:
            self._widget_state_based_on_n_iters(
                fitting_result_iterations_options['n_iters'])
            self.index_selection = {
                'min': 0, 'step': 1, 'index': 0,
                'max': fitting_result_iterations_options['n_iters'] - 1}
            self.index_animation.set_widget_state(self.index_selection,
                                                  allow_callback=False)
            self.index_slider.value = (0, 0)
            self.index_slider.max = \
                fitting_result_iterations_options['n_iters'] - 1

        # Update common figure toggle
        if fitting_result_iterations_options['subplots_enabled'] is not None:
            self.common_figure.value = \
                fitting_result_iterations_options['subplots_enabled']

        # Update plot errors
        if fitting_result_iterations_options['image_has_gt_shape'] is not None:
            self._widget_state_based_on_image_has_gt_shape(
                fitting_result_iterations_options['image_has_gt_shape'])

        # Update displacements menu
        if (fitting_result_iterations_options['n_points'] is not None and
                not fitting_result_iterations_options['n_points'] ==
                self.selected_values['n_points']):
            dropdown_menu = OrderedDict()
            dropdown_menu['mean'] = 'mean'
            dropdown_menu['median'] = 'median'
            dropdown_menu['max'] = 'max'
            dropdown_menu['min'] = 'min'
            for p in range(fitting_result_iterations_options['n_points']):
                dropdown_menu["point {}".format(p)] = p
            self.plot_displacements_menu.options = dropdown_menu

        # Re-assign render callback
        self.add_render_function(render_function)

        # Assign new options dict to selected_values
        self.selected_values = fitting_result_iterations_options

        # trigger render function if allowed
        if allow_callback:
            self._render_function('', True)
