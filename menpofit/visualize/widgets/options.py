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

    def replace_variancer_function(self, variance_function):
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
    Creates a widget for selecting parameters values when visualizing a linear
    model (e.g. PCA model). The widget consists of the following parts from
    `IPython.html.widgets`:

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
    fitting_result_options : `list`
        The dictionary with the initial options. For example
        ::

            fitting_result_options = {'all_groups': ['initial', 'final',
                                                     'ground'],
                                      'selected_groups': ['final'],
                                      'render_image': True,
                                      'subplots_enabled': True}

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
        fitting_result_options : `list`
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


def _convert_iterations_to_groups(from_iter, to_iter, iter_str):
    r"""
    Function that generates a `list` of group labels given the range bounds and
    the `str` to be used.
    """
    return ["{}{}".format(iter_str, i) for i in range(from_iter, to_iter + 1)]


class FittingResultIterationsOptionsWidget(ipywidgets.FlexBox):
    r"""
    Creates a widget for selecting parameters values when visualizing a linear
    model (e.g. PCA model). The widget consists of the following parts from
    `IPython.html.widgets`:

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
    fitting_result_options : `list`
        The dictionary with the initial options. For example
        ::

            fitting_result_options = {'all_groups': ['initial', 'final',
                                                     'ground'],
                                      'selected_groups': ['final'],
                                      'render_image': True,
                                      'subplots_enabled': True}

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
    def __init__(self, fitting_result_iterations_options, render_function=None,
                 plot_errors_function=None, plot_displacements_function=None,
                 style='minimal'):
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
            value=(0, 0), description='', margin='0.15cm')
        self.index_slider.width = '6cm'
        self.common_figure = ipywidgets.ToggleButton(
            description='Common figure', margin='0.15cm',
            value=not fitting_result_iterations_options['subplots_enabled'])
        self.render_image = ipywidgets.Checkbox(
            description='Render image',
            value=fitting_result_iterations_options['render_image'])
        self.plot_errors_button = ipywidgets.Button(description='Errors')
        self.plot_displacements_button = ipywidgets.Button(
            description='Displacements')
        dropdown_menu = OrderedDict()
        dropdown_menu['mean'] = 'mean'
        dropdown_menu['median'] = 'median'
        dropdown_menu['max'] = 'max'
        dropdown_menu['min'] = 'min'
        for p in range(fitting_result_iterations_options['n_points']):
            dropdown_menu["point {}".format(p)] = p
        self.plot_displacements_menu = ipywidgets.Select(
            options=dropdown_menu, height='2cm', width='2.5cm', visible=False,
            value=fitting_result_iterations_options['displacement_type'])

        # Group widgets
        self.index_slider_and_common_figure = ipywidgets.HBox(
            children=[self.index_slider, self.common_figure], align='start',
            visible=False)
        self.index_box = ipywidgets.VBox(
            children=[self.index_animation,
                      self.index_slider_and_common_figure])
        self.iterations_mode_and_sliders = ipywidgets.HBox(
            children=[self.iterations_mode, self.index_box], align='center')
        self.plot_displacements = ipywidgets.VBox(
            children=[self.plot_displacements_button,
                      self.plot_displacements_menu], align='center')
        self.render_image_and_plot = ipywidgets.HBox(
            children=[self.render_image, self.plot_errors_button,
                      self.plot_displacements])
        super(FittingResultIterationsOptionsWidget, self).__init__(
            children=[self.iterations_mode_and_sliders,
                      self.render_image_and_plot])

        # Assign output
        #self.selected_values = index

        # Set style
        #self.predefined_style(style)

        # Set functionality
        self._plot_displacements_function = None
        self.add_plot_displacements_function(plot_displacements_function)

        def displacements_button_function(name):
            self.plot_displacements_menu.visible = True
            if self._plot_displacements_function is not None:
                self._plot_displacements_function(
                    '', self.plot_displacements_menu.value)
        self.plot_displacements_button.on_click(displacements_button_function)

        self._plot_errors_function = None
        self.add_plot_errors_function(plot_errors_function)

        def iterations_mode_function(name, value):
            if value == 'animation':
                # Get value that needs to be assigned
                val = self.index_slider.value[0]
                # Update visibility
                self.index_animation.visible = True
                self.index_slider_and_common_figure.visible = False
                # Set correct values
                self.index_animation.index_wid.slider.value = val
                #animation_wid.selected_values['index'] = val
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

        def render_image_function(name, value):
            self.selected_values['render_image'] = value
        self.render_image.on_trait_change(render_image_function, 'value')

        def common_figure_function(name, value):
            self.selected_values['subplots_enabled'] = not value
        self.common_figure.on_trait_change(common_figure_function, 'value')

        # def get_groups(name, value):
        #     if self.iterations_mode.value == 'animation':
        #         self.selected_values['selected_groups'] = \
        #             _convert_iterations_to_groups(
        #                 self.index_animation.selected_values['index'],
        #                 self.index_animation.selected_values['index'],
        #                 self.selected_values['iter_str'])
        #     else:
        #         self.selected_values['selected_groups'] = \
        #             _convert_iterations_to_groups(
        #                 self.index_slider.value[0], self.index_slider.value[1],
        #                 self.selected_values['iter_str'])
        # self.index_slider.on_trait_change(get_groups, 'value')
        # self.index_animation.index_wid.slider.on_trait_change(get_groups, 'value')

    def _get_selected_options(self):
        if self.iterations_mode.value == 'animation':
            self.selected_values['selected_groups'] = \
                _convert_iterations_to_groups(
                    self.index_animation.selected_values['index'],
                    self.index_animation.selected_values['index'],
                    self.selected_values['iter_str'])
        else:
            self.selected_values['selected_groups'] = \
                _convert_iterations_to_groups(
                    self.index_slider.value[0], self.index_slider.value[1],
                    self.selected_values['iter_str'])


    def _widget_state_based_on_n_iters(self, n_iters):
        if n_iters == 1:
            self.iterations_mode.value = 'animation'
        self.iterations_mode.visible = n_iters == 1
        self.index_slider.visible = n_iters == 1
        self.plot_errors_button.visible = n_iters == 1
        self.plot_displacements_button.visible = n_iters == 1
        self.plot_displacements_menu.visible = n_iters == 1

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
        if plot_errors_function is None:
            self._plot_errors_function = None
        else:
            def plot_errors_function_mine(name):
                self.plot_displacements_menu.visible = False
            self._plot_errors_function

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


# def iterations_result_options(iterations_result_options_default,
#                               plot_function=None, plot_errors_function=None,
#                               plot_displacements_function=None,
#                               title='Iterations Result',
#                               toggle_show_default=True,
#                               toggle_show_visible=True):
#     r"""
#     Creates a widget with Iterations Result Options. Specifically, it has:
#         1) Two radio buttons that select an options mode, depending on whether
#            the user wants to visualize iterations in ``Animation`` or ``Static``
#            mode.
#         2) If mode is ``Animation``, an animation options widget appears.
#            If mode is ``Static``, the iterations range is selected by two
#            sliders and there is an update plot button.
#         3) A checkbox that controls the visibility of the image.
#         4) A set of radio buttons that define whether subplots are enabled.
#         5) A button to plot the error evolution.
#         6) A button to plot the landmark points' displacement.
#         7) A drop down menu to select which displacement to plot.
#         8) A toggle button that controls the visibility of all the above, i.e.
#            the final result options.
#
#     The structure of the widgets is the following:
#         iterations_result_wid.children = [toggle_button, all_options]
#         all_options.children = [iterations_mode_and_sliders, options]
#         iterations_mode_and_sliders.children = [iterations_mode_radio_buttons,
#                                                 all_sliders]
#         all_sliders.children = [animation_slider, first_slider, second_slider,
#                                 update_and_axes]
#         update_and_axes.children = [same_axes_checkbox, update_button]
#         options.children = [render_image_checkbox, plot_errors_button,
#                             plot_displacements]
#         plot_displacements.children = [plot_displacements_button,
#                                        plot_displacements_drop_down_menu]
#
#     The returned widget saves the selected values in the following fields:
#         iterations_result_wid.selected_values
#
#     To fix the alignment within this widget please refer to
#     `format_iterations_result_options()` function.
#
#     To update the state of this widget, please refer to
#     `update_iterations_result_options()` function.
#
#     Parameters
#     ----------
#     iterations_result_options_default : `dict`
#         The default options. For example:
#             iterations_result_options_default = {'n_iters': 10,
#                                                  'image_has_gt_shape': True,
#                                                  'n_points': 68,
#                                                  'iter_str': 'iter_',
#                                                  'selected_groups': [0],
#                                                  'render_image': True,
#                                                  'subplots_enabled': True,
#                                                  'displacement_type': 'mean'}
#     plot_function : `function` or None, optional
#         The plot function that is executed when a widgets' value changes.
#         If None, then nothing is assigned.
#     plot_errors_function : `function` or None, optional
#         The plot function that is executed when the 'Plot Errors' button is
#         pressed.
#         If None, then nothing is assigned.
#     plot_displacements_function : `function` or None, optional
#         The plot function that is executed when the 'Plot Displacements' button
#         is pressed.
#         If None, then nothing is assigned.
#     title : `str`, optional
#         The title of the widget printed at the toggle button.
#     toggle_show_default : `bool`, optional
#         Defines whether the options will be visible upon construction.
#     toggle_show_visible : `bool`, optional
#         The visibility of the toggle button.
#     """
#     import IPython.html.widgets as ipywidgets
#     # Create all necessary widgets
#     but = ipywidgets.ToggleButton(description=title,
#                                         value=toggle_show_default,
#                                         visible=toggle_show_visible)
#     iterations_mode = ipywidgets.RadioButtons(
#         options={'Animation': 'animation', 'Static': 'static'},
#         value='animation', description='Mode:', visible=toggle_show_default)
#     # Don't assign the plot function to the animation_wid at this point. We
#     # first need to assign the get_groups function and then the plot_function()
#     # for synchronization reasons.
#     index_selection_default = {
#         'min': 0, 'max': iterations_result_options_default['n_iters'] - 1,
#         'step': 1, 'index': 0}
#     animation_wid = animation_options(
#         index_selection_default, plot_function=None, update_function=None,
#         index_description='Iteration', index_style='slider', loop_default=False,
#         interval_default=0.2, toggle_show_default=toggle_show_default,
#         toggle_show_visible=False)
#     first_slider_wid = ipywidgets.IntSlider(
#         min=0, max=iterations_result_options_default['n_iters'] - 1, step=1,
#         value=0, description='From', visible=False)
#     second_slider_wid = ipywidgets.IntSlider(
#         min=0, max=iterations_result_options_default['n_iters'] - 1, step=1,
#         value=iterations_result_options_default['n_iters'] - 1,
#         description='To', visible=False)
#     same_axes = ipywidgets.Checkbox(
#         description='Same axes',
#         value=not iterations_result_options_default['subplots_enabled'],
#         visible=False)
#     update_but = ipywidgets.Button(description='Update Plot',
#                                          visible=False)
#     render_image = ipywidgets.Checkbox(
#         description='Render image',
#         value=iterations_result_options_default['render_image'])
#     plot_errors_button = ipywidgets.Button(description='Plot Errors')
#     plot_displacements_button = ipywidgets.Button(
#         description='Plot Displacements')
#     dropdown_menu = OrderedDict()
#     dropdown_menu['mean'] = 'mean'
#     dropdown_menu['median'] = 'median'
#     dropdown_menu['max'] = 'max'
#     dropdown_menu['min'] = 'min'
#     for p in range(iterations_result_options_default['n_points']):
#         dropdown_menu["point {}".format(p)] = p
#     plot_displacements_menu = ipywidgets.Dropdown(
#         options=dropdown_menu,
#         value=iterations_result_options_default['displacement_type'])
#
#     # if just one iteration, disable multiple options
#     if iterations_result_options_default['n_iters'] == 1:
#         iterations_mode.value = 'animation'
#         iterations_mode.disabled = True
#         first_slider_wid.disabled = True
#         animation_wid.children[1].children[0].disabled = True
#         animation_wid.children[1].children[1].children[0].children[0].\
#             disabled = True
#         animation_wid.children[1].children[1].children[0].children[1].\
#             disabled = True
#         animation_wid.children[1].children[1].children[0].children[2].\
#             disabled = True
#         second_slider_wid.disabled = True
#         plot_errors_button.disabled = True
#         plot_displacements_button.disabled = True
#         plot_displacements_menu.disabled = True
#
#     # Group widgets
#     update_and_subplots = ipywidgets.Box(
#         children=[same_axes, update_but])
#     sliders = ipywidgets.Box(
#         children=[animation_wid, first_slider_wid, second_slider_wid,
#                   update_and_subplots])
#     iterations_mode_and_sliders = ipywidgets.Box(
#         children=[iterations_mode, sliders])
#     plot_displacements = ipywidgets.Box(
#         children=[plot_displacements_button, plot_displacements_menu])
#     opts = ipywidgets.Box(
#         children=[render_image, plot_errors_button, plot_displacements])
#     all_options = ipywidgets.Box(
#         children=[iterations_mode_and_sliders, opts])
#
#     # Widget container
#     iterations_result_wid = ipywidgets.Box(children=[but,
#                                                                  all_options])
#
#     # Initialize variables
#     iterations_result_options_default['selected_groups'] = \
#         _convert_iterations_to_groups(
#             0, 0, iterations_result_options_default['iter_str'])
#     iterations_result_wid.selected_values = iterations_result_options_default
#
#     # Define iterations mode visibility
#     def iterations_mode_selection(name, value):
#         if value == 'animation':
#             # get val that needs to be assigned
#             val = first_slider_wid.value
#             # update visibility
#             animation_wid.visible = True
#             first_slider_wid.visible = False
#             second_slider_wid.visible = False
#             same_axes.visible = False
#             update_but.visible = False
#             # set correct values
#             animation_wid.children[1].children[0].value = val
#             animation_wid.selected_values['index'] = val
#             first_slider_wid.value = 0
#             second_slider_wid.value = \
#                 iterations_result_wid.selected_values['n_iters'] - 1
#         else:
#             # get val that needs to be assigned
#             val = animation_wid.selected_values['index']
#             # update visibility
#             animation_wid.visible = False
#             first_slider_wid.visible = True
#             second_slider_wid.visible = True
#             same_axes.visible = True
#             update_but.visible = True
#             # set correct values
#             second_slider_wid.value = val
#             first_slider_wid.value = val
#             animation_wid.children[1].children[0].value = 0
#             animation_wid.selected_values['index'] = 0
#     iterations_mode.on_trait_change(iterations_mode_selection, 'value')
#
#     # Check first slider's value
#     def first_slider_val(name, value):
#         if value > second_slider_wid.value:
#             first_slider_wid.value = second_slider_wid.value
#     first_slider_wid.on_trait_change(first_slider_val, 'value')
#
#     # Check second slider's value
#     def second_slider_val(name, value):
#         if value < first_slider_wid.value:
#             second_slider_wid.value = first_slider_wid.value
#     second_slider_wid.on_trait_change(second_slider_val, 'value')
#
#     # Convert slider values to groups
#     def get_groups(name, value):
#         if iterations_mode.value == 'animation':
#             iterations_result_wid.selected_values['selected_groups'] = \
#                 _convert_iterations_to_groups(
#                     animation_wid.selected_values['index'],
#                     animation_wid.selected_values['index'],
#                     iterations_result_wid.selected_values['iter_str'])
#         else:
#             iterations_result_wid.selected_values['selected_groups'] = \
#                 _convert_iterations_to_groups(
#                     first_slider_wid.value, second_slider_wid.value,
#                     iterations_result_wid.selected_values['iter_str'])
#     first_slider_wid.on_trait_change(get_groups, 'value')
#     second_slider_wid.on_trait_change(get_groups, 'value')
#
#     # assign get_groups() to the slider of animation_wid
#     animation_wid.children[1].children[0].on_trait_change(get_groups, 'value')
#
#     # Render image function
#     def render_image_fun(name, value):
#         iterations_result_wid.selected_values['render_image'] = value
#     render_image.on_trait_change(render_image_fun, 'value')
#
#     # Same axes function
#     def same_axes_fun(name, value):
#         iterations_result_wid.selected_values['subplots_enabled'] = not value
#     same_axes.on_trait_change(same_axes_fun, 'value')
#
#     # Displacement type function
#     def displacement_type_fun(name, value):
#         iterations_result_wid.selected_values['displacement_type'] = value
#     plot_displacements_menu.on_trait_change(displacement_type_fun, 'value')
#
#     # Toggle button function
#     def show_options(name, value):
#         iterations_mode.visible = value
#         render_image.visible = value
#         plot_errors_button.visible = \
#             iterations_result_wid.selected_values['image_has_gt_shape'] and value
#         plot_displacements.visible = value
#         if value:
#             if iterations_mode.value == 'animation':
#                 animation_wid.visible = True
#             else:
#                 first_slider_wid.visible = True
#                 second_slider_wid.visible = True
#                 same_axes.visible = True
#                 update_but.visible = True
#         else:
#             animation_wid.visible = False
#             first_slider_wid.visible = False
#             second_slider_wid.visible = False
#             same_axes.visible = False
#             update_but.visible = False
#     show_options('', toggle_show_default)
#     but.on_trait_change(show_options, 'value')
#
#     # assign general plot_function
#     if plot_function is not None:
#         def plot_function_but(name):
#             plot_function(name, 0)
#
#         update_but.on_click(plot_function_but)
#         # Here we assign plot_function() to the slider of animation_wid, as
#         # we didn't do it at its creation.
#         animation_wid.children[1].children[0].on_trait_change(plot_function,
#                                                               'value')
#         render_image.on_trait_change(plot_function, 'value')
#         iterations_mode.on_trait_change(plot_function, 'value')
#
#     # assign plot function of errors button
#     if plot_errors_function is not None:
#         plot_errors_button.on_click(plot_errors_function)
#
#     # assign plot function of displacements button
#     if plot_displacements_function is not None:
#         plot_displacements_button.on_click(plot_displacements_function)
#
#     return iterations_result_wid
#
#
# def format_iterations_result_options(iterations_result_wid,
#                                      container_padding='6px',
#                                      container_margin='6px',
#                                      container_border='1px solid black',
#                                      toggle_button_font_weight='bold',
#                                      border_visible=True):
#     r"""
#     Function that corrects the align (style format) of a given
#     iterations_result_options widget. Usage example:
#         iterations_result_wid = iterations_result_options()
#         display(iterations_result_wid)
#         format_iterations_result_options(iterations_result_wid)
#
#     Parameters
#     ----------
#     iterations_result_wid :
#         The widget object generated by the `iterations_result_options()`
#         function.
#     container_padding : `str`, optional
#         The padding around the widget, e.g. '6px'
#     container_margin : `str`, optional
#         The margin around the widget, e.g. '6px'
#     container_border : `str`, optional
#         The border around the widget, e.g. '1px solid black'
#     toggle_button_font_weight : `str`
#         The font weight of the toggle button, e.g. 'bold'
#     border_visible : `bool`, optional
#         Defines whether to draw the border line around the widget.
#     """
#     # format animations options
#     format_animation_options(
#         iterations_result_wid.children[1].children[0].children[1].children[0],
#         index_text_width='0.5cm', container_padding=container_padding,
#         container_margin=container_margin, container_border=container_border,
#         toggle_button_font_weight=toggle_button_font_weight,
#         border_visible=False)
#
#     # align displacement button and drop down menu
#     remove_class(iterations_result_wid.children[1].children[1].children[2], 'vbox')
#     add_class(iterations_result_wid.children[1].children[1].children[2], 'hbox')
#     add_class(iterations_result_wid.children[1].children[1].children[2], 'align-center')
#     iterations_result_wid.children[1].children[1].children[2].children[0].\
#         margin_right = '0px'
#     iterations_result_wid.children[1].children[1].children[2].children[1].\
#         margin_left = '0px'
#
#     # align options
#     remove_class(iterations_result_wid.children[1].children[1], 'vbox')
#     add_class(iterations_result_wid.children[1].children[1], 'hbox')
#     add_class(iterations_result_wid.children[1].children[1], 'align-center')
#     iterations_result_wid.children[1].children[1].children[0].\
#         margin_right = '30px'
#     iterations_result_wid.children[1].children[1].children[1].\
#         margin_right = '30px'
#
#     # align update button and same axes checkbox
#     remove_class(iterations_result_wid.children[1].children[0].children[1].children[3], 'vbox')
#     add_class(iterations_result_wid.children[1].children[0].children[1].children[3], 'hbox')
#     iterations_result_wid.children[1].children[0].children[1].children[3].children[0].\
#         margin_right = '20px'
#
#     # align sliders
#     add_class(iterations_result_wid.children[1].children[0].children[1], 'align-end')
#     iterations_result_wid.children[1].children[0].children[1].\
#         margin_bottom = '20px'
#
#     # align sliders and iterations_mode
#     remove_class(iterations_result_wid.children[1].children[0], 'vbox')
#     add_class(iterations_result_wid.children[1].children[0], 'hbox')
#     add_class(iterations_result_wid.children[1].children[0], 'align-start')
#
#     # align sliders and options
#     add_class(iterations_result_wid.children[1], 'align-end')
#
#     # set toggle button font bold
#     iterations_result_wid.children[0].font_weight = toggle_button_font_weight
#
#     # margin and border around container widget
#     iterations_result_wid.padding = container_padding
#     iterations_result_wid.margin = container_margin
#     if border_visible:
#         iterations_result_wid.border = container_border
#
#
# def update_iterations_result_options(iterations_result_wid,
#                                      iterations_result_default):
#     r"""
#     Function that updates the state of a given iterations_result_options widget. Usage example:
#         iterations_result_options_default = {'n_iters': 10,
#                                              'image_has_gt_shape': True,
#                                              'n_points': 68,
#                                              'iter_str': 'iter_',
#                                              'selected_groups': [0],
#                                              'render_image': True,
#                                              'subplots_enabled': True,
#                                              'displacement_type': 'mean'}
#         iterations_result_wid = iterations_result_options(iterations_result_options_default)
#         display(iterations_result_wid)
#         format_iterations_result_options(iterations_result_wid)
#         iterations_result_options_default = {'n_iters': 100,
#                                              'image_has_gt_shape': False,
#                                              'n_points': 15,
#                                              'iter_str': 'iter_',
#                                              'selected_groups': [0],
#                                              'render_image': False,
#                                              'subplots_enabled': False,
#                                              'displacement_type': 'median'}
#         update_iterations_result_options(iterations_result_wid, iterations_result_options_default)
#
#     Parameters
#     ----------
#     iterations_result_wid :
#         The widget generated by `iterations_result_options()` function.
#     iterations_result_options_default : `dict`
#         The default options. For example:
#             iterations_result_options_default = {'n_iters': 10,
#                                                  'image_has_gt_shape': True,
#                                                  'n_points': 68,
#                                                  'iter_str': 'iter_',
#                                                  'selected_groups': [0],
#                                                  'render_image': True,
#                                                  'subplots_enabled': True,
#                                                  'displacement_type': 'mean'}
#     """
#     # if image_has_gt_shape flag has actually changed from the previous value
#     if ('image_has_gt_shape' in iterations_result_default and
#         iterations_result_default['image_has_gt_shape'] !=
#                 iterations_result_wid.selected_values['image_has_gt_shape']):
#         # set the plot errors visibility
#         iterations_result_wid.children[1].children[1].children[1].visible = \
#             (iterations_result_wid.children[0].value and
#              iterations_result_default['image_has_gt_shape'])
#         # store the flag
#         iterations_result_wid.selected_values['image_has_gt_shape'] = \
#             iterations_result_default['image_has_gt_shape']
#
#     # if n_points has actually changed from the previous value
#     if ('n_points' in iterations_result_default and
#         iterations_result_default['n_points'] !=
#                 iterations_result_wid.selected_values['n_points']):
#         # change the contents of the displacement types
#         select_menu = OrderedDict()
#         select_menu['mean'] = 'mean'
#         select_menu['median'] = 'median'
#         select_menu['max'] = 'max'
#         select_menu['min'] = 'min'
#         for p in range(iterations_result_default['n_points']):
#             select_menu["point {}".format(p + 1)] = p
#         iterations_result_wid.children[1].children[1].children[2].children[1].\
#             values = select_menu
#         # store the number of points
#         iterations_result_wid.selected_values['n_points'] = \
#             iterations_result_default['n_points']
#
#     # if displacement_type has actually changed from the previous value
#     if ('displacement_type' in iterations_result_default and
#         iterations_result_default['displacement_type'] !=
#                 iterations_result_wid.selected_values['displacement_type']):
#         iterations_result_wid.children[1].children[1].children[2].children[1].\
#             value = iterations_result_default['displacement_type']
#
#     # if iter_str are actually different from the previous value
#     if ('iter_str' in iterations_result_default and
#         iterations_result_default['iter_str'] !=
#                 iterations_result_wid.selected_values['iter_str']):
#         iterations_result_wid.selected_values['iter_str'] = \
#             iterations_result_default['iter_str']
#
#     # if render_image are actually different from the previous value
#     if ('render_image' in iterations_result_default and
#         iterations_result_default['render_image'] !=
#                 iterations_result_wid.selected_values['render_image']):
#         iterations_result_wid.children[1].children[1].children[0].value = \
#             iterations_result_default['render_image']
#
#     # if subplots_enabled are actually different from the previous value
#     if ('subplots_enabled' in iterations_result_default and
#         iterations_result_default['subplots_enabled'] !=
#                 iterations_result_wid.selected_values['subplots_enabled']):
#         iterations_result_wid.children[1].children[0].children[1].children[3].children[0].value = \
#             not iterations_result_default['subplots_enabled']
#
#     # if n_iters are actually different from the previous value
#     if ('n_iters' in iterations_result_default and
#         iterations_result_default['n_iters'] !=
#                 iterations_result_wid.selected_values['n_iters']):
#         # change the iterations_result_wid output
#         iterations_result_wid.selected_values['n_iters'] = \
#             iterations_result_default['n_iters']
#         iterations_result_wid.selected_values['selected_groups'] = \
#             _convert_iterations_to_groups(
#                 0, 0, iterations_result_wid.selected_values['iter_str'])
#
#         animation_options_wid = iterations_result_wid.children[1].children[0].children[1].children[0]
#         # set the iterations options state
#         if iterations_result_default['n_iters'] == 1:
#             # set sliders values and visibility
#             for t in range(4):
#                 if t == 0:
#                     # first slider
#                     iterations_result_wid.children[1].children[0].children[1].children[1].value = 0
#                     iterations_result_wid.children[1].children[0].children[1].children[1].max = 0
#                     iterations_result_wid.children[1].children[0].children[1].children[1].visible = False
#                 elif t == 1:
#                     # second slider
#                     iterations_result_wid.children[1].children[0].children[1].children[2].value = 0
#                     iterations_result_wid.children[1].children[0].children[1].children[2].max = 0
#                     iterations_result_wid.children[1].children[0].children[1].children[2].visible = False
#                 elif t == 2:
#                     # animation slider
#                     animation_options_wid.selected_values['index'] = 0
#                     animation_options_wid.selected_values['max'] = 0
#                     animation_options_wid.children[1].children[0].value = 0
#                     animation_options_wid.children[1].children[0]. max = 0
#                     animation_options_wid.children[1].children[0].disabled = True
#                     animation_options_wid.children[1].children[1].children[0].children[0].disabled = True
#                     animation_options_wid.children[1].children[1].children[0].children[1].disabled = True
#                     animation_options_wid.children[1].children[1].children[0].children[2].disabled = True
#                 else:
#                     # iterations mode
#                     iterations_result_wid.children[1].children[0].children[0].value = 'animation'
#                     #iterations_result_wid.groups = [iter_str + "0"]
#                     iterations_result_wid.children[1].children[0].children[0].disabled = True
#         else:
#             # set sliders max and min values
#             for t in range(4):
#                 if t == 0:
#                     # first slider
#                     iterations_result_wid.children[1].children[0].children[1].children[1].value = 0
#                     iterations_result_wid.children[1].children[0].children[1].children[1].max = \
#                         iterations_result_default['n_iters'] - 1
#                     iterations_result_wid.children[1].children[0].children[1].children[1].visible = False
#                 elif t == 1:
#                     # second slider
#                     iterations_result_wid.children[1].children[0].children[1].children[2].value = \
#                         iterations_result_default['n_iters'] - 1
#                     iterations_result_wid.children[1].children[0].children[1].children[2].max = \
#                         iterations_result_default['n_iters'] - 1
#                     iterations_result_wid.children[1].children[0].children[1].children[2].visible = False
#                 elif t == 2:
#                     # animation slider
#                     animation_options_wid.children[1].children[0].value = 0
#                     animation_options_wid.children[1].children[0].max = \
#                         iterations_result_default['n_iters'] - 1
#                     animation_options_wid.selected_values['index'] = 0
#                     animation_options_wid.selected_values['max'] = \
#                         iterations_result_default['n_iters'] - 1
#                     animation_options_wid.children[1].children[0].disabled = \
#                         False
#                     animation_options_wid.children[1].children[1].children[0].children[0].disabled = False
#                     animation_options_wid.children[1].children[1].children[0].children[1].disabled = True
#                     animation_options_wid.children[1].children[1].children[0].children[2].disabled = False
#                 else:
#                     # iterations mode
#                     iterations_result_wid.children[1].children[0].children[0].\
#                         value = 'animation'
#                     #iterations_result_wid.groups = [iter_str + "0"]
#                     iterations_result_wid.children[1].children[0].children[0].\
#                         disabled = False
#
#
# def plot_options(plot_options_default, plot_function=None,
#                  toggle_show_visible=True, toggle_show_default=True):
#     r"""
#     Creates a widget with Plot Options. Specifically, it has:
#         1) A drop down menu for curve selection.
#         2) A text area for the legend entry.
#         3) A checkbox that controls line's visibility.
#         4) A checkbox that controls markers' visibility.
#         5) Options for line colour, style and width.
#         6) Options for markers face colour, edge colour, size, edge width and
#            style.
#         7) A toggle button that controls the visibility of all the above, i.e.
#            the plot options.
#
#     The structure of the widgets is the following:
#         plot_options_wid.children = [toggle_button, options]
#         options.children = [curve_menu, per_curve_options_wid]
#         per_curve_options_wid = ipywidgets.Box(children=[legend_entry,
#                                                           line_marker_wid])
#         line_marker_wid = ipywidgets.Box(children=[line_widget, marker_widget])
#         line_widget.children = [show_line_checkbox, line_options]
#         marker_widget.children = [show_marker_checkbox, marker_options]
#         line_options.children = [linestyle, linewidth, linecolour]
#         marker_options.children = [markerstyle, markersize, markeredgewidth,
#                                    markerfacecolour, markeredgecolour]
#
#     The returned widget saves the selected values in the following dictionary:
#         plot_options_wid.selected_options
#
#     To fix the alignment within this widget please refer to
#     `format_plot_options()` function.
#
#     Parameters
#     ----------
#     plot_options_default : list of `dict`
#         A list of dictionaries with the initial selected plot options per curve.
#         Example:
#             plot_options_1={'show_line':True,
#                             'linewidth':2,
#                             'linecolour':'r',
#                             'linestyle':'-',
#                             'show_marker':True,
#                             'markersize':20,
#                             'markerfacecolour':'r',
#                             'markeredgecolour':'b',
#                             'markerstyle':'o',
#                             'markeredgewidth':1,
#                             'legend_entry':'final errors'}
#             plot_options_2={'show_line':False,
#                             'linewidth':3,
#                             'linecolour':'r',
#                             'linestyle':'-',
#                             'show_marker':True,
#                             'markersize':60,
#                             'markerfacecolour':[0.1, 0.2, 0.3],
#                             'markeredgecolour':'k',
#                             'markerstyle':'x',
#                             'markeredgewidth':1,
#                             'legend_entry':'initial errors'}
#             plot_options_default = [plot_options_1, plot_options_2]
#
#     plot_function : `function` or None, optional
#         The plot function that is executed when a widgets' value changes.
#         If None, then nothing is assigned.
#
#     toggle_show_default : `boolean`, optional
#         Defines whether the options will be visible upon construction.
#
#     toggle_show_visible : `boolean`, optional
#         The visibility of the toggle button.
#     """
#     import IPython.html.widgets as ipywidgets
#     # make sure that plot_options_default is a list even with one member
#     if not isinstance(plot_options_default, list):
#         plot_options_default = [plot_options_default]
#
#     # find number of curves
#     n_curves = len(plot_options_default)
#
#     # Create widgets
#     # toggle button
#     but = ipywidgets.ToggleButton(description='Plot Options',
#                                         value=toggle_show_default,
#                                         visible=toggle_show_visible)
#
#     # select curve drop down menu
#     curves_dict = OrderedDict()
#     for k in range(n_curves):
#         curves_dict['Curve ' + str(k)] = k
#     curve_selection = ipywidgets.Dropdown(options=curves_dict,
#                                                 value=0,
#                                                 description='Select curve',
#                                                 visible=n_curves > 1)
#
#     # legend entry
#     legend_entry = ipywidgets.Text(description='Legend entry',
#                                          value=plot_options_default[0][
#                                              'legend_entry'])
#
#     # show line, show markers checkboxes
#     show_line = ipywidgets.Checkbox(description='Show line',
#                                           value=plot_options_default[0][
#                                               'show_line'])
#     show_marker = ipywidgets.Checkbox(description='Show markers',
#                                             value=plot_options_default[0][
#                                                 'show_marker'])
#
#     # linewidth, markersize
#     linewidth = ipywidgets.FloatText(description='Width',
#                                            value=plot_options_default[0][
#                                                'linewidth'])
#     markersize = ipywidgets.IntText(description='Size',
#                                           value=plot_options_default[0][
#                                               'markersize'])
#     markeredgewidth = ipywidgets.FloatText(
#         description='Edge width',
#         value=plot_options_default[0]['markeredgewidth'])
#
#     # markerstyle
#     markerstyle_dict = OrderedDict()
#     markerstyle_dict['point'] = '.'
#     markerstyle_dict['pixel'] = ','
#     markerstyle_dict['circle'] = 'o'
#     markerstyle_dict['triangle down'] = 'v'
#     markerstyle_dict['triangle up'] = '^'
#     markerstyle_dict['triangle left'] = '<'
#     markerstyle_dict['triangle right'] = '>'
#     markerstyle_dict['tri down'] = '1'
#     markerstyle_dict['tri up'] = '2'
#     markerstyle_dict['tri left'] = '3'
#     markerstyle_dict['tri right'] = '4'
#     markerstyle_dict['octagon'] = '8'
#     markerstyle_dict['square'] = 's'
#     markerstyle_dict['pentagon'] = 'p'
#     markerstyle_dict['star'] = '*'
#     markerstyle_dict['hexagon 1'] = 'h'
#     markerstyle_dict['hexagon 2'] = 'H'
#     markerstyle_dict['plus'] = '+'
#     markerstyle_dict['x'] = 'x'
#     markerstyle_dict['diamond'] = 'D'
#     markerstyle_dict['thin diamond'] = 'd'
#     markerstyle = ipywidgets.Dropdown(options=markerstyle_dict,
#                                             value=plot_options_default[0][
#                                                 'markerstyle'],
#                                             description='Style')
#
#     # linestyle
#     linestyle_dict = OrderedDict()
#     linestyle_dict['solid'] = '-'
#     linestyle_dict['dashed'] = '--'
#     linestyle_dict['dash-dot'] = '-.'
#     linestyle_dict['dotted'] = ':'
#     linestyle = ipywidgets.Dropdown(options=linestyle_dict,
#                                           value=plot_options_default[0][
#                                               'linestyle'],
#                                           description='Style')
#
#     # colours
#     # do not assign the plot_function here
#     linecolour = colour_selection(plot_options_default[0]['linecolour'],
#                                   title='Colour')
#     markerfacecolour = colour_selection(
#         plot_options_default[0]['markerfacecolour'],
#         title='Face Colour')
#     markeredgecolour = colour_selection(
#         plot_options_default[0]['markeredgecolour'],
#         title='Edge Colour')
#
#     # Group widgets
#     line_options = ipywidgets.Box(
#         children=[linestyle, linewidth, linecolour])
#     marker_options = ipywidgets.Box(
#         children=[markerstyle, markersize,
#                   markeredgewidth,
#                   markerfacecolour,
#                   markeredgecolour])
#     line_wid = ipywidgets.Box(children=[show_line, line_options])
#     marker_wid = ipywidgets.Box(
#         children=[show_marker, marker_options])
#     line_options_options_wid = ipywidgets.Box(
#         children=[line_wid, marker_wid])
#     options_wid = ipywidgets.Box(children=[legend_entry,
#                                                        line_options_options_wid])
#     options_and_curve_wid = ipywidgets.Box(
#         children=[curve_selection,
#                   options_wid])
#     plot_options_wid = ipywidgets.Box(
#         children=[but, options_and_curve_wid])
#
#     # initialize output
#     plot_options_wid.selected_options = plot_options_default
#
#     # line options visibility
#     def line_options_visible(name, value):
#         linestyle.disabled = not value
#         linewidth.disabled = not value
#         linecolour.children[0].disabled = not value
#         linecolour.children[1].children[0].disabled = not value
#         linecolour.children[1].children[1].disabled = not value
#         linecolour.children[1].children[2].disabled = not value
#     show_line.on_trait_change(line_options_visible, 'value')
#
#     # marker options visibility
#     def marker_options_visible(name, value):
#         markerstyle.disabled = not value
#         markersize.disabled = not value
#         markeredgewidth.disabled = not value
#         markerfacecolour.children[0].disabled = not value
#         markerfacecolour.children[1].children[0].disabled = not value
#         markerfacecolour.children[1].children[1].disabled = not value
#         markerfacecolour.children[1].children[2].disabled = not value
#         markeredgecolour.children[0].disabled = not value
#         markeredgecolour.children[1].children[0].disabled = not value
#         markeredgecolour.children[1].children[1].disabled = not value
#         markeredgecolour.children[1].children[2].disabled = not value
#     show_marker.on_trait_change(marker_options_visible, 'value')
#
#     # function that gets colour selection
#     def get_colour(colour_wid):
#         if colour_wid.children[0].value == 'custom':
#             return [float(colour_wid.children[1].children[0].value),
#                     float(colour_wid.children[1].children[1].value),
#                     float(colour_wid.children[1].children[2].value)]
#         else:
#             return colour_wid.children[0].value
#
#     # assign options
#     def save_legend_entry(name, value):
#         plot_options_wid.selected_options[curve_selection.value][
#             'legend_entry'] = str(value)
#
#     legend_entry.on_trait_change(save_legend_entry, 'value')
#
#     def save_show_line(name, value):
#         plot_options_wid.selected_options[curve_selection.value][
#             'show_line'] = value
#
#     show_line.on_trait_change(save_show_line, 'value')
#
#     def save_show_marker(name, value):
#         plot_options_wid.selected_options[curve_selection.value][
#             'show_marker'] = value
#
#     show_marker.on_trait_change(save_show_marker, 'value')
#
#     def save_linewidth(name, value):
#         plot_options_wid.selected_options[curve_selection.value][
#             'linewidth'] = float(value)
#
#     linewidth.on_trait_change(save_linewidth, 'value')
#
#     def save_linestyle(name, value):
#         plot_options_wid.selected_options[curve_selection.value][
#             'linestyle'] = value
#
#     linestyle.on_trait_change(save_linestyle, 'value')
#
#     def save_markersize(name, value):
#         plot_options_wid.selected_options[curve_selection.value][
#             'markersize'] = int(value)
#
#     markersize.on_trait_change(save_markersize, 'value')
#
#     def save_markeredgewidth(name, value):
#         plot_options_wid.selected_options[curve_selection.value][
#             'markeredgewidth'] = float(value)
#
#     markeredgewidth.on_trait_change(save_markeredgewidth, 'value')
#
#     def save_markerstyle(name, value):
#         plot_options_wid.selected_options[curve_selection.value][
#             'markerstyle'] = value
#
#     markerstyle.on_trait_change(save_markerstyle, 'value')
#
#     def save_linecolour(name, value):
#         plot_options_wid.selected_options[curve_selection.value][
#             'linecolour'] = get_colour(linecolour)
#
#     linecolour.children[0].on_trait_change(save_linecolour, 'value')
#     linecolour.children[1].children[0].on_trait_change(save_linecolour, 'value')
#     linecolour.children[1].children[1].on_trait_change(save_linecolour, 'value')
#     linecolour.children[1].children[2].on_trait_change(save_linecolour, 'value')
#
#     def save_markerfacecolour(name, value):
#         plot_options_wid.selected_options[curve_selection.value][
#             'markerfacecolour'] = get_colour(markerfacecolour)
#
#     markerfacecolour.children[0].on_trait_change(save_markerfacecolour, 'value')
#     markerfacecolour.children[1].children[0].on_trait_change(
#         save_markerfacecolour, 'value')
#     markerfacecolour.children[1].children[1].on_trait_change(
#         save_markerfacecolour, 'value')
#     markerfacecolour.children[1].children[2].on_trait_change(
#         save_markerfacecolour, 'value')
#
#     def save_markeredgecolour(name, value):
#         plot_options_wid.selected_options[curve_selection.value][
#             'markeredgecolour'] = get_colour(markeredgecolour)
#
#     markeredgecolour.children[0].on_trait_change(save_markeredgecolour, 'value')
#     markeredgecolour.children[1].children[0].on_trait_change(
#         save_markeredgecolour, 'value')
#     markeredgecolour.children[1].children[1].on_trait_change(
#         save_markeredgecolour, 'value')
#     markeredgecolour.children[1].children[2].on_trait_change(
#         save_markeredgecolour, 'value')
#
#     # set correct value to slider when drop down menu value changes
#     def set_options(name, value):
#         legend_entry.value = plot_options_wid.selected_options[value][
#             'legend_entry']
#         show_line.value = plot_options_wid.selected_options[value]['show_line']
#         show_marker.value = plot_options_wid.selected_options[value][
#             'show_marker']
#         linewidth.value = plot_options_wid.selected_options[value]['linewidth']
#         linestyle.value = plot_options_wid.selected_options[value]['linestyle']
#         markersize.value = plot_options_wid.selected_options[value][
#             'markersize']
#         markerstyle.value = plot_options_wid.selected_options[value][
#             'markerstyle']
#         markeredgewidth.value = plot_options_wid.selected_options[value][
#             'markeredgewidth']
#         default_colour = plot_options_wid.selected_options[value]['linecolour']
#         if not isinstance(default_colour, str):
#             r_val = default_colour[0]
#             g_val = default_colour[1]
#             b_val = default_colour[2]
#             default_colour = 'custom'
#             linecolour.children[1].children[0].value = r_val
#             linecolour.children[1].children[1].value = g_val
#             linecolour.children[1].children[2].value = b_val
#         linecolour.children[0].value = default_colour
#         default_colour = plot_options_wid.selected_options[value][
#             'markerfacecolour']
#         if not isinstance(default_colour, str):
#             r_val = default_colour[0]
#             g_val = default_colour[1]
#             b_val = default_colour[2]
#             default_colour = 'custom'
#             markerfacecolour.children[1].children[0].value = r_val
#             markerfacecolour.children[1].children[1].value = g_val
#             markerfacecolour.children[1].children[2].value = b_val
#         markerfacecolour.children[0].value = default_colour
#         default_colour = plot_options_wid.selected_options[value][
#             'markeredgecolour']
#         if not isinstance(default_colour, str):
#             r_val = default_colour[0]
#             g_val = default_colour[1]
#             b_val = default_colour[2]
#             default_colour = 'custom'
#             markeredgecolour.children[1].children[0].value = r_val
#             markeredgecolour.children[1].children[1].value = g_val
#             markeredgecolour.children[1].children[2].value = b_val
#         markeredgecolour.children[0].value = default_colour
#     curve_selection.on_trait_change(set_options, 'value')
#
#     # Toggle button function
#     def toggle_fun(name, value):
#         options_and_curve_wid.visible = value
#     toggle_fun('', toggle_show_default)
#     but.on_trait_change(toggle_fun, 'value')
#
#     # assign plot_function
#     if plot_function is not None:
#         legend_entry.on_trait_change(plot_function, 'value')
#         show_line.on_trait_change(plot_function, 'value')
#         linestyle.on_trait_change(plot_function, 'value')
#         linewidth.on_trait_change(plot_function, 'value')
#         show_marker.on_trait_change(plot_function, 'value')
#         markerstyle.on_trait_change(plot_function, 'value')
#         markeredgewidth.on_trait_change(plot_function, 'value')
#         markersize.on_trait_change(plot_function, 'value')
#         linecolour.children[0].on_trait_change(plot_function, 'value')
#         linecolour.children[1].children[0].on_trait_change(plot_function,
#                                                            'value')
#         linecolour.children[1].children[1].on_trait_change(plot_function,
#                                                            'value')
#         linecolour.children[1].children[2].on_trait_change(plot_function,
#                                                            'value')
#         markerfacecolour.children[0].on_trait_change(plot_function, 'value')
#         markerfacecolour.children[1].children[0].on_trait_change(plot_function,
#                                                                  'value')
#         markerfacecolour.children[1].children[1].on_trait_change(plot_function,
#                                                                  'value')
#         markerfacecolour.children[1].children[2].on_trait_change(plot_function,
#                                                                  'value')
#         markeredgecolour.children[0].on_trait_change(plot_function, 'value')
#         markeredgecolour.children[1].children[0].on_trait_change(plot_function,
#                                                                  'value')
#         markeredgecolour.children[1].children[1].on_trait_change(plot_function,
#                                                                  'value')
#         markeredgecolour.children[1].children[2].on_trait_change(plot_function,
#                                                                  'value')
#
#     return plot_options_wid
#
#
# def format_plot_options(plot_options_wid, container_padding='6px',
#                         container_margin='6px',
#                         container_border='1px solid black',
#                         toggle_button_font_weight='bold', border_visible=True,
#                         suboptions_border_visible=True):
#     r"""
#     Function that corrects the align (style format) of a given figure_options
#     widget. Usage example:
#         plot_options_wid = plot_options()
#         display(plot_options_wid)
#         format_plot_options(figure_options_wid)
#
#     Parameters
#     ----------
#     plot_options_wid :
#         The widget object generated by the `figure_options()` function.
#
#     container_padding : `str`, optional
#         The padding around the widget, e.g. '6px'
#
#     container_margin : `str`, optional
#         The margin around the widget, e.g. '6px'
#
#     container_border : `str`, optional
#         The border around the widget, e.g. '1px solid black'
#
#     toggle_button_font_weight : `str`
#         The font weight of the toggle button, e.g. 'bold'
#
#     border_visible : `boolean`, optional
#         Defines whether to draw the border line around the widget.
#
#     suboptions_border_visible : `boolean`, optional
#         Defines whether to draw the border line around the per curve options.
#     """
#     # align line options with checkbox
#     add_class(plot_options_wid.children[1].children[1].children[1].children[0], 'align-end')
#
#     # align marker options with checkbox
#     add_class(plot_options_wid.children[1].children[1].children[1].children[1], 'align-end')
#
#     # set text boxes width
#     plot_options_wid.children[1].children[1].children[1].children[0].children[
#         1].children[1]. \
#         width = '1cm'
#     plot_options_wid.children[1].children[1].children[1].children[1].children[
#         1].children[1]. \
#         width = '1cm'
#     plot_options_wid.children[1].children[1].children[1].children[1].children[
#         1].children[2]. \
#         width = '1cm'
#
#     # align line and marker options
#     remove_class(plot_options_wid.children[1].children[1].children[1], 'vbox')
#     add_class(plot_options_wid.children[1].children[1].children[1], 'hbox')
#     if suboptions_border_visible:
#         plot_options_wid.children[1].children[1].margin = container_margin
#         plot_options_wid.children[1].children[1].border = container_border
#
#     # align curve selection with line and marker options
#     add_class(plot_options_wid.children[1], 'align-start')
#
#     # format colour options
#     format_colour_selection(
#         plot_options_wid.children[1].children[1].children[1].children[
#             0].children[1].children[2])
#     format_colour_selection(
#         plot_options_wid.children[1].children[1].children[1].children[
#             1].children[1].children[3])
#     format_colour_selection(
#         plot_options_wid.children[1].children[1].children[1].children[
#             1].children[1].children[4])
#
#     # set toggle button font bold
#     plot_options_wid.children[0].font_weight = toggle_button_font_weight
#
#     # margin and border around container widget
#     plot_options_wid.padding = container_padding
#     plot_options_wid.margin = container_margin
#     if border_visible:
#         plot_options_wid.border = container_border

