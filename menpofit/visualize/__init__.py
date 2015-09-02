# If IPython is not installed, then access to the widgets should be blocked.
try:
    from .widgets import (visualize_shape_model, visualize_appearance_model,
                          visualize_patch_appearance_model, visualize_aam,
                          visualize_patch_aam, visualize_atm, plot_ced,
                          visualize_fitting_result)
except ImportError:
    pass
from .textutils import print_progress
