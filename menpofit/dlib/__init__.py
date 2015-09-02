try:
    from .fitter import DlibERT
except ImportError:
    # If dlib is not installed then we shouldn't import anything into this
    # module.
    pass
