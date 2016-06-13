from menpofit.io import load_fitter


def load_balanced_frontal_face_fitter():
    r"""
    Loads a frontal face fitter that is a good compromise between model size,
    fitting time, and fitting performance.

    Note that the first time you invoke this function, menpofit will
    download the fitter from Menpo server's. The fitter will then be stored
    locally for future use.

    Returns
    -------

    fitter : `Fitter`
        A pretrained menpofit `Fitter` that performs iBUG68 facial landmark
        localization.

    """
    return load_fitter('balanced_frontal_face_aam')
