from menpofit.io import load_fitter


def load_balanced_frontal_face_fitter():
    r"""
    Loads a frontal face patch-based AAM fitter that is a good compromise
    between model size, fitting time and fitting performance. The model is a
    :map:`PatchAAM` that was trained using the following parameters:

        =================== =================================
        Parameter           Value
        =================== =================================
        `diagonal`          110
        `scales`            (0.5, 1.0)
        `patch_shape`       [(13, 13), (15, 15)]
        `holistic_features` `menpo.feature.fast_dsift()`
        `n_shape`           [5, 20]
        `n_appearance`      [30, 150]
        `lk_algorithm_cls`  :map:`WibergInverseCompositional`
        =================== =================================

    Note that the first time you invoke this function, menpofit will
    download the fitter from Menpo's server. The fitter will then be stored
    locally for future use.

    Returns
    -------
    fitter : :map:`LucasKanadeAAMFitter`
        A pre-trained :map:`LucasKanadeAAMFitter` based on a :map:`PatchAAM`
        that performs iBUG68 facial landmark localization.
    """
    return load_fitter('balanced_frontal_face_aam')
