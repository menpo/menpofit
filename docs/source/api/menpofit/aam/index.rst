.. _api-aam-index:

:mod:`menpofit.aam`
===================

Active Appearance Models
------------------------
AAM is a generative model that consists of a statistical parametric model of
the shape and the appearance of an object. MenpoFit has several AAMs which
differ in the manner that they compute the warp (thus represent the
appearance features).

.. toctree::
    :maxdepth: 1

    AAM
    HolisticAAM
    MaskedAAM
    LinearAAM
    LinearMaskedAAM
    PatchAAM

Fitters
-------
An AAM can be optimised either in a gradient descent manner (Lucas-Kanade) or
using cascaded regression (Supervised Descent).

.. toctree::
    :maxdepth: 1

    LucasKanadeAAMFitter
    SupervisedDescentAAMFitter

Lucas-Kanade Optimisation Algorithms
------------------------------------

.. toctree::
    :maxdepth: 1

    AlternatingForwardCompositional
    AlternatingInverseCompositional
    ModifiedAlternatingForwardCompositional
    ModifiedAlternatingInverseCompositional
    ProjectOutForwardCompositional
    ProjectOutInverseCompositional
    SimultaneousForwardCompositional
    SimultaneousInverseCompositional
    WibergForwardCompositional
    WibergInverseCompositional

Supervised Descent Optimisation Algorithms
------------------------------------------

.. toctree::
    :maxdepth: 1

    AppearanceWeightsNewton
    AppearanceWeightsGaussNewton
    MeanTemplateNewton
    MeanTemplateGaussNewton
    ProjectOutNewton
    ProjectOutGaussNewton

Fitting Result
--------------

.. toctree::
    :maxdepth: 1

    AAMResult
    AAMAlgorithmResult
