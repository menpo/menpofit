.. _api-atm-index:

:mod:`menpofit.atm`
===================

Active Template Model
---------------------
ATM is a generative model that performs deformable alignment between a
template image and a test image with respect to a statistical parametric
shape model. MenpoFit has several ATMs which differ in the manner that they
compute the warp (thus represent the appearance features).

.. toctree::
    :maxdepth: 1

    ATM
    HolisticATM
    MaskedATM
    LinearATM
    LinearMaskedATM
    PatchATM

Fitter
------

.. toctree::
    :maxdepth: 1

    LucasKanadeATMFitter

Lucas-Kanade Optimisation Algorithms
------------------------------------

.. toctree::
    :maxdepth: 1

    ForwardCompositional
    InverseCompositional

