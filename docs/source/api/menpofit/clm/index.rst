.. _api-clm-index:

:mod:`menpofit.clm`
===================

Constrained Local Models
------------------------
Deformable model that consists of a generative parametric shape model and
discriminatively trained experts per part.

.. toctree::
    :maxdepth: 1

    CLM

CLM Fitters
-----------
A CLM is optimised either a gradient descent manner.

.. toctree::
    :maxdepth: 1

    GradientDescentCLMFitter

Optimisation Algorithms
-----------------------

.. toctree::
    :maxdepth: 1

    ActiveShapeModel
    RegularisedLandmarkMeanShift

Experts
-------

.. toctree::
    :maxdepth: 1

    CorrelationFilterExpertEnsemble
    IncrementalCorrelationFilterThinWrapper
