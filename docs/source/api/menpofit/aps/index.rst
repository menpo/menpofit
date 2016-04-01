.. _api-aps-index:

:mod:`menpofit.aps`
===================

Active Pictorial Structures
---------------------------
APS is a generative model that consists of a statistical parametric model of
the shape and the appearance of an object. MenpoFit has several AAMs which
differ in the manner that they compute the warp (thus represent the
appearance features).

.. toctree::
    :maxdepth: 1

    GenerativeAPS

Fitters
-------
An APS can be optimised in a gradient descent manner.

.. toctree::
    :maxdepth: 1

    GaussNewtonAPSFitter

Gauss-Newton Optimisation Algorithms
------------------------------------

.. toctree::
    :maxdepth: 1

    Inverse
    Forward

Fitting Result
--------------

.. toctree::
    :maxdepth: 1

    APSResult
    APSAlgorithmResult
