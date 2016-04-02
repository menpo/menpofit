.. _api-aps-index:

:mod:`menpofit.aps`
===================

Active Pictorial Structures
---------------------------
APS is a model that utilises a Gaussian Markov Random Field (GMRF) for 
learning an appearance model with pairwise distributions based on a graph.
It also has a parametric statitical shape model (either using PCA or GMRF),
as well as a spring-like deformation prior term. The optimisation is performed
using a weighted Gauss-Newton algorithm with fixed Jacobian and Hessian.

.. toctree::
    :maxdepth: 1

    GenerativeAPS

Fitters
-------

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
