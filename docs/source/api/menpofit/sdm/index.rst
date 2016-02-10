.. _api-sdm-index:

:mod:`menpofit.sdm`
===================

Supervised Descent Method
-------------------------
SDM is a cascaded-regression deformable model that learns average descent
directions that minimise a given cost function.

.. toctree::
    :maxdepth: 1

    SupervisedDescentFitter

Pre-defined Models
------------------
Models with pre-defined algorithms that are commonly-used in literature.

.. toctree::
    :maxdepth: 1

    SDM
    RegularizedSDM

Non-Parametric Algorithms
-------------------------
The cascaded regression of these algorithms is performed between landmark
coordinates and image-based features.

.. toctree::
    :maxdepth: 1

    NonParametricNewton
    NonParametricGaussNewton
    NonParametricPCRRegression
    NonParametricOptimalRegression
    NonParametricOPPRegression

Parametric Shape Algorithms
---------------------------
The cascaded regression of these algorithms is performed between the
parameters of a statistical shape model and image-based features.

.. toctree::
    :maxdepth: 1

    ParametricShapeNewton
    ParametricShapeGaussNewton
    ParametricShapePCRRegression
    ParametricShapeOptimalRegression
    ParametricShapeOPPRegression

Parametric Appearance Algorithms
--------------------------------
The cascaded regression of these algorithms is performed between landmark
coordinates and features that are based on a statistical parametric
appearance model.

.. toctree::
    :maxdepth: 1

    ParametricAppearanceProjectOutNewton
    ParametricAppearanceProjectOutGuassNewton
    ParametricAppearanceMeanTemplateNewton
    ParametricAppearanceMeanTemplateGuassNewton
    ParametricAppearanceWeightsNewton
    ParametricAppearanceWeightsGuassNewton

Fully Parametric Algorithms
---------------------------
The cascaded regression is performed between the parameters of a statistical
shape model and features that are based on a statistical parametric
appearance model.

.. toctree::
    :maxdepth: 1

    FullyParametricProjectOutNewton
    FullyParametricProjectOutGaussNewton
    FullyParametricMeanTemplateNewton
    FullyParametricWeightsNewton
    FullyParametricProjectOutOPP
