=======
Welcome
=======

**Welcome to the MenpoFit documentation!**

MenpoFit is a Python package for building, fitting and manipulating deformable
models. It includes state-of-the-art deformable modelling techniques implemented
on top of the **Menpo** project. Currently, the techniques that have been
implemented include:

* :ref:`Active Appearance Model (AAM) <api-aam-index>`

  * :ref:`Holistic <menpofit-aam-HolisticAAM>`, :ref:`Patch-based <menpofit-aam-PatchAAM>`, :ref:`Masked <menpofit-aam-MaskedAAM>`, :ref:`Linear <menpofit-aam-LinearAAM>`, :ref:`Linear Masked <menpofit-aam-LinearMaskedAAM>`
  * Lucas-Kanade Optimisation
  * Cascaded-Regression Optimisation

* :ref:`Active Pictorial Structures (APS) <api-aps-index>`

  * Weighted Gauss-Newton Optimisation with fixed Jacobian and Hessian

* :ref:`Active Template Model (ATM) <api-atm-index>`

  * :ref:`Holistic <menpofit-atm-HolisticATM>`, :ref:`Patch-based <menpofit-atm-PatchATM>`, :ref:`Masked <menpofit-atm-MaskedATM>`, :ref:`Linear <menpofit-atm-LinearATM>`, :ref:`Linear Masked <menpofit-atm-LinearMaskedATM>`
  * Lucas-Kanade Optimisation

* :ref:`Lucas-Kanade Image Alignment (LK) <api-lk-index>`

  * Forward Additive, Forward Compositional, Inverse Compositional
  * Residuals: SSD, Fourier SSD, ECC, Gradient Correlation, Gradient Images

* :ref:`Constrained Local Model (CLM) <api-clm-index>`

  * Active Shape Model
  * Regularised Landmark Mean Shift

* :ref:`Ensemble of Regression Trees (ERT) <api-dlib-index>` [provided by `DLib <http://dlib.net/>`_]
* :ref:`Supervised Descent Method (SDM) <api-sdm-index>`

  * Non Parametric
  * Parametric Shape
  * Parametric Appearance
  * Fully Parametric

Please see the to :ref:`References <ug-references>` for an indicative list of
papers that are relevant to the methods implemented in MenpoFit.

.. toctree::
    :maxdepth: 2
    :hidden:

    userguide/index
    api/index
