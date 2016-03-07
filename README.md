menpofit - A deformable model toolkit
=====================================
State-of-the-art deformable modelling techniques implemented on top of the
Menpo project. Currently, the techniques that have been implemented include:

  - **Active Appearance Model**
    - Holistic, Patch-based, Masked, Linear, Linear Masked
    - Lucas-Kanade Optimisation (Alternating, Modified Alternating, Project Out, Simultaneous, Wiberg) 
    - Casaded-Regression Optimisation
  - **Lucas-Kanade Image Alignment**
    - Forward Additive, Forward Compositional, Inverse Additive, Inverse Compositional
  - **Active Template Model**
    - Holistic, Patch-based, Masked, Linear, Linear Masked
    - Lucas-Kanade Optimisation (Inverse Compositional, Forward Compositional)
  - **Constrained Local Model**
    - Active Shape Models
    - Regularized Landmark Mean-Shift
  - **Ensemble of Regression Trees (ERT)** \[provided by [DLib](http://dlib.net/)\]
  - **Supervised Descent Method**
    - Non Parametric
    - Parametric Shape
    - Parametric Appearance
    - Fully Parametric

Installation
------------
Here in the Menpo team, we are firm believers in making installation as simple 
as possible. Unfortunately, we are a complex project that relies on satisfying 
a number of complex 3rd party library dependencies. The default Python packing 
environment does not make this an easy task. Therefore, we evangelise the use 
of the conda ecosystem, provided by 
[Anaconda](https://store.continuum.io/cshop/anaconda/). In order to make things 
as simple as possible, we suggest that you use conda too! To try and persuade 
you, go to the [Menpo website](http://www.menpo.io/installation/) to find 
installation instructions for all major platforms.
