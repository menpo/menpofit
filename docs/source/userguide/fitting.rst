.. _ug-fitting:

Fitting Models
==============

Fitter Objects
--------------
MenpoFit has specialised classes for performing a fitting process that are
called `Fitters`. All `Fitter` objects are subclasses of :map:`MultiFitter` and
:map:`ModelFitter` and their behaviour differs depending on the deformable
model. For example, a Lucas-Kanade AAM fitter (:map:`LucasKanadeAAMFitter`)
assumes that you have trained an AAM model (assume the `aam` we trained in the
:ref:`Building <ug-building>` section) and can be created as:

.. code-block:: python

    from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional

    fitter = LucasKanadeAAMFitter(aam,
                                  lk_algorithm_cls=WibergInverseCompositional,
                                  n_shape=[5, 10, 15], n_appearance=150)

The constructor of the `Fitter` will set the active shape and appearance
components based on `n_shape` and `n_appearance` respectively, and will also
perform all the necessary pre-computations based on the selected algorithm.

However, there are deformable models that are directly defined through a
`Fitter` object, which is responsible for training the model as well.
:map:`SupervisedDescentFitter` is a good example. The reason for that is that
the fitting process is utilised during the building procedure, thus the
functionality of a `Fitter` is required. Such models can be built as:

.. code-block:: python

    from menpofit.sdm import SupervisedDescentFitter, NonParametricNewton

    fitter = SupervisedDescentFitter(training_images, group='PTS',
                                     sd_algorithm_cls=NonParametricNewton,
                                     verbose=True)

Information about a `Fitter` can be retrieved by:

.. code-block:: python

    print(fitter)

Fitting Methods
---------------
All the deformable models that are currently implemented in MenpoFit, which
are the state-of-the-art approaches in current literature, aim to find a
*local optimum* of the cost function that they try to optimise, given an
initialisation. The initialisation can be seen as an initial estimation of
the target shape. MenpoFit's `Fitter` objects provide two functions for fitting
the model to an image:

.. code-block:: python

    result = fitter.fit_from_shape(image, initial_shape, max_iters=20, gt_shape=None,
                                   crop_image=None, "**"kwargs)

or

.. code-block:: python

    result = fitter.fit_from_bb(image, bounding_box, max_iters=20, gt_shape=None,
                                crop_image=None, "**"kwargs)

They only differ on the type of initialisation. ``fit_from_shape`` expects a
`PointCloud` as the `initial_shape`. On the other hand, the `bounding_box`
argument of ``fit_from_bb`` is a `PointDirectedGraph` of 4 vertices that
represents the initial bounding box. The bounding box is used in order to
align the model's reference shape and use the resulting `PointCloud` as the
initial shape. Such a bounding box can be retrieved using the detection
methods of **menpodetect**. The rest of the options are:

**max_iters** (`int` or `list` of `int`)
  Defines the maximum number of iterations. If `int`, then it specifies the
  maximum number of iterations over all scales. If `list` of `int`, then it
  specifies the maximum number of iterations per scale. Note that this does
  not apply on all deformable models. For example, it can control the number
  of iterations of a Lucas-Kanade optimisation algorithm, but it does not
  affect the fitting of a cascaded-regression method (e.g. SDM) which has a
  predefined number of cascades (iterations).
**gt_shape** (`PointCloud` or `None`)
  The ground truth shape associated to the image. This is *only* useful to
  compute the final fitting error. It is *not* used, of course, at any
  internal stage of the optimisation.
**crop_image** (`None` or `float`)
  If `float`, it specifies the proportion of the border wrt the initial shape
  to which the image will be internally cropped around the initial shape
  range. If `None` , no cropping is performed. This limits the fitting
  algorithm search region but is likely to speed up its running time,
  specially when the modeled object occupies a small portion of the image.
**kwargs** (`dict`)
  Additional keyword arguments that can be passed to specific models.

The next section (:ref:`Result <ug-result>`) presents the basics of the
fitting `result`.
