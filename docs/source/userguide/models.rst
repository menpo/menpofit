.. _ug-models:

Every Model, Built and Fitted
=============================
One example for each family of models in MenpoFit: how to build it and how to
fit it. Each example assumes the same things, which are explained in the
:ref:`Building <ug-building>` and :ref:`Fitting <ug-fitting>` pages:

- ``training_images``: a `list` of `menpo.image.Image` with a landmark group
  called ``'PTS'`` (e.g. from ``menpo.io.import_images('/path/to/*.png')``);
- ``image``: the `menpo.image.Image` to fit;
- ``initial_shape``: a `menpo.shape.PointCloud` that is the initial estimate of
  the shape in ``image``, or ``bounding_box``: the box of the object (e.g. from
  `menpodetect`) to start from.

All the fitters have ``fit_from_shape(image, initial_shape)`` and
``fit_from_bb(image, bounding_box)``, which return a result with
``final_shape``, ``initial_shape`` and, when you give ``gt_shape``,
``final_error()`` (see :ref:`Result <ug-result>`). The examples use whichever
is natural; the other works as well.

The numbers in the examples (``diagonal``, ``scales``, the sizes of patches and
the numbers of components) are reasonable starting points for faces, not
requirements: change them to fit your data and your memory.

Active Appearance Models
------------------------
The shape and the appearance of the object are modelled by PCA. Fit with one of
the Lucas-Kanade algorithms (``ProjectOut``, ``Simultaneous``, ``Alternating``,
``Wiberg``, each ``InverseCompositional`` or ``ForwardCompositional``).
``n_shape`` and ``n_appearance`` are the number of components that the fitter
uses at each scale.

.. code-block:: python

    from menpofit.aam import HolisticAAM, LucasKanadeAAMFitter, WibergInverseCompositional

    aam = HolisticAAM(training_images, group='PTS', diagonal=100,
                      scales=(0.5, 1.0), max_shape_components=20,
                      max_appearance_components=100, verbose=True)
    fitter = LucasKanadeAAMFitter(aam,
                                  lk_algorithm_cls=WibergInverseCompositional,
                                  n_shape=[5, 10], n_appearance=[20, 50])
    result = fitter.fit_from_shape(image, initial_shape, max_iters=20)
    print(result.final_shape)

The other model classes take the same arguments: ``PatchAAM`` (features of
patches around the landmarks, instead of the whole image, and a ``patch_shape``),
``MaskedAAM`` and ``LinearAAM`` / ``LinearMaskedAAM`` (the appearance model of
the linear shape and appearance warp). They are fitted with the same fitter:

.. code-block:: python

    from menpofit.aam import PatchAAM, LucasKanadeAAMFitter

    aam = PatchAAM(training_images, group='PTS', diagonal=100,
                   scales=(0.5, 1.0), patch_shape=(9, 9))
    fitter = LucasKanadeAAMFitter(aam, n_shape=10, n_appearance=50)
    result = fitter.fit_from_shape(image, initial_shape, max_iters=20)

An AAM can also be fitted by cascaded regression instead of Lucas-Kanade, in
which case the fitter is trained on the images as well:

.. code-block:: python

    from functools import partial
    from menpofit.aam import HolisticAAM, ProjectOutNewton, SupervisedDescentAAMFitter

    aam = HolisticAAM(training_images, group='PTS', diagonal=100,
                      scales=(0.5, 1.0))
    fitter = SupervisedDescentAAMFitter(training_images, aam, group='PTS',
                                        sd_algorithm_cls=partial(ProjectOutNewton, alpha=100),
                                        n_shape=10, n_appearance=50,
                                        n_iterations=3, n_perturbations=5)
    result = fitter.fit_from_bb(image, bounding_box)

Active Template Models
----------------------
A template (one image) instead of the appearance of many, and a model of the
shapes. Align the template to an image with Lucas-Kanade
(``InverseCompositional`` or ``ForwardCompositional``).

.. code-block:: python

    from menpofit.atm import HolisticATM, LucasKanadeATMFitter, InverseCompositional

    template = training_images[0]
    shapes = [i.landmarks['PTS'] for i in training_images]
    atm = HolisticATM(template, shapes, group='PTS', diagonal=100,
                      scales=(0.5, 1.0), max_shape_components=20)
    fitter = LucasKanadeATMFitter(atm, lk_algorithm_cls=InverseCompositional,
                                  n_shape=10)
    result = fitter.fit_from_shape(image, initial_shape, max_iters=20)

Constrained Local Models
------------------------
Experts that say where each landmark is (a correlation filter around each
landmark), and a shape model that regularises them.

.. code-block:: python

    from menpofit.clm import CLM, GradientDescentCLMFitter, RegularisedLandmarkMeanShift

    clm = CLM(training_images, group='PTS', diagonal=100, scales=(0.5, 1.0),
              patch_shape=(9, 9), context_shape=(18, 18),
              max_shape_components=20)
    fitter = GradientDescentCLMFitter(clm,
                                      gd_algorithm_cls=RegularisedLandmarkMeanShift,
                                      n_shape=10)
    result = fitter.fit_from_shape(image, initial_shape, max_iters=20)

Unified AAM and CLM
-------------------
Both the appearance model and the experts, fitted together. This fitter needs
the noise variance of the models, so they must keep fewer components than
there are images: a ``float`` for the number of components is the ratio of the
variance to keep (here 95%), which works for any number of images.

.. code-block:: python

    from menpofit.unified_aam_clm import UnifiedAAMCLM, UnifiedAAMCLMFitter

    model = UnifiedAAMCLM(training_images, group='PTS', diagonal=100,
                          scales=(0.5, 1.0), patch_shape=(9, 9),
                          context_shape=(18, 18), max_shape_components=0.95,
                          max_appearance_components=0.95)
    fitter = UnifiedAAMCLMFitter(model)
    result = fitter.fit_from_shape(image, initial_shape, max_iters=20)

Active Pictorial Structures
---------------------------
Appearance and deformation modelled by Gaussian Markov random fields over a
graph of the landmarks (here, the Delaunay triangulation of the mean shape).

.. code-block:: python

    from menpo.shape import UndirectedGraph, delaunay_graph
    from menpofit.aps import GenerativeAPS, GaussNewtonAPSFitter, Inverse

    graph = delaunay_graph(training_images[0].landmarks['PTS'])
    appearance_graph = UndirectedGraph(graph.adjacency_matrix)
    aps = GenerativeAPS(training_images, group='PTS',
                        appearance_graph=appearance_graph, diagonal=100,
                        scales=(0.5, 1.0), patch_shape=(9, 9),
                        max_shape_components=20)
    fitter = GaussNewtonAPSFitter(aps, gn_algorithm_cls=Inverse, n_shape=10)
    result = fitter.fit_from_shape(image, initial_shape, max_iters=20)

Supervised Descent
------------------
A cascade of regressors from features at the current shape to the update of the
shape. The model *is* the fitter: it is trained when it is built. ``SDM`` and
``RegularizedSDM`` are ready-made choices of the algorithm. The regression has a
weight for every feature, so it needs many more samples than features, or a ridge
penalty (``alpha``, which ``RegularizedSDM`` takes): without one, ``SDM`` fails on a
system that is not positive definite and asks for it.

.. code-block:: python

    from functools import partial
    from menpofit.sdm import SupervisedDescentFitter, NonParametricNewton

    fitter = SupervisedDescentFitter(training_images, group='PTS',
                                     sd_algorithm_cls=partial(NonParametricNewton, alpha=100),
                                     diagonal=100, scales=(0.5, 1.0),
                                     patch_shape=(9, 9), n_iterations=3,
                                     n_perturbations=5)
    result = fitter.fit_from_bb(image, bounding_box)

.. code-block:: python

    from menpofit.sdm import RegularizedSDM

    fitter = RegularizedSDM(training_images, group='PTS', alpha=100,
                            diagonal=100, scales=(0.5, 1.0),
                            patch_shape=(9, 9), n_iterations=3, n_perturbations=5)
    result = fitter.fit_from_bb(image, bounding_box)

Training takes the bounding boxes it perturbs from the ground truth landmarks
unless you give ``bounding_box_group_glob``. Mind the memory: the regression is
a matrix of (landmarks x pixels of a patch x channels) squared.

Lucas-Kanade Image Alignment
----------------------------
Aligning a template to an image, with no model of a shape or of the
appearance. The residual is ``SSD``, ``FourierSSD``, ``ECC``,
``GradientImages`` or ``GradientCorrelation``.

.. code-block:: python

    from menpo.transform import Affine
    from menpofit.lk import LucasKanadeFitter, InverseCompositional, SSD

    template = training_images[0]
    fitter = LucasKanadeFitter(template, group='PTS', scales=(0.5, 1.0),
                               algorithm_cls=InverseCompositional,
                               residual_cls=SSD)
    result = fitter.fit_from_shape(image, initial_shape, max_iters=20)

Ensemble of Regression Trees
----------------------------
Provided by `dlib <http://dlib.net/>`_ (``pip install "menpofit[dlib]"``).

.. code-block:: python

    # (needs dlib: not run in the tests)
    from menpofit.dlib import DlibERT

    fitter = DlibERT(training_images, group='PTS', scales=(1.0,),
                     n_iterations=10, n_perturbations=5)
    result = fitter.fit_from_bb(image, bounding_box)

Saving and loading a model
--------------------------
Models and fitters are saved with pickle, which menpo wraps:

.. code-block:: python

    import menpo.io as mio
    from menpofit.aam import HolisticAAM, LucasKanadeAAMFitter

    aam = HolisticAAM(training_images, group='PTS', diagonal=100)
    fitter = LucasKanadeAAMFitter(aam, n_shape=10, n_appearance=50)

    mio.export_pickle(fitter, 'fitter.pkl', overwrite=True)
    fitter = mio.import_pickle('fitter.pkl')
    result = fitter.fit_from_shape(image, initial_shape, max_iters=20)
