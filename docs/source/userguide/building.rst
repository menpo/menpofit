.. _ug-building:

Building Models
===============
All MenpoFit's models are built in a **multi-scale** manner, i.e. in multiple
resolutions. In all our core classes, this is controlled using the following
three parameters:

**reference_shape** (`PointCloud`)
  First, the size of the training images is normalized by rescaling them so
  that the scale of their ground truth shapes matches the scale of this
  reference shape. In case no reference shape is provided, then the mean of
  the ground shapes is used. This step is essential in order to ensure
  consistency between the extracted features of the images.
**diagonal** (`int`)
  This parameter is used to rescale the reference shape so that the diagonal
  of its bounding box matches the provided value. This rescaling takes place
  before normalizing the training images' size. Thus, `diagonal` controls the
  size of the model at the highest scale.
**scales** (`tuple` of `float`)
  A `tuple` with the scale value at each level, provided in ascending order,
  i.e. from lowest to highest scale. These values are proportional to the
  final resolution achieved through the reference shape normalization.

Additionally, all models have a **holistic_features** argument which expects
the `callable` that will be used for extracting features from the training
images.

Given the above assumptions, an example of a typical call for building a
deformable model using :map:`HolisticAAM` is:

.. code-block:: python

    from menpofit.aam import HolisticAAM
    from menpo.feature import fast_dsift

    aam = HolisticAAM(training_images, group='PTS', reference_shape=None,
                      diagonal=200, scales=(0.25, 0.5, 1.0),
                      holistic_features=fast_dsift, verbose=True)

Information about any kind of model can be retrieved by:

.. code-block:: python

    print(aam)

The next section (:ref:`Fitting <ug-fitting>`) explains the basics of
fitting such a deformable model.
