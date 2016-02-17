.. _api-error-index:

:mod:`menpofit.error`
=====================

Normalisers
-----------
Functions that compute a metric which can be used to normalise the error
between two shapes.

Bounding Box Normalisers
^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 1

    bb_area
    bb_perimeter
    bb_avg_edge_length
    bb_diagonal

Distance Normalisers
^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 1

    distance_two_indices


Errors
------
Functions that compute the error between two shapes.

Root Mean Square Error
^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 1

    root_mean_square_error
    root_mean_square_bb_normalised_error
    root_mean_square_distance_normalised_error
    root_mean_square_distance_indexed_normalised_error

Euclidean Distance Error
^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 1

    euclidean_error
    euclidean_bb_normalised_error
    euclidean_distance_normalised_error
    euclidean_distance_indexed_normalised_error


Cumulative Error Distribution
-----------------------------
.. toctree::
    :maxdepth: 1

    compute_cumulative_error
