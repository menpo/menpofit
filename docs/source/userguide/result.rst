.. _ug-result:

Fitting Result
==============

Objects
-------
The fitting methods of the `Fitters` presented in the previous section return
a result object. MenpoFit has three basic fitting result objects:

* :map:`Result` : Basic fitting result object that holds the final shape, and
  optionally, the initial shape, ground truth shape and the image.

* :map:`MultiScaleNonParametricIterativeResult` : The result of a multi-scale
  iterative fitting procedure. Apart from the final shape, it also stores the
  shapes acquired at each fitting iteration.

* :map:`MultiScaleParametricIterativeResult` : The same as :map:`MultiScaleNonParametricIterativeResult`
  with the difference that the optimisation was performed over the parameters
  of a statistical parametric shape model. Thus, apart from the actual
  shapes, it also stores the shape parameters acquired per iteration.

Attributes
----------
The above result objects can provide some very useful information regarding
the fitting procedure. For example, the various shapes can be retrieved as:

`result.final_shape`
  The final shape of the fitting procedure.
`result.initial_shape`
  The initial shape of the fitting procedure.
`result.image`
  The image on which the fitting procedure was applied.
`result.gt_shape`
  The ground truth shape associated to the image.
`result.shapes`
  The `list` of shapes acquired at each fitting iteration. It only applies on
  :map:`MultiScaleNonParametricIterativeResult` and
  :map:`MultiScaleParametricIterativeResult`.

Also, a result can compute some error metrics, in case the `gt_shape` of the
image exists:

`result.final_error()`
  The final fitting error.
`result.initial_error()`
  The initial fitting error.
`result.errors()`
  The `list` of errors acquired at each fitting iteration. It only applies on
  :map:`MultiScaleNonParametricIterativeResult` and
  :map:`MultiScaleParametricIterativeResult`.
