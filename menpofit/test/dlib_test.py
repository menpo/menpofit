"""menpofit.dlib: the conversion of a dlib detection (needs no dlib) and DlibERT end to end (needs dlib)."""

import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

import menpo.io as mio


def test_a_dlib_detection_becomes_a_pointcloud_of_row_column_points(monkeypatch):
    # dlib is not installed in CI: a stand-in module is enough for the
    # conversion, which only calls .parts() and reads .x and .y. (It went through
    # np.vstack of a generator, which NumPy no longer takes.)
    monkeypatch.setitem(sys.modules, "dlib", types.ModuleType("dlib"))
    monkeypatch.delitem(sys.modules, "menpofit.dlib.conversion", raising=False)
    conversion = importlib.import_module("menpofit.dlib.conversion")
    try:
        detection = SimpleNamespace(
            parts=lambda: [
                SimpleNamespace(x=10, y=2),
                SimpleNamespace(x=11, y=3),
                SimpleNamespace(x=12, y=4),
            ]
        )
        points = conversion.dlib_full_object_detection_to_pointcloud(detection)
        np.testing.assert_array_equal(points.points, [[2, 10], [3, 11], [4, 12]])
    finally:
        sys.modules.pop("menpofit.dlib.conversion", None)  # do not keep the stand-in's


def test_dlib_ert_trains_and_fits():
    pytest.importorskip("dlib")
    from menpofit.dlib import DlibERT

    image = mio.import_builtin_asset("takeo.ppm").as_greyscale()
    training = [
        image.rotate_ccw_about_centre(a).rescale(0.5) for a in (-6, -3, 0, 3, 6, 9)
    ]
    np.random.seed(0)
    fitter = DlibERT(
        training,
        group="PTS",
        scales=(1.0,),
        diagonal=60,
        n_iterations=3,
        n_perturbations=2,
        n_trees=5,
        n_tree_levels=2,
    )
    target = training[2]
    ground_truth = target.landmarks["PTS"]
    result = fitter.fit_from_bb(
        target, ground_truth.bounding_box(), gt_shape=ground_truth
    )
    assert result.final_shape.n_points == ground_truth.n_points
    assert result.final_error() < 0.5
