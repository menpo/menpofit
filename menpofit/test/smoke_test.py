"""Smoke tests: the package imports and the main numerical pipeline runs end to end.

These are deliberately tiny (a few small images) so they stay fast and light; they exist to catch
breakage from newer Python/NumPy/SciPy/menpo releases, which the unit tests alone did not.
"""

import importlib

import numpy as np
import pytest

import menpo.io as mio
from menpo.shape import PointCloud

SUBPACKAGES = [
    "menpofit.aam",
    "menpofit.aps",
    "menpofit.atm",
    "menpofit.clm",
    "menpofit.error",
    "menpofit.lk",
    "menpofit.math",
    "menpofit.sdm",
    "menpofit.transform",
    "menpofit.unified_aam_clm",
    "menpofit.visualize",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    importlib.import_module(name)


def test_version_is_available():
    import menpofit

    assert isinstance(menpofit.__version__, str) and menpofit.__version__


def test_area_under_curve_uses_scipy_simpson():
    # this went through scipy.integrate.simps, which SciPy 1.14 removed (and broke `import menpofit`)
    from menpofit.error import area_under_curve_and_failure_rate

    # every error is 0: the CED is 1 everywhere, so the area is 1 and nothing fails
    auc, failure_rate = area_under_curve_and_failure_rate(
        [0.0] * 10, step_error=0.01, max_error=0.08
    )
    assert auc == pytest.approx(1.0)
    assert failure_rate == pytest.approx(0.0)
    # half of the errors are beyond max_error: those are the failures
    errors = [0.0] * 5 + [0.5] * 5
    auc, failure_rate = area_under_curve_and_failure_rate(
        errors, step_error=0.01, max_error=0.08
    )
    assert failure_rate == pytest.approx(0.5)
    assert 0.0 < auc < 1.0


def test_aam_trains_and_fits():
    from menpofit.aam import HolisticAAM, LucasKanadeAAMFitter

    image = mio.import_builtin_asset("takeo.ppm")
    group = "PTS"
    training = [image.rotate_ccw_about_centre(a).rescale(0.5) for a in (-6, -3, 0, 3, 6, 9)]
    aam = HolisticAAM(
        training,
        group=group,
        verbose=False,
        scales=1,
        diagonal=60,
        max_shape_components=3,
        max_appearance_components=3,
    )
    fitter = LucasKanadeAAMFitter(aam, n_shape=3, n_appearance=3)
    target = training[2]
    ground_truth = target.landmarks[group]
    rng = np.random.RandomState(0)
    initial = PointCloud(ground_truth.points + rng.randn(*ground_truth.points.shape))
    result = fitter.fit_from_shape(target, initial, max_iters=5)
    error = np.sqrt(((result.final_shape.points - ground_truth.points) ** 2).sum(1)).mean()
    assert result.final_shape.n_points == ground_truth.n_points
    assert error < 1.0
