"""Training SDM in batches of images gives the model of training on all of them at once.

A level of the cascade is a ridge regression, whose solution only depends on ``X^T X`` and
``X^T Y``. They are sums over the samples, so they are accumulated over the batches and solved
once (`IRLRegression.accumulate` and `solve`); the images are visited once per level, because the
samples of a level are the shapes that the previous levels have updated.
"""

import numpy as np
import pytest

import menpo.io as mio
from menpo.feature import fast_dsift
from menpofit import sdm
from menpofit.math import IIRLRegression, IRLRegression

RIDGE = 5.0


def test_accumulate_and_solve_equal_train():
    rng = np.random.RandomState(0)
    X = rng.randn(500, 40)
    Y = X.dot(rng.randn(40, 6)) + 0.1 * rng.randn(500, 6)
    for bias in (True, False):
        full = IRLRegression(alpha=3.0, bias=bias)
        full.train(X, Y)
        batched = IRLRegression(alpha=3.0, bias=bias)
        for start in range(0, 500, 70):  # the last batch is smaller
            batched.accumulate(X[start : start + 70], Y[start : start + 70])
        batched.solve()
        np.testing.assert_allclose(batched.W, full.W, atol=1e-10)
        np.testing.assert_allclose(batched.predict(X), full.predict(X), atol=1e-9)


def test_solve_needs_samples_and_frees_the_matrix():
    regression = IRLRegression(alpha=1.0)
    with pytest.raises(ValueError):
        regression.solve()
    regression.accumulate(np.random.RandomState(0).randn(20, 5), np.ones((20, 2)))
    regression.solve()
    assert regression._XX is None and regression._XY is None


def test_only_ridge_regression_trains_in_batches():
    assert IRLRegression._supports_batches
    assert not IIRLRegression._supports_batches
    with pytest.raises(ValueError):
        IRLRegression(incrementable=True).accumulate(np.ones((3, 2)), np.ones((3, 1)))


@pytest.fixture(scope="module")
def training():
    image = mio.import_builtin_asset("takeo.ppm").as_greyscale()
    return [image.rotate_ccw_about_centre(a).rescale(0.5) for a in (-8, -6, -3, 0, 3, 6, 9)]


def build(images, batch_size, **kwargs):
    np.random.seed(0)  # the perturbations are random
    return sdm.RegularizedSDM(
        images,
        group="PTS",
        alpha=RIDGE,
        diagonal=50,
        patch_shape=(5, 5),
        n_iterations=3,
        n_perturbations=4,
        batch_size=batch_size,
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(scales=(1.0,)),
        dict(scales=(0.5, 1.0)),
        dict(scales=(0.5, 1.0), holistic_features=fast_dsift),
    ],
    ids=["one scale", "two scales", "dense sift"],
)
def test_batches_give_the_same_model(training, kwargs):
    full = build(training, None, **kwargs)
    # a batch size that does not divide the images: the last batch has two
    batched = build(training, 5, **kwargs)
    np.testing.assert_allclose(batched.reference_shape.points, full.reference_shape.points)
    for a, b in zip(full.algorithms, batched.algorithms):
        assert len(a.regressors) == len(b.regressors)
        for ra, rb in zip(a.regressors, b.regressors):
            scale = np.abs(ra.W).max()
            np.testing.assert_allclose(rb.W, ra.W, atol=1e-8 * max(scale, 1e-3))
    ground_truth = training[0].landmarks["PTS"]
    args = (training[0], ground_truth.bounding_box())
    error_full = full.fit_from_bb(*args, gt_shape=ground_truth).final_error()
    error_batched = batched.fit_from_bb(*args, gt_shape=ground_truth).final_error()
    assert error_batched == pytest.approx(error_full, abs=1e-6)


def test_batch_size_of_a_whole_set_or_more(training):
    full = build(training, None, scales=(1.0,))
    for batch_size in (len(training), 100):
        batched = build(training, batch_size, scales=(1.0,))
        np.testing.assert_allclose(
            batched.algorithms[0].regressors[0].W, full.algorithms[0].regressors[0].W, atol=1e-10
        )


def test_batches_with_other_regressions_are_refused(training):
    with pytest.raises(NotImplementedError, match="ridge"):
        sdm.SupervisedDescentFitter(
            training,
            group="PTS",
            sd_algorithm_cls=sdm.NonParametricGaussNewton,
            diagonal=50,
            scales=(1.0,),
            patch_shape=(5, 5),
            n_iterations=2,
            n_perturbations=2,
            batch_size=3,
        )
