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


def normal_equations(X, Y, alpha, bias):
    """The ridge solution from the normal equations, with a general solver."""
    if bias:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
    A = X.T.dot(X)
    A[np.diag_indices_from(A)] += alpha
    return np.linalg.solve(A, X.T.dot(Y))


def regression_data():
    rng = np.random.RandomState(0)
    X = rng.randn(500, 40)
    return X, X.dot(rng.randn(40, 6)) + 0.1 * rng.randn(500, 6)


@pytest.mark.parametrize("bias", [True, False])
def test_train_and_batches_equal_the_normal_equations(bias):
    X, Y = regression_data()
    expected = normal_equations(X, Y, 3.0, bias)
    full = IRLRegression(alpha=3.0, bias=bias)
    full.train(X, Y)
    batched = IRLRegression(alpha=3.0, bias=bias)
    for start in range(0, 500, 70):  # the last batch is smaller
        batched.accumulate(X[start : start + 70], Y[start : start + 70])
    batched.solve()
    np.testing.assert_allclose(full.W, expected, atol=1e-10)
    np.testing.assert_allclose(batched.W, expected, atol=1e-10)
    np.testing.assert_allclose(batched.predict(X), full.predict(X), atol=1e-9)


def test_float32_features_give_a_float32_model():
    X, Y = regression_data()
    X32 = X.astype(np.float32)
    regression = IRLRegression(alpha=3.0)
    regression.train(X32, Y)
    assert regression.W.dtype == np.float32
    expected = normal_equations(X32.astype(np.float64), Y, 3.0, True)
    np.testing.assert_allclose(regression.W, expected, atol=1e-4)


def test_train_without_ridge_on_a_singular_problem_still_solves():
    # fewer samples than features, and no regularization: the matrix is not
    # positive definite, which the Cholesky solve refuses, and the LU solve is
    # the fallback
    rng = np.random.RandomState(0)
    X, Y = rng.randn(10, 30), rng.randn(10, 2)
    regression = IRLRegression(alpha=0.0)
    regression.train(X, Y)
    assert regression.W.shape == (31, 2)
    assert np.isfinite(regression.W).all()


def test_incrementable_regression_is_unchanged():
    X, Y = regression_data()
    regression = IRLRegression(alpha=3.0, incrementable=True)
    regression.train(X, Y)
    np.testing.assert_allclose(regression.W, normal_equations(X, Y, 3.0, True), atol=1e-10)
    assert regression.V.shape == (41, 41)


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
