"""Training SDM on batches of images increments its ridge regressions.

The solution of ridge regression only depends on ``X^T X`` and ``X^T Y``, which are sums over
the samples, so incrementing a regression with new samples gives exactly the model of training
on all of them. Training an SDM on batches of images increments the regressors of every level
with the samples of each batch.
"""

import numpy as np
import pytest

import menpo.io as mio
from menpo.feature import igo
from menpofit import sdm
from menpofit.math import IRLRegression


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
def test_train_and_increment_equal_the_normal_equations(bias):
    X, Y = regression_data()
    expected = normal_equations(X, Y, 3.0, bias)
    full = IRLRegression(alpha=3.0, bias=bias)
    full.train(X, Y)
    incremented = IRLRegression(alpha=3.0, bias=bias, incrementable=True)
    incremented.train(X[:100], Y[:100])
    for start in range(100, 500, 70):  # the last batch is smaller
        incremented.increment(X[start : start + 70], Y[start : start + 70])
    np.testing.assert_allclose(full.W, expected, atol=1e-10)
    np.testing.assert_allclose(incremented.W, expected, atol=1e-10)
    np.testing.assert_allclose(incremented.predict(X), full.predict(X), atol=1e-9)


def test_float32_features_give_a_float32_model():
    X, Y = regression_data()
    X32 = X.astype(np.float32)
    regression = IRLRegression(alpha=3.0)
    regression.train(X32, Y)
    assert regression.W.dtype == np.float32
    expected = normal_equations(X32.astype(np.float64), Y, 3.0, True)
    np.testing.assert_allclose(regression.W, expected, atol=1e-4)


def test_only_an_incrementable_regression_is_incremented():
    X, Y = regression_data()
    regression = IRLRegression(alpha=3.0)
    regression.train(X, Y)
    with pytest.raises(ValueError, match="not incrementable"):
        regression.increment(X, Y)


def test_a_singular_problem_asks_for_a_ridge_penalty():
    rng = np.random.RandomState(0)
    X, Y = rng.randn(10, 30), rng.randn(10, 2)  # fewer samples than features
    with pytest.raises(np.linalg.LinAlgError, match="ridge"):
        IRLRegression(alpha=0.0).train(X, Y)
    IRLRegression(alpha=1.0).train(X, Y)


@pytest.fixture(scope="module")
def training():
    image = mio.import_builtin_asset("takeo.ppm").as_greyscale()
    return [
        image.rotate_ccw_about_centre(a).rescale(0.5) for a in (-8, -6, -3, 0, 3, 6, 9)
    ]


def build(images, batch_size, **kwargs):
    np.random.seed(0)  # the perturbations are random
    return sdm.RegularizedSDM(
        images,
        group="PTS",
        alpha=5.0,
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
        dict(scales=(0.5, 1.0), holistic_features=igo),
    ],
    ids=["one scale", "two scales", "multi-channel features"],
)
@pytest.mark.parametrize("batch_size", [3, 5, 100])
def test_batches_give_the_model_of_training_on_all_the_images(
    training, kwargs, batch_size
):
    # the 7 images in batches of 3 (the last has one image), of 5 and in one batch
    full = build(training, None, **kwargs)
    batched = build(training, batch_size, **kwargs)
    np.testing.assert_allclose(
        batched.reference_shape.points, full.reference_shape.points
    )
    for a, b in zip(full.algorithms, batched.algorithms):
        assert len(a.regressors) == len(b.regressors)
        for ra, rb in zip(a.regressors, b.regressors):
            np.testing.assert_allclose(
                rb.W, ra.W, atol=1e-8 * max(np.abs(ra.W).max(), 1e-3)
            )
    ground_truth = training[0].landmarks["PTS"]
    args = (training[0], ground_truth.bounding_box())
    assert batched.fit_from_bb(
        *args, gt_shape=ground_truth
    ).final_error() == pytest.approx(
        full.fit_from_bb(*args, gt_shape=ground_truth).final_error(), abs=1e-6
    )


def test_a_generator_of_images_can_be_trained_in_batches(training):
    full = build(training, None, scales=(1.0,))
    batched = build((image for image in training), 3, scales=(1.0,))
    np.testing.assert_allclose(
        batched.algorithms[0].regressors[0].W,
        full.algorithms[0].regressors[0].W,
        atol=1e-10,
    )


def test_batches_with_other_regressions_are_refused(training):
    with pytest.raises(NotImplementedError, match="batch_size"):
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


def test_the_supervised_descent_of_an_aam_trains_in_batches(training):
    from functools import partial

    from menpofit import aam

    model = aam.HolisticAAM(
        training,
        group="PTS",
        scales=(1.0,),
        diagonal=50,
        max_shape_components=3,
        max_appearance_components=3,
    )

    def build_aam_fitter(batch_size):
        np.random.seed(0)
        return aam.SupervisedDescentAAMFitter(
            training,
            model,
            group="PTS",
            sd_algorithm_cls=partial(aam.ProjectOutNewton, alpha=5.0),
            n_iterations=2,
            n_perturbations=3,
            batch_size=batch_size,
        )

    full, batched = build_aam_fitter(None), build_aam_fitter(3)
    for ra, rb in zip(full.algorithms[0].regressors, batched.algorithms[0].regressors):
        np.testing.assert_allclose(
            rb.W, ra.W, atol=1e-8 * max(np.abs(ra.W).max(), 1e-3)
        )
    with pytest.raises(NotImplementedError, match="batch_size"):
        aam.SupervisedDescentAAMFitter(
            training,
            model,
            group="PTS",
            sd_algorithm_cls=aam.ProjectOutGaussNewton,
            n_iterations=2,
            n_perturbations=2,
            batch_size=3,
        )
