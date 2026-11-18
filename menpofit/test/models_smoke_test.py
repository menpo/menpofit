"""Train a tiny model of every family and variant, and fit with it.

Everything is deliberately small (six greyscale 60 x 43 pixel images from menpo's built-in
``takeo`` asset, one scale, 5 x 5 patches, a handful of components), so the whole module takes a
few seconds and very little memory. It checks that every model *runs* on the installed
Python/NumPy/SciPy/menpo (it caught ``np.lib.pad`` being removed in NumPy 2, which broke the
CLM), not how accurate it is: with six images the models are not meant to be good.

Do keep the models small. The SDM regression builds a (features x features) matrix where
features = landmarks x patch pixels x channels, so the default 17 x 17 patches on an RGB image
(about 59,000 features for 68 landmarks) need tens of GB.
"""

import numpy as np
import pytest

import menpo.io as mio
from menpofit import aam, atm, clm, sdm

GROUP = "PTS"
COMMON = dict(group=GROUP, scales=(1.0,), diagonal=50, verbose=False)
PATCH = dict(patch_shape=(5, 5))
LK = dict(n_shape=3)


@pytest.fixture(scope="module")
def training():
    image = mio.import_builtin_asset("takeo.ppm").as_greyscale()
    return [image.rotate_ccw_about_centre(a).rescale(0.5) for a in (-6, -3, 0, 3, 6, 9)]


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(0)  # the SDM perturbations are random


def build_aam(cls, extra):
    def build(images):
        model = cls(
            images,
            max_shape_components=3,
            max_appearance_components=3,
            **COMMON,
            **extra,
        )
        return aam.LucasKanadeAAMFitter(model, n_shape=3, n_appearance=3)

    return build


def build_atm(cls, extra):
    def build(images):
        shapes = [i.landmarks[GROUP] for i in images]
        model = cls(images[0], shapes, max_shape_components=3, **COMMON, **extra)
        return atm.LucasKanadeATMFitter(model, **LK)

    return build


def build_clm(images):
    model = clm.CLM(
        images,
        patch_shape=(5, 5),
        context_shape=(10, 10),
        max_shape_components=3,
        **COMMON,
    )
    return clm.GradientDescentCLMFitter(model, **LK)


def build_sdm(cls=None, algorithm=None, **extra):
    def build(images):
        kwargs = dict(patch_shape=(5, 5), n_iterations=2, n_perturbations=4, **COMMON)
        if algorithm is not None:
            return sdm.SupervisedDescentFitter(
                images, sd_algorithm_cls=algorithm, **kwargs
            )
        kwargs.update(extra)
        return cls(images, **kwargs)

    return build


MODELS = {
    "aam-holistic": build_aam(aam.HolisticAAM, {}),
    "aam-patch": build_aam(aam.PatchAAM, PATCH),
    "aam-masked": build_aam(aam.MaskedAAM, PATCH),
    "aam-linear": build_aam(aam.LinearAAM, {}),
    "aam-linear-masked": build_aam(aam.LinearMaskedAAM, PATCH),
    "atm-holistic": build_atm(atm.HolisticATM, {}),
    "atm-patch": build_atm(atm.PatchATM, PATCH),
    "atm-masked": build_atm(atm.MaskedATM, PATCH),
    "atm-linear": build_atm(atm.LinearATM, {}),
    "atm-linear-masked": build_atm(atm.LinearMaskedATM, PATCH),
    "clm": build_clm,
    "sdm": build_sdm(sdm.SDM),
    "sdm-regularized": build_sdm(sdm.RegularizedSDM),
    "sdm-parametric-shape": build_sdm(algorithm=sdm.ParametricShapeNewton),
    "sdm-parametric-appearance": build_sdm(
        algorithm=sdm.ParametricAppearanceWeightsNewton
    ),
    "sdm-fully-parametric": build_sdm(algorithm=sdm.FullyParametricProjectOutNewton),
}
# With six images this regression is ill-conditioned and blows up numerically (also with the
# previous NumPy/SciPy), so only check that it runs and returns the right number of points.
MAY_DIVERGE = {"sdm-fully-parametric"}


@pytest.mark.parametrize("name", MODELS)
def test_model_trains_and_fits(training, name):
    fitter = MODELS[name](training)
    image = training[2]
    ground_truth = image.landmarks[GROUP]
    result = fitter.fit_from_bb(image, ground_truth.bounding_box(), max_iters=5)
    assert result.final_shape.n_points == ground_truth.n_points
    if name not in MAY_DIVERGE:
        assert np.all(np.isfinite(result.final_shape.points))
