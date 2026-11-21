"""A fitter works on its own trimmed copy of the model it is given: it does not change that one.

The models can be gigabytes, so the copy shares the memory of the models (menpo's ``trimmed()``).
"""

from types import SimpleNamespace

import menpo.io as mio
import numpy as np
import pytest
from menpo.base import MenpoDeprecationWarning
from menpo.model import PCAModel
from menpo.shape import PointCloud
from numpy.testing import assert_allclose, assert_array_equal

from menpofit import aam, atm, checks, clm
from menpofit.modelinstance import PDM, GlobalPDM, OrthoPDM
from menpofit.transform import DifferentiableAlignmentSimilarity


def shapes(seed=0, n=12):
    rs = np.random.RandomState(seed)
    return [PointCloud(rs.randn(10, 2) + 2 * np.arange(10)[:, None]) for _ in range(n)]


def a_model():
    # what a fitter is given: a shape model (PDM) and an appearance model (PCAModel) per scale
    return SimpleNamespace(
        shape_models=[PDM(shapes(0)), OrthoPDM(shapes(1))],
        appearance_models=[PCAModel(shapes(2)), PCAModel(shapes(3))],
        reference_shape="shared",
    )


def n_comps(models):
    return [m.n_components for m in models]


# ---- checks.trimmed_copy
def test_trimmed_copy_one_number_for_all_scales_and_leaves_the_model_alone():
    model = a_model()
    new = checks.trimmed_copy(model, shape_models=3, appearance_models=4)
    assert n_comps(new.shape_models) == [3, 3] and n_comps(new.appearance_models) == [
        4,
        4,
    ]
    assert n_comps(model.shape_models) == [11, 11]  # as it was
    assert n_comps(model.appearance_models) == [11, 11]
    assert new is not model and new.reference_shape is model.reference_shape


def test_trimmed_copy_number_per_scale_one_in_a_list_and_a_ratio():
    model = a_model()
    assert n_comps(checks.trimmed_copy(model, shape_models=[2, 5]).shape_models) == [
        2,
        5,
    ]
    assert n_comps(checks.trimmed_copy(model, shape_models=[4]).shape_models) == [
        4,
        4,
    ]
    ratio = checks.trimmed_copy(model, appearance_models=0.5).appearance_models
    assert all(1 <= n < 11 for n in n_comps(ratio))


def test_trimmed_copy_none_gives_new_models_with_all_the_components_and_a_wrong_length_is_refused():
    model = a_model()
    new = checks.trimmed_copy(model, shape_models=None, appearance_models=None)
    assert new is not model
    for old_models, new_models in (
        (model.shape_models, new.shape_models),
        (model.appearance_models, new.appearance_models),
    ):
        assert n_comps(new_models) == n_comps(old_models) == [11, 11]
        assert all(a is not b for a, b in zip(old_models, new_models))

    with pytest.raises(ValueError, match="n_components can be an integer"):
        checks.trimmed_copy(model, shape_models=[1, 2, 3])
    assert n_comps(model.shape_models) == [11, 11]


def test_trimmed_copy_only_changes_what_it_is_asked_to():
    model = a_model()
    new = checks.trimmed_copy(model, shape_models=3)
    assert new.appearance_models is model.appearance_models  # the same list
    assert n_comps(new.shape_models) == [3, 3]


def test_trimmed_copy_shares_the_memory_of_the_models():
    model = a_model()
    new = checks.trimmed_copy(model, shape_models=3, appearance_models=3)
    for old, small in zip(model.appearance_models, new.appearance_models):
        assert np.shares_memory(old.components, small.components)
    # a PDM shares its model's memory too (an OrthoPDM orthonormalizes its components
    # again, which gives it arrays of its own, the size of what it keeps)
    assert np.shares_memory(
        model.shape_models[0].model.components, new.shape_models[0].model.components
    )


def test_a_second_trimmed_copy_can_have_more_components_than_the_first():
    # this is what setting the number on the models themselves could not give a second fitter
    model = a_model()
    first = checks.trimmed_copy(model, shape_models=2, appearance_models=2)
    second = checks.trimmed_copy(model, shape_models=6, appearance_models=6)
    assert n_comps(first.shape_models) == [2, 2] and n_comps(second.shape_models) == [
        6,
        6,
    ]
    assert n_comps(second.appearance_models) == [6, 6]


# ---- PDM.trimmed: the same as a PDM that was built with fewer components
PDMS = pytest.mark.parametrize(
    "make",
    [
        lambda **kw: PDM(shapes(), **kw),
        lambda **kw: GlobalPDM(shapes(), DifferentiableAlignmentSimilarity, **kw),
        lambda **kw: OrthoPDM(shapes(), **kw),
    ],
    ids=["PDM", "GlobalPDM", "OrthoPDM"],
)


@PDMS
def test_pdm_trimmed_is_what_a_pdm_built_with_fewer_components_is(make):
    k = 4
    small, built = make().trim_components(k), make(max_n_components=k)
    probe = shapes(7, n=1)[0]
    pts = np.random.RandomState(1).rand(7, 2)
    for pdm in (small, built):
        pdm.set_target(probe)
    assert small.n_components == small.n_weights == k
    assert_allclose(
        np.abs(small.model.components), np.abs(built.model.components), atol=1e-12
    )
    assert_allclose(np.abs(small.weights), np.abs(built.weights), atol=1e-12)
    assert_allclose(small.target.points, built.target.points, atol=1e-12)
    assert_allclose(np.abs(small.d_dp(pts)), np.abs(built.d_dp(pts)), atol=1e-12)
    assert small.as_vector().shape == built.as_vector().shape


@PDMS
def test_pdm_trimmed_is_a_new_instance_that_leaves_this_one_alone(make):
    full = make()
    full.set_target(shapes(8, n=1)[0])
    target, weights = full.target.points.copy(), full.weights.copy()
    components = full.model.components.copy()
    small = full.trim_components(4)
    assert type(small) is type(full) and small is not full
    assert_allclose(
        small.target.points, small.model.mean().points
    )  # a new one starts at the mean
    small.set_target(shapes(9, n=1)[0])
    assert_array_equal(full.target.points, target)
    assert_array_equal(full.weights, weights)
    assert_array_equal(full.model.components, components)
    assert full.n_components == 11 and full.n_weights == 11
    if hasattr(full, "global_transform"):
        assert small.global_transform is not full.global_transform


def test_a_trimmed_pdm_shares_the_memory_of_the_model():
    full = PDM(shapes())
    assert np.shares_memory(
        full.trim_components(4).model.components, full.model.components
    )


@PDMS
def test_incrementing_a_trimmed_pdm_leaves_the_one_it_came_from_alone(make):
    full = make()
    components = full.model.components.copy()
    small = full.trim_components(4)
    small.increment(shapes(5, n=3))
    assert not np.shares_memory(small.model.components, full.model.components)
    assert_array_equal(full.model.components, components)
    assert full.n_components == 11


def test_an_ortho_pdm_made_from_a_trimmed_model_leaves_the_model_it_came_from_alone():
    # what LinearOrthoMDTransform does with the shape model of an AAM: an OrthoPDM
    # orthonormalizes the components of the model it is given, by giving it new arrays
    full = OrthoPDM(shapes())
    components = full.model.components.copy()
    OrthoPDM(full.model.trim_components(4))
    assert_array_equal(full.model.components, components)


# ---- the fitters
@pytest.fixture(scope="module")
def training():
    image = mio.import_builtin_asset("takeo.ppm").as_greyscale()
    return [image.rotate_ccw_about_centre(a).rescale(0.5) for a in (-6, -3, 0, 3, 6, 9)]


COMMON = dict(group="PTS", scales=(1.0,), diagonal=50, verbose=False)


@pytest.fixture(scope="module")
def an_aam(training):
    return aam.HolisticAAM(training, **COMMON)


def test_an_aam_fitter_works_on_its_own_trimmed_copy(an_aam, training):
    full = n_comps(an_aam.shape_models), n_comps(an_aam.appearance_models)
    shape_model = an_aam.shape_models[0]
    fitter = aam.LucasKanadeAAMFitter(an_aam, n_shape=3, n_appearance=2)
    # the AAM it was given is as it was, and the fitter has the numbers it was asked for
    assert (n_comps(an_aam.shape_models), n_comps(an_aam.appearance_models)) == full
    assert an_aam.shape_models[0] is shape_model
    assert fitter.aam is not an_aam
    assert n_comps(fitter.aam.shape_models) == [3] and n_comps(
        fitter.aam.appearance_models
    ) == [2]
    assert "3 active shape components" in str(fitter) and "2 active appearance" in str(
        fitter
    )
    # and it fits
    gt = training[2].landmarks["PTS"]
    result = fitter.fit_from_bb(training[2], gt.bounding_box(), max_iters=3)
    assert np.all(np.isfinite(result.final_shape.points))


def test_a_second_aam_fitter_can_ask_for_more_components_than_the_first(an_aam):
    n_shape, n_appearance = (
        n_comps(an_aam.shape_models)[0],
        n_comps(an_aam.appearance_models)[0],
    )
    assert n_shape >= 2 and n_appearance >= 3
    first = aam.LucasKanadeAAMFitter(an_aam, n_shape=1, n_appearance=2)
    second = aam.LucasKanadeAAMFitter(
        an_aam, n_shape=n_shape, n_appearance=n_appearance
    )
    none = aam.LucasKanadeAAMFitter(an_aam)  # all of them
    assert n_comps(first.aam.shape_models) == [1]
    assert n_comps(second.aam.shape_models) == [
        n_shape
    ]  # more than the first, from the same AAM
    assert n_comps(second.aam.appearance_models) == [n_appearance]
    assert n_comps(none.aam.shape_models) == [n_shape]


def test_an_atm_and_a_clm_fitter_leave_the_model_they_are_given_alone(training):
    shapes_ = [i.landmarks["PTS"] for i in training]
    a_atm = atm.HolisticATM(training[0], shapes_, **COMMON)
    full = n_comps(a_atm.shape_models)
    assert full[0] >= 3
    atm_fitter = atm.LucasKanadeATMFitter(a_atm, n_shape=2)
    assert n_comps(a_atm.shape_models) == full
    assert n_comps(atm_fitter.atm.shape_models) == [2]

    a_clm = clm.CLM(training, patch_shape=(5, 5), context_shape=(10, 10), **COMMON)
    full = n_comps(a_clm.shape_models)
    assert full[0] >= 3
    clm_fitter = clm.GradientDescentCLMFitter(a_clm, n_shape=2)
    assert n_comps(a_clm.shape_models) == full
    assert n_comps(clm_fitter.clm.shape_models) == [2]


def test_random_instance_of_an_aam_with_fewer_appearance_than_shape_components(an_aam):
    # the appearance weights were drawn with the number of *shape* components
    model = checks.trimmed_copy(an_aam, shape_models=3, appearance_models=1)
    assert n_comps(model.appearance_models) == [1] and n_comps(model.shape_models) == [
        3
    ]
    instance = model.random_instance()
    assert np.all(np.isfinite(instance.pixels))


@pytest.mark.parametrize("cls", [aam.HolisticAAM, aam.LinearAAM])
def test_fitting_does_not_change_the_aam_also_without_n_shape(cls, training):
    # without n_shape / n_appearance a fitter used the AAM's own models: fitting moved the
    # target of its shape models, and a LinearAAM fitter orthonormalized its shape model
    model = cls(training, **COMMON)
    shape_model = model.shape_models[0]
    pdm_state = (shape_model.target.points.copy(), shape_model.weights.copy())
    components = shape_model.model.components.copy()
    arrays = shape_model.model._components
    fitter = aam.LucasKanadeAAMFitter(model)
    gt = training[2].landmarks["PTS"]
    fitter.fit_from_bb(training[2], gt.bounding_box(), max_iters=3)
    assert model.shape_models[0] is shape_model
    assert_array_equal(shape_model.target.points, pdm_state[0])
    assert_array_equal(shape_model.weights, pdm_state[1])
    assert shape_model.model._components is arrays
    assert_array_equal(shape_model.model.components, components)
