"""The ``scales`` argument is documented as a float or a tuple of floats for every model.

The AAM, APS, SDM and LK classes normalise it with ``checks.check_scales`` before using its length;
the ATM and CLM classes called ``len(scales)`` on the raw argument first, so a plain number such as
``scales=1`` raised ``TypeError: object of type 'int' has no len()``.
"""

import pytest

import menpo.io as mio
from menpofit import atm, clm

GROUP = "PTS"
SCALES = [1, 1.0, (1,), (1.0,), [1.0]]
ATM_VARIANTS = [
    ("holistic", atm.HolisticATM, {}),
    ("patch", atm.PatchATM, {"patch_shape": (5, 5)}),
    ("masked", atm.MaskedATM, {"patch_shape": (5, 5)}),
    ("linear", atm.LinearATM, {}),
    ("linear_masked", atm.LinearMaskedATM, {"patch_shape": (5, 5)}),
]


@pytest.fixture(scope="module")
def images():
    image = mio.import_builtin_asset("takeo.ppm").as_greyscale()
    return [image.rotate_ccw_about_centre(a).rescale(0.4) for a in (-6, -3, 0, 3, 6)]


@pytest.mark.parametrize("scales", SCALES, ids=repr)
@pytest.mark.parametrize(
    "name, cls, extra", ATM_VARIANTS, ids=[v[0] for v in ATM_VARIANTS]
)
def test_atm_accepts_a_single_scale(images, name, cls, extra, scales):
    shapes = [i.landmarks[GROUP] for i in images]
    model = cls(
        images[0], shapes, group=GROUP, scales=scales, diagonal=40,
        max_shape_components=3, **extra,
    )  # fmt: skip
    assert model.n_scales == 1
    assert model.scales == [1]


@pytest.mark.parametrize("scales", SCALES, ids=repr)
def test_clm_accepts_a_single_scale(images, scales):
    model = clm.CLM(
        images, group=GROUP, scales=scales, diagonal=40, patch_shape=(5, 5),
        context_shape=(10, 10), max_shape_components=3,
    )  # fmt: skip
    assert model.n_scales == 1
    assert model.scales == [1]


def test_several_scales_still_work(images):
    model = atm.PatchATM(
        images[0], [i.landmarks[GROUP] for i in images], group=GROUP, scales=(0.5, 1.0),
        patch_shape=[(5, 5), (7, 7)], diagonal=40, max_shape_components=3,
    )  # fmt: skip
    assert model.n_scales == 2
    assert model.scales == [0.5, 1.0]


def test_invalid_scales_are_rejected(images):
    with pytest.raises(ValueError):
        clm.CLM(
            images,
            group=GROUP,
            scales="fast",
            patch_shape=(5, 5),
            context_shape=(10, 10),
        )
