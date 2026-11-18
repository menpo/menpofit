import numpy as np
import pytest

from menpo.image import Image
from menpofit.clm.expert.base import probability_map


def random_maps(seed=0, shape=(2, 3, 4, 5)):
    return np.random.RandomState(seed).rand(*shape)


def test_every_map_is_normalised_to_sum_one():
    result = probability_map(random_maps())
    assert np.all(result >= 0)
    assert result.sum(axis=(-2, -1)) == pytest.approx(np.ones((2, 3)))


def test_flat_map_among_normal_ones_gets_a_uniform_distribution():
    # a constant response map has no variance: this used to raise an IndexError
    # (a boolean index of shape (2, 3, 1, 1) applied to an array of shape (2, 3, 4, 5))
    x = random_maps(1)
    x[0, 1] = 0.7
    reference = probability_map(random_maps(1))  # the other maps are not affected
    with pytest.warns(UserWarning, match="0 variance"):
        result = probability_map(x)
    # uniform over the 4 x 5 map (and not 1 / prod(axes) = 1 / 2)
    assert result[0, 1] == pytest.approx(np.full((4, 5), 1 / 20))
    assert result.sum(axis=(-2, -1)) == pytest.approx(np.ones((2, 3)))
    mask = np.ones((2, 3), dtype=bool)
    mask[0, 1] = False
    assert result[mask] == pytest.approx(reference[mask])


def test_all_maps_flat():
    with pytest.warns(UserWarning):
        result = probability_map(np.full((2, 2, 3, 3), 5.0))
    assert result == pytest.approx(np.full((2, 2, 3, 3), 1 / 9))


def test_other_axes():
    x = random_maps(2, shape=(3, 6))
    x[1] = 1.0
    with pytest.warns(UserWarning):
        result = probability_map(x, axes=-1)
    assert result[1] == pytest.approx(np.full(6, 1 / 6))
    assert result.sum(axis=-1) == pytest.approx(np.ones(3))
    flat = np.full((2, 3, 4), 2.0)
    with pytest.warns(UserWarning):
        assert probability_map(flat, axes=None) == pytest.approx(
            np.full((2, 3, 4), 1 / 24)
        )


def test_images_are_supported():
    pixels = random_maps(3, shape=(3, 4, 5))
    pixels[1] = 0.0
    with pytest.warns(UserWarning):
        result = probability_map(Image(pixels))
    assert isinstance(result, Image)
    assert result.pixels[1] == pytest.approx(np.full((4, 5), 1 / 20))
