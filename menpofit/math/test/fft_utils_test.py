import numpy as np
import pytest

from menpofit.math.fft_utils import crop, fft_convolve2d, pad


def test_pad_centres_the_pixels_and_keeps_leading_dimensions():
    # np.lib.pad, which this used to call, was removed in NumPy 2
    x = np.ones((2, 3, 3))
    result = pad(x, (6, 7))
    assert result.shape == (2, 6, 7)
    # rows: (6 - 3) // 2 = 1, plus one more at the top because 3 + 2 < 6; columns: 2 on both sides
    assert result[:, 2:5, 2:5] == pytest.approx(np.ones((2, 3, 3)))
    assert result.sum() == pytest.approx(2 * 9)


def test_pad_symmetric_boundary():
    x = np.arange(9.0).reshape(1, 3, 3)
    result = pad(x, (5, 5), boundary="symmetric")
    assert result.shape == (1, 5, 5)
    assert result[:, 1:4, 1:4] == pytest.approx(x)
    assert result[0, 0, 1:4] == pytest.approx(x[0, 0])  # mirrored first row


@pytest.mark.parametrize("shape", [(8, 9), (9, 8), (7, 7), (10, 11)])
def test_crop_returns_the_centre(shape):
    x = np.random.RandomState(0).rand(2, 11, 12)
    result = crop(x, shape)
    assert result.shape == (2,) + shape
    top = (11 - shape[0]) // 2 + (1 if (11 - shape[0]) % 2 else 0)
    left = (12 - shape[1]) // 2 + (1 if (12 - shape[1]) % 2 else 0)
    assert np.array_equal(result, x[:, top : top + shape[0], left : left + shape[1]])


def test_crop_to_the_same_shape_is_the_identity():
    # a margin of 0 used to give the slice [0:-0], which is empty
    x = np.random.RandomState(1).rand(1, 6, 7)
    assert np.array_equal(crop(x, (6, 7)), x)


@pytest.mark.parametrize("filter_size", [3, 5, 9, 17])
def test_fft_convolution_with_a_centred_delta_filter_is_the_identity(filter_size):
    x = np.random.RandomState(2).rand(2, 12, 13)
    f = np.zeros((2, filter_size, filter_size))
    f[:, filter_size // 2, filter_size // 2] = 1.0
    result = fft_convolve2d(x, f, mode="same")
    assert result.shape == x.shape
    assert result == pytest.approx(x, abs=1e-9)
