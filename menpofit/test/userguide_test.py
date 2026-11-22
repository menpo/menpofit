"""The examples of docs/source/userguide/models.rst run.

Every ``code-block:: python`` of that page is executed on a few small images, so
that the examples cannot go out of date without a test failing. The page's
examples are written for real data (more images, larger sizes): here ``image``
is a greyscale picture of 60 x 43 pixels and there are 13 training images, which
is enough to build and to fit every model, not to make them good.
"""

import re
import textwrap
from pathlib import Path

import numpy as np
import pytest

import menpo.io as mio
from menpo.shape import PointCloud

PAGE = (
    Path(__file__).resolve().parents[2] / "docs" / "source" / "userguide" / "models.rst"
)
SKIP_MARKER = "not run in the tests"


def code_blocks(text):
    blocks = re.findall(
        r"^\.\. code-block:: python\n\n((?:(?:    .*)?\n)+)", text, flags=re.M
    )
    return [textwrap.dedent(b).rstrip() + "\n" for b in blocks]


# The tests run from the installed package as well (the sdist job), where the docs
# are not: only a source checkout (it has the pyproject.toml next to docs/) must
# have the page.
IN_SOURCE_CHECKOUT = (PAGE.parents[3] / "pyproject.toml").exists()
if not PAGE.exists() and not IN_SOURCE_CHECKOUT:
    pytest.skip("the documentation is not installed", allow_module_level=True)

BLOCKS = code_blocks(PAGE.read_text()) if PAGE.exists() else []
RUNNABLE = [b for b in BLOCKS if SKIP_MARKER not in b]


def block_id(block):
    # what the example builds, named by its first import: from menpofit.x import A, B
    line = next(l for l in block.splitlines() if l.startswith(("from", "import")))
    return re.sub(
        r"[^A-Za-z0-9_.]+", "_", line.replace("from ", "").replace(" import ", ".")
    )[:50]


@pytest.fixture(scope="module")
def data():
    # takeo rotated (the appearance varies) and with the landmarks moved about
    # (a shape model of rotations alone has no variance left out when it is
    # trimmed, which the models that need a noise variance can not take)
    image = mio.import_builtin_asset("takeo.ppm").as_greyscale()
    rng = np.random.RandomState(0)
    images = []
    for angle in range(-12, 13, 2):
        img = image.rotate_ccw_about_centre(angle).rescale(0.5)
        points = img.landmarks["PTS"].points
        img.landmarks["PTS"] = PointCloud(points + rng.randn(*points.shape) * 1.5)
        images.append(img)
    target = images[len(images) // 2]
    gt = target.landmarks["PTS"]
    initial = PointCloud(gt.points + rng.randn(*gt.points.shape))
    return dict(
        training_images=images,
        image=target,
        initial_shape=initial,
        bounding_box=gt.bounding_box(),
    )


def test_the_page_has_examples_for_every_family():
    assert PAGE.exists()
    text = " ".join(BLOCKS)
    for name in (
        "HolisticAAM",
        "PatchAAM",
        "SupervisedDescentAAMFitter",
        "HolisticATM",
        "CLM",
        "UnifiedAAMCLM",
        "GenerativeAPS",
        "SupervisedDescentFitter",
        "SDM",
        "LucasKanadeFitter",
        "DlibERT",
        "export_pickle",
    ):
        assert name in text, name


@pytest.mark.parametrize(
    "block",
    RUNNABLE,
    ids=["{}-{}".format(i, block_id(b)) for i, b in enumerate(RUNNABLE)],
)
def test_example_runs(block, data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the pickle is written here
    np.random.seed(0)
    namespace = dict(data)
    exec(compile(block, "models.rst", "exec"), namespace)
    result = namespace.get("result")
    if result is not None:
        assert result.final_shape.n_points == data["initial_shape"].n_points
