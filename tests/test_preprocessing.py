import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import preprocessing as pp  # noqa: E402


def _img(w=300, h=260):
    d = tempfile.mkdtemp()
    fp = os.path.join(d, "a.jpg")
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(fp)
    return fp


def test_maps_shape_and_modes():
    fp = _img()
    for mode in ("paper", "multi_ela"):
        m = pp.forensic_maps(fp, mode)
        assert m.shape == (260, 300, 3) and m.dtype == np.uint8


def test_crops():
    m = pp.forensic_maps(_img(700, 500), "multi_ela")
    crops = pp.grid_crops(m, 9)
    assert 1 < len(crops) <= 9 and all(c.shape == (224, 224, 3) for c in crops)
    assert pp.center_crop(m).shape == (224, 224, 3)
    small = pp.forensic_maps(_img(100, 80), "multi_ela")
    assert pp.center_crop(small).shape == (224, 224, 3)
    assert pp.to_model_input(crops[0]).dtype == np.float32
