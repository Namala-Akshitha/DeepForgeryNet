"""Shared inference helpers: multi-crop (+flip) test-time augmentation."""
import numpy as np

import config
import preprocessing as pp


def predict_maps(model, maps: np.ndarray, max_crops=config.TTA_MAX_CROPS,
                 flips=config.TTA_FLIPS, batch_size=32) -> float:
    """Probability that an image is forged, averaged over crops covering the whole image."""
    crops = pp.grid_crops(maps, max_crops)
    if flips:
        crops = crops + [c[:, ::-1] for c in crops]
    x = np.stack([pp.to_model_input(np.ascontiguousarray(c)) for c in crops])
    p = model.predict(x, batch_size=batch_size, verbose=0).ravel()
    return float(p.mean())


def predict_path(model, path, mode, cache_dir=None, **kw) -> float:
    return predict_maps(model, pp.cached_maps(path, mode, cache_dir), **kw)
