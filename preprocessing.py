"""Artifact-aware forensic preprocessing (ELA / denoise / Sobel).

Deliberately free of TensorFlow so it can be unit-tested anywhere.
Maps are computed at the ORIGINAL image resolution (no resizing), because
resizing destroys the fine JPEG-compression traces that ELA relies on.
"""
import hashlib
import io
import os

import cv2
import numpy as np
from PIL import Image

import config

# ResNet50 "caffe" preprocessing: RGB -> BGR, subtract ImageNet mean.
_BGR_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def _ela(img_rgb: Image.Image, quality: int) -> np.ndarray:
    """Error Level Analysis: re-save at `quality`, amplify the difference."""
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    diff = np.abs(np.asarray(img_rgb, np.float32) - np.asarray(resaved, np.float32))
    m = float(diff.max())
    if m > 0:
        diff *= 255.0 / m
    return diff.mean(axis=2)  # H x W


def forensic_maps(path: str, mode: str = config.MODE) -> np.ndarray:
    """Return an H x W x 3 uint8 stack of forensic maps for the image at `path`."""
    img = Image.open(path).convert("RGB")
    if mode == "paper":
        ela = _ela(img, config.ELA_QUALITY)
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        den = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.5)
        gx = cv2.Sobel(den, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(den, cv2.CV_32F, 0, 1)
        mag = np.hypot(gx, gy)
        if mag.max() > 0:
            mag = mag * (255.0 / mag.max())
        stack = np.stack([ela, den.astype(np.float32), mag], axis=-1)
    elif mode == "multi_ela":
        stack = np.stack([_ela(img, q) for q in config.ELA_QUALITIES], axis=-1)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return np.clip(stack, 0, 255).astype(np.uint8)


def cached_maps(path: str, mode: str, cache_dir: str | None) -> np.ndarray:
    """Like forensic_maps but memoised on disk (maps are expensive to build)."""
    if not cache_dir:
        return forensic_maps(path, mode)
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(path))[0]
    fp = os.path.join(cache_dir, f"{mode}_{stem}_{key}.npy")
    if os.path.exists(fp):
        try:
            return np.load(fp)
        except Exception:
            pass
    arr = forensic_maps(path, mode)
    np.save(fp, arr)
    return arr


def pad_to_crop(arr: np.ndarray, crop: int = config.CROP) -> np.ndarray:
    h, w = arr.shape[:2]
    ph, pw = max(0, crop - h), max(0, crop - w)
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="constant")
    return arr


def center_crop(arr: np.ndarray, crop: int = config.CROP) -> np.ndarray:
    arr = pad_to_crop(arr, crop)
    h, w = arr.shape[:2]
    y, x = (h - crop) // 2, (w - crop) // 2
    return arr[y:y + crop, x:x + crop]


def random_crop(arr: np.ndarray, rng: np.random.Generator, crop: int = config.CROP) -> np.ndarray:
    arr = pad_to_crop(arr, crop)
    h, w = arr.shape[:2]
    y = int(rng.integers(0, h - crop + 1))
    x = int(rng.integers(0, w - crop + 1))
    return arr[y:y + crop, x:x + crop]


def grid_crops(arr: np.ndarray, max_crops: int = config.TTA_MAX_CROPS,
               crop: int = config.CROP) -> list[np.ndarray]:
    """Crops on a (up to 3x3) grid so the WHOLE image is covered, not just the centre."""
    arr = pad_to_crop(arr, crop)
    h, w = arr.shape[:2]
    side = int(np.ceil(np.sqrt(max_crops)))
    ny = min(side, int(np.ceil(h / crop)))
    nx = min(side, int(np.ceil(w / crop)))
    ys = np.unique(np.linspace(0, h - crop, ny).round().astype(int))
    xs = np.unique(np.linspace(0, w - crop, nx).round().astype(int))
    return [arr[y:y + crop, x:x + crop] for y in ys for x in xs][:max_crops]


def to_model_input(crop_u8: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 normalised like ResNet50 (BGR, mean-subtracted)."""
    x = crop_u8.astype(np.float32)[..., ::-1]
    return x - _BGR_MEAN
