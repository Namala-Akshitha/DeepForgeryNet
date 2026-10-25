"""Artifact-aware preprocessing v3: ELA + Gaussian denoising + Sobel. No resizing; random crop + flip when augment=True."""
import io
import random

import cv2
import numpy as np
from PIL import Image, ImageChops

from config import ELA_QUALITY, GAUSSIAN_SIGMA, IMG_SIZE

_IMAGENET_BGR_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def ela_map(img, quality=ELA_QUALITY):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    diff = np.asarray(ImageChops.difference(img, resaved), dtype=np.float32)
    max_diff = diff.max()
    scale = 255.0 / max_diff if max_diff > 0 else 1.0
    return np.clip(diff * scale, 0, 255).mean(axis=2)


def _offsets(h, w, size, augment):
    if augment:
        return random.randint(0, max(h - size, 0)), random.randint(0, max(w - size, 0))
    return max((h - size) // 2, 0), max((w - size) // 2, 0)


def _crop(arr, top, left, size):
    arr = arr[top:top + size, left:left + size]
    ph, pw = size - arr.shape[0], size - arr.shape[1]
    if ph > 0 or pw > 0:
        arr = np.pad(arr, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)), mode="constant")
    return arr


def preprocess(source, augment=False):
    img = Image.open(source).convert("RGB")
    ela = ela_map(img)
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    denoised = cv2.GaussianBlur(gray, (0, 0), sigmaX=GAUSSIAN_SIGMA)
    gx = cv2.Sobel(denoised, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(denoised, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.sqrt(gx ** 2 + gy ** 2)
    sobel = sobel / sobel.max() * 255.0 if sobel.max() > 0 else sobel

    h, w = ela.shape
    top, left = _offsets(h, w, IMG_SIZE, augment)
    stacked = np.stack([_crop(m, top, left, IMG_SIZE) for m in (ela, denoised, sobel)],
                       axis=-1).astype(np.float32)
    if augment and random.random() < 0.5:
        stacked = stacked[:, ::-1, :]
    return (stacked[..., ::-1] - _IMAGENET_BGR_MEAN).astype(np.float32)
