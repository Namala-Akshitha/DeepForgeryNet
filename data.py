"""Dataset listing, splitting and the Keras batch generator."""
import json
import os
import re

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import config
import preprocessing as pp

_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_ID = re.compile(r"(ani|art|cha|ind|nat|pla|sec|txt)_?(\d{5})", re.I)


def list_casia(root: str):
    """Find CASIA V2.0 (folders Au/ and Tp/) under `root`. Returns paths, labels (Tp=1)."""
    for base in (root, os.path.join(root, "CASIA2"), os.path.join(root, "CASIA 2.0"),
                 os.path.join(root, "CASIA2.0_revised")):
        au, tp = os.path.join(base, "Au"), os.path.join(base, "Tp")
        if os.path.isdir(au) and os.path.isdir(tp):
            break
    else:
        raise FileNotFoundError(f"Could not find Au/ and Tp/ folders under {root}")
    paths, labels = [], []
    for folder, lab in ((au, 0), (tp, 1)):
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(_EXT):
                paths.append(os.path.join(folder, f))
                labels.append(lab)
    return np.array(paths), np.array(labels)


def group_key(path: str) -> str:
    """Source-image id so a tampered image and its authentic original stay together."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _ID.search(stem)
    return (m.group(1) + m.group(2)).lower() if m else stem


def make_splits(paths, labels, group_split=False, seed=config.SEED):
    """70/15/15 split. group_split=True prevents leakage between related images."""
    idx = np.arange(len(paths))
    if group_split:
        groups = np.array([group_key(p) for p in paths])
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        tr, rest = next(gss.split(idx, labels, groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        va_r, te_r = next(gss2.split(rest, labels[rest], groups[rest]))
        va, te = rest[va_r], rest[te_r]
    else:
        tr, rest = train_test_split(idx, test_size=0.30, stratify=labels, random_state=seed)
        va, te = train_test_split(rest, test_size=0.50, stratify=labels[rest], random_state=seed)
    return {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}


def save_splits(path, paths, labels, splits):
    out = {k: {"paths": [str(paths[i]) for i in v], "labels": [int(labels[i]) for i in v]}
           for k, v in splits.items()}
    with open(path, "w") as f:
        json.dump(out, f)


def load_splits(path):
    with open(path) as f:
        d = json.load(f)
    return {k: (np.array(v["paths"]), np.array(v["labels"])) for k, v in d.items()}


class ForgerySequence(tf.keras.utils.Sequence):
    """Yields (batch, labels). Train: random crop + flips. Eval: centre crop."""

    def __init__(self, paths, labels, mode, cache_dir, train, batch_size=config.BATCH_SIZE,
                 seed=config.SEED):
        try:
            super().__init__(workers=2, use_multiprocessing=False, max_queue_size=8)
        except TypeError:  # tf.keras 2.x Sequence takes no arguments
            super().__init__()
        self.paths, self.labels = np.asarray(paths), np.asarray(labels)
        self.mode, self.cache_dir, self.train = mode, cache_dir, train
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.paths))
        if train:
            self.rng.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, i):
        ids = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        xs = []
        for j in ids:
            arr = pp.cached_maps(self.paths[j], self.mode, self.cache_dir)
            if self.train:
                c = pp.random_crop(arr, self.rng)
                if self.rng.random() < 0.5:
                    c = c[:, ::-1]
                if self.rng.random() < 0.5:
                    c = c[::-1, :]
            else:
                c = pp.center_crop(arr)
            xs.append(pp.to_model_input(np.ascontiguousarray(c)))
        return np.stack(xs), self.labels[ids].astype("float32")

    def on_epoch_end(self):
        if self.train:
            self.rng.shuffle(self.order)
