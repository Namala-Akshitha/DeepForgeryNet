import os
import random

import numpy as np
from keras.utils import PyDataset

from config import LABELS
from src.preprocessing import preprocess

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_split(split_dir):
    paths, labels = [], []
    for cls, lab in LABELS.items():
        folder = os.path.join(split_dir, cls)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing folder: {folder}")
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(EXTS):
                paths.append(os.path.join(folder, name))
                labels.append(lab)
    if not paths:
        raise RuntimeError(f"No images found in {split_dir}")
    return paths, labels


class ImageSequence(PyDataset):
    def __init__(self, paths, labels, batch_size=16, shuffle=False, augment=False, workers=4, **kwargs):
        super().__init__(workers=workers, use_multiprocessing=False, max_queue_size=8, **kwargs)
        self.items = list(zip(paths, labels))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        if shuffle:
            random.shuffle(self.items)

    def __len__(self):
        return int(np.ceil(len(self.items) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.items[idx * self.batch_size:(idx + 1) * self.batch_size]
        xs, ys = [], []
        for path, lab in batch:
            try:
                xs.append(preprocess(path, augment=self.augment))
                ys.append(lab)
            except Exception as e:
                print(f"[warn] skipped {path}: {e}")
        if not xs:
            return self.__getitem__((idx + 1) % len(self))
        return np.stack(xs), np.array(ys, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.items)
