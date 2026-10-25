"""Split data/raw/{real,fake} into data/{train,val,test}/{real,fake} (70/15/15).

NOTE: for video datasets (FaceForensics++, Celeb-DF, DFDC) split by VIDEO before extracting
frames, otherwise frames of the same video land in train and test and the metrics are inflated.
"""
import argparse
import os
import random
import shutil

from src.data import EXTS

p = argparse.ArgumentParser()
p.add_argument("--raw-dir", default="data/raw")
p.add_argument("--out-dir", default="data")
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

random.seed(a.seed)
for cls in ("real", "fake"):
    src = os.path.join(a.raw_dir, cls)
    files = sorted(f for f in os.listdir(src) if f.lower().endswith(EXTS))
    random.shuffle(files)
    n = len(files)
    cuts = {"train": files[: int(0.70 * n)],
            "val": files[int(0.70 * n): int(0.85 * n)],
            "test": files[int(0.85 * n):]}
    for split, names in cuts.items():
        dst = os.path.join(a.out_dir, split, cls)
        os.makedirs(dst, exist_ok=True)
        for f in names:
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    print(cls, {k: len(v) for k, v in cuts.items()})
