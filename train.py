"""Two-stage training: (1) frozen backbone, (2) fine-tune the top of ResNet50.

Example:
    python train.py --data_root /content/CASIA2 --out outputs
    python train.py --data_root /content/CASIA2 --out outputs --group_split
"""
import argparse
import os

import numpy as np
import tensorflow as tf

import config
import data
import model as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--mode", default=config.MODE, choices=["paper", "multi_ela"])
    ap.add_argument("--epochs_head", type=int, default=config.EPOCHS_HEAD)
    ap.add_argument("--epochs_ft", type=int, default=config.EPOCHS_FINETUNE)
    ap.add_argument("--unfreeze", type=int, default=config.UNFREEZE_LAST)
    ap.add_argument("--group_split", action="store_true",
                    help="keep tampered images and their source images in the same split")
    a = ap.parse_args()

    tf.keras.utils.set_random_seed(config.SEED)
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "cache")

    paths, labels = data.list_casia(a.data_root)
    print(f"Found {len(paths)} images ({(labels == 0).sum()} authentic, {(labels == 1).sum()} tampered)")
    splits = data.make_splits(paths, labels, group_split=a.group_split)
    splits_fp = os.path.join(a.out, "splits.json")
    data.save_splits(splits_fp, paths, labels, splits)
    sp = data.load_splits(splits_fp)

    tr = data.ForgerySequence(*sp["train"], a.mode, cache, train=True)
    va = data.ForgerySequence(*sp["val"], a.mode, cache, train=False)

    n0, n1 = (sp["train"][1] == 0).sum(), (sp["train"][1] == 1).sum()
    n = n0 + n1
    class_weight = {0: n / (2 * n0), 1: n / (2 * n1)}
    print("class weights:", class_weight)

    model_fp = os.path.join(a.out, "model.keras")
    callbacks = lambda: [
        tf.keras.callbacks.ModelCheckpoint(model_fp, monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6,
                                         restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(os.path.join(a.out, "training_log.csv"), append=True),
    ]

    model = M.build_model()
    print("== Stage 1: training head (ResNet50 frozen) ==")
    M.compile_model(model, config.LR_HEAD)
    model.fit(tr, validation_data=va, epochs=a.epochs_head, class_weight=class_weight,
              callbacks=callbacks())

    print(f"== Stage 2: fine-tuning last {a.unfreeze} ResNet50 layers ==")
    M.set_finetune(model, a.unfreeze)
    M.compile_model(model, config.LR_FINETUNE)
    model.fit(tr, validation_data=va, epochs=a.epochs_ft, class_weight=class_weight,
              callbacks=callbacks())

    model.save(model_fp)
    with open(os.path.join(a.out, "mode.txt"), "w") as f:
        f.write(a.mode)
    print("Saved", model_fp)


if __name__ == "__main__":
    main()
