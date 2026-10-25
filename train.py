import argparse
import os

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam

from config import MODEL_PATH
from src.data import ImageSequence, list_split
from src.model import build_model


def callbacks():
    return [EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)]


def compile_model(model, lr):
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", AUC(name="auc")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--fine-tune-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    tr_p, tr_y = list_split(os.path.join(a.data_dir, "train"))
    va_p, va_y = list_split(os.path.join(a.data_dir, "val"))
    train = ImageSequence(tr_p, tr_y, a.batch_size, shuffle=True, augment=True, workers=a.workers)
    val = ImageSequence(va_p, va_y, a.batch_size, workers=a.workers)

    n_fake = int(np.sum(tr_y)); n_real = len(tr_y) - n_fake
    class_weight = {0: len(tr_y) / (2 * n_real), 1: len(tr_y) / (2 * n_fake)}
    print(f"train: {n_real} real / {n_fake} fake | class weights {class_weight}")

    model, base = build_model()

    print("\n== Stage 1: train LSTM + dense head (ResNet50 frozen) ==")
    compile_model(model, 1e-4)
    model.fit(train, validation_data=val, epochs=a.epochs,
              class_weight=class_weight, callbacks=callbacks())

    if a.fine_tune_epochs > 0:
        print("\n== Stage 2: fine-tune last 30 ResNet50 layers ==")
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        compile_model(model, 1e-5)
        model.fit(train, validation_data=val, epochs=a.fine_tune_epochs,
                  class_weight=class_weight, callbacks=callbacks())

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
