"""Evaluate the trained model on a test folder and write REAL metrics to results/.

    python evaluate.py                           # data/test
    python evaluate.py --test-dir data_celebdf/test --name celebdf   # cross-dataset
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from tensorflow.keras.models import load_model

from config import MODEL_PATH, RESULTS_DIR, THRESHOLD
from src.data import ImageSequence, list_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="data/test")
    ap.add_argument("--name", default="test")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    paths, labels = list_split(a.test_dir)
    seq = ImageSequence(paths, labels, a.batch_size)
    model = load_model(MODEL_PATH)

    probs = model.predict(seq, verbose=1).ravel()
    y_true = np.array(labels)[: len(probs)]
    y_pred = (probs >= THRESHOLD).astype(int)

    m = {
        "dataset": a.name,
        "num_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, probs)),
        "confusion_matrix [[TN, FP], [FN, TP]]": confusion_matrix(y_true, y_pred).tolist(),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"metrics_{a.name}.json"), "w") as f:
        json.dump(m, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4)); plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks([0, 1], ["Real", "Fake"]); plt.yticks([0, 1], ["Real", "Fake"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{a.name}.png"), dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(4, 4)); plt.plot(fpr, tpr, label=f"AUC = {m['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray"); plt.xlabel("False positive rate")
    plt.ylabel("True positive rate"); plt.legend(); plt.title("ROC curve"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"roc_curve_{a.name}.png"), dpi=150); plt.close()

    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
