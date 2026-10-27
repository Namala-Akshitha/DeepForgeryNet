"""Evaluate on the held-out test set. The decision threshold is tuned on the
VALIDATION set only (never on test), then applied to test.

    python evaluate.py --out outputs
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

import config
import data
import infer


def metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": accuracy_score(y, yh),
        "precision": precision_score(y, yh, zero_division=0),
        "recall": recall_score(y, yh),
        "f1": f1_score(y, yh),
        "auc": roc_auc_score(y, p),
        "confusion_matrix": confusion_matrix(y, yh).tolist(),
    }


def tune_threshold(y, p, criterion):
    best, best_t = -1, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        yh = (p >= t).astype(int)
        if criterion == "f1":
            s = f1_score(y, yh)
        elif criterion == "accuracy":
            s = accuracy_score(y, yh)
        else:  # youden = balanced accuracy
            s = recall_score(y, yh) + recall_score(1 - y, 1 - yh) - 1
        if s > best:
            best, best_t = s, t
    return float(best_t)


def score(model, paths, mode, cache, tta):
    out = []
    for i, p in enumerate(paths):
        kw = {} if tta else {"max_crops": 1, "flips": False}
        out.append(infer.predict_path(model, p, mode, cache, **kw))
        if (i + 1) % 200 == 0:
            print(f"  scored {i + 1}/{len(paths)}")
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--criterion", default="youden", choices=["youden", "f1", "accuracy"])
    ap.add_argument("--no_tta", action="store_true")
    a = ap.parse_args()

    mode = open(os.path.join(a.out, "mode.txt")).read().strip()
    cache = os.path.join(a.out, "cache")
    model = tf.keras.models.load_model(os.path.join(a.out, "model.keras"))
    sp = data.load_splits(os.path.join(a.out, "splits.json"))

    print("Scoring validation set ...")
    pv = score(model, sp["val"][0], mode, cache, not a.no_tta)
    thr = tune_threshold(sp["val"][1], pv, a.criterion)
    print(f"Threshold tuned on validation ({a.criterion}): {thr:.2f}")

    print("Scoring test set ...")
    yt = sp["test"][1]
    pt = score(model, sp["test"][0], mode, cache, not a.no_tta)

    res = {"at_0.5": metrics(yt, pt, 0.5), "at_tuned_threshold": metrics(yt, pt, thr),
           "tta": not a.no_tta, "mode": mode, "n_test": int(len(yt))}
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(a.out, "threshold.json"), "w") as f:
        json.dump({"threshold": thr, "mode": mode}, f)

    print("\n| Setting | Acc | Prec | Rec | F1 | AUC |\n|---|---|---|---|---|---|")
    for k in ("at_0.5", "at_tuned_threshold"):
        m = res[k]
        print(f"| {k} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
              f"{m['f1']:.3f} | {m['auc']:.3f} |")
    print("confusion matrix (tuned):", res["at_tuned_threshold"]["confusion_matrix"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(yt, pt)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"ROC (AUC={res['at_0.5']['auc']:.3f})")
    plt.savefig(os.path.join(a.out, "roc_curve.png"), dpi=150); plt.close()
    cm = np.array(res["at_tuned_threshold"]["confusion_matrix"])
    plt.figure(); plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, v, ha="center", va="center")
    plt.xticks([0, 1], ["Real", "Fake"]); plt.yticks([0, 1], ["Real", "Fake"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.savefig(os.path.join(a.out, "confusion_matrix.png"), dpi=150); plt.close()


if __name__ == "__main__":
    main()
