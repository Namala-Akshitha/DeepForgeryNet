"""Print the saved DeepForgeryNet results (no model, dataset or TensorFlow needed)."""
import json
import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "results.json")
with open(path) as f:
    r = json.load(f)

cm = r["confusion_matrix"]
print("=" * 55)
print(" DeepForgeryNet - Test Results")
print("=" * 55)
print(f" Model    : {r['model']}")
print(f" Dataset  : {r['dataset']}")
print(f" Threshold: {r['threshold']}")
print("-" * 55)
print(f" Accuracy : {r['accuracy'] * 100:.1f}%")
print(f" Precision: {r['precision'] * 100:.1f}%")
print(f" Recall   : {r['recall'] * 100:.1f}%")
print(f" F1-score : {r['f1'] * 100:.1f}%")
print(f" AUC      : {r['auc']:.3f}")
print("-" * 55)
print(" Confusion matrix:")
print(f"   Authentic correct : {cm['true_authentic']}")
print(f"   Authentic flagged fake: {cm['false_fake']}")
print(f"   Tampered missed   : {cm['missed_tampered']}")
print(f"   Tampered correct  : {cm['true_tampered']}")
print("=" * 55)