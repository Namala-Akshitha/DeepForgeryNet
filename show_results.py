"""Print the saved DeepForgeryNet test results (no model, dataset or TensorFlow needed).
The numbers come from results/results.json, written by evaluate.py."""
import json
import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "results.json")
with open(path) as f:
    r = json.load(f)

print("=" * 62)
print(" DeepForgeryNet - Test Results (CASIA V2.0)")
print("=" * 62)
print(f" Test images: {r['n_test']}   Input mode: {r['mode']}   Multi-crop: {r['tta']}")
print("-" * 62)
print(f" {'Setting':<26}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}{'AUC':>7}")
for key, name in (("at_0.5", "Threshold 0.50"), ("at_tuned_threshold", None)):
    m = r[key]
    label = name or f"Tuned threshold {m['threshold']:.2f}"
    print(f" {label:<26}{m['accuracy'] * 100:>6.1f}%{m['precision'] * 100:>6.1f}%"
          f"{m['recall'] * 100:>6.1f}%{m['f1'] * 100:>6.1f}%{m['auc']:>7.3f}")
(tn, fp), (fn, tp) = r["at_tuned_threshold"]["confusion_matrix"]
print("-" * 62)
print(" Confusion matrix (tuned threshold):")
print(f"   Authentic correct: {tn}   Authentic flagged fake: {fp}")
print(f"   Tampered missed  : {fn}   Tampered correct      : {tp}")
print("=" * 62)