# DeepForgeryNet – Real vs Fake Image Detection (Web App)

A student implementation of **DeepForgeryNet** (Sardhara et al., 2026): a hybrid CNN–LSTM with transfer learning for image forgery detection, with a web page where you upload an image and get **Real** or **Fake**.

**Team:** N. Akshitha, Y. Manasa, B. Soumya, L. Navyasri, G. Harini


## See the accuracy (quick start)

After cloning, run this first. It needs no model, dataset or TensorFlow:

    python show_results.py

It prints the saved test results (accuracy 94.8%, precision 91.2%, recall 96.5%, F1 93.7%, AUC 0.989). To reproduce them, follow the Train and Evaluate steps below.

## How it works
Input image -> forensic maps (multi-quality ELA at JPEG quality 95/90/75, computed at full resolution) -> ImageNet-pretrained ResNet50 (7x7x2048, last 60 layers fine-tuned) -> sequence of 49 regions -> LayerNorm -> Bidirectional LSTM (256) -> Dense (128) -> Dropout (0.5) -> Sigmoid -> Real / Fake

At test time the probability is averaged over up to 9 crops (plus flips) covering the whole image, and the decision threshold is tuned on the validation set.

## Results (our own, CASIA V2.0 test set, 1,893 unseen images)

| Run | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Run 1 (resized maps, fine-tuned) | 74.5% | 67.7% | 71.4% | 69.5% | 0.805 |
| Run 2 (full-resolution maps, centre crop) | 77.4% | 70.9% | 75.3% | 73.0% | 0.852 |
| Run 3 (+ random-crop/flip augmentation) | 81.0% | 72.1% | 86.9% | 78.8% | 0.886 |
| **Run 4 (this version, threshold 0.58)** | **94.8%** | **91.2%** | **96.5%** | **93.7%** | **0.989** |

Run 4 at the default threshold 0.5: accuracy 94.4%, precision 89.4%, recall 97.8%, F1 93.4%, AUC 0.989.
Run 4 confusion matrix (tuned threshold): 1,052 authentic and 742 tampered images correct; 72 authentic flagged as fake; 27 tampered missed.

Dataset: CASIA V2.0 (7,491 authentic, 5,123 tampered), random 70/15/15 split (seed 42). Trained on Google Colab (Tesla T4).

**Important:** these numbers use a random image-level split. CASIA tampered images are derived from authentic ones, so related images can appear in both train and test, which can make the results optimistic. Use `--group_split` (below) for a leakage-reduced evaluation. We changed several things at once between Run 3 and Run 4 and did not run ablations, so we cannot say which change helped most. Results are from a single run per setting.

## Project structure
- app.py - Flask web app (upload page and /predict)
- train.py - two-stage training (head, then fine-tuning)
- evaluate.py - metrics, threshold tuning on validation, ROC / confusion-matrix plots
- infer.py - multi-crop test-time augmentation
- preprocessing.py - ELA / forensic maps and cropping
- data.py - dataset listing, splits, batch generator
- model.py - ResNet50 + BiLSTM model
- config.py - settings
- tests/ - preprocessing tests
- templates/, static/ - web page

## Setup (Python 3.10-3.12)
```bash
python -m venv venv
venv\Scripts\activate          # Windows (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## 1. Data
Download CASIA V2.0. You need the folder that contains `Au/` (authentic) and `Tp/` (tampered). It is passed to `train.py` as `--data_root`; no manual splitting is needed.

## 2. Train (GPU strongly recommended, e.g. Google Colab T4)
```bash
python train.py --data_root /path/to/CASIA2 --out outputs
# leakage-reduced split (tampered image and its source stay together):
python train.py --data_root /path/to/CASIA2 --out outputs_group --group_split
```
Options: `--mode paper` (original ELA + Gaussian + Sobel input) or `--mode multi_ela` (default), `--epochs_head`, `--epochs_ft`, `--unfreeze`.
This creates `outputs/model.keras`, `splits.json`, `mode.txt` and `training_log.csv`. The model file is not in this repository (file size).

## 3. Evaluate
```bash
python evaluate.py --out outputs
```
Writes `results.json`, `threshold.json`, `roc_curve.png`, `confusion_matrix.png` in `outputs/`. The threshold is tuned on the validation set only.

## 4. Run the web app
On first run the app downloads the trained model automatically from the GitHub Release (about 100+ MB). 
Put `model.keras`, `threshold.json` and `mode.txt` in `outputs/`, then:
```bash  
python app.py --out outputs
```
Open http://127.0.0.1:5000, upload an image and click "Check image".

## Tests
```bash
pip install pytest && pytest tests
```

## Limitations
- Trained and tested only on CASIA V2.0; no cross-dataset or deepfake evaluation yet.
- No baseline (CNN-only / LSTM-only) or ablation experiments yet.
- The reported test numbers use a random image-level split; related images may appear in both train and test.
- ELA depends on JPEG compression, so PNG, heavily compressed or screenshot images can be handled poorly. The model can be wrong on image types not seen in training.
- Single training run per setting; no estimate of variation between runs.

## Reference
Sardhara, A., Vekariya, V., Pathak, A. R., & Dash, S. (2026). "DeepForgeryNet: A hybrid CNN-LSTM and transfer learning framework for robust image forgery and deepfake detection." Frontiers in Artificial Intelligence.
