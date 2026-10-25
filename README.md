# DeepForgeryNet – Real vs Fake Image Detection (Web App)

A student implementation of **DeepForgeryNet** (Sardhara et al., 2026): a hybrid CNN–LSTM with transfer learning for image forgery detection, with a web page where you upload an image and get **Real** or **Fake**.

**Team:** N. Akshitha, Y. Manasa, B. Soumya, L. Navyasri, G. Harini

## How it works
Input image -> ELA + Gaussian denoising (sigma 0.5) + Sobel edges -> ImageNet-pretrained ResNet50 (7x7x2048) -> sequence of 49 regions -> LSTM (256) -> Dense (128) -> Dropout (0.5) -> Sigmoid -> Real / Fake

## Results (our own, CASIA V2.0 test set, 1,893 unseen images)

| Metric | Result |
|---|---|
| Accuracy | 81.0% |
| Precision | 72.1% |
| Recall | 86.9% |
| F1-score | 78.8% |
| AUC | 0.886 |

Dataset: CASIA V2.0 (7,491 authentic, 5,123 tampered), random 70/15/15 split. Trained on Google Colab (Tesla T4).
These results are lower than those reported in the original paper; we did not reproduce its full experimental protocol.

## Project structure
- app.py - Flask web app (upload page and /predict)
- train.py - training
- evaluate.py - computes accuracy, precision, recall, F1, AUC, confusion matrix
- split_dataset.py - 70/15/15 split helper
- config.py - settings
- src/ - preprocessing.py, model.py, data.py
- templates/, static/ - web page

## Setup (Python 3.10-3.12)
```bash
python -m venv venv
venv\Scripts\activate          # Windows (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## 1. Prepare data
Download CASIA V2.0 and put the images in `data/raw/real` (Au folder) and `data/raw/fake` (Tp folder), then run:
```bash
python split_dataset.py
```

## 2. Train
```bash
python train.py --epochs 10 --fine-tune-epochs 0 --batch-size 32 --workers 2
```
This creates `models/deepforgerynet.keras`. A GPU (for example Google Colab) is strongly recommended. The trained model is not included in this repository (file size).

## 3. Evaluate
```bash
python evaluate.py
```
Metrics are saved in `results/`.

## 4. Run the web app
```bash
python app.py
```
Open http://127.0.0.1:5000, upload an image and click "Check image".

## Limitations
- Trained and tested only on CASIA V2.0; no cross-dataset or deepfake evaluation yet.
- No baseline (CNN-only / LSTM-only) or ablation experiments yet.
- Random image-level split; related images may appear in both train and test.
- The model can be wrong, especially on image types not seen in training.

## Reference
Sardhara, A., Vekariya, V., Pathak, A. R., & Dash, S. (2026). "DeepForgeryNet: A hybrid CNN-LSTM and transfer learning framework for robust image forgery and deepfake detection." Frontiers in Artificial Intelligence.
