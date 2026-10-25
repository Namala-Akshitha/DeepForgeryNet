import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 224
ELA_QUALITY = 90
GAUSSIAN_SIGMA = 0.5
MODEL_PATH = os.path.join(BASE_DIR, "models", "deepforgerynet.keras")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
THRESHOLD = 0.5          # probability >= threshold -> FAKE
LABELS = {"real": 0, "fake": 1}
