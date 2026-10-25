"""Flask web app: upload an image -> REAL or FAKE."""
import os

from flask import Flask, jsonify, render_template, request
from PIL import UnidentifiedImageError

from config import MODEL_PATH, THRESHOLD
from src.preprocessing import preprocess

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not trained yet. Run `python train.py` first.")
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
    return _model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="Please choose an image."), 400
    if os.path.splitext(file.filename)[1].lower() not in ALLOWED:
        return jsonify(error="Unsupported file type. Upload JPG, PNG, BMP, WEBP or TIFF."), 400
    try:
        x = preprocess(file.stream)[None, ...]
        prob_fake = float(get_model().predict(x, verbose=0)[0][0])
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503
    except UnidentifiedImageError:
        return jsonify(error="That file is not a valid image."), 400
    return jsonify(label="FAKE" if prob_fake >= THRESHOLD else "REAL")


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is too large (max 10 MB)."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
