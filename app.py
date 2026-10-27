"""Flask web app: upload an image -> REAL or FAKE.

    python app.py              # expects outputs/model.keras, threshold.json, mode.txt
    python app.py --out outputs --port 5000
"""
import argparse
import json
import os
import tempfile

from flask import Flask, jsonify, render_template, request

import infer

ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
STATE = {"out": "outputs", "model": None, "threshold": 0.5, "mode": "multi_ela"}

import urllib.request

RELEASE_URL = "https://github.com/Namala-Akshitha/DeepForgeryNet/releases/download/v2/"


def ensure_files(out):
    os.makedirs(out, exist_ok=True)
    for name in ("model.keras", "threshold.json", "mode.txt"):
        fp = os.path.join(out, name)
        if not os.path.exists(fp):
            print(f"Downloading {name} ...")
            try:
                urllib.request.urlretrieve(RELEASE_URL + name, fp)
            except Exception as e:
                print("Download failed:", e)


def get_model():
    if STATE["model"] is None:
        out = STATE["out"]
        ensure_files(out)
        fp = os.path.join(out, "model.keras")
        if not os.path.exists(fp):
            raise FileNotFoundError(
                "Model not found. Train it (python train.py) or put model.keras, "
                f"threshold.json and mode.txt in '{out}/'.")
        import tensorflow as tf
        STATE["model"] = tf.keras.models.load_model(fp)
        meta = os.path.join(out, "threshold.json")
        if os.path.exists(meta):
            with open(meta) as f:
                m = json.load(f)
            STATE["threshold"], STATE["mode"] = m["threshold"], m["mode"]
    return STATE["model"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="Please choose an image."), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify(error="Unsupported file type. Upload JPG, PNG, BMP, WEBP or TIFF."), 400
    try:
        model = get_model()
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "upload" + ext)
            file.save(fp)
            prob = infer.predict_path(model, fp, STATE["mode"])
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503
    except Exception:
        return jsonify(error="That file is not a valid image."), 400
    return jsonify(label="FAKE" if prob >= STATE["threshold"] else "REAL")


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is too large (max 10 MB)."), 413


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    STATE["out"] = a.out
    app.run(host="0.0.0.0", port=a.port, debug=False)
