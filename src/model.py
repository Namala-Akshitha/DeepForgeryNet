"""DeepForgeryNet: ResNet50 (ImageNet) -> 49-step sequence -> LSTM(256) -> Dense(128) -> Dropout(0.5) -> Sigmoid."""
from tensorflow.keras import Model, applications, layers

from config import IMG_SIZE


def build_model():
    base = applications.ResNet50(include_top=False, weights="imagenet",
                                 input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False  # stage 1: train only the head

    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = base(inp, training=False)              # (7, 7, 2048)
    x = layers.Reshape((49, 2048))(x)          # 49 spatial regions as a sequence
    x = layers.LSTM(256)(x)                    # contextual modelling
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)   # probability of FAKE
    return Model(inp, out, name="DeepForgeryNet"), base
