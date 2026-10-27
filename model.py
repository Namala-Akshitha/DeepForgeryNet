"""ResNet50 -> 49-step sequence -> (Bi)LSTM -> Dense -> sigmoid."""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

import config

BACKBONE_NAME = "resnet50_backbone"


def build_model(crop=config.CROP, lstm_units=config.LSTM_UNITS, dense_units=config.DENSE_UNITS,
                dropout=config.DROPOUT, bidirectional=config.BIDIRECTIONAL):
    base = ResNet50(include_top=False, weights="imagenet", input_shape=(crop, crop, 3),
                    name=BACKBONE_NAME)
    base.trainable = False
    inp = layers.Input((crop, crop, 3))
    x = base(inp, training=False)                       # 7 x 7 x 2048
    x = layers.Reshape((-1, 2048))(x)                   # 49 spatial tokens
    x = layers.LayerNormalization()(x)
    lstm = layers.LSTM(lstm_units)
    x = layers.Bidirectional(lstm)(x) if bidirectional else lstm(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inp, out, name="DeepForgeryNet")


def set_finetune(model, n_layers=config.UNFREEZE_LAST):
    """Unfreeze the last `n_layers` of ResNet50. BatchNorm layers stay frozen."""
    base = model.get_layer(BACKBONE_NAME)
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
