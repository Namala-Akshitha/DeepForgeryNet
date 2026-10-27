"""Central configuration for DeepForgeryNet (v2)."""

SEED = 42
CROP = 224                 # network input size
BATCH_SIZE = 32

# Forensic preprocessing
#   "paper"     : [ELA(q90), Gaussian-denoised gray, Sobel edges]  (base paper)
#   "multi_ela" : [ELA(q95), ELA(q90), ELA(q75)]  (multi-quality ELA, usually
#                 stronger on CASIA because it keeps three views of the
#                 compression-error signal instead of one)
MODE = "multi_ela"
ELA_QUALITY = 90
ELA_QUALITIES = (95, 90, 75)

# Model
LSTM_UNITS = 256
DENSE_UNITS = 128
DROPOUT = 0.5
BIDIRECTIONAL = True

# Training schedule
LR_HEAD = 1e-4
LR_FINETUNE = 1e-5
EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 25
UNFREEZE_LAST = 60         # number of ResNet50 layers to unfreeze (BatchNorm stays frozen)
LABEL_SMOOTHING = 0.05

# Test-time augmentation
TTA_MAX_CROPS = 9          # up to 3x3 grid of crops covering the full image
TTA_FLIPS = True
