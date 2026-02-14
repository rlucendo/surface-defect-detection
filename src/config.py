import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
IMAGE_SIZE = (224, 224) # Standard for ResNet
RANDOM_SEED = 42

# Device (Auto-detect CUDA)
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Classes (NEU dataset specific)
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']