"""
utils.py
Shared utilities: emotion label maps, reproducibility, config loading, logging helpers.
"""
import os
import json
import random
import yaml
import numpy as np

# ----------------------------------------------------------------------
# Canonical 8-class emotion label map used across the whole project.
# Individual dataset loaders map their native labels onto this scheme.
# ----------------------------------------------------------------------
EMOTION_LABELS = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised",
}
LABEL_TO_ID = {v: k for k, v in EMOTION_LABELS.items()}
NUM_CLASSES = len(EMOTION_LABELS)


def load_config(path="config.yaml"):
    """Load YAML config file into a plain dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed=42):
    """Make runs reproducible across numpy / random / torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_label_map(path):
    ensure_dir(os.path.dirname(path) if os.path.dirname(path) else ".")
    with open(path, "w") as f:
        json.dump(EMOTION_LABELS, f, indent=2)


def load_label_map(path):
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def get_device(preference="auto"):
    import torch
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
