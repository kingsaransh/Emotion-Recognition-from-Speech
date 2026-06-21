"""
dataset_loader.py
Parses the three supported datasets (RAVDESS, TESS, EMO-DB) and returns a unified
list of (filepath, emotion_label_str) pairs using the canonical 8-class scheme
defined in utils.EMOTION_LABELS.

Each dataset encodes emotion differently in its filenames -- this module is the
single place that knows those conventions, so the rest of the pipeline never
has to care which dataset the audio came from.
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import LABEL_TO_ID


# ----------------------------------------------------------------------
# RAVDESS
# Filename pattern: 03-01-06-01-02-01-12.wav
# Position 3 (index 2) = emotion code:
# 01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust 08=surprised
# ----------------------------------------------------------------------
RAVDESS_CODE_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}


def parse_ravdess(data_dir):
    pairs = []
    files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    for fp in files:
        fname = os.path.basename(fp)
        parts = fname.replace(".wav", "").split("-")
        if len(parts) < 3:
            continue
        code = parts[2]
        label = RAVDESS_CODE_MAP.get(code)
        if label:
            pairs.append((fp, label))
    return pairs


# ----------------------------------------------------------------------
# TESS
# Filename pattern: OAF_back_angry.wav / YAF_date_happy.wav
# Last underscore-separated token (before extension) = emotion word.
# TESS uses "ps" for pleasant surprise -> map to "surprised"; "fear" -> "fearful"
# ----------------------------------------------------------------------
TESS_WORD_MAP = {
    "neutral": "neutral", "calm": "calm", "happy": "happy", "sad": "sad",
    "angry": "angry", "fear": "fearful", "fearful": "fearful",
    "disgust": "disgust", "ps": "surprised", "surprise": "surprised", "surprised": "surprised",
}


def parse_tess(data_dir):
    pairs = []
    files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    for fp in files:
        fname = os.path.basename(fp).replace(".wav", "").lower()
        token = fname.split("_")[-1]
        label = TESS_WORD_MAP.get(token)
        if label:
            pairs.append((fp, label))
    return pairs


# ----------------------------------------------------------------------
# EMO-DB (Berlin Database of Emotional Speech)
# Filename pattern: 03a01Fa.wav -> 6th character = emotion code (German)
# W=anger(Wut) L=boredom(neutral-ish->we map to calm) E=disgust(Ekel)
# A=fear(Angst) F=happy(Freude) T=sad(Trauer) N=neutral
# ----------------------------------------------------------------------
EMODB_CODE_MAP = {
    "W": "angry", "L": "calm", "E": "disgust", "A": "fearful",
    "F": "happy", "T": "sad", "N": "neutral",
}


def parse_emodb(data_dir):
    pairs = []
    files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    for fp in files:
        fname = os.path.basename(fp)
        if len(fname) < 6:
            continue
        code = fname[5]  # 6th character, 0-indexed position 5
        label = EMODB_CODE_MAP.get(code)
        if label:
            pairs.append((fp, label))
    return pairs


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------
DATASET_PARSERS = {
    "ravdess": parse_ravdess,
    "tess": parse_tess,
    "emodb": parse_emodb,
}


def load_dataset(name, data_dir):
    """
    Returns list of (filepath, label_str, label_id) tuples for the given dataset.
    """
    name = name.lower()
    if name not in DATASET_PARSERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASET_PARSERS)}")
    pairs = DATASET_PARSERS[name](data_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No labeled .wav files found in '{data_dir}' for dataset '{name}'. "
            f"Check that the dataset was downloaded and unzipped there, preserving original filenames."
        )
    result = [(fp, label, LABEL_TO_ID[label]) for fp, label in pairs]
    return result


def dataset_summary(triples):
    """Print a quick class-distribution summary, useful right after loading."""
    from collections import Counter
    counts = Counter(label for _, label, _ in triples)
    print(f"Total samples: {len(triples)}")
    for label, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label:>10s}: {n}")
    return counts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick test of dataset loader")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_PARSERS))
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    triples = load_dataset(args.dataset, args.data_dir)
    dataset_summary(triples)
