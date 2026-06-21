"""
feature_extraction.py
Extracts speech features (MFCC + delta + delta-delta, chroma, mel-spectrogram,
zero-crossing rate, RMS energy) from raw audio for emotion recognition.

Can be run standalone to pre-compute and cache features for an entire dataset:
    python src/feature_extraction.py --dataset ravdess --data_dir data/RAVDESS

This writes data/processed/<dataset>_features.npz containing:
    X: (N, n_features, max_pad_len) float32 array
    y: (N,) int labels
    label_names: (N,) str labels
"""
import os
import sys
import argparse
import numpy as np
import librosa
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, ensure_dir
from src.dataset_loader import load_dataset, dataset_summary


def load_audio(filepath, sample_rate=22050, duration_sec=3.5, trim_silence=True):
    """Load an audio file, resample, trim silence, and pad/truncate to fixed duration."""
    y, sr = librosa.load(filepath, sr=sample_rate, mono=True)

    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=25)

    target_len = int(sample_rate * duration_sec)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, max(0, target_len - len(y))), mode="constant")

    return y, sr


def extract_features(y, sr, cfg_features):
    """
    Extract a stacked feature matrix from a single audio waveform.
    Returns array of shape (n_feature_channels, n_time_frames).
    """
    n_mfcc = cfg_features["n_mfcc"]
    n_fft = cfg_features["n_fft"]
    hop_length = cfg_features["hop_length"]
    n_mels = cfg_features["n_mels"]
    max_pad_len = cfg_features["max_pad_len"]

    feats = []

    # --- MFCCs ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    feats.append(mfcc)

    if cfg_features.get("use_delta", True):
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feats.append(delta)
        feats.append(delta2)

    # --- Chroma (pitch class energy, related to prosody) ---
    if cfg_features.get("use_chroma", True):
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        feats.append(chroma)

    # --- Mel-spectrogram (log scale) ---
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Downscale mel bands to keep total feature dim reasonable (avg-pool every 4 bands)
    mel_db_reduced = mel_db.reshape(n_mels // 4, 4, -1).mean(axis=1)
    feats.append(mel_db_reduced)

    # --- Zero-crossing rate & RMS energy (voice intensity/sharpness) ---
    if cfg_features.get("use_zcr_rms", True):
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
        feats.append(zcr)
        feats.append(rms)

    # Align all feature arrays to the same number of time frames (min across stack)
    min_frames = min(f.shape[1] for f in feats)
    feats = [f[:, :min_frames] for f in feats]
    stacked = np.vstack(feats)  # (total_channels, time_frames)

    # Pad or truncate along time axis to fixed length
    if stacked.shape[1] < max_pad_len:
        pad_width = max_pad_len - stacked.shape[1]
        stacked = np.pad(stacked, ((0, 0), (0, pad_width)), mode="constant")
    else:
        stacked = stacked[:, :max_pad_len]

    return stacked.astype(np.float32)


def build_feature_dataset(triples, cfg, augment_fn=None):
    """
    triples: list of (filepath, label_str, label_id)
    Returns X (N, C, T), y (N,), label_names (N,)
    """
    audio_cfg = cfg["audio"]
    feat_cfg = cfg["features"]

    X, y, label_names = [], [], []
    for filepath, label_str, label_id in tqdm(triples, desc="Extracting features"):
        try:
            waveform, sr = load_audio(
                filepath,
                sample_rate=audio_cfg["sample_rate"],
                duration_sec=audio_cfg["duration_sec"],
                trim_silence=audio_cfg["trim_silence"],
            )
        except Exception as e:
            print(f"  [WARN] Skipping {filepath}: {e}")
            continue

        waveforms_to_process = [waveform]
        if augment_fn is not None:
            waveforms_to_process.append(augment_fn(waveform, sr))

        for wf in waveforms_to_process:
            feat = extract_features(wf, sr, feat_cfg)
            X.append(feat)
            y.append(label_id)
            label_names.append(label_str)

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    label_names = np.array(label_names)
    return X, y, label_names


def main():
    parser = argparse.ArgumentParser(description="Pre-extract and cache features for a dataset")
    parser.add_argument("--dataset", required=True, choices=["ravdess", "tess", "emodb"])
    parser.add_argument("--data_dir", required=True, help="Path to raw dataset directory")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--augment", action="store_true", help="Also generate augmented copies")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = ensure_dir(cfg["dataset"]["processed_dir"])

    print(f"Loading dataset '{args.dataset}' from '{args.data_dir}' ...")
    triples = load_dataset(args.dataset, args.data_dir)
    dataset_summary(triples)

    augment_fn = None
    if args.augment:
        from src.augmentation import augment_waveform
        augment_fn = augment_waveform
        print("Augmentation ENABLED: dataset size will roughly double.")

    print("Extracting features (this may take a few minutes)...")
    X, y, label_names = build_feature_dataset(triples, cfg, augment_fn=augment_fn)

    out_path = os.path.join(processed_dir, f"{args.dataset}_features.npz")
    np.savez_compressed(out_path, X=X, y=y, label_names=label_names)
    print(f"\nSaved features: {out_path}")
    print(f"Feature tensor shape: {X.shape}  (N samples, C channels, T time-frames)")


if __name__ == "__main__":
    main()
