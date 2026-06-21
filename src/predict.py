"""
predict.py
Run inference on a single .wav file using a trained checkpoint.

Usage:
    python src/predict.py --file path/to/audio.wav --model_path outputs/best_model.pt
"""
import os
import sys
import argparse
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, get_device, EMOTION_LABELS
from src.models import build_model
from src.feature_extraction import load_audio, extract_features


def load_inference_artifacts(cfg, model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["model_type"], checkpoint["n_channels"], checkpoint["n_time"],
        checkpoint["num_classes"], checkpoint["model_cfg"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(cfg["paths"]["scaler_save_path"], "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, checkpoint["n_channels"]


def predict_emotion(filepath, model, scaler, n_channels, cfg, device):
    audio_cfg = cfg["audio"]
    feat_cfg = cfg["features"]

    waveform, sr = load_audio(
        filepath,
        sample_rate=audio_cfg["sample_rate"],
        duration_sec=audio_cfg["duration_sec"],
        trim_silence=audio_cfg["trim_silence"],
    )
    feat = extract_features(waveform, sr, feat_cfg)  # (C, T)

    # Normalize using the saved scaler (fit during training)
    t = feat.shape[1]
    feat_flat = feat.T  # (T, C)
    feat_norm = scaler.transform(feat_flat).T  # (C, T)

    X = torch.from_numpy(feat_norm[None, :, :]).float().to(device)  # (1, C, T)

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_label = EMOTION_LABELS[pred_id]

    prob_dict = {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(probs))}
    return pred_label, prob_dict


def main():
    parser = argparse.ArgumentParser(description="Predict emotion from a single audio file")
    parser.add_argument("--file", required=True, help="Path to a .wav (or other librosa-readable) audio file")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = args.model_path or cfg["paths"]["model_save_path"]

    if not os.path.exists(model_path):
        print(f"ERROR: model not found at '{model_path}'. Train a model first with src/train.py.")
        sys.exit(1)
    if not os.path.exists(args.file):
        print(f"ERROR: audio file not found at '{args.file}'.")
        sys.exit(1)

    device = get_device(cfg["training"]["device"])
    model, scaler, n_channels = load_inference_artifacts(cfg, model_path, device)

    pred_label, prob_dict = predict_emotion(args.file, model, scaler, n_channels, cfg, device)

    print(f"\nFile: {args.file}")
    print(f"Predicted emotion: {pred_label.upper()}\n")
    print("Class probabilities:")
    for label, p in sorted(prob_dict.items(), key=lambda x: -x[1]):
        bar = "#" * int(p * 40)
        print(f"  {label:>10s}: {p:.3f} {bar}")


if __name__ == "__main__":
    main()
