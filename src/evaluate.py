"""
evaluate.py
Evaluates a trained model checkpoint on the held-out test split saved by train.py.
Prints accuracy, macro-F1, full classification report, and saves a confusion matrix plot.

Usage:
    python src/evaluate.py --model_path outputs/best_model.pt
"""
import os
import sys
import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, get_device, EMOTION_LABELS
from src.models import build_model
from src.train import FeatureDataset
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SER model on the test split")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--test_split_path", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = args.model_path or cfg["paths"]["model_save_path"]
    test_split_path = args.test_split_path or os.path.join(cfg["paths"]["output_dir"], "test_split.npz")

    if not os.path.exists(model_path):
        print(f"ERROR: model checkpoint not found at '{model_path}'. Train a model first with src/train.py.")
        sys.exit(1)
    if not os.path.exists(test_split_path):
        print(f"ERROR: test split not found at '{test_split_path}'. It is created automatically by src/train.py.")
        sys.exit(1)

    device = get_device(cfg["training"]["device"])

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["model_type"], checkpoint["n_channels"], checkpoint["n_time"],
        checkpoint["num_classes"], checkpoint["model_cfg"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model '{checkpoint['model_type']}' (val_acc during training: {checkpoint['val_acc']:.4f})")

    test_data = np.load(test_split_path)
    X_test, y_test = test_data["X"], test_data["y"]
    test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=64, shuffle=False)

    all_preds, all_true, all_probs = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    acc = accuracy_score(all_true, all_preds)
    macro_f1 = f1_score(all_true, all_preds, average="macro")

    present_labels = sorted(set(all_true.tolist()) | set(all_preds.tolist()))
    target_names = [EMOTION_LABELS[i] for i in present_labels]

    print(f"\n{'='*50}")
    print(f"TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(all_true, all_preds, labels=present_labels, target_names=target_names, zero_division=0))

    # ---- Confusion matrix plot ----
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(all_true, all_preds, labels=present_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix (Accuracy={acc:.3f})")
        plt.tight_layout()
        out_path = os.path.join(cfg["paths"]["output_dir"], "confusion_matrix.png")
        plt.savefig(out_path, dpi=120)
        print(f"\nSaved confusion matrix: {out_path}")
    except ImportError:
        print("\n(matplotlib/seaborn not available - skipping confusion matrix plot)")


if __name__ == "__main__":
    main()
