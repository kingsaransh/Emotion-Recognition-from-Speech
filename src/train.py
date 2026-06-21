"""
train.py
Trains a speech emotion recognition model end-to-end.

Usage:
    # Standard run (assumes features already extracted via feature_extraction.py)
    python src/train.py --model cnn_lstm --epochs 50 --batch_size 32

    # Quick smoke-test with synthetic data (no dataset download needed)
    python src/train.py --synthetic --model cnn --epochs 5

    # Use a different dataset / pre-extracted feature cache
    python src/train.py --dataset tess --model lstm
"""
import os
import sys
import json
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_seed, ensure_dir, get_device, save_label_map, EMOTION_LABELS, NUM_CLASSES
from src.models import build_model


# ----------------------------------------------------------------------
# Synthetic dataset generator (for pipeline smoke-testing without real audio)
# ----------------------------------------------------------------------
def make_synthetic_dataset(n_samples=400, n_channels=85, n_time=174, num_classes=8, seed=42):
    rng = np.random.RandomState(seed)
    X = np.zeros((n_samples, n_channels, n_time), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        label = i % num_classes
        # Give each class a distinct mean/frequency signature so the model can
        # actually learn something nontrivial, validating the full pipeline.
        base = rng.randn(n_channels, n_time) * 0.5
        base += np.sin(np.linspace(0, (label + 1) * np.pi, n_time))[None, :] * (label + 1) * 0.3
        X[i] = base
        y[i] = label
    label_names = np.array([EMOTION_LABELS[l] for l in y])
    return X, y, label_names


# ----------------------------------------------------------------------
# PyTorch Dataset wrapper
# ----------------------------------------------------------------------
class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def normalize_features(X_train, X_val, X_test):
    """Z-score normalize per feature channel, fit only on train split."""
    n_channels = X_train.shape[1]
    scaler = StandardScaler()

    def flatten_for_scaler(X):
        # (N, C, T) -> (N*T, C) so StandardScaler normalizes per-channel
        return X.transpose(0, 2, 1).reshape(-1, n_channels)

    def unflatten(X_flat, n, t):
        return X_flat.reshape(n, t, n_channels).transpose(0, 2, 1)

    n_tr, _, t_tr = X_train.shape
    scaler.fit(flatten_for_scaler(X_train))

    X_train_norm = unflatten(scaler.transform(flatten_for_scaler(X_train)), n_tr, t_tr)
    X_val_norm, X_test_norm = None, None
    if X_val is not None:
        n_v, _, t_v = X_val.shape
        X_val_norm = unflatten(scaler.transform(flatten_for_scaler(X_val)), n_v, t_v)
    if X_test is not None:
        n_te, _, t_te = X_test.shape
        X_test_norm = unflatten(scaler.transform(flatten_for_scaler(X_test)), n_te, t_te)

    return X_train_norm.astype(np.float32), X_val_norm, X_test_norm, scaler


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train a speech emotion recognition model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default=None, help="ravdess | tess | emodb (overrides config)")
    parser.add_argument("--model", default=None, choices=["cnn", "lstm", "cnn_lstm"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for a pipeline smoke test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["dataset"]["random_seed"])

    # ---- CLI overrides ----
    dataset_name = args.dataset or cfg["dataset"]["name"]
    model_type = args.model or cfg["model"]["type"]
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["learning_rate"]

    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    device = get_device(cfg["training"]["device"])
    print(f"Using device: {device}")

    # ---- Load data ----
    if args.synthetic:
        print("Using SYNTHETIC dataset for pipeline smoke-test (no real audio needed).")
        X, y, label_names = make_synthetic_dataset(
            n_channels=cfg["features"]["n_mfcc"] * 3 + 12 + cfg["features"]["n_mels"] // 4 + 2,
            n_time=cfg["features"]["max_pad_len"],
            num_classes=NUM_CLASSES,
        )
    else:
        feat_path = os.path.join(cfg["dataset"]["processed_dir"], f"{dataset_name}_features.npz")
        if not os.path.exists(feat_path):
            print(f"ERROR: Feature cache not found at '{feat_path}'.")
            print(f"Run this first:\n  python src/feature_extraction.py --dataset {dataset_name} --data_dir <path_to_raw_audio>")
            sys.exit(1)
        data = np.load(feat_path, allow_pickle=True)
        X, y, label_names = data["X"], data["y"], data["label_names"]
        print(f"Loaded cached features: {feat_path}  shape={X.shape}")

    n_channels, n_time = X.shape[1], X.shape[2]
    num_classes = NUM_CLASSES

    # ---- Train / Val / Test split (stratified) ----
    test_size = cfg["dataset"]["test_size"]
    val_size = cfg["dataset"]["val_size"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=cfg["dataset"]["random_seed"]
    )
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val), stratify=y_temp, random_state=cfg["dataset"]["random_seed"]
    )

    print(f"Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

    # ---- Normalize ----
    X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)

    with open(cfg["paths"]["scaler_save_path"], "wb") as f:
        pickle.dump(scaler, f)
    save_label_map(cfg["paths"]["label_map_path"])

    # Save test split too, so evaluate.py can reuse the exact same held-out set
    np.savez_compressed(
        os.path.join(output_dir, "test_split.npz"), X=X_test, y=y_test
    )

    # ---- DataLoaders ----
    train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FeatureDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # ---- Model ----
    model = build_model(model_type, n_channels, n_time, num_classes, cfg["model"]).to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ---- Class-weighted loss (handles emotion class imbalance) ----
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=cfg["training"]["lr_scheduler_patience"]
    )

    # ---- Training loop with early stopping ----
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": model_type,
                "n_channels": n_channels,
                "n_time": n_time,
                "num_classes": num_classes,
                "model_cfg": cfg["model"],
                "val_acc": val_acc,
            }, cfg["paths"]["model_save_path"])
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch} epochs (no improvement for "
                      f"{cfg['training']['early_stopping_patience']} epochs).")
                break

    # ---- Save training history + plot ----
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["train_loss"], label="train")
        axes[0].plot(history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(history["train_acc"], label="train")
        axes[1].plot(history["val_acc"], label="val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=120)
        print(f"Saved training curves: {plot_path}")
    except ImportError:
        pass

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {cfg['paths']['model_save_path']}")
    print(f"Run evaluation with:\n  python src/evaluate.py --model_path {cfg['paths']['model_save_path']}")


if __name__ == "__main__":
    main()
