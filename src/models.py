"""
models.py
PyTorch model architectures for speech emotion recognition.

Input shape convention: (batch, channels, time_frames)
  channels = stacked feature dim (MFCC + delta + delta2 + chroma + mel + zcr + rms)
  time_frames = fixed-length time axis after padding/truncation

Three architectures:
  1. EmotionCNN      - treats the feature matrix as a 2D "image" (channels x time)
  2. EmotionLSTM     - treats time as a sequence, feature channels as per-step input
  3. EmotionCNNLSTM  - CNN front-end for local spectro-temporal patterns,
                       followed by a bidirectional LSTM + attention pooling.
                       Recommended default; best accuracy in practice.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. Pure CNN
# ----------------------------------------------------------------------
class EmotionCNN(nn.Module):
    def __init__(self, n_channels, n_time, num_classes=8, conv_channels=(32, 64, 128), dropout=0.4):
        super().__init__()
        c1, c2, c3 = conv_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout * 0.5),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout * 0.5),

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, channels, time) -> add a dummy "image channel" dim
        x = x.unsqueeze(1)  # (batch, 1, channels, time)
        x = self.conv_block(x)
        return self.classifier(x)


# ----------------------------------------------------------------------
# 2. Pure LSTM
# ----------------------------------------------------------------------
class EmotionLSTM(nn.Module):
    def __init__(self, n_channels, n_time, num_classes=8, hidden_size=128,
                 num_layers=2, bidirectional=True, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )
        direction_mult = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * direction_mult, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, channels, time) -> LSTM expects (batch, time, channels)
        x = x.permute(0, 2, 1)
        out, (h_n, _) = self.lstm(x)
        # Use mean pooling over time for a robust summary representation
        pooled = out.mean(dim=1)
        return self.classifier(pooled)


# ----------------------------------------------------------------------
# 3. CNN + LSTM hybrid with attention pooling (recommended)
# ----------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """Learns a weighted average over time steps instead of plain mean pooling."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, time, hidden_dim)
        weights = torch.softmax(self.attn(x), dim=1)  # (batch, time, 1)
        pooled = (x * weights).sum(dim=1)              # (batch, hidden_dim)
        return pooled


class EmotionCNNLSTM(nn.Module):
    def __init__(self, n_channels, n_time, num_classes=8, conv_channels=(32, 64),
                 lstm_hidden_size=128, lstm_layers=2, bidirectional=True, dropout=0.4):
        super().__init__()
        c1, c2 = conv_channels

        # CNN front-end: convolve over (feature_channel, time) treating feature
        # channels like "height" -- this captures local spectro-temporal patterns
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # pool over feature axis only, preserve time resolution
            nn.Dropout(dropout * 0.5),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(dropout * 0.5),
        )

        reduced_feat_dim = n_channels // 4  # after two (2,1) poolings on the feature axis
        lstm_input_size = c2 * max(reduced_feat_dim, 1)

        direction_mult = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.attention_pool = AttentionPooling(lstm_hidden_size * direction_mult)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * direction_mult, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, channels, time)
        x = x.unsqueeze(1)              # (batch, 1, channels, time)
        x = self.conv_block(x)          # (batch, c2, reduced_channels, time)

        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (batch, time, c2*reduced_channels)

        out, _ = self.lstm(x)           # (batch, time, hidden*dirs)
        pooled = self.attention_pool(out)
        return self.classifier(pooled)


# ----------------------------------------------------------------------
# Registry so train.py / predict.py can build models by name from config
# ----------------------------------------------------------------------
MODEL_REGISTRY = {
    "cnn": EmotionCNN,
    "lstm": EmotionLSTM,
    "cnn_lstm": EmotionCNNLSTM,
}


def build_model(model_type, n_channels, n_time, num_classes, cfg_model):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(MODEL_REGISTRY)}")

    cls = MODEL_REGISTRY[model_type]

    if model_type == "cnn":
        return cls(
            n_channels=n_channels, n_time=n_time, num_classes=num_classes,
            conv_channels=tuple(cfg_model.get("cnn_channels", [32, 64, 128])),
            dropout=cfg_model.get("dropout", 0.4),
        )
    elif model_type == "lstm":
        return cls(
            n_channels=n_channels, n_time=n_time, num_classes=num_classes,
            hidden_size=cfg_model.get("lstm_hidden_size", 128),
            num_layers=cfg_model.get("lstm_layers", 2),
            bidirectional=cfg_model.get("bidirectional", True),
            dropout=cfg_model.get("dropout", 0.4),
        )
    else:  # cnn_lstm
        cnn_channels = cfg_model.get("cnn_channels", [32, 64, 128])[:2]
        return cls(
            n_channels=n_channels, n_time=n_time, num_classes=num_classes,
            conv_channels=tuple(cnn_channels),
            lstm_hidden_size=cfg_model.get("lstm_hidden_size", 128),
            lstm_layers=cfg_model.get("lstm_layers", 2),
            bidirectional=cfg_model.get("bidirectional", True),
            dropout=cfg_model.get("dropout", 0.4),
        )
