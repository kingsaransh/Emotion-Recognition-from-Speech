# Speech Emotion Recognition (SER)

Recognize human emotions (happy, sad, angry, fearful, calm, disgust, surprised, neutral) from speech audio using deep learning and signal processing.

This project supports **RAVDESS**, **TESS**, and **EMO-DB** datasets, extracts **MFCC** (and other) audio features, and trains **CNN, LSTM, or CNN+LSTM hybrid** models. It includes a full pipeline: data loading → feature extraction → training → evaluation → real-time/file-based inference.

---

## 1. Project Structure

```
speech-emotion-recognition/
├── README.md                  <- you are here
├── requirements.txt            <- pip dependencies
├── config.yaml                  <- all settings (dataset, model, training)
├── data/
│   ├── RAVDESS/                <- put RAVDESS audio here (Actor_01 ... Actor_24)
│   ├── TESS/                    <- put TESS audio here
│   ├── EMODB/                   <- put EMO-DB .wav files here
│   └── processed/                <- cached extracted features (auto-generated)
├── src/
│   ├── __init__.py
│   ├── dataset_loader.py        <- parses filenames/labels for each dataset
│   ├── feature_extraction.py    <- MFCC / chroma / mel-spectrogram extraction
│   ├── augmentation.py          <- noise/pitch/stretch audio augmentation
│   ├── models.py                <- CNN, LSTM, CNN-LSTM architectures (PyTorch)
│   ├── train.py                 <- training loop, CLI entry point
│   ├── evaluate.py              <- confusion matrix, classification report
│   ├── predict.py                <- run inference on a single .wav file
│   └── utils.py                  <- label maps, seed, helpers
├── app.py                        <- Streamlit web demo (upload/record audio)
├── outputs/                       <- saved models, plots, logs (auto-generated)
└── notebooks/
    └── quickstart.ipynb           <- end-to-end walkthrough in a notebook
```

---

## 2. Setup

```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Download a dataset (pick at least one)

| Dataset | Size | Link | Notes |
|---|---|---|---|
| **RAVDESS** (speech) | ~1 GB | https://zenodo.org/record/1188976 | 24 actors, 8 emotions |
| **TESS** | ~200 MB | https://tspace.library.utoronto.ca/handle/1807/24487 | 2 actresses, 7 emotions |
| **EMO-DB** | ~50 MB | http://emodb.bilderbar.info/download/ | German, 10 speakers, 7 emotions |

Unzip them into `data/RAVDESS/`, `data/TESS/`, `data/EMODB/` respectively, preserving original filenames (the loader parses emotion labels straight from filenames).

Expected layout examples:
```
data/RAVDESS/Actor_01/03-01-05-01-01-01-01.wav
data/TESS/OAF_back_angry.wav
data/EMODB/03a01Fa.wav
```

---

## 3. Quick Start

```bash
# Step 1: Extract features (caches to data/processed/features.npz)
python src/feature_extraction.py --dataset ravdess --data_dir data/RAVDESS

# Step 2: Train a model (cnn_lstm is the recommended default)
python src/train.py --model cnn_lstm --epochs 50 --batch_size 32

# Step 3: Evaluate on the held-out test split
python src/evaluate.py --model_path outputs/best_model.pt

# Step 4: Predict the emotion of a single audio file
python src/predict.py --file path/to/your_audio.wav --model_path outputs/best_model.pt

# Step 5 (optional): Launch the interactive web demo
streamlit run app.py
```

Everything also works **without downloading any dataset** — running `train.py` with `--synthetic` generates a small synthetic dataset so you can verify the whole pipeline runs end-to-end before plugging in real data.

```bash
python src/train.py --synthetic --model cnn --epochs 5
```

---

## 4. How It Works

### 4.1 Feature Extraction (`src/feature_extraction.py`)
For every audio clip, we extract a stack of complementary features using `librosa`:
- **MFCCs** (40 coefficients) — captures timbre / vocal-tract shape, the primary emotion cue.
- **Delta & Delta-Delta MFCCs** — captures how MFCCs change over time (speech dynamics).
- **Chroma STFT** — pitch class energy, related to prosody/intonation.
- **Mel-spectrogram** — perceptual loudness across frequency bands.
- **Zero-Crossing Rate & RMS Energy** — captures voice intensity/sharpness (useful for angry vs. calm).

All features are stacked into a single 2D matrix (`features × time`) per clip, padded/truncated to a fixed length, then z-score normalized.

### 4.2 Models (`src/models.py`) — implemented in PyTorch
1. **CNN** — 2D convolutions treat the MFCC matrix like an image, learning local time-frequency patterns. Fast, good baseline.
2. **LSTM** — treats MFCC frames as a sequence over time, modeling temporal evolution of speech.
3. **CNN-LSTM (recommended)** — CNN layers extract local spectro-temporal patterns, then a bidirectional LSTM models how those patterns evolve, followed by an attention-pooling layer and dense classifier. This combination consistently outperforms either alone on SER benchmarks.

### 4.3 Training (`src/train.py`)
- Stratified train/val/test split (default 70/15/15).
- Class-weighted loss to handle emotion class imbalance.
- Optional data augmentation: additive noise, pitch shift, time stretch (`src/augmentation.py`).
- Learning rate scheduling (`ReduceLROnPlateau`) + early stopping.
- Saves best checkpoint (by val accuracy) to `outputs/best_model.pt`, plus training curves.

### 4.4 Evaluation (`src/evaluate.py`)
Prints accuracy, F1 (macro), full classification report, and saves a confusion matrix plot to `outputs/confusion_matrix.png`.

### 4.5 Inference (`src/predict.py`)
Loads a trained checkpoint + the saved feature scaler, extracts features from a new `.wav` file, and prints the predicted emotion with class probabilities.

---

## 5. Emotion Label Map

| Code | Emotion |
|---|---|
| 0 | neutral |
| 1 | calm |
| 2 | happy |
| 3 | sad |
| 4 | angry |
| 5 | fearful |
| 6 | disgust |
| 7 | surprised |

(Not every dataset contains all 8 — TESS has no "calm", EMO-DB uses German labels mapped to this scheme. See `src/dataset_loader.py` for exact per-dataset mappings.)

---

## 6. Configuration

All key parameters live in `config.yaml` — sample rate, MFCC count, max audio duration, model hyperparameters, training settings — so you can tune the project without touching code.

---

## 7. Extending This Project

- **Add a new dataset**: write a new `parse_<name>()` function in `dataset_loader.py` returning `(filepath, emotion_label)` pairs.
- **Add a new model**: add a class to `models.py` implementing `forward(x)`, register it in `MODEL_REGISTRY`.
- **Use spectrograms + a pretrained vision backbone**: swap feature extraction to output mel-spectrogram images and feed into a CNN like ResNet (transfer learning) — the codebase's modular design supports this with minimal changes.
- **Real-time microphone inference**: `app.py` includes a Streamlit recorder widget showing how to capture mic audio and feed it into `predict.py`'s pipeline live.

---

## 8. Troubleshooting

- **`librosa` install issues on Windows**: install `soundfile` first (`pip install soundfile`), then `librosa`.
- **CUDA not detected**: training automatically falls back to CPU; CNN-LSTM on RAVDESS (~1,440 clips) trains in a few minutes on CPU.
- **Out of memory**: lower `batch_size` in `config.yaml` or reduce `max_pad_len`.
- **Class imbalance / low minority-class recall**: enable augmentation (`--augment`) and class-weighted loss (on by default).
