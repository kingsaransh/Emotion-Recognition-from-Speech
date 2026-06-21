"""
app.py
Interactive Streamlit web demo for Speech Emotion Recognition.

Run with:
    streamlit run app.py

Lets the user upload a .wav file, see its waveform, and get the predicted emotion
with a probability breakdown.
"""
import os
import sys
import tempfile

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, get_device
from src.predict import load_inference_artifacts, predict_emotion

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")

EMOTION_EMOJI = {
    "neutral": "😐", "calm": "😌", "happy": "😄", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲",
}


@st.cache_resource
def get_model_and_scaler():
    cfg = load_config("config.yaml")
    model_path = cfg["paths"]["model_save_path"]
    if not os.path.exists(model_path):
        return None, None, None, cfg
    device = get_device(cfg["training"]["device"])
    model, scaler, n_channels = load_inference_artifacts(cfg, model_path, device)
    return model, scaler, n_channels, cfg


def main():
    st.title("🎙️ Speech Emotion Recognition")
    st.write(
        "Upload a short speech audio clip (`.wav`) and the trained model will "
        "predict the speaker's emotion (happy, sad, angry, calm, neutral, fearful, disgust, surprised)."
    )

    model, scaler, n_channels, cfg = get_model_and_scaler()

    if model is None:
        st.error(
            "No trained model found at `outputs/best_model.pt`. "
            "Train one first:\n\n"
            "```\npython src/feature_extraction.py --dataset ravdess --data_dir data/RAVDESS\n"
            "python src/train.py --model cnn_lstm\n```\n\n"
            "Or run a quick smoke-test with synthetic data:\n"
            "```\npython src/train.py --synthetic --model cnn --epochs 5\n```"
        )
        return

    uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav", "mp3", "ogg", "flac"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.audio(uploaded_file)

        # Plot waveform
        y, sr = librosa.load(tmp_path, sr=cfg["audio"]["sample_rate"])
        fig, ax = plt.subplots(figsize=(8, 2.2))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#6C63FF")
        ax.set_title("Waveform")
        ax.set_xlabel("Time (s)")
        st.pyplot(fig)

        with st.spinner("Analyzing emotion..."):
            device = get_device(cfg["training"]["device"])
            pred_label, prob_dict = predict_emotion(tmp_path, model, scaler, n_channels, cfg, device)

        emoji = EMOTION_EMOJI.get(pred_label, "")
        st.markdown(f"## Predicted Emotion: {emoji} **{pred_label.upper()}**")

        st.subheader("Confidence Breakdown")
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: -x[1]))
        for label, p in sorted_probs.items():
            st.write(f"{EMOTION_EMOJI.get(label, '')} {label.capitalize()}")
            st.progress(min(max(p, 0.0), 1.0))

        os.unlink(tmp_path)

    st.markdown("---")
    st.caption(
        "Model: CNN-LSTM trained on stacked MFCC + delta + chroma + mel-spectrogram features. "
        "See README.md for full pipeline details."
    )


if __name__ == "__main__":
    main()
