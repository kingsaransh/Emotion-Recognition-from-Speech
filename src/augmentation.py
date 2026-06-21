"""
augmentation.py
Lightweight audio augmentation to improve generalization and address class imbalance.
Applied at the waveform level, before feature extraction.
"""
import numpy as np
import librosa


def add_noise(y, noise_level=0.005):
    noise = np.random.randn(len(y))
    return y + noise_level * noise


def pitch_shift(y, sr, n_steps=None):
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def time_stretch(y, rate=None):
    if rate is None:
        rate = np.random.uniform(0.85, 1.15)
    stretched = librosa.effects.time_stretch(y, rate=rate)
    # Restore original length (pad or truncate) so downstream shapes stay consistent
    if len(stretched) > len(y):
        stretched = stretched[: len(y)]
    else:
        stretched = np.pad(stretched, (0, len(y) - len(stretched)), mode="constant")
    return stretched


def augment_waveform(y, sr, noise_prob=0.3, pitch_prob=0.3, stretch_prob=0.3):
    """
    Randomly applies one or more augmentations to a waveform.
    Returns an augmented copy (the original `y` is left untouched by the caller).
    """
    out = y.copy()
    if np.random.rand() < noise_prob:
        out = add_noise(out)
    if np.random.rand() < pitch_prob:
        out = pitch_shift(out, sr)
    if np.random.rand() < stretch_prob:
        out = time_stretch(out)
    return out
