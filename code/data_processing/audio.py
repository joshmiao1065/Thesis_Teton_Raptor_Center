"""Field-audio loading: resample to 32 kHz and cut 5 s windows every 1 s in memory."""
import librosa
import numpy as np
import torch

SAMPLE_RATE = 32_000
WINDOW = 5 * SAMPLE_RATE
STEP = 1 * SAMPLE_RATE


def load_32k(path, quantize16=True):
    """Load a wav resampled to 32 kHz (librosa default resampler, as in the baseline).

    quantize16 mimics the baseline writing each window to a 16-bit wav before scoring.
    """
    y, _ = librosa.load(path, sr=SAMPLE_RATE)
    if quantize16:
        y = np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.float32) / 32768.0
    return torch.from_numpy(y)


def windows(y):
    """(n_windows, WINDOW) view over y; window i starts at i seconds. Empty if y is short."""
    if y.numel() < WINDOW:
        return y.new_empty((0, WINDOW))
    return y.unfold(0, WINDOW, STEP)


def gpu_resample(x, orig_sr, new_sr, device="cuda"):
    """Sinc-resample a (samples,) or (B, samples) tensor on the GPU; returns a CPU-free tensor on device."""
    import torchaudio.functional as AF

    return AF.resample(x.to(device), orig_sr, new_sr)


def read_wav(path):
    """Read a mono wav as float32 tensor and its sample rate (first channel if stereo)."""
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(y[:, 0].copy()), sr
