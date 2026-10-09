"""Acoustic diagnostics of a single window, independent of either model's score.

Two signal-processing features over a frequency band: how much louder a window is than the quiet
moments around it (`band_snr`), and how rhythmic its energy envelope is (`pulse_regularity`). Neither
uses a trained model or any label; both only need the recording and a band of interest.
"""
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

PAD = 15  # default seconds of audio on each side used as the local noise-floor estimate


def band_snr(path, t0, dur, band, pad=PAD):
    """How much louder the window [t0, t0+dur] is than the quiet moments around it, in `band` Hz.

    Returns (band_snr_db, tonality, peak_hz, peak_time_s), or None if there isn't enough real audio
    around the window to judge (too close to the start/end of the file).
      - band_snr_db: the loudest frame in the window, in the band, minus the band's median loudness
        over `pad` seconds before and after (captures that moment's own background: wind, insects,
        a stream -- not a fixed noise floor for the whole recording).
      - tonality: at that loudest frame, the band's share of power across the whole spectrum (0-1;
        high = a concentrated tone stands out from a broadband hiss, low = broadband noise or several
        bands active at once).
      - peak_hz, peak_time_s: where in frequency and time the loudest moment was.
    """
    info = sf.info(path)
    sr = info.samplerate
    a, b = max(t0 - pad, 0), min(t0 + dur + pad, info.frames / sr)
    if b - a < dur + 2:  # not enough real audio around it to judge
        return None
    y, _ = sf.read(path, start=int(a * sr), frames=int((b - a) * sr), dtype="float32", always_2d=True)
    y = y.mean(1)  # mono mixdown (stereo recordings)
    fr, t, S = spectrogram(y, sr, nperseg=512, noverlap=384)
    db = 10 * np.log10(S + 1e-12)
    lo, hi = np.searchsorted(fr, band[0]), np.searchsorted(fr, band[1])
    lo, hi = max(lo, 1), max(hi, lo + 1)  # skip DC bin
    tl, th = t0 - a, t0 + dur - a
    in_win = (t >= tl) & (t <= th)
    in_pad = ~in_win
    if not in_win.any() or not in_pad.any():
        return None
    band_power_t = S[lo:hi].sum(0)  # (frames,) linear power summed over the band
    total_power_t = S[1:].sum(0)
    noise_floor = np.median(10 * np.log10(band_power_t[in_pad] + 1e-12))
    k = np.argmax(band_power_t[in_win])
    peak_db = 10 * np.log10(band_power_t[in_win][k] + 1e-12)
    peak_hz = fr[lo:hi][np.argmax(db[lo:hi, in_win][:, k])]
    tonality = band_power_t[in_win][k] / (total_power_t[in_win][k] + 1e-12)
    return float(peak_db - noise_floor), float(tonality), float(peak_hz), float(t[in_win][k] + a)


def pulse_regularity(path, t0, dur, band, min_hz=0.8, max_hz=10):
    """How regular and how fast the window's band-limited energy pulses, via autocorrelation.

    Returns (regularity, pulse_hz), or None if the window is too short or silent. `regularity` is the
    autocorrelation peak (0-1) in the [min_hz, max_hz] pulse-rate range: high means a strongly periodic
    on/off pattern (e.g. a rhythmic call), low means an irregular or steady sound. `pulse_hz` is the
    rate at that peak.
    """
    info = sf.info(path)
    sr = info.samplerate
    if dur * sr < 1024:
        return None
    y, _ = sf.read(path, start=int(t0 * sr), frames=int(dur * sr), dtype="float32", always_2d=True)
    y = y.mean(1)
    fr, t, S = spectrogram(y, sr, nperseg=512, noverlap=448)
    lo, hi = np.searchsorted(fr, band[0]), np.searchsorted(fr, band[1])
    env = S[max(lo, 1):hi].sum(0)
    env = env - env.mean()
    if env.std() < 1e-12 or len(t) < 3:
        return None
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    ac = ac / (ac[0] + 1e-12)
    hop = t[1] - t[0]
    lag_lo, lag_hi = int(1 / max_hz / hop), min(int(1 / min_hz / hop), len(ac) - 1)
    if lag_hi <= lag_lo:
        return None
    k = lag_lo + np.argmax(ac[lag_lo:lag_hi])
    return float(ac[k]), float(1 / (k * hop))
