import numpy as np
import soundfile as sf

from evaluation.acoustic_features import band_snr, pulse_regularity


def _write(path, y, sr=8000):
    sf.write(path, y.astype(np.float32), sr)


def test_band_snr_detects_a_loud_tone_in_band_and_not_out_of_band(tmp_path):
    sr = 8000
    t = np.arange(30 * sr) / sr  # 30 s: quiet throughout except a tone burst in the middle
    y = 0.001 * np.random.default_rng(0).standard_normal(len(t)).astype(np.float32)
    burst = (t >= 14) & (t < 16)
    tone = 0.5 * np.sin(2 * np.pi * 700 * t[burst])
    fade = np.ones(burst.sum(), np.float32)
    n_fade = int(0.05 * sr)
    fade[:n_fade] = np.linspace(0, 1, n_fade)  # fade in/out so the burst edges aren't broadband clicks
    fade[-n_fade:] = np.linspace(1, 0, n_fade)
    y[burst] += tone * fade  # loud, narrow tone at 700 Hz
    path = tmp_path / "tone.wav"
    _write(path, y, sr)

    in_band = band_snr(str(path), t0=14, dur=2, band=(500, 900), pad=10)
    assert in_band is not None
    snr_db, tonality, peak_hz, _ = in_band
    assert snr_db > 15  # the burst is much louder than the surrounding noise floor
    assert tonality > 0.8  # a pure tone concentrates power in a narrow band
    assert 600 < peak_hz < 800

    out_of_band = band_snr(str(path), t0=14, dur=2, band=(2000, 3000), pad=10)
    assert out_of_band is not None
    assert out_of_band[0] < 5  # no energy at 2-3 kHz: near-zero SNR there


def test_band_snr_needs_enough_real_audio_around_the_window(tmp_path):
    sr = 8000
    y = np.zeros(2 * sr, dtype=np.float32)  # only 2 s total: far short of dur + 2 s of context
    path = tmp_path / "short.wav"
    _write(path, y, sr)
    assert band_snr(str(path), t0=0.5, dur=1, band=(500, 900)) is None


def test_pulse_regularity_recovers_a_known_pulse_rate(tmp_path):
    sr = 8000
    dur = 10
    t = np.arange(dur * sr) / sr
    rate_hz = 3.0
    envelope = (np.sin(2 * np.pi * rate_hz * t) > 0.5).astype(np.float32)  # short duty-cycle pulses
    y = envelope * 0.3 * np.sin(2 * np.pi * 1000 * t)
    path = tmp_path / "pulses.wav"
    _write(path, y.astype(np.float32), sr)

    res = pulse_regularity(str(path), t0=0, dur=dur, band=(800, 1200))
    assert res is not None
    regularity, pulse_hz = res
    assert regularity > 0.3  # a clean periodic pulse train should autocorrelate strongly
    assert abs(pulse_hz - rate_hz) < 0.5


def test_pulse_regularity_none_on_silence(tmp_path):
    sr = 8000
    y = np.zeros(5 * sr, dtype=np.float32)
    path = tmp_path / "silence.wav"
    _write(path, y, sr)
    assert pulse_regularity(str(path), t0=0, dur=5, band=(500, 900)) is None
