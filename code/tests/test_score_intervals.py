import numpy as np
import soundfile as sf

from evaluation.score_intervals import read_span, window_starts


def test_read_span_pads_outside_the_file_and_mixes_to_mono(tmp_path):
    sr = 8000
    left = np.linspace(-0.5, 0.5, 2 * sr, dtype=np.float32)
    x = np.stack([left, np.zeros_like(left)], 1)  # 2 s stereo
    path = tmp_path / "a.wav"
    sf.write(path, x, sr, subtype="FLOAT")
    y, rate, dur = read_span(str(path), -1, 3)
    assert rate == sr and abs(dur - 2) < 1e-9 and len(y) == 4 * sr
    assert np.all(y[:sr] == 0) and np.all(y[3 * sr:] == 0)
    np.testing.assert_allclose(y[sr:3 * sr], left / 2, atol=1e-7)


def test_window_starts_inclusive():
    assert window_starts(3, 5).tolist() == [3, 4, 5]
    assert window_starts(-1, -1).tolist() == [-1]
