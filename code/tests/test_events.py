import pandas as pd

from evaluation.events import baseline_detections, merge_events


def test_merge_events_joins_consecutive_windows_and_splits_gaps():
    det = pd.DataFrame({
        "file": ["a.wav"] * 5 + ["b.wav"],
        "species": ["brdowl"] * 5 + ["brdowl"],
        "start_s": [10, 11, 12, 20, 21, 10],
        "p": [0.2, 0.9, 0.3, 0.5, 0.4, 0.6],
    })
    ev = merge_events(det).sort_values(["file", "start_s"]).reset_index(drop=True)
    assert list(ev.n_windows) == [3, 2, 1]
    assert list(ev.start_s) == [10, 20, 10]
    assert list(ev.end_s) == [12 + 5, 21 + 5, 10 + 5]
    assert ev.p_max[0] == 0.9


def test_baseline_rule_needs_target_top1_and_threshold():
    w = pd.DataFrame({
        "file": ["a"] * 4, "start_s": range(4),
        "top1": ["brdowl", "brdowl", "blksco1", "borowl"],
        "top1_p": [0.5, 0.05, 0.9, 0.11],
    })
    d = baseline_detections(w)
    assert list(d.start_s) == [0, 3]
