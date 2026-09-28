"""Score chosen windows of recordings with ConvNeXT or BirdNET and keep the logits of every class.

Input: a CSV with one row per item and columns `path` (wav), `first`, `last` (start second of the
first and last window, 1 s hop; may lie partly outside the file, the missing audio is zero-padded).
ConvNeXT windows are 5 s, BirdNET windows 3 s. Stereo files are mixed to mono. ConvNeXT resampling:
`--resampler gpu` (torchaudio sinc, as in the feature extraction) or `librosa` (soxr, then 16-bit
quantization, as in the baseline scoring); ConvNeXT probabilities differ by a few hundredths between them.

    python -m evaluation.score_intervals convnext --items items.csv --out cx.npz [--resampler librosa]  # .venv
    python -m evaluation.score_intervals birdnet  --items items.csv --out bn.npz     # .venv-birdnet

Output npz, one row per window: `item` (row of the CSV), `start_s` (window start in the recording),
`padded` (window reaches outside the file), `logits` (float16, all classes), `labels` (class names);
ConvNeXT also `emb` (pooled 1024-d embedding, float16).
"""
import argparse

import numpy as np
import pandas as pd
import soundfile as sf

MARGIN = 2  # seconds of real audio kept around the windows so resampling edges stay outside them


def read_span(path, a, b):
    """Mono float32 audio of [a, b) seconds at the file's rate, zero-padded outside the file."""
    info = sf.info(path)
    sr = info.samplerate
    i0, i1 = round(float(a) * sr), round(float(b) * sr)
    r0, r1 = max(i0, 0), min(i1, info.frames)
    y = np.zeros(i1 - i0, np.float32)
    if r1 > r0:
        x, _ = sf.read(path, start=r0, frames=r1 - r0, dtype="float32", always_2d=True)
        y[r0 - i0: r1 - i0] = x.mean(1)
    return y, sr, info.frames / sr


def window_starts(first, last):
    return np.arange(int(first), int(last) + 1)


def score_convnext(items, batch=32, resampler="gpu"):
    import torch
    from transformers import AutoConfig

    from data_processing.audio import gpu_resample
    from models.convnext import MODEL_ID, Preprocessor, load_model

    model, prep = load_model(), Preprocessor().cuda()
    id2label = AutoConfig.from_pretrained(MODEL_ID).id2label
    labels = [id2label[i] for i in range(model.config.num_labels)]
    rows, start, padded, logits, emb = [], [], [], [], []
    buf, meta = [], []

    @torch.no_grad()
    def flush():
        if not buf:
            return
        w = torch.stack(buf)
        e = model.convnext(prep(w)).pooler_output
        logits.append(model.classifier(e).half().cpu().numpy())
        emb.append(e.half().cpu().numpy())
        for k, s, p in meta:
            rows.append(k)
            start.append(s)
            padded.append(p)
        buf.clear()
        meta.clear()

    for k, it in enumerate(items.itertuples()):
        starts = window_starts(it.first, it.last)
        a, b = starts[0] - MARGIN, starts[-1] + 5 + MARGIN
        y, sr, dur = read_span(it.path, a, b)
        if resampler == "librosa":
            import librosa

            y32 = librosa.resample(y, orig_sr=sr, target_sr=32000)
            y32 = np.clip(np.round(y32 * 32768.0), -32768, 32767).astype(np.float32) / 32768.0
            y32 = torch.from_numpy(y32).cuda()
        else:
            y32 = gpu_resample(torch.from_numpy(y), sr, 32000)
        for s in starts:
            i = int((s - a) * 32000)
            buf.append(y32[i: i + 160000])
            meta.append((k, s, bool(s < 0 or s + 5 > dur + 1e-6)))
            if len(buf) == batch:
                flush()
    flush()
    return dict(item=np.array(rows), start_s=np.array(start, np.float32), padded=np.array(padded),
                logits=np.concatenate(logits), emb=np.concatenate(emb), labels=np.array(labels))


def score_birdnet(items, device="GPU:0", chunk=256):
    import birdnet

    model = birdnet.load("acoustic", "2.4", "pb")
    labels = list(model.species_list)
    rows, start, padded, logits = [], [], [], []
    for c0 in range(0, len(items), chunk):
        part = items.iloc[c0: c0 + chunk]
        arrays, spans = [], []
        for it in part.itertuples():
            starts = window_starts(it.first, it.last)
            y, sr, dur = read_span(it.path, starts[0], starts[-1] + 3)
            arrays.append((y, sr))
            spans.append((starts, dur))
        r = model.predict_arrays(arrays, top_k=None, n_workers=1, batch_size=32, overlap_duration_s=2.0,
                                 apply_sigmoid=False, default_confidence_threshold=None, device=device)
        vals, ids = np.asarray(r.species_probs), np.asarray(r.species_ids)
        assert abs(r.hop_duration_s - 1.0) < 1e-6
        for j, (starts, dur) in enumerate(spans):
            v = np.empty((len(starts), vals.shape[2]), np.float32)  # class order differs per window
            np.put_along_axis(v, ids[j, : len(starts)].astype(np.int64), vals[j, : len(starts)].astype(np.float32), axis=1)
            logits.append(v.astype(np.float16))
            rows += [c0 + j] * len(starts)
            start += list(starts)
            padded += [bool(s < 0 or s + 3 > dur + 1e-6) for s in starts]
        print(f"birdnet {min(c0 + chunk, len(items))}/{len(items)}", flush=True)
    return dict(item=np.array(rows), start_s=np.array(start, np.float32), padded=np.array(padded),
                logits=np.concatenate(logits), labels=np.array(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["convnext", "birdnet"])
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resampler", choices=["gpu", "librosa"], default="gpu", help="ConvNeXT only")
    a = ap.parse_args()
    items = pd.read_csv(a.items)
    res = score_convnext(items, resampler=a.resampler) if a.model == "convnext" else score_birdnet(items)
    np.savez(a.out, **res)
    print(f"{a.model}: {len(res['item'])} windows for {len(items)} items -> {a.out}")


if __name__ == "__main__":
    main()
