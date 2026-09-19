# Master's thesis: reducing false positives in owl and raptor detection from field audio

Work in progress. This repository holds the code for a master's thesis on automatic detection of
five owl and raptor species (flammulated, great gray, boreal and barred owl; northern goshawk) in
long-term field recordings, using a pretrained bird-sound classifier (ConvNeXT, BirdSet XCL).

**Question.** Why does the classifier produce many false positives on field recordings from one
site, and can they be reduced without losing performance at other sites and on benchmark data?

**Approach.**
1. Score every 5 s window (1 s hop) of the recordings and characterise the detections.
2. Test candidate explanations with controlled experiments (for example the 8 kHz band limit).
3. Train small "rescorer" heads on the frozen embeddings, using calls mixed into real background
   noise, and compare methods at equal false-alarm rates on held-out days, held-out sites and a
   synthetic benchmark. A gate on the original model's probability keeps the heads from learning
   background shortcuts.

## Layout (`code/`)
| Folder | Contents |
|---|---|
| `data_processing/` | audio loading and resampling, call-in-background mixtures, path configuration |
| `models/` | model loading and batched preprocessing |
| `evaluation/` | window scoring, feature extraction, event building, overview / band-limit / gallery notebooks |
| `training/` | rescorer heads, training scripts, evaluation notebooks (cross-site, benchmark, gated) |
| `analysis/` | plotting helpers and a blind audit-sheet generator for human labelling |
| `environments/` | pinned `requirements-*.txt` |

Notebooks are [jupytext](https://jupytext.readthedocs.io) pairs; the tracked source is the `.py`
file (percent format), and `.ipynb` files with outputs are kept locally.

## Setup
Data locations are not stored in the repository. Copy the paths your machine uses into
`code/local_paths.json` (keys `murie_audio`, `kaleidoscope`, `synthetic_audio`, `region_audio`,
relative to `THESIS_ROOT`), then `pip install -e code`.
