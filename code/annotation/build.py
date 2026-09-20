"""Build the annotation directory: queue (main items), reference library and practice set.

    python -m annotation.build --out <dir> [--sites murie taylor butler riverview] [--seed 0]

Writes ``manifest.json`` (what the server reads, including the hidden model context of every item),
the reference wav files, and prints a summary. Run again with the same seed to get the same items.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from annotation.reference import build_reference
from annotation.sampling import build_queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites", nargs="+", default=["murie", "taylor", "butler", "riverview"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    queue = build_queue(a.sites, out, a.seed)
    ref, prac = build_reference(queue, out)
    rng = np.random.default_rng(a.seed)
    prac = [prac[i] for i in rng.permutation(len(prac))]
    (out / "manifest.json").write_text(json.dumps(dict(queue=queue, reference=ref, practice=prac)))
    print(f"queue {len(queue)} (tier 1: {sum(i['tier'] == 1 for i in queue)}), reference {len(ref)}, practice {len(prac)}")


if __name__ == "__main__":
    main()
