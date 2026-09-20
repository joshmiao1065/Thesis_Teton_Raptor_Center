"""Read the labels collected by the annotation tool and join them with what the sampler knew.

Each labeled item gets its stratum, the stratum's population size and the model scores at the focal
window (``cx_<code>``, ``bn_<code>``, ``hd_<code>``). ``weight`` = stratum population / number of
labeled items in that stratum, so weighted means estimate rates over the whole population of
candidates of that stratum (repeat clips and practice items are excluded from the weighting).
"""
import json
from pathlib import Path

import pandas as pd

CODES = ["flaowl", "grgowl", "norgos", "brdowl", "borowl"]


def load_labels(folder, labeler=None):
    """One row per (labeler, main queue item); the latest label wins. Practice items are left out."""
    folder = Path(folder)
    man = json.loads((folder / "manifest.json").read_text())
    items = {it["id"]: it for it in man["queue"]}
    latest = {}
    path = folder / "labels.jsonl"
    for line in (path.read_text().splitlines() if path.exists() else []):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["id"] in items and (labeler is None or r["labeler"] == labeler):
            latest[(r["labeler"], r["id"])] = r
    rows = []
    for (who, iid), r in latest.items():
        it = items[iid]
        row = dict(id=iid, labeler=who, site=it["site"], stratum=it["stratum"], pop=it["pop"], tier=it["tier"],
                   dup_of=it.get("dup_of"), hour=it["hidden"]["hour"], species="|".join(sorted(r["species"])),
                   none=r["none"], unsure=r["unsure"], certainty=r.get("certainty"), tags="|".join(r["tags"]),
                   edge=r["edge"], discuss=r["discuss"], notes=r["notes"], ms=r["ms"])
        for k, code in enumerate(CODES):
            row[f"cx_{code}"] = it["hidden"]["cx_p"][k]
            row[f"bn_{code}"] = it["hidden"]["bn_p"][k]
            row[f"hd_{code}"] = it["hidden"]["hd_p"][k]
            row[f"is_{code}"] = code in r["species"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    main = df[df.dup_of.isna()]
    n = main.groupby(["labeler", "stratum"]).id.transform("size")
    df["weight"] = (main["pop"] / n).reindex(df.index)
    return df


def weighted_rate(df, col, by):
    """Weighted share of rows with df[col] true within each group of `by` (main items only)."""
    d = df[df.weight.notna()]
    g = d.assign(w=d.weight, x=d[col].astype(float) * d.weight).groupby(by)
    return (g.x.sum() / g.w.sum()).rename(col)
