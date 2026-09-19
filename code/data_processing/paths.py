"""Project locations. Machine-specific data paths live in `local_paths.json` (not tracked)."""
import json
import os
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("THESIS_ROOT", CODE.parent))
INTERMEDIATE = ROOT / "intermediate"
RESULTS = ROOT / "results"

_cfg_file = CODE / "local_paths.json"
_cfg = json.loads(_cfg_file.read_text()) if _cfg_file.exists() else {}


def _get(key):
    if key not in _cfg:
        raise FileNotFoundError(f"set '{key}' (path relative to THESIS_ROOT) in {_cfg_file}")
    return ROOT / _cfg[key]


def __getattr__(name):  # lazy, so importing this module never needs the data to exist
    keys = {"MURIE_AUDIO": "murie_audio", "KALEIDOSCOPE": "kaleidoscope", "SYNTHETIC": "synthetic_audio"}
    if name in keys:
        return _get(keys[name])
    raise AttributeError(name)
