"""Save and load pre-extracted feature vectors (NumPy + JSON only).

``prepare_features.py`` writes one ``.npz`` per split here; ``train.py`` and
``evaluate.py`` read them back with ``--features-dir``. Deliberately free of
``datasets``/``soundfile`` so a GPU box can train without the corpus or the
audio decoding stack.

Layout::

    features/mfcc-mean_std/
        manifest.json
        train.npz        X (N, D) float32, y (N) int64, names, systems
        validation.npz
        test.npz
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

MANIFEST_NAME = "manifest.json"


def default_dir(feature: str, pooling: str) -> Path:
    """Conventional output folder, e.g. ``features/mfcc-mean_std``."""
    return Path("features") / f"{feature}-{pooling}"


def split_path(directory: str | Path, split: str) -> Path:
    return Path(directory) / f"{split}.npz"


def save_split(
    directory: str | Path,
    split: str,
    features: np.ndarray,
    labels: np.ndarray,
    names: Optional[Sequence[str]] = None,
    systems: Optional[Sequence[str]] = None,
) -> Path:
    """Write one split to ``<directory>/<split>.npz``."""
    path = split_path(directory, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {
        "X": np.asarray(features, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.int64),
    }
    if names is not None:
        arrays["names"] = np.asarray(names, dtype=str)
    if systems is not None:
        arrays["systems"] = np.asarray(systems, dtype=str)
    np.savez_compressed(path, **arrays)
    return path


def load_split(
    directory: str | Path, split: str, with_names: bool = False
):
    """Read one split; returns ``(X, y)`` or ``(X, y, names)``."""
    path = split_path(directory, split)
    if not path.exists():
        raise FileNotFoundError(
            f"no pre-extracted features at {path}\n"
            "run `python prepare_features.py` first (or fix --features-dir)"
        )
    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["X"], dtype=np.float32)
        labels = np.asarray(data["y"], dtype=np.int64)
        if not with_names:
            return features, labels
        names = (
            data["names"].astype(str) if "names" in data.files else np.empty(0, dtype=object)
        )
    return features, labels, names


def write_manifest(directory: str | Path, payload: Dict[str, Any]) -> Path:
    path = Path(directory) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def read_manifest(directory: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text())
