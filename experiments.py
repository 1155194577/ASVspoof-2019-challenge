"""Numbered experiment folders.

Every training run gets ``experiment_log/exp_NNN/`` containing everything about
it, so a result is never ambiguous:

    experiment_log/exp_007/
        config.json    resolved arguments, timestamp, host, git revision
        history.json   per-epoch train/val metrics (what plot.py draws)
        model.pt       weights (+ standardisation stats, model kwargs)
        plot.png       written by plot.py
        slurm.out      slurm output, copied back by submit.sh --fetch

The number is allocated atomically (``mkdir`` fails if taken), so parallel
submissions cannot collide.

Used by:
* ``train.py``  -- allocates a folder when ``--exp-dir`` is not given
* ``plot.py``   -- defaults to the most recent folder and writes its ``plot.png``
* ``evaluate.py`` -- defaults to the most recent folder's ``model.pt``
* ``submit.sh`` -- allocates the folder locally, passes it to the job, pulls it back
"""

from __future__ import annotations

import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_ROOT = "experiment_log"
PREFIX = "exp_"
WIDTH = 3

CONFIG_NAME = "config.json"
HISTORY_NAME = "history.json"
MODEL_NAME = "model.pt"
PLOT_NAME = "plot.png"
SLURM_NAME = "slurm.out"


def list_experiments(root: str | Path = DEFAULT_ROOT) -> List[Path]:
    """Existing experiment folders, oldest first."""
    base = Path(root)
    if not base.is_dir():
        return []
    found = []
    for path in base.iterdir():
        if path.is_dir() and path.name.startswith(PREFIX):
            suffix = path.name[len(PREFIX):]
            if suffix.isdigit():
                found.append((int(suffix), path))
    return [path for _, path in sorted(found)]


def experiment_name(index: int) -> str:
    return f"{PREFIX}{index:0{WIDTH}d}"


def create(
    root: str | Path = DEFAULT_ROOT,
    tag: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Allocate the next ``exp_NNN`` folder and return it."""
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    existing = list_experiments(base)
    next_index = (int(existing[-1].name[len(PREFIX):]) + 1) if existing else 1

    for index in range(next_index, next_index + 1000):
        path = base / experiment_name(index)
        try:
            path.mkdir()  # atomic: raises if another process grabbed it
        except FileExistsError:
            continue
        if config is not None:
            write_config(path, config, tag=tag)
        return path
    raise RuntimeError(f"could not allocate an experiment folder under {base}")


def peek_next(root: str | Path = DEFAULT_ROOT) -> str:
    """Next experiment path without creating it (for --dry-run)."""
    existing = list_experiments(root)
    index = (int(existing[-1].name[len(PREFIX):]) + 1) if existing else 1
    return str(Path(root) / experiment_name(index))


def git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def write_config(
    path: str | Path, config: Dict[str, Any], tag: Optional[str] = None
) -> Path:
    """Write ``config.json`` (arguments + provenance) into an experiment."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": path.name,
        "tag": tag,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "git": git_revision(),
        "args": config,
    }
    out = path / CONFIG_NAME
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return out


def read_config(path: str | Path) -> Dict[str, Any]:
    return json.loads((Path(path) / CONFIG_NAME).read_text())


def latest(root: str | Path = DEFAULT_ROOT) -> Optional[Path]:
    experiments = list_experiments(root)
    return experiments[-1] if experiments else None


def resolve(spec: Optional[str | int], root: str | Path = DEFAULT_ROOT) -> Path:
    """Turn ``None`` / ``7`` / ``exp_007`` / ``some/path`` into an experiment dir."""
    if spec is None:
        latest_path = latest(root)
        if latest_path is None:
            raise SystemExit(
                f"no experiments under {root}/ -- run `python train.py` first "
                "(or pass --exp)"
            )
        return latest_path

    text = str(spec)
    if text.isdigit():
        path = Path(root) / experiment_name(int(text))
        if not path.is_dir():
            raise SystemExit(f"experiment {text} not found ({path})")
        return path
    path = Path(text)
    if not path.is_dir():
        raise SystemExit(f"experiment folder not found: {path}")
    return path


def write_metrics(path: str | Path, split: str, payload: Dict[str, Any]) -> Path:
    """Write ``<split>.json`` (test.json / validation.json) into an experiment."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    out = path / f"{split}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return out


def read_metrics(path: str | Path, split: str) -> Optional[Dict[str, Any]]:
    out = Path(path) / f"{split}.json"
    if not out.exists():
        return None
    return json.loads(out.read_text())


def describe(path: Path) -> str:
    """One-line summary, e.g. ``exp_007 (svm_rff lfcc, EER 6.33%)``."""
    try:
        config = read_config(path)
        args = config.get("args", {})
        label = f"{args.get('model', '?')} {args.get('feature', '?')}"
    except Exception:
        label = "?"
    eer = ""
    history = path / HISTORY_NAME
    if history.exists():
        try:
            rows = json.loads(history.read_text())["history"]
            values = [r["val_eer"] for r in rows if r.get("val_eer") is not None]
            if values:
                eer = f", best EER {min(values) * 100:.2f}%"
        except Exception:
            pass
    return f"{path.name} ({label}{eer})"
