"""Visualise an experiment: learning curves, score distributions, t-SNE.

    conda activate ml
    python plot.py                         # checkpoint + features from config.py
    python plot.py --split validation --out figures/svm-report.png
    python plot.py --show                  # also open a window

Four panels:

* **accuracy** — train vs validation accuracy per epoch (from the history JSON
  ``train.py`` writes next to the checkpoint)
* **EER** — validation EER per epoch (the metric the checkpoint is chosen on),
  with the best epoch marked
* **scores** — distribution of the SVM's spoof score for bonafide vs spoof,
  with the decision threshold, the EER, and the experiment's test EER when
  ``test.json`` exists
* **embedding** — t-SNE of the feature vectors, coloured by true label with
  misclassified utterances ringed in black

Everything is read from the checkpoint (standardisation, model, front-end) and
the pre-extracted ``.npz`` features, so no dataset or audio is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

import matplotlib

if "--show" not in sys.argv:  # headless by default (cluster, CI, ssh)
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments
import features_io
from config import TrainingConfig
from evaluate import load_checkpoint, score_batches
from model import build_model
from train import compute_eer, select_device

CFG = TrainingConfig()
LABEL_NAMES = ("bonafide", "spoof")
BONAFIDE, SPOOF = 0, 1
COLORS = {BONAFIDE: "#4C72B0", SPOOF: "#C44E52"}


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot scores, t-SNE and learning curves",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--exp", default=None,
                        help="experiment number or folder (default: the latest one)")
    parser.add_argument("--checkpoint", default=None,
                        help="override the model file (default: <experiment>/model.pt)")
    parser.add_argument("--history", default=None,
                        help="override the history file (default: <experiment>/history.json)")
    parser.add_argument("--features-dir", default=CFG.features_dir,
                        help="pre-extracted .npz folder")
    parser.add_argument("--split", default=CFG.eval_split,
                        choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=CFG.limit,
                        help="use only the first N utterances")
    parser.add_argument("--out", default=None,
                        help="figure path (default: <experiment>/plot.png)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-scatter", type=int, default=4000,
                        help="points embedded / drawn in the 2-D panel")
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE perplexity")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    return parser.parse_args(argv)


def resolve_features_dir(explicit: Optional[str], ckpt: Dict[str, Any]) -> str:
    """Prefer features/<feature>-<pooling>, else the first folder that has data."""
    args = ckpt.get("args", {})
    conventional = f"features/{args.get('feature', CFG.feature)}-" \
                   f"{args.get('pooling', CFG.pooling)}"
    if explicit:
        if not Path(explicit).exists():
            raise SystemExit(f"features folder not found: {explicit}")
        return explicit
    if Path(conventional, "train.npz").exists():
        return conventional
    candidates = sorted(p for p in Path("features").glob("*/") if (p / "train.npz").exists())
    if not candidates:
        raise SystemExit("no pre-extracted features found; run `python prepare_features.py`")
    return str(candidates[0]).rstrip("/")


def find_history(explicit: Optional[str], ckpt_path: Path, ckpt: Dict[str, Any]) -> Optional[Dict]:
    path: Optional[Path] = Path(explicit) if explicit else None
    if path is None:
        args = ckpt.get("args", {})
        named = ckpt_path.parent / (
            f"history-{args.get('model', CFG.model)}-"
            f"{args.get('feature', CFG.feature)}-{args.get('pooling', CFG.pooling)}.json"
        )
        if named.exists():
            path = named
        else:
            matches = sorted(ckpt_path.parent.glob("history-*.json"))
            path = matches[-1] if matches else None
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def plot_curve(ax, epochs, series, ylabel: str, title: str, saved_epoch: int,
               as_percent: bool) -> None:
    for name, values in series:
        if as_percent:
            values = np.asarray(values, dtype=float) * 100.0
        ax.plot(epochs, values, marker="o", ms=4, lw=1.8, label=name)
    if saved_epoch:
        ax.axvline(saved_epoch, color="0.5", ls=":", lw=1.2)
        ax.annotate(f"saved epoch {saved_epoch}", xy=(saved_epoch, ax.get_ylim()[1]),
                    xytext=(4, -10), textcoords="offset points", fontsize=8, color="0.4")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)


def plot_scores(ax, labels: np.ndarray, scores: np.ndarray, eer: float) -> None:
    lo, hi = float(scores.min()), float(scores.max())
    pad = 0.05 * (hi - lo + 1e-9)
    bins = np.linspace(lo - pad, hi + pad, 70)
    for klass in (BONAFIDE, SPOOF):
        mask = labels == klass
        if mask.any():
            ax.hist(scores[mask], bins=bins, density=True, alpha=0.55,
                    color=COLORS[klass], label=f"{LABEL_NAMES[klass]} (n={int(mask.sum())})")
    ax.axvline(0.0, color="k", ls="--", lw=1.2, label="threshold (score = 0)")
    ax.set_xlabel("spoof score  (> 0 predicts spoof)")
    ax.set_ylabel("density")
    eer_text = "n/a" if np.isnan(eer) else f"{eer * 100:.2f}%"
    ax.set_title(f"score distributions  •  EER {eer_text}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)


def tsne_project(x: np.ndarray, max_points: int, perplexity: float):
    """2-D t-SNE embedding of a random subset (returns points + their indices)."""
    from sklearn.manifold import TSNE

    n = len(x)
    keep = (np.random.default_rng(0).choice(n, max_points, replace=False)
            if n > max_points else np.arange(n))
    subset = np.asarray(x[keep], dtype=np.float32)
    perp = float(min(perplexity, max(5.0, (len(subset) - 1) / 4.0)))
    kwargs = dict(n_components=2, init="pca", learning_rate="auto",
                  random_state=0, perplexity=perp)
    try:
        tsne = TSNE(max_iter=750, **kwargs)
    except TypeError:  # sklearn < 1.5 spells it n_iter
        tsne = TSNE(n_iter=750, **kwargs)
    return tsne.fit_transform(subset), keep


def plot_tsne(ax, x: np.ndarray, labels: np.ndarray, predictions: np.ndarray,
              max_points: int, perplexity: float) -> None:
    embedded, keep = tsne_project(x, max_points, perplexity)
    labels_sub = labels[keep]
    wrong = labels_sub != predictions[keep]

    for klass in (BONAFIDE, SPOOF):
        mask = labels_sub == klass
        ax.scatter(embedded[mask, 0], embedded[mask, 1], s=7, alpha=0.45,
                   color=COLORS[klass], label=LABEL_NAMES[klass], linewidths=0)
    if wrong.any():
        ax.scatter(embedded[wrong, 0], embedded[wrong, 1], s=26, facecolors="none",
                   edgecolors="k", linewidths=0.8,
                   label=f"misclassified ({wrong.mean() * 100:.1f}%)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE of the {x.shape[1]}-d features")
    ax.legend(frameon=False, fontsize=9, markerscale=2.5)
    ax.grid(alpha=0.2)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> str:
    args = parse_args(argv)
    exp_dir = experiments.resolve(args.exp, CFG.experiment_root)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else exp_dir / experiments.MODEL_NAME
    ckpt = load_checkpoint(ckpt_path)

    saved = ckpt.get("args", {})
    feature = saved.get("feature", CFG.feature)
    pooling = saved.get("pooling", CFG.pooling)
    model_key = ckpt.get("model", saved.get("model", "svm"))
    saved_epoch = int(ckpt.get("epoch") or 0)

    features_dir = resolve_features_dir(args.features_dir, ckpt)
    x, labels = features_io.load_split(features_dir, args.split)
    if args.limit:
        x, labels = x[: args.limit], labels[: args.limit]
    if len(x) == 0:
        raise SystemExit(f"split {args.split!r} is empty in {features_dir}")

    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(ckpt["std"], dtype=np.float32), 1e-6)
    x_scaled = (x - mean) / std

    device = select_device("cpu")  # tiny linear model; cpu keeps this deterministic
    model = build_model(model_key, input_dim=int(ckpt["input_dim"]),
                        **ckpt.get("model_kwargs", {})).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scores = score_batches(model, x_scaled, device, batch_size=512)
    labels = np.asarray(labels, dtype=np.int64)
    eer = compute_eer(labels, scores)

    history_path = Path(args.history) if args.history else exp_dir / experiments.HISTORY_NAME
    history = (
        json.loads(history_path.read_text())
        if history_path.exists()
        else find_history(args.history, ckpt_path, ckpt)
    )
    rows = (history or {}).get("history") or []

    if rows:  # learning curves only when train.py left a history file
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
        ax_acc, ax_eer = axes[0, 0], axes[0, 1]
        ax_score, ax_embed = axes[1, 0], axes[1, 1]
        epochs = [row["epoch"] for row in rows]
        plot_curve(
            ax_acc, epochs,
            [("train", [r.get("train_acc") for r in rows]),
             ("validation", [r.get("val_acc") for r in rows])],
            "accuracy (%)", "accuracy per epoch", saved_epoch, as_percent=True,
        )
        eer_curve = [r.get("val_eer") for r in rows]
        plot_curve(
            ax_eer, epochs, [("validation", eer_curve)],
            "EER (%)", "validation EER per epoch", saved_epoch, as_percent=True,
        )
        finite = [(i, v) for i, v in enumerate(eer_curve) if v is not None]
        if finite:
            best_i, best_v = min(finite, key=lambda pair: pair[1])
            ax_eer.scatter([epochs[best_i]], [best_v * 100], s=70, facecolor="none",
                            edgecolor="k", linewidths=1.4, zorder=5)
            ax_eer.annotate(f"best {best_v * 100:.2f}% (epoch {epochs[best_i]})",
                             xy=(epochs[best_i], best_v * 100), xytext=(6, 10),
                             textcoords="offset points", fontsize=8)
    else:
        fig, (ax_score, ax_embed) = plt.subplots(1, 2, figsize=(13.5, 5.2))
        print("note: no history-*.json found; plotting scores and t-SNE only")

    predictions = (scores > 0).astype(np.int64)
    plot_scores(ax_score, labels, scores, eer)
    plot_tsne(ax_embed, x_scaled, labels, predictions,
              args.max_scatter, args.perplexity)

    eer_text = "n/a" if np.isnan(eer) else f"EER {eer * 100:.2f}%"
    test_metrics = experiments.read_metrics(exp_dir, "test")
    if test_metrics and test_metrics.get("eer") is not None:
        eer_text += f"  ·  test EER {test_metrics['eer'] * 100:.2f}%"
    fig.suptitle(
        f"{type(model).__name__} · {feature} + {pooling} · {args.split}: "
        f"{len(labels)} utterances ({int((labels == BONAFIDE).sum())} bonafide / "
        f"{int((labels == SPOOF).sum())} spoof) · {eer_text}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94 if rows else 0.90))

    out = Path(args.out) if args.out else exp_dir / experiments.PLOT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"experiment : {exp_dir}")
    print(f"checkpoint : {ckpt_path}")
    print(f"features   : {features_dir}/{args.split}.npz")
    print(f"history    : {'found' if history else 'missing'}")
    print(f"EER        : {eer_text}")
    print(f"figure     : {out}")
    if args.show:
        plt.show()
    plt.close(fig)
    return str(out)


if __name__ == "__main__":
    main()
