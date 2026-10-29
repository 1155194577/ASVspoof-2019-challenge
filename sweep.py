"""Compare front-ends / learning rates on the same pre-extracted features.

Trains the SVM in-process for each configuration on the ``.npz`` vectors, then
draws the EER-per-epoch curves and a summary bar chart in one figure.

    conda activate ml
    python sweep.py                                # mfcc vs lfcc, lr 0.05 vs 0.01
    python sweep.py --lrs 0.05 0.02 0.01 --epochs 40
    python sweep.py --features lfcc --out figures/lfcc-sweep.png

Everything runs on CPU in seconds (40 epochs x 25k utterances ≈ 8 s per config),
so this is a cheap way to pick a front-end/lr before burning cluster time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import features_io
from model import build_model, hinge_loss
from train import (
    class_weights_from_labels,
    compute_eer,
    make_loader,
    set_seed,
    standardise,
    train_one_epoch,
)

LABEL_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep features / learning rates for the SVM baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--features", nargs="+", default=["mfcc", "lfcc"],
                        choices=["mfcc", "lfcc", "logmel"])
    parser.add_argument("--pooling", default="mean_std")
    parser.add_argument("--lrs", nargs="+", type=float, default=[0.05, 0.01])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--features-root", default="features")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="figures/sweep.png")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args(argv)


def load_pooled(features_root: str, feature: str, pooling: str, split: str
                ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    directory = Path(features_root) / f"{feature}-{pooling}"
    try:
        return features_io.load_split(directory, split)
    except FileNotFoundError:
        print(f"  ! {directory}/{split}.npz missing -- skipping {feature}")
        return None


def run_config(x_train, y_train, x_eval, y_eval, lr: float, args) -> List[Dict]:
    """One SGD run; returns per-epoch metrics."""
    set_seed(args.seed)
    x_tr, x_ev, _, _ = standardise(x_train, x_eval)
    train_loader = make_loader(x_tr, y_train, args.batch_size, True, args.seed, 0)
    eval_loader = make_loader(x_ev, y_eval, args.batch_size * 4, False, args.seed, 0)

    model = build_model("svm", input_dim=x_tr.shape[1])
    optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    weights = class_weights_from_labels(y_train, "balanced")

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimiser,
                                        torch.device("cpu"), weights, args.margin)
        eval_metrics = evaluate_cpu(model, eval_loader, args.margin)
        history.append({"epoch": epoch, **train_metrics, **eval_metrics})
    return history


@torch.no_grad()
def evaluate_cpu(model, loader, margin: float) -> Dict[str, float]:
    model.eval()
    total, correct, seen = 0.0, 0, 0
    scores, labels = [], []
    for xb, yb in loader:
        logits = model(xb)
        loss = hinge_loss(logits, yb, margin=margin)
        batch = yb.size(0)
        total += float(loss) * batch
        correct += int((logits.argmax(dim=1) == yb).sum())
        seen += batch
        scores.append(model.decision_function(xb).float().numpy())
        labels.append(yb.numpy())
    y_true, y_score = np.concatenate(labels), np.concatenate(scores)
    return {
        "val_loss": total / max(seen, 1),
        "val_acc": correct / max(seen, 1),
        "val_eer": compute_eer(y_true, y_score),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    results: List[Dict] = []

    cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for feature in args.features:
        train_data = load_pooled(args.features_root, feature, args.pooling, args.train_split)
        eval_data = load_pooled(args.features_root, feature, args.pooling, args.eval_split)
        if train_data is None or eval_data is None:
            continue
        cache[feature] = (train_data, eval_data)
        print(f"{feature:<6} train {train_data[0].shape}  eval {eval_data[0].shape}")

    for feature, (train_data, eval_data) in cache.items():
        for lr in args.lrs:
            label = f"{feature}  lr={lr:g}"
            print(f"running {label} ...", end="", flush=True)
            history = run_config(*train_data, *eval_data, lr=lr, args=args)
            eers = np.array([h["val_eer"] for h in history]) * 100
            results.append({
                "label": label,
                "history": history,
                "best": float(np.nanmin(eers)),
                "best_epoch": int(np.nanargmin(eers)) + 1,
                "last10": float(np.nanmean(eers[-10:])),
                "median": float(np.nanmedian(eers)),
            })
            print(f" best {results[-1]['best']:.2f}%  last10 {results[-1]['last10']:.2f}%")

    if not results:
        raise SystemExit("nothing to plot -- run prepare_features.py first")

    fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(15, 5.6),
                                           gridspec_kw={"width_ratios": [1.35, 1]})
    for i, res in enumerate(results):
        eers = np.array([h["val_eer"] for h in res["history"]]) * 100
        ax_curve.plot(np.arange(1, len(eers) + 1), eers, lw=1.4, alpha=0.9,
                      color=LABEL_COLORS[i % len(LABEL_COLORS)], label=res["label"])
    ax_curve.set_xlabel("epoch")
    ax_curve.set_ylabel("validation EER (%)")
    ax_curve.set_title(f"EER per epoch ({args.epochs} epochs, momentum {args.momentum})")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(frameon=False, fontsize=8, ncol=2)

    order = sorted(results, key=lambda r: r["last10"])
    positions = np.arange(len(order))
    ax_bar.barh(positions - 0.2, [r["last10"] for r in order], height=0.38,
                color=[LABEL_COLORS[results.index(r) % len(LABEL_COLORS)] for r in order],
                label="mean of last 10 epochs")
    ax_bar.barh(positions + 0.2, [r["best"] for r in order], height=0.38,
                color=[LABEL_COLORS[results.index(r) % len(LABEL_COLORS)] for r in order],
                alpha=0.45, hatch="//", label="best epoch")
    ax_bar.set_yticks(positions)
    ax_bar.set_yticklabels([r["label"] for r in order], fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("validation EER (%) -- lower is better")
    ax_bar.set_title("summary")
    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("SVM baseline: front-end / learning-rate comparison", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("\nconfig                         best EER   last-10   median")
    for res in sorted(results, key=lambda r: r["last10"]):
        print(f"{res['label']:<28} {res['best']:7.2f}%  {res['last10']:7.2f}%  {res['median']:6.2f}%")
    print(f"\nfigure: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
