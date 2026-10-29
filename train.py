"""Training loop for the ASVspoof 2019 LA baselines.

Trains a PyTorch model (by default the linear SVM from :mod:`model`) on
pooled MFCC/LFCC statistics produced by :mod:`dataPreprocessor`, for a few
epochs, reporting **EER** on the validation split every epoch.

Typical use
-----------
    conda activate ml
    python train.py                             # defaults come from config.py
    python train.py --epochs 10 --feature lfcc --pooling mean_std
    python train.py --limit 400 --streaming     # quick smoke test, no big download

The first run has to pull the split from the Hub (~1.4 GB for ``train``) and
featurise every file; ``--cache-dir`` (default ``cache/``) makes later runs
instant because the (C, T) maps are reused instead of recomputed.

Notes
-----
* Features are pooled **before** training, so each utterance is one vector
  (40 dims for MFCC + ``mean_std``). Standardisation is fit on the training
  split only and stored in the checkpoint, so ``evaluate.py`` can reuse it.
* ``normalize="none"`` in the preprocessor is deliberate: per-utterance CMVN
  would make the pooled mean/std identical for every utterance.
* Use EER, not accuracy: the corpus is heavily spoof-heavy.
* Validation decides the checkpoint (``history.json``); test is scored **once**
  with that frozen model and stored as ``test.json``. Scoring test every epoch
  would leak it into model selection.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import experiments
import features_io
from config import TrainingConfig
from model import MODEL_REGISTRY, build_model, hinge_loss

N_CLASSES = 2
LABEL_NAMES = ("bonafide", "spoof")
CFG = TrainingConfig()   # defaults come from config.py


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str = "auto") -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Equal error rate, with ``spoof = 1`` as the positive class.

    ``scores`` should be the spoof score (higher = more likely spoof).
    Returns ``nan`` if only one class is present.
    """
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")  # highest score first
    y = labels[order]
    tpr = np.cumsum(y == 1) / n_pos
    fpr = np.cumsum(y == 0) / n_neg
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def standardise(
    train_x: np.ndarray, other: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Zero-mean / unit-variance scaling fit on ``train_x`` only."""
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    scaled = (train_x - mean) / std
    scaled_other = None if other is None else (other - mean) / std
    return scaled.astype(np.float32), (
        None if scaled_other is None else scaled_other.astype(np.float32)
    ), mean, std


def make_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int, num_workers: int
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )


def write_history(path: Path, args: argparse.Namespace, history: list) -> None:
    """Persist per-epoch metrics so plot.py can draw the learning curves."""
    def clean(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None  # NaN has no JSON representation
        return value

    payload = {
        "args": vars(args),
        "history": [{k: clean(v) for k, v in row.items()} for row in history],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def class_weights_from_labels(y: np.ndarray, mode: str) -> Optional[torch.Tensor]:
    """Inverse-frequency weights, like ``class_weight="balanced"`` in sklearn."""
    if mode == "none":
        return None
    counts = np.bincount(y.astype(np.int64), minlength=N_CLASSES).astype(np.float64)
    weights = counts.sum() / (N_CLASSES * np.maximum(counts, 1.0))
    weights[counts == 0] = 1.0  # absent classes never contribute to the loss
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Epoch loop
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
    margin: float = 1.0,
    loss_name: str = "hinge",
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0
    criterion = None
    if loss_name == "ce":
        weight = None if class_weights is None else class_weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)
        logits = model(xb)
        if criterion is not None:
            loss = criterion(logits, yb)
        else:
            weight = None if class_weights is None else class_weights.to(device)[yb]
            loss = hinge_loss(logits, yb, margin=margin, weight=weight)
        loss.backward()
        optimiser.step()

        batch = yb.size(0)
        total_loss += float(loss.detach()) * batch
        correct += int((logits.argmax(dim=1) == yb).sum())
        seen += batch

    return {
        "loss": total_loss / max(seen, 1),
        "acc": correct / max(seen, 1),
        "n": seen,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    margin: float = 1.0,
    loss_name: str = "hinge",
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    scores, labels = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        if loss_name == "ce":
            loss = torch.nn.functional.cross_entropy(logits, yb)
        else:
            loss = hinge_loss(logits, yb, margin=margin)

        batch = yb.size(0)
        total_loss += float(loss) * batch
        correct += int((logits.argmax(dim=1) == yb).sum())
        seen += batch

        if hasattr(model, "decision_function"):
            scores.append(model.decision_function(xb).detach().float().cpu().numpy())
        else:
            scores.append(logits[:, 1].detach().float().cpu().numpy())
        labels.append(yb.detach().cpu().numpy())

    y_true = np.concatenate(labels) if labels else np.empty(0, dtype=np.int64)
    y_score = np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)
    return {
        "loss": total_loss / max(seen, 1),
        "acc": correct / max(seen, 1),
        "eer": compute_eer(y_true, y_score),
        "n": seen,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an SVM baseline on pooled MFCC/LFCC features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data
    parser.add_argument("--model", default=CFG.model, choices=sorted(MODEL_REGISTRY),
                        help="model registry key from model.build_model")
    parser.add_argument("--feature", default=CFG.feature, choices=["mfcc", "lfcc", "logmel"],
                        help="front-end feature type")
    parser.add_argument("--gamma", type=float, default=CFG.gamma,
                        help="RBF bandwidth for --model svm_rff (default 1/input_dim)")
    parser.add_argument("--n-features", type=int, default=CFG.n_features,
                        help="random Fourier features for --model svm_rff")
    parser.add_argument("--pooling", default=CFG.pooling,
                        choices=["mean", "std", "mean_std", "min_max", "flat"],
                        help="how the time axis is collapsed into one vector")
    parser.add_argument("--n-mfcc", type=int, default=CFG.n_mfcc,
                        help="MFCC coefficients (2x this with mean_std pooling)")
    parser.add_argument("--n-lfcc", type=int, default=CFG.n_lfcc,
                        help="LFCC coefficients")
    parser.add_argument("--n-mels", type=int, default=CFG.n_mels,
                        help="mel/linear filters for logmel/LFCC front-ends")
    parser.add_argument("--train-split", default=CFG.train_split,
                        help="split used for fitting")
    parser.add_argument("--eval-split", default=CFG.eval_split,
                        help="split used for per-epoch EER")
    parser.add_argument("--limit", type=int, default=CFG.limit,
                        help="keep only the first N utterances of each split (debugging)")
    parser.add_argument("--streaming", action="store_true", default=CFG.streaming,
                        help="stream rows from the Hub instead of downloading the split")
    parser.add_argument("--cache-dir", default=CFG.cache_dir,
                        help="where to persist featurised splits ('' disables)")
    parser.add_argument("--num-workers", type=int, default=CFG.num_workers,
                        help="DataLoader worker processes")

    # optimisation
    parser.add_argument("--epochs", type=int, default=CFG.epochs,
                        help="number of passes over the training split")
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--lr", type=float, default=CFG.lr,
                        help="SGD learning rate")
    parser.add_argument("--momentum", type=float, default=CFG.momentum)
    parser.add_argument("--weight-decay", type=float, default=CFG.weight_decay,
                        help="L2 penalty; acts like 1/C of an SVM")
    parser.add_argument("--margin", type=float, default=CFG.margin,
                        help="hinge margin")
    parser.add_argument("--loss", default=CFG.loss, choices=["hinge", "ce"],
                        help="training objective")
    parser.add_argument("--optimizer", default=CFG.optimizer, choices=["sgd", "adam"],
                        help="optimiser")
    parser.add_argument("--class-weight", default=CFG.class_weight,
                        choices=["balanced", "none"],
                        help="inverse-frequency weights for the loss")

    # runtime
    parser.add_argument("--device", default=CFG.device, choices=["auto", "cpu", "mps", "cuda"],
                        help="'auto' picks cuda > mps > cpu")
    parser.add_argument("--seed", type=int, default=CFG.seed,
                        help="seed for shuffling and weight init")
    parser.add_argument("--features-dir", default=CFG.features_dir,
                        help="pre-extracted .npz folder from prepare_features.py "
                             "(skips the dataset entirely)")
    parser.add_argument("--exp-dir", default=None,
                        help="experiment folder (default: auto-numbered experiment_log/exp_NNN)")
    parser.add_argument("--exp-root", default=CFG.experiment_root,
                        help="root for auto-numbered experiments")
    parser.add_argument("--tag", default=None,
                        help="free-form label stored in config.json")
    parser.add_argument("--eval-test", action=argparse.BooleanOptionalAction,
                        default=CFG.eval_test,
                        help="score the TEST split once after training (never during)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, float]:
    args = parse_args(argv)
    set_seed(args.seed)
    device = select_device(args.device)

    # every run gets experiment_log/exp_NNN/ with config + history + weights
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        exp_dir = experiments.create(args.exp_root, tag=args.tag, config=vars(args))
    if not (exp_dir / experiments.CONFIG_NAME).exists():
        experiments.write_config(exp_dir, vars(args), tag=args.tag)
    print(f"experiment  : {exp_dir}")

    print(f"device      : {device}")
    print(f"front-end   : {args.feature} + {args.pooling} pooling, normalize='none'")
    print(f"model       : {args.model}")
    print(f"optimiser   : SGD(lr={args.lr}, momentum={args.momentum}, "
          f"weight_decay={args.weight_decay}) for {args.epochs} epochs, "
          f"batch {args.batch_size}")

    print(f"\nloading {args.train_split!r} and {args.eval_split!r} features ...")
    if args.features_dir:
        # no `datasets` import needed: the vectors come from prepare_features.py
        manifest = features_io.read_manifest(args.features_dir)
        if manifest:
            if (manifest.get("feature"), manifest.get("pooling")) != (
                args.feature, args.pooling
            ):
                print(f"note        : {args.features_dir} holds "
                      f"{manifest.get('feature')}/{manifest.get('pooling')}, using that")
            args.feature = manifest.get("feature", args.feature)
            args.pooling = manifest.get("pooling", args.pooling)
        print(f"source      : pre-extracted {args.features_dir}")
        # the manifest may have corrected feature/pooling -- keep config.json honest
        experiments.write_config(exp_dir, vars(args), tag=args.tag)
        train_x, train_y = features_io.load_split(args.features_dir, args.train_split)
        eval_x, eval_y = features_io.load_split(args.features_dir, args.eval_split)
    else:
        from dataPreprocessor import ASVspoofPreprocessor

        pre = ASVspoofPreprocessor(
            feature=args.feature,
            n_mfcc=args.n_mfcc,
            n_lfcc=args.n_lfcc,
            n_mels=args.n_mels,
            normalize="none",  # pooling needs un-normalised features
            streaming=args.streaming,
            limit=args.limit,
            cache_dir=args.cache_dir or None,
            splits=(args.train_split, args.eval_split),
        )
        train_x, train_y = pre.get_sklearn_dataset(args.train_split, pooling=args.pooling)
        eval_x, eval_y = pre.get_sklearn_dataset(args.eval_split, pooling=args.pooling)
        print(f"source      : extracted on the fly (cache: {args.cache_dir or 'off'})")

    if len(train_x) == 0 or len(eval_x) == 0:
        raise SystemExit("one of the splits is empty -- check --limit/--streaming")

    train_x, eval_x, mean, std = standardise(train_x, eval_x)
    print(f"train       : {train_x.shape[0]} utterances x {train_x.shape[1]} dims "
          f"({np.bincount(train_y, minlength=N_CLASSES).tolist()} = bonafide/spoof)")
    print(f"eval        : {eval_x.shape[0]} utterances x {eval_x.shape[1]} dims "
          f"({np.bincount(eval_y, minlength=N_CLASSES).tolist()} = bonafide/spoof)")

    train_loader = make_loader(
        train_x, train_y, args.batch_size, True, args.seed, args.num_workers
    )
    eval_loader = make_loader(
        eval_x, eval_y, args.batch_size, False, args.seed, args.num_workers
    )

    model_kwargs: Dict[str, Any] = {}
    if args.model == "svm_rff":
        model_kwargs["n_features"] = args.n_features
        if args.gamma is not None:
            model_kwargs["gamma"] = args.gamma
    elif args.model == "cnn2d":
        n_coeffs = {"lfcc": args.n_lfcc, "mfcc": args.n_mfcc, "logmel": args.n_mels}[args.feature]
        if train_x.shape[1] % n_coeffs:
            raise SystemExit(
                f"--model cnn2d needs flat pooling: {train_x.shape[1]} dims is not "
                f"divisible by {n_coeffs} coefficients"
            )
        model_kwargs["shape"] = (n_coeffs, train_x.shape[1] // n_coeffs)
    model = build_model(args.model, input_dim=train_x.shape[1], **model_kwargs).to(device)
    print(f"architecture: {model}")
    if args.optimizer == "adam":
        optimiser = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimiser = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    class_weights = class_weights_from_labels(train_y, args.class_weight)
    if class_weights is not None:
        print(f"class weight: {class_weights.tolist()} (bonafide, spoof)")

    ckpt_path = exp_dir / experiments.MODEL_NAME
    history_path = exp_dir / experiments.HISTORY_NAME

    best_eer = float("nan")  # stays NaN if a split has only one class
    best_epoch = 0
    history = []

    def save_checkpoint(epoch: int, eer: float) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "model": args.model,                 # registry key, e.g. "svm"
                "model_kwargs": model_kwargs,        # needed to rebuild svm_rff
                "model_name": type(model).__name__,
                "input_dim": int(train_x.shape[1]),
                "num_classes": N_CLASSES,
                # tensors (not numpy) keep `torch.load(...)` on the
                # safest pickling path
                "mean": torch.from_numpy(mean),
                "std": torch.from_numpy(std),
                "args": vars(args),
                "epoch": epoch,
                "eer": eer,
            },
            ckpt_path,
        )

    print()
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimiser, device, class_weights, args.margin, args.loss
        )
        eval_metrics = evaluate(model, eval_loader, device, args.margin, args.loss)
        seconds = time.time() - start

        eer = eval_metrics["eer"]
        eer_text = "n/a" if np.isnan(eer) else f"{eer * 100:6.2f}%"
        print(
            f"epoch {epoch}/{args.epochs} | train loss {train_metrics['loss']:.4f} "
            f"acc {train_metrics['acc'] * 100:5.1f}% | "
            f"val loss {eval_metrics['loss']:.4f} acc {eval_metrics['acc'] * 100:5.1f}% "
            f"EER {eer_text} | {seconds:5.1f}s"
        )
        history.append(
            {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in eval_metrics.items()},
            }
        )
        write_history(history_path, args, history)

        if np.isnan(eer):
            if best_epoch == 0:  # degenerate split: at least keep epoch 1
                save_checkpoint(epoch, eer)
        elif np.isnan(best_eer) or eer < best_eer:
            best_eer, best_epoch = eer, epoch
            save_checkpoint(epoch, eer)

    # ---- one-shot TEST evaluation (after the checkpoint is frozen) -------
    test_metrics = None
    if args.eval_test and args.features_dir:
        test_features = Path(args.features_dir) / "test.npz"
        if test_features.exists():
            test_x, test_y = features_io.load_split(args.features_dir, "test")
            test_x = ((test_x - mean) / np.maximum(std, 1e-6)).astype(np.float32)
            test_loader = make_loader(test_x, test_y, args.batch_size * 4, False,
                                      args.seed, args.num_workers)
            best_state = torch.load(ckpt_path, map_location=device,
                                    weights_only=True)["model_state"]
            model.load_state_dict(best_state)
            test_metrics = evaluate(model, test_loader, device, args.margin, args.loss)
            experiments.write_metrics(exp_dir, "test", {
                "split": "test", **test_metrics,
                "checkpoint_epoch": best_epoch,
                "checkpoint_eer": best_eer,
                "n_bonafide": int((test_y == 0).sum()),
                "n_spoof": int((test_y == 1).sum()),
            })
        else:
            print(f"\ntest       : {test_features} not found -- skipped "
                  "(run `python prepare_features.py --splits test`)")

    final_eer = history[-1]["val_eer"]
    best_eer_text = "n/a" if np.isnan(best_eer) else f"{best_eer * 100:.2f}%"
    final_eer_text = "n/a" if np.isnan(final_eer) else f"{final_eer * 100:.2f}%"
    print(f"\nbest checkpoint: {ckpt_path} (epoch {max(best_epoch, 1)}, EER {best_eer_text})")
    print(f"final val EER  : {final_eer_text}")
    print(f"labels         : 0={LABEL_NAMES[0]}, 1={LABEL_NAMES[1]}")
    print(f"history        : {history_path}")
    if test_metrics is not None:
        print(f"test EER       : {test_metrics['eer'] * 100:.2f}% "
              f"({test_metrics['n']} utterances, saved to {exp_dir}/test.json)")
    print(f"experiment dir : {exp_dir}  (config.json, history.json, model.pt)")

    return {
        "experiment": str(exp_dir),
        "best_eer": best_eer,
        "final_val_eer": final_eer,
        "test_eer": None if test_metrics is None else test_metrics["eer"],
        "checkpoint": str(ckpt_path),
        "train_size": int(train_x.shape[0]),
        "dim": int(train_x.shape[1]),
    }


if __name__ == "__main__":
    main()
