"""Evaluate a trained checkpoint on a split (EER, accuracy, optional CSV).

    conda activate ml
    python evaluate.py                              # validation, newest default checkpoint
    python evaluate.py --split test                 # official ASVspoof LA eval split
    python evaluate.py --checkpoint checkpoints/svm-lfcc-mean_std.pt --out scores.csv

The checkpoint written by ``train.py`` carries everything needed to reproduce
the pipeline: the front-end arguments, the standardisation statistics
(``mean``/``std``) and the model class key, so no extra bookkeeping is needed.

Metrics follow the usual anti-spoofing convention: ``spoof`` is the positive
class, and **EER** is what matters (the corpus is heavily spoof-heavy, so
accuracy is misleading).
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

import features_io
from config import TrainingConfig
from model import build_model
from train import compute_eer, select_device

CFG = TrainingConfig()
LABEL_NAMES = ("bonafide", "spoof")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint (EER)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=None,
                        help=f"defaults to {CFG.checkpoint_path}")
    parser.add_argument("--split", default=CFG.eval_split,
                        choices=["train", "validation", "test"],
                        help="split to score ('test' is the official eval set)")
    parser.add_argument("--limit", type=int, default=CFG.limit,
                        help="keep only the first N utterances (debugging)")
    parser.add_argument("--streaming", action="store_true", default=CFG.streaming,
                        help="stream rows from the Hub instead of downloading the split")
    parser.add_argument("--cache-dir", default=CFG.cache_dir,
                        help="where featurised splits are persisted ('' disables)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="inference batch size")
    parser.add_argument("--device", default=CFG.device, choices=["auto", "cpu", "mps", "cuda"],
                        help="'auto' picks cuda > mps > cpu")
    parser.add_argument("--features-dir", default=CFG.features_dir,
                        help="pre-extracted .npz folder from prepare_features.py")
    parser.add_argument("--out", default=None,
                        help="optional CSV with per-utterance scores")
    return parser.parse_args(argv)


def load_checkpoint(path: Path) -> Dict:
    """Load a checkpoint written by ``train.py`` (tensors + plain values only)."""
    if not path.exists():
        raise SystemExit(
            f"checkpoint not found: {path}\n"
            "train first with `python train.py`, or pass --checkpoint"
        )
    try:
        # our checkpoints hold tensors + plain values only, so the safe path works
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only kwarg
        return torch.load(path, map_location="cpu")
    except Exception:  # older checkpoints pickling numpy objects
        warnings.warn(
            "checkpoint needs the legacy unpickler (weights_only=False)",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


@torch.no_grad()
def score_batches(
    model: torch.nn.Module, features: np.ndarray, device: torch.device, batch_size: int
) -> np.ndarray:
    """Spoof score for every row, computed batch by batch."""
    model.eval()
    scores = []
    for start in range(0, len(features), max(1, batch_size)):
        xb = torch.from_numpy(features[start : start + batch_size]).to(device)
        logits = model(xb)
        if hasattr(model, "decision_function"):
            batch_scores = model.decision_function(xb)
        else:
            batch_scores = logits[:, 1]
        scores.append(batch_scores.float().cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    device = select_device(args.device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(CFG.checkpoint_path)
    ckpt = load_checkpoint(ckpt_path)

    saved_args = dict(ckpt.get("args", {}))
    model_key = ckpt.get("model", saved_args.get("model", "svm"))
    feature = saved_args.get("feature", CFG.feature)
    pooling = saved_args.get("pooling", CFG.pooling)
    input_dim = int(ckpt["input_dim"])

    print(f"checkpoint  : {ckpt_path}")
    print(f"trained     : {ckpt.get('model_name', model_key)} on {feature} + {pooling}, "
          f"input_dim={input_dim}, epoch={ckpt.get('epoch')}, "
          f"saved EER={_fmt_eer(ckpt.get('eer'))}")
    print(f"split       : {args.split} | device: {device}")

    names = None
    if args.features_dir:
        print(f"source      : pre-extracted {args.features_dir}")
        features, labels, names = features_io.load_split(
            args.features_dir, args.split, with_names=True
        )
        if names is not None and len(names) == 0:
            names = None
    else:
        from dataPreprocessor import ASVspoofPreprocessor

        pre = ASVspoofPreprocessor(
            feature=feature,
            n_mfcc=saved_args.get("n_mfcc", CFG.n_mfcc),
            n_lfcc=saved_args.get("n_lfcc", CFG.n_lfcc),
            n_mels=saved_args.get("n_mels", CFG.n_mels),
            normalize="none",  # must match training
            streaming=args.streaming,
            limit=args.limit,
            cache_dir=args.cache_dir or None,
            splits=(args.split,),
        )
        features, labels = pre.get_sklearn_dataset(args.split, pooling=pooling)
        if args.out and not args.streaming:
            names = pre.prepare(args.split)["audio_file_name"]

    if len(features) == 0:
        raise SystemExit(f"split {args.split!r} is empty -- check --limit/--streaming")

    # standardisation must be the one fitted on the training split
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    features = (features - mean) / np.maximum(std, 1e-6)

    model = build_model(model_key, input_dim=input_dim).to(device)
    model.load_state_dict(ckpt["model_state"])

    scores = score_batches(model, features, device, args.batch_size)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = (scores > 0).astype(np.int64)

    eer = compute_eer(labels, scores)
    accuracy = float((predictions == labels).mean())
    counts = np.bincount(labels, minlength=2)

    print(f"\nutterances  : {len(labels)} ({counts[0]} bonafide / {counts[1]} spoof)")
    for klass in (0, 1):
        mask = labels == klass
        if mask.any():
            print(f"  {LABEL_NAMES[klass]:>8} score: mean {scores[mask].mean():+.3f} "
                  f"std {scores[mask].std():.3f}")
    print(f"\nEER         : {_fmt_eer(eer)}")
    print(f"accuracy    : {accuracy * 100:.2f}% (threshold 0)")
    print(f"TPR / FPR   : {_tpr_fpr(labels, scores)}")

    if args.out:
        if names is None or len(names) == 0:
            raise SystemExit(
                "--out needs per-utterance file names; they are stored only when "
                "prepare_features.py runs without --streaming (or drop --streaming)"
            )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["audio_file_name", "label", "label_name", "score", "prediction"])
            for name, label, score, pred in zip(names, labels, scores, predictions):
                writer.writerow([name, int(label), LABEL_NAMES[int(label)],
                                 f"{score:.6f}", int(pred)])
        print(f"scores      : {out_path}")

    return {
        "eer": eer,
        "accuracy": accuracy,
        "n": int(len(labels)),
        "checkpoint": str(ckpt_path),
    }


def _fmt_eer(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _tpr_fpr(labels: np.ndarray, scores: np.ndarray) -> str:
    predictions = (scores > 0).astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    if not pos.any() or not neg.any():
        return "n/a (single-class split)"
    tpr = float((predictions[pos] == 1).mean())
    fpr = float((predictions[neg] == 1).mean())
    return f"{tpr * 100:.1f}% / {fpr * 100:.1f}%"


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, float]:
    return evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
