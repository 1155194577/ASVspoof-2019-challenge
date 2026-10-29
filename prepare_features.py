"""Pre-extract pooled feature vectors, once, before moving to a GPU server.

Running this locally writes one small ``.npz`` per split (a few MB each) so the
GPU machine needs neither the 7 GB ASVspoof download nor ``datasets`` /
``soundfile`` -- only ``numpy`` and ``torch``.

    conda activate ml
    python prepare_features.py                          # train + validation
    python prepare_features.py --splits train validation test   # + official test
    python prepare_features.py --feature lfcc           # features/lfcc-mean_std/

Then copy the folder over and train there:

    scp -r features/mfcc-mean_std gpu:~/asvspoof/features/
    # on the GPU box
    python train.py    --features-dir features/mfcc-mean_std
    python evaluate.py --features-dir features/mfcc-mean_std --split test

The first local run has to featurise every file (that is the expensive part);
``--cache-dir`` keeps the pooled vectors so re-running is instant.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import features_io
from config import TrainingConfig
from dataPreprocessor import ASVspoofPreprocessor

CFG = TrainingConfig()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-extract pooled features into .npz files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--feature", default=CFG.feature, choices=["mfcc", "lfcc", "logmel"],
                        help="front-end feature type")
    parser.add_argument("--pooling", default=CFG.pooling,
                        choices=["mean", "std", "mean_std", "min_max", "flat"],
                        help="how the time axis is collapsed")
    parser.add_argument("--n-mfcc", type=int, default=CFG.n_mfcc)
    parser.add_argument("--n-lfcc", type=int, default=CFG.n_lfcc)
    parser.add_argument("--n-mels", type=int, default=CFG.n_mels)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"],
                        choices=["train", "validation", "test"],
                        help="splits to extract ('test' is the 4 GB one)")
    parser.add_argument("--out", default=None,
                        help="output folder (defaults to features/<feature>-<pooling>)")
    parser.add_argument("--limit", type=int, default=CFG.limit,
                        help="keep only the first N utterances (debugging)")
    parser.add_argument("--streaming", action="store_true", default=CFG.streaming,
                        help="stream from the Hub instead of downloading a split")
    parser.add_argument("--cache-dir", default=CFG.cache_dir,
                        help="where to persist pooled splits ('' disables)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = parse_args(argv)
    out_dir = Path(args.out) if args.out else features_io.default_dir(args.feature, args.pooling)

    pre = ASVspoofPreprocessor(
        feature=args.feature,
        n_mfcc=args.n_mfcc,
        n_lfcc=args.n_lfcc,
        n_mels=args.n_mels,
        normalize="none",  # the pooled vectors must stay un-normalised
        streaming=args.streaming,
        limit=args.limit,
        cache_dir=args.cache_dir or None,
        splits=tuple(args.splits),
    )

    print(f"front-end   : {args.feature} + {args.pooling} pooling")
    print(f"output      : {out_dir}")
    print(f"splits      : {', '.join(args.splits)}\n")

    manifest = {
        "dataset": pre.config.dataset_name,
        "feature": args.feature,
        "pooling": args.pooling,
        "normalize": "none",
        "n_mfcc": args.n_mfcc,
        "n_lfcc": args.n_lfcc,
        "n_mels": args.n_mels,
        "sample_rate": pre.config.sample_rate,
        "duration": pre.config.duration,
        "hop_length": pre.config.hop_length,
        "n_fft": pre.config.n_fft,
        "dim": None,
        "splits": {},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    for split in args.splits:
        start = time.time()
        features, labels = pre.get_sklearn_dataset(split, pooling=args.pooling)
        if len(features) == 0:
            print(f"{split:<11} skipped (empty)")
            continue

        names = systems = None
        if not args.streaming:
            dataset = pre.prepare(split, pooling=args.pooling)
            names = dataset["audio_file_name"]
            systems = dataset["system_id"]

        path = features_io.save_split(
            out_dir, split, features, labels, names=names, systems=systems
        )
        counts = np.bincount(labels, minlength=2)
        manifest["dim"] = int(features.shape[1])
        manifest["splits"][split] = {
            "n": int(len(labels)),
            "bonafide": int(counts[0]),
            "spoof": int(counts[1]),
            "dim": int(features.shape[1]),
            "file": path.name,
            "bytes": path.stat().st_size,
        }
        print(
            f"{split:<11} {len(labels):6d} x {features.shape[1]:3d} "
            f"({counts[0]} bonafide / {counts[1]} spoof) -> {path.name} "
            f"[{path.stat().st_size / 1e6:.1f} MB, {time.time() - start:.0f}s]"
        )

    manifest_path = features_io.write_manifest(out_dir, manifest)
    total = sum(item["bytes"] for item in manifest["splits"].values())
    print(f"\nmanifest    : {manifest_path}")
    print(f"total size  : {total / 1e6:.1f} MB -- copy '{out_dir}' to the GPU server")
    print(f"next        : python train.py --features-dir {out_dir}")
    return manifest


if __name__ == "__main__":
    main()
