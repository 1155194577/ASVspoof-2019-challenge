"""Project-wide defaults for training and evaluation.

Everything that ``train.py`` / ``evaluate.py`` expose on the command line is
declared here, so an experiment can be described in one place. The CLI is
built with these values as defaults: any flag you pass overrides the dataclass.

    >>> from config import TrainingConfig
    >>> cfg = TrainingConfig()
    >>> cfg.epochs, cfg.lr, cfg.batch_size
    (3, 0.05, 64)

Changing a value here changes the default for every script:

    # config.py
    @dataclass
    class TrainingConfig:
        feature: str = "lfcc"
        epochs: int = 10

or override per run: ``python train.py --epochs 10 --feature lfcc``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Defaults shared by ``train.py`` and ``evaluate.py``."""

    # ---- model -------------------------------------------------------------
    # TODO: "hubert" / "wavlm" / "wav2vec2" backbones, see model.build_model
    model: str = "svm"

    # ---- optimisation ------------------------------------------------------
    epochs: int = 3
    # Hinge-loss SGD needs a step size ~10x larger than cross-entropy's
    # 1e-3 default to move the margin in 3 epochs.
    lr: float = 0.05
    batch_size: int = 64
    momentum: float = 0.9
    weight_decay: float = 1e-4       # plays the role of 1/C in an SVM
    margin: float = 1.0
    class_weight: str = "balanced"   # bonafide is the minority class

    # ---- front-end ---------------------------------------------------------
    feature: str = "mfcc"            # "mfcc" | "lfcc" | "logmel"
    pooling: str = "mean_std"        # utterance-level vector for SVM-style models
    n_mfcc: int = 20
    n_lfcc: int = 20
    n_mels: int = 40

    # ---- data --------------------------------------------------------------
    train_split: str = "train"
    eval_split: str = "validation"
    limit: Optional[int] = None      # keep only N utterances per split
    streaming: bool = False          # stream from the Hub instead of downloading
    cache_dir: str = "cache"         # where featurised splits are persisted
    num_workers: int = 0
    # Folder written by prepare_features.py; when set, train/evaluate load the
    # .npz vectors instead of touching the dataset (no `datasets` needed).
    features_dir: Optional[str] = None

    # ---- runtime -----------------------------------------------------------
    device: str = "auto"             # "auto" | "cpu" | "mps" | "cuda"
    seed: int = 0
    checkpoint_dir: str = "checkpoints"

    @property
    def checkpoint_name(self) -> str:
        """File name used for the best checkpoint of this configuration."""
        return f"{self.model}-{self.feature}-{self.pooling}.pt"

    @property
    def checkpoint_path(self) -> str:
        return f"{self.checkpoint_dir}/{self.checkpoint_name}"
