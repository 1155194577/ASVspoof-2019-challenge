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
    # "svm" (linear kernel) | "svm_rff" (RBF kernel, random Fourier features)
    # TODO: "hubert" / "wavlm" / "wav2vec2" backbones, see model.build_model
    model: str = "svm"

    # ---- optimisation ------------------------------------------------------
    epochs: int = 50
    # Hinge-loss SGD needs a step size ~10x larger than cross-entropy's
    # 1e-3 default to move the margin in 3 epochs.
    lr: float = 0.01
    batch_size: int = 64
    momentum: float = 0.9
    weight_decay: float = 1e-4       # plays the role of 1/C in an SVM
    margin: float = 1.0
    loss: str = "hinge"              # "hinge" | "ce" (cross-entropy)
    optimizer: str = "sgd"           # "sgd" | "adam"
    gamma: Optional[float] = None    # RBF bandwidth for svm_rff (None -> 1/dim)
    n_features: int = 512            # random Fourier features for svm_rff
    class_weight: str = "balanced"   # bonafide is the minority class

    # ---- front-end ---------------------------------------------------------
    feature: str = "lfcc"            # "mfcc" | "lfcc" | "logmel"
    pooling: str = "mean_std"        # utterance-level vector for SVM-style models
    n_mfcc: int = 20
    n_lfcc: int = 20
    n_mels: int = 40

    # ---- data --------------------------------------------------------------
    train_split: str = "train"
    eval_split: str = "validation"
    eval_test: bool = True          # score test once after training, stored in test.json
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
    experiment_root: str = "experiment_log"   # exp_NNN/ folders live here

    @property
    def checkpoint_name(self) -> str:
        """Model file name inside an experiment folder."""
        return "model.pt"

    @property
    def checkpoint_path(self) -> str:
        """Where the *latest* experiment keeps its weights."""
        return f"{self.experiment_root}/latest/{self.checkpoint_name}"
