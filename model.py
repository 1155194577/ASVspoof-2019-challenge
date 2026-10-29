"""Models for the ASVspoof 2019 LA project.

The first model is a **linear SVM implemented in PyTorch**: a single affine
map :math:`f(x) = Wx + b` trained with the (Crammer-Singer) hinge loss instead
of cross-entropy. That gives the classical SVM baseline while reusing the same
optimiser / training loop that the neural models will use later.

    >>> from model import LinearSVM, hinge_loss
    >>> model = LinearSVM(input_dim=40)
    >>> logits = model(torch.randn(8, 40))
    >>> loss = hinge_loss(logits, torch.tensor([0, 1] * 4))

Label convention matches the dataset: ``0 = bonafide``, ``1 = spoof``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class LinearSVM(nn.Module):
    """Binary linear SVM over a fixed-length utterance vector.

    Parameters
    ----------
    input_dim:
        Feature dimension, e.g. ``40`` for ``mfcc`` (20) with ``mean_std``
        pooling (2 x 20).
    num_classes:
        Fixed at 2 so that :meth:`decision_function` is a logit difference.

    Notes
    -----
    The SVM's regularisation strength ``C`` maps roughly to the optimiser's
    ``weight_decay``: larger decay = smaller ``C`` = flatter margin.
    """

    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError("LinearSVM currently supports binary problems only")
        self.input_dim = int(input_dim)
        self.num_classes = num_classes
        self.linear = nn.Linear(self.input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw class scores, shape ``(batch, 2)``."""
        return self.linear(x)

    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        """Spoof score, shape ``(batch,)``: ``> 0`` means spoof."""
        logits = self.forward(x)
        return logits[:, 1] - logits[:, 0]

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Hard labels, shape ``(batch,)``."""
        return (self.decision_function(x) > 0).long()

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, num_classes={self.num_classes}"


def hinge_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    margin: float = 1.0,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Multi-class hinge loss (Crammer-Singer) used by :class:`LinearSVM`.

    ``loss_i = sum_{j != y_i} max(0, s_j - s_{y_i} + margin)``

    Parameters
    ----------
    logits: ``(batch, num_classes)`` class scores.
    target: ``(batch,)`` integer labels.
    margin: hinge margin (1.0 is the textbook default).
    weight: optional ``(batch,)`` per-sample weights, e.g. inverse class
        frequencies for the imbalanced bonafide/spoof split.
    reduction: ``"mean"``, ``"sum"`` or ``"none"`` (per-sample losses).
    """
    if logits.dim() != 2:
        raise ValueError(f"expected (batch, classes) logits, got {tuple(logits.shape)}")
    if target.dim() != 1:
        target = target.reshape(-1)

    correct = logits.gather(1, target[:, None]).squeeze(1)
    margins = logits - correct[:, None] + margin
    margins = margins.scatter(1, target[:, None], 0.0).clamp(min=0.0)
    losses = margins.sum(dim=1)

    if weight is not None:
        weight = weight.reshape(-1).to(losses.dtype)
        losses = losses * weight
        denominator = weight.sum().clamp(min=1e-8)
    else:
        denominator = torch.tensor(
            max(int(losses.numel()), 1), dtype=losses.dtype, device=losses.device
        )

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.sum() / denominator
    raise ValueError(f"unknown reduction {reduction!r}")




class RBFSVM(nn.Module):
    """RBF-kernel SVM, approximated with random Fourier features.

    By Bochner's theorem ``exp(-gamma * ||x - y||^2)`` is the characteristic
    function of a Gaussian, so drawing ``omega ~ N(0, 2*gamma)`` and
    ``phase ~ U(0, 2*pi)`` gives

        phi(x) = sqrt(2/n_features) * cos(omega^T x + phase)
        <phi(x), phi(y)>  ~=  exp(-gamma * ||x - y||^2)

    The same affine head + hinge loss then trains an *approximate* kernel SVM.
    Unlike a true kernel SVM (a QP with a Gram matrix), this stays trainable
    with mini-batch SGD, so epochs / lr / batch size keep their usual meaning.

    ``omega`` and ``phase`` are registered buffers: fixed at construction,
    saved in the state_dict and moved by ``.to(device)``, so a checkpoint
    reproduces the same feature map exactly.

    Parameters
    ----------
    input_dim: feature dimension (40 for LFCC + mean_std).
    n_features: number of random features D; larger = closer to the exact
        kernel, more compute (512 is plenty for 40-d input).
    gamma: RBF bandwidth; ``None`` -> ``1 / input_dim`` (the "scale"
        heuristic for standardised features).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        n_features: int = 512,
        gamma: Optional[float] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError("RBFSVM currently supports binary problems only")
        self.input_dim = int(input_dim)
        self.num_classes = num_classes
        self.n_features = int(n_features)
        self.gamma = float(gamma) if gamma is not None else 1.0 / max(self.input_dim, 1)

        generator = torch.Generator().manual_seed(seed)
        omega = torch.randn(self.input_dim, self.n_features, generator=generator)
        omega *= (2.0 * self.gamma) ** 0.5
        phase = torch.rand(self.n_features, generator=generator) * 2.0 * math.pi

        self.register_buffer("omega", omega)
        self.register_buffer("phase", phase)
        self.register_buffer("scale", torch.tensor((2.0 / self.n_features) ** 0.5))
        self.linear = nn.Linear(self.n_features, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Random Fourier feature map, shape ``(batch, n_features)``."""
        return torch.cos(x @ self.omega + self.phase) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.features(x))

    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        """Spoof score: ``> 0`` means spoof."""
        logits = self.forward(x)
        return logits[:, 1] - logits[:, 0]

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return (self.decision_function(x) > 0).long()

    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, n_features={self.n_features}, "
                f"gamma={self.gamma:g}")


class CNN2D(nn.Module):
    """Small 2-D CNN over a frame-level feature map, e.g. LFCC ``(20, 401)``.

    ``train.py`` feeds flat vectors of length ``C * T`` (the ``--pooling flat``
    layout: coefficient-major, i.e. ``[c0_t0 .. c0_tT, c1_t0 ..]``), so the first
    thing the model does is ``view(B, 1, C, T)``. Nothing else in the pipeline
    changes: same loops, same EER metric, same checkpointing.

    Three conv blocks (16/32/64 channels, batch-norm + ReLU + max-pool), then
    global average pooling over the remaining ``(C, T)`` grid and a linear
    classifier. ~40 k parameters -- deliberately small, so it trains from
    scratch on 25 k utterances without overfitting to dev speakers.

    Parameters
    ----------
    input_dim: ``C * T`` (8020 for LFCC 20 x 401).
    shape: ``(C, T)`` of the map; inferred as ``(20, input_dim // 20)`` when
        omitted (correct for the 20-coefficient MFCC/LFCC front-ends).
    channels: conv widths per block.
    dropout: applied before the classifier.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        shape: Optional[tuple] = None,
        channels: tuple = (16, 32, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError("CNN2D currently supports binary problems only")
        self.input_dim = int(input_dim)
        self.num_classes = num_classes
        if shape is None:
            if self.input_dim % 20:
                raise ValueError(
                    f"cannot infer (C, T) from input_dim={self.input_dim}; pass shape="
                )
            shape = (20, self.input_dim // 20)
        shape = tuple(int(v) for v in shape)
        if shape[0] * shape[1] != self.input_dim:
            raise ValueError(f"shape {shape} does not match input_dim {self.input_dim}")
        self.shape = shape

        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(c3, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.shape[0], 1, *self.shape)
        elif x.dim() == 3:  # already (B, C, T)
            x = x.unsqueeze(1)
        return self.classifier(self.pool(self.features(x)))

    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return logits[:, 1] - logits[:, 0]

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return (self.decision_function(x) > 0).long()

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, shape={self.shape}"


# --------------------------------------------------------------------------- #
# Registry: lets config.model pick a backbone without touching train.py
# --------------------------------------------------------------------------- #
MODEL_REGISTRY = {
    "svm": LinearSVM,        # linear kernel
    "svm_rff": RBFSVM,       # RBF kernel via random Fourier features
    "cnn2d": CNN2D,          # small 2-D CNN over the (C, T) map (needs flat pooling)
    # TODO: "hubert": HuBERTClassifier, "wavlm": WavLMClassifier, ...
}


def register_model(name: str, cls: type) -> None:
    """Add a model class so ``build_model(name, ...)`` can find it."""
    MODEL_REGISTRY[name.lower()] = cls


def build_model(name: str, input_dim: int, **kwargs) -> nn.Module:
    """Instantiate a registered model, e.g. ``build_model("svm", 40)``."""
    key = (name or "svm").lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key](input_dim=input_dim, **kwargs)
