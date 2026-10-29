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


# --------------------------------------------------------------------------- #
# Registry: lets config.model pick a backbone without touching train.py
# --------------------------------------------------------------------------- #
MODEL_REGISTRY = {
    "svm": LinearSVM,
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
