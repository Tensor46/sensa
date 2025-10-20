from typing import Literal

import torch
import torch.nn.functional as F

from sensa.layers import HeadWithTargetMining
from sensa.loss.base import BaseLoss
from sensa.loss.registry import register_loss


@register_loss("CrossEntropyLoss")
class CrossEntropyLoss(BaseLoss):
    """See `torch.nn.functional.cross_entropy`."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.kwargs = {
            "weight": weight,
            "size_average": size_average,
            "ignore_index": ignore_index,
            "reduce": reduce,
            "reduction": reduction,
            "label_smoothing": label_smoothing,
        }

    def forward(self, tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(tensor, target, **self.kwargs)


@register_loss("CrossEntropyWithTargetMining")
class CrossEntropyWithTargetMining(BaseLoss):
    """Cross-entropy loss with target mining.

    Args:
        dim: dimension of the input features.
        num_labels: number of labels.
        weight (torch.Tensor | None = None): See `torch.nn.functional.cross_entropy`.
        size_average (bool | None = None): See `torch.nn.functional.cross_entropy`.
        ignore_index (int = -100): See `torch.nn.functional.cross_entropy`.
        reduce (bool | None = None): See `torch.nn.functional.cross_entropy`.
        reduction (str = "mean"): See `torch.nn.functional.cross_entropy`.
        label_smoothing (float = 0.0): See `torch.nn.functional.cross_entropy`.
        keep_ratio: fraction of labels to keep (0 <= r <= 1) (a minimum of 1 class is kept).
        batch_size: chunk size when scanning labels to limit memory.
        distance_type: "cosine" or "euclidean".
            - "cosine" uses max cosine similarity (higher = closer)
            - "euclidean" uses min Euclidean distance (lower = closer)
        normalize: whether to normalize the input features and weight vectors.

    Returns:
        loss: (B,) loss tensor.
        predictions: (B, num_labels) predictions tensor.
        target_new: (B,) targets remapped into the subsampled class index space.
    """

    def __init__(
        self,
        dim: int,
        num_labels: int,
        weight: torch.Tensor | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        keep_ratio: float = 0.5,
        batch_size: int = 50_000,
        distance_mining: Literal["cosine", "euclidean"] = "cosine",
        distance_prediction: Literal["dot", "cosine"] = "dot",
    ):
        """Initialize."""
        super().__init__()
        self.head = HeadWithTargetMining(dim, num_labels, keep_ratio, batch_size, distance_mining, distance_prediction)
        self.kwargs = {
            "weight": weight,
            "size_average": size_average,
            "ignore_index": ignore_index,
            "reduce": reduce,
            "reduction": reduction,
            "label_smoothing": label_smoothing,
        }

    def forward(self, tensor: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Definition."""
        predictions, target_new = self.head(tensor, target)
        loss = self.fn_loss(predictions, target_new)
        return {"loss": loss, "predictions": predictions, "target": target_new}

    def fn_loss(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(predictions, target, **self.kwargs)
