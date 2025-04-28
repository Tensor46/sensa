import torch

from sensa.loss.base import BaseLoss
from sensa.loss.registry import register_loss


@register_loss("CrossEntropyLoss")
class CrossEntropyLoss(BaseLoss):
    """See `torch.nn.functional.mse_loss`."""

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
