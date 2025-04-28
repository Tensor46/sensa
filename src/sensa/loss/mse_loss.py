import torch

from sensa.loss.base import BaseLoss
from sensa.loss.registry import register_loss


@register_loss("MSELoss")
class MSELoss(BaseLoss):
    """See `torch.nn.functional.mse_loss`."""

    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.kwargs = {"size_average": size_average, "reduce": reduce, "reduction": reduction, "weight": weight}

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(input, target, **self.kwargs)
