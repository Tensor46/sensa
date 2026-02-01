import torch
import torch.nn.functional as F

from sensa.loss.cross_entropy import CrossEntropyWithTargetMining
from sensa.loss.registry import register_loss


@register_loss("CosFace")
class CosFace(CrossEntropyWithTargetMining):
    """CosFace: Large Margin Cosine Loss for Deep Face Recognition
    URL: https://arxiv.org/pdf/1801.09414

    Args:
        SEE CrossEntropyWithTargetMining for other arguments.

        m (float): Margin applied to loss for better separation between classes.
        s (float): Scale factor for the logits before computing loss.
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
        m: float = 0.3,
        s: float = 32,
    ):
        # setting distance mining and prediction to cosine
        kwargs = {"distance_mining": "cosine", "distance_prediction": "cosine"}
        # margin & scale
        self.m: float = m
        self.s: float = s
        super().__init__(
            dim=dim,
            num_labels=num_labels,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
            keep_ratio=keep_ratio,
            batch_size=batch_size,
            **kwargs,
        )

    def forward(self, tensor: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Definition."""
        if self.head.weight.grad is None:
            with torch.no_grad():
                self.head.weight.data = F.normalize(self.head.weight.data, dim=-1)
        return super().forward(tensor, target)

    def fn_loss(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            margin = torch.zeros_like(predictions)
            margin.scatter_(1, target.view(-1, 1), -self.m, reduce="add")

        loss = F.cross_entropy(self.s * (predictions + margin), target)
        return loss
