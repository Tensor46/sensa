import math

import torch
import torch.nn.functional as F

from sensa.loss.cross_entropy import CrossEntropyWithTargetMining
from sensa.loss.registry import register_loss


@register_loss("SphereFace2")
class SphereFace2(CrossEntropyWithTargetMining):
    """SPHEREFACE2: BINARY CLASSIFICATION IS ALL YOU NEED FOR DEEP FACE RECOGNITION
    URL: https://arxiv.org/pdf/2108.01513

    Args:
        SEE CrossEntropyWithTargetMining for other arguments.

        lam (float): Lambda mixing parameter controlling balance between positive and negative examples.
        m (float): Margin applied to loss for better separation between classes.
        r (float): Scale factor for the logits before computing loss.
        t (float): Temperature parameter for controlling the shape of the loss.

    Notes:
        - The 'bias' parameter is initialized analytically based on hyperparameters.
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
        lam: float = 0.7,
        m: float = 0.4,
        r: float = 40,
        t: float = 3.0,
    ):
        # setting distance mining and prediction to cosine
        kwargs = {"distance_mining": "cosine", "distance_prediction": "cosine"}
        # weighting factor to balance gradients
        self.lam: float = lam
        # margin
        self.m: float = m
        # radius of the hypersphere
        self.r: float = r
        # t controls the strength of similarity adjustmen
        self.t: float = t
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
        # bias from Section-G in Appendix
        z = self.lam / (1 - self.lam) / (self.head.num_labels - 1)
        ay = self.r * (2 * 0.5**self.t - 1 - self.m)
        ai = self.r * (2 * 0.5**self.t - 1 + self.m)
        tmp = (1.0 - z) ** 2 + 4 * z * math.exp(ay - ai)
        b = math.log(2 * z) - ai - math.log(1.0 - z + math.sqrt(tmp))
        self.bias = torch.nn.Parameter(torch.tensor(b).float())

    def forward(self, tensor: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Definition."""
        if self.head.weight.grad is None:
            with torch.no_grad():
                self.head.weight.data = F.normalize(self.head.weight.data, dim=-1)
        return super().forward(tensor, target)

    def fn_loss(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cosine = predictions
        n, n_labels = cosine.shape
        with torch.no_grad():  # onehot and pos mask
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, target.view(-1, 1), 1.0)
            pos_mask = one_hot.bool().reshape(-1)

            # compute gz
            gz = (cosine + 1).div_(2).pow_(self.t).mul_(2).sub_(1)
            # add margin
            gz.sub_(self.m * (2.0 * one_hot - 1.0))
            # get delta (yet we are calling it gz)
            gz.sub_(cosine)

        # gz again
        gz = cosine + gz
        gz.mul_(self.r)
        gz = gz + self.bias

        # split pos and negs
        # ==================
        gz = gz.view(-1)
        gz_pos = gz[pos_mask]
        gz_neg = gz[~pos_mask].reshape(n, n_labels - 1)
        gz_pos = (1 + torch.exp(-gz_pos)).log_().mul_(self.lam / self.r)
        gz_neg = (1 + torch.exp(+gz_neg)).log_().mul_((1 - self.lam) / self.r)

        # loss
        return (gz_pos + gz_neg.sum(1)).mean()
