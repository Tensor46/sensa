from collections import OrderedDict

import torch

from sensa.models.base import BaseModel
from sensa.models.registry import register_model


@register_model("DinoHead")
class DinoHead(BaseModel):
    """A projection head used in DINO (self-distillation) architectures.

    This implementation is ported from https://github.com/facebookresearch/dino/.
    Paper:
        Emerging Properties in Self-Supervised Vision Transformers [https://arxiv.org/pdf/2104.14294]

    Args:
        dim (int): Dimensionality of the input features.
        hidden_dim (int, optional): Number of hidden units in the MLP. Default: 2048.
        bottleneck_dim (int, optional): Output dimensionality of the MLP bottleneck. Default: 256.
        output_dim (int, optional): Output dimensionality of the final linear layer. Default: 2**16.
        batch_norm (bool, optional): If True, apply BatchNorm1d after each hidden linear layer. Default: False.
        last_layer_norm (bool, optional): If True, freeze the weight_g parameter of the final weight-norm layer.
            Default: True.
        epoch_to_cancel_last_layer_grads (int, optional):
            If the current epoch is below this value, last-layer gradients are zeroed. Default: -1.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 2**16,
        batch_norm: bool = False,
        last_layer_norm: bool = True,
        epoch_to_cancel_last_layer_grads: int = 1,
    ):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [
                    ("linear1", torch.nn.Linear(dim, hidden_dim)),
                    ("norm1", torch.nn.BatchNorm1d(hidden_dim) if batch_norm else torch.nn.Identity()),
                    ("nlinear1", torch.nn.GELU()),
                    ("linear2", torch.nn.Linear(hidden_dim, hidden_dim)),
                    ("norm2", torch.nn.BatchNorm1d(hidden_dim) if batch_norm else torch.nn.Identity()),
                    ("nlinear2", torch.nn.GELU()),
                    ("linear3", torch.nn.Linear(hidden_dim, bottleneck_dim)),
                ]
            )
        )
        # epoch threshold for cancelling gradients on the last layer
        self.epoch_to_cancel_last_layer_grads = epoch_to_cancel_last_layer_grads
        # final weight-normalized linear layer without bias
        self.last_layer = torch.nn.utils.parametrizations.weight_norm(
            torch.nn.Linear(bottleneck_dim, output_dim, bias=False)
        )
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        if last_layer_norm:
            self.last_layer.parametrizations.weight.original0.requires_grad = False

    def cancel_last_layer_gradients(self, epoch: int) -> None:
        """Zero out gradients for the last layer if current epoch is below threshold."""
        if epoch >= self.epoch_to_cancel_last_layer_grads:
            return
        for param in self.last_layer.parameters():
            param.grad = None

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DINO head.

        Args:
            tensor (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.forward_features(tensor)["features"]

    def forward_features(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        o = self.mlp(tensor)
        o = torch.nn.functional.normalize(o, dim=-1, p=2)
        o = self.last_layer(o)
        return {"features": o}
