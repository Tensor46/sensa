import torch

from sensa.layers.base import BaseLayer


class DyT(BaseLayer):
    """Replacement for LayerNorm using DyT(x) = gamma * tanh(alpha * x) + beta.

    Reference:
        “Transformers without Normalization” (ArXiv:2503.10622).

    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))
        self.gamma = torch.nn.Parameter(torch.ones(dim))
        self.beta = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the DyT normalization element-wise."""
        self.assert_once(tensor.size(-1) == self.dim, f"Last dimension must be {self.dim}, got {tensor.shape[-1]}")
        return (tensor.mul_(self.alpha).tanh_() * self.gamma).add_(self.beta)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class DyT2D(BaseLayer):
    """2D version of DyT for feature maps: DyT(x) = gamma * tanh(alpha * x) + beta applied per channel.

    Reference:
        “Transformers without Normalization” (ArXiv:2503.10622).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))
        self.gamma = torch.nn.Parameter(torch.ones(dim))
        self.beta = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the DyT normalization to a 4D tensor (BxCxHxW)."""
        self.assert_once(tensor.size(1) == self.dim, f"Channels must be {self.dim}, got {tensor.shape[1]}")
        self.assert_once(tensor.dim() == 4, f"Expected BxCxHxW, got {tuple(tensor.shape)}")
        return (tensor.mul_(self.alpha).tanh_() * self.gamma[:, None, None]).add_(self.beta[:, None, None])

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
