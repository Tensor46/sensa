import torch

from sensa.layers.base import BaseLayer


class RMSNorm(BaseLayer):
    """Replacement for LayerNorm using RMSNorm(x) = gamma * x / sqrt(x^2 + epsilon).

    Reference:
        “Root Mean Square Layer Normalization” (ArXiv:2002.07514).
    """

    def __init__(self, dim: int, eps: float = 1e-6, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dim = dim
        self.dtype = dtype
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * tensor.pow(2).mean(-1, True).add(self.eps).rsqrt()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the RMSNorm normalization element-wise."""
        self.assert_once(tensor.size(-1) == self.dim, f"Last dimension must be {self.dim}, got {tensor.shape[-1]}.")
        dtype = tensor.dtype
        return self.weight * self._norm(tensor.type(self.dtype)).type(dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class RMSNorm2D(RMSNorm):
    """See RMSNorm, but applied per channel."""

    def _norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * tensor.pow(2).mean((2, 3), True).add(self.eps).rsqrt()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the RMSNorm normalization element-wise."""
        self.assert_once(tensor.size(1) == self.dim, f"Channels must be {self.dim}, got {tensor.shape[1]}.")
        dtype = tensor.dtype
        return self.weight[None, :, None, None] * self._norm(tensor.type(self.dtype)).type(dtype)
