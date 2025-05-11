import logging
from typing import Literal

import einops
import torch


class LastPool(torch.nn.Module):
    """Flexible pooling over the token dimension of a tensor.

    Supports four modes:
      - "avg": sum over tokens, output shape (B, C)
      - "full": flatten all tokens, output shape (B, C*N)
      - "half": reshape tokens to HxW and maxpool to (H/2 x W/2), then flatten
      - "token": select the first token, output shape (B, C)

    Args:
        pool ("avg" | "full" | "half" | "token", default="token"):
            Pooling strategy
        size (tuple[int, int] | None, default=None):
            Grid dims for "half" mode (must be provided).
    """

    def __init__(
        self,
        pool: Literal["avg", "full", "half", "token"] = "token",
        size: tuple[int, int] | None = None,
    ):
        super().__init__()
        if not isinstance(pool, str):
            logging.error(f"LastPool: pool must be 'avg' | 'full' | 'half' | 'token' - {pool}.")
            raise TypeError(f"LastPool: pool must be 'avg' | 'full' | 'half' | 'token' - {pool}")
        if pool not in ("avg", "full", "half", "token"):
            logging.error(f"LastPool: pool must be 'avg' | 'full' | 'half' | 'token' - {pool}.")
            raise ValueError(f"LastPool: pool must be 'avg' | 'full' | 'half' | 'token' - {pool}")
        self.pool = pool
        self.size = size

        if self.pool == "avg":
            self.fn = self.fn_avg
        elif self.pool == "full":
            self.fn = self.fn_full
        elif self.pool == "half":
            if not (isinstance(size, list | tuple) and all(isinstance(x, int) for x in size)):
                logging.error(f"LastPool: size must be tuple[int, int] - {size}.")
                raise TypeError(f"LastPool: size must be tuple[int, int] - {size}")
            if len(size) != 2:
                logging.error(f"LastPool: size must be tuple[int, int] - {size}.")
                raise ValueError(f"LastPool: size must be tuple[int, int] - {size}")
            self.size_after_pool = torch.nn.functional.max_pool2d(torch.randn(1, 1, *self.size), 2, 2)[2:]
            self.fn = self.fn_half
        elif self.pool == "token":
            self.fn = self.fn_token

    def forward(self, tensor: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        """Apply the selected pooling function."""
        kwargs = {"flatten": flatten} if self.pool in ("full", "half") else {}
        return self.fn(tensor, **kwargs)

    def fn_avg(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sum over the token dimension → (B, C)."""
        return tensor.sum(1)

    def fn_full(self, tensor: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        """Flatten all tokens → (B, C·N)."""
        return tensor.permute(0, 2, 1).flatten(1) if flatten else tensor.permute(0, 2, 1)

    def fn_half(self, tensor: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        """Reshape tokens to (H, W), maxpool by 2, then flatten."""
        o = einops.rearrange(tensor, "b (h w) c -> b c h w", h=self.size[0], w=self.size[1])
        o = torch.nn.functional.max_pool2d(o, 2, 2)
        return o.flatten(1) if flatten else o

    def fn_token(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Select the first token (class token) → (B, C)."""
        return tensor[:, 0, :]
