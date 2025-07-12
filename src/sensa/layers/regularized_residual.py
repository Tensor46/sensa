__all__ = ["RegularizedResidual"]

from collections.abc import Callable

import torch
import torch.nn as nn

from sensa.layers.regularizer import RegularizeDP, RegularizeJA


class RegularizedResidual(nn.Module):
    """Base regularized residual layer.

    Args:
        residue (nn.Module): residue module
        non_residue (nn.Module): non-residue module
        non_linear (nn.Module | None): non-linear module (default is None)
        p_regularize_dp (float): dropout rate for the DP regularization (default is 0.0)
        p_regularize_ja (float): dropout rate for the JA regularization (default is 0.0)
    """

    def __init__(
        self,
        residue: nn.Module | None,
        non_residue: nn.Module,
        non_linear: Callable[..., nn.Module] = nn.Identity,
        p_regularize_dp: float = 0.0,
        p_regularize_ja: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.residue = residue
        self.non_residue = non_residue
        non_linear = nn.Identity if non_linear is None else non_linear
        self.non_linear = non_linear()

        self.p_regularize_dp = p_regularize_dp
        self.p_regularize_ja = p_regularize_ja

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        residue = self.residue(tensor) if self.residue is not None else tensor
        o = self.non_residue(tensor)

        if self.training and self.p_regularize_dp > 0:  # drop path
            o = RegularizeDP.apply(o, self.p_regularize_dp)

        if residue.shape != o.shape:  # skip connection
            torch._assert(
                self.residue is None,
                f"{self.__class__.__name__}: Residue must be None if the shapes do not match.",
            )
            return self.non_linear(o)

        if self.training and self.p_regularize_ja > 0:  # jitter and add
            o = RegularizeJA.apply(residue, o, self.p_regularize_ja)
            return self.non_linear(o)

        return self.non_linear(residue + o)

    def extra_repr(self) -> str:
        return f"[p_regularize_dp={self.p_regularize_dp}, p_regularize_ja={self.p_regularize_ja}]"
