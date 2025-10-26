__all__ = ["RegularizeDP", "RegularizeJA"]

import torch


class RegularizeDP(torch.autograd.Function):
    """Similar to drop path (p must be lower for deeper networks).

    Args:
        x (torch.Tensor): input tensor
        p (float): dropout rate (default is 0.05)
        scale (bool): whether to scale the output by the number of non-zero elements (default is True).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, p: float = 0.05, scale: bool = True) -> torch.Tensor:
        with torch.no_grad():
            shape = [1 if i else x.size(0) for i in range(x.ndim)]
            picks = torch.ones(x.size(0), device=x.device)
            n = min(int(x.size(0) * p), x.size(0) - 1)
            picks[torch.randperm(x.size(0), device=x.device)[:n]] = 0
            picks = picks.view(shape)
            if scale and n > 0:
                picks.mul_(x.size(0) / (x.size(0) - n))

        ctx.save_for_backward(picks)
        return x * picks

    @staticmethod
    def backward(ctx, grads: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        gx = grads * ctx.saved_tensors[0]
        return gx, None, None


class RegularizeJA(torch.autograd.Function):
    """Something in the lines of Shake-Drop, Shake-Shake and others.

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): input tensor
        gamma (float): gamma parameter (default is 0.25)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, gamma: float = 0.25) -> torch.Tensor:
        with torch.no_grad():
            shape = [1 if i else x.size(0) for i in range(x.ndim)]
            alpha = torch.rand(shape, dtype=x.dtype, device=x.device)
            alpha = alpha * gamma - gamma / 2

            # Support for RegularizeDP
            # When drop path is enabled, few non residual samples in a batch
            # are zeroed -- this addition will avoid RegularizeJA on such
            # samples.
            valid = x.clone().detach().flatten(1).abs().sum(1).ge(1e-12)
            valid *= y.clone().detach().flatten(1).abs().sum(1).ge(1e-12)
            valid = valid.reshape(*shape)
            alpha *= valid.to(dtype=x.dtype)
            ctx.save_for_backward(valid)

        ctx.gamma = gamma
        return (1 - alpha) * x + (1 + alpha) * y

    @staticmethod
    def backward(ctx, grads: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        with torch.no_grad():
            (valid,) = ctx.saved_tensors
            gamma = ctx.gamma
            shape = [1 if i else grads.size(0) for i in range(grads.ndim)]
            alpha = torch.rand(shape, dtype=grads.dtype, device=grads.device)
            alpha = alpha * gamma - gamma / 2
            alpha *= valid.to(dtype=grads.dtype)

        gx = (1 - alpha) * grads
        gy = (1 + alpha) * grads
        return gx, gy, None
