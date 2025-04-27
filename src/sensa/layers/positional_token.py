import einops
import torch


class PositionalTokenOn3D(torch.nn.Module):
    def __init__(self, dim: int, extra_tokens: int = 0, size: tuple[int, int] = (8, 8)):
        super().__init__()
        h, w = size
        x, y = torch.linspace(-1, 1, w), torch.linspace(-1, 1, h)
        self.register_buffer("grid", torch.stack(torch.meshgrid(y, x, indexing="ij"))[None])
        if extra_tokens:
            self.register_buffer("token", torch.zeros(1, extra_tokens, dim))
        self.projection = torch.nn.Conv2d(2, dim, 1, bias=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        encoding = self.projection(self.grid)
        encoding = einops.rearrange(encoding, "b c h w -> b (h w) c")
        if hasattr(self, "token"):
            encoding = torch.cat((self.token, encoding), 1)
        encoding = encoding.repeat(tensor.size(0), 1, 1)
        return encoding + tensor


class PositionalTokenOn4D(torch.nn.Module):
    def __init__(self, dim: int, size: tuple[int, int] = (8, 8)):
        super().__init__()
        h, w = size
        x, y = torch.linspace(-1, 1, w), torch.linspace(-1, 1, h)
        self.register_buffer("grid", torch.stack(torch.meshgrid(y, x, indexing="ij"))[None])
        self.projection = torch.nn.Conv2d(2, dim, 1, bias=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        encoding = self.projection(self.grid)
        *_, h, w = tensor.shape
        if h != self.grid.size(-2) or w != self.grid.size(-1):
            encoding = torch.nn.functional.interpolate(
                encoding,
                size=(h, w),
                mode="bilinear",
                align_corners=True,
                antialias=True,
            )
        encoding = encoding.repeat(tensor.size(0), 1, 1, 1)
        return encoding + tensor
