import torch


def sincos_2d_positional_encoding(dim: int, extra_tokens: int, size: tuple[int, int]) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"sincos_2d_positional_encoding: dim ({dim}) must be divisible by 4.")
    h, w = size
    ys = torch.arange(h, dtype=torch.float32)
    xs = torch.arange(w, dtype=torch.float32)
    grid = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack(grid, axis=0).reshape([2, 1, h, w])
    pos_emb = torch.cat(
        (
            sincos_1d_positional_encoding(dim // 2, grid[0]),
            sincos_1d_positional_encoding(dim // 2, grid[1]),
        ),
        dim=1,
    )
    if extra_tokens:
        pos_emb = torch.cat((torch.zeros(extra_tokens, dim), pos_emb))
    return pos_emb


def sincos_1d_positional_encoding(dim: int, grid: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(dim // 2, dtype=torch.float32)
    omega /= dim / 2.0
    omega = 10000**-omega

    sin = torch.sin(grid.reshape(-1, 1) * omega)
    cos = torch.cos(grid.reshape(-1, 1) * omega)
    return torch.cat([sin, cos], axis=1)
