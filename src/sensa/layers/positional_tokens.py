import torch


def sincos_2d_positional_encoding(dim: int, extra_tokens: int, size: tuple[int, int]) -> torch.Tensor:
    """Generate a 2D sinusoidal positional encoding.

    This creates a `(H·W + extra_tokens) x dim` tensor where each of the
    H·W spatial positions receives a `dim`-dimensional embedding formed by
    concatenating two 1D sin-cos encodings (one over rows, one over columns).
    Any `extra_tokens` (e.g. classification tokens) are prepended as zeros.

    Parameters
    ----------
    dim : int
        Total embedding dimension. Must be divisible by 4, since half of the
        dimensions go to height and half to width, and each is split into
        sin and cos components.
    extra_tokens : int
        Number of leading zero embeddings to insert (e.g. `[CLS]` tokens).
    size : tuple[int, int]
        Spatial dimensions of the grid as `(height, width)`.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(extra_tokens + height*width, dim)`, suitable for
        adding to flattened image patches or tokens.

    Raises
    ------
    ValueError
        If `dim` is not divisible by 4.
    """
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
    """Compute a 1D sinusoidal positional encoding over a flattened grid.

    Given a `grid` of positions, this returns a tensor of shape
    `(N, dim)`, where `N = grid.numel()`. Half of the dimensions contain
    sine values and half contain cosine values at different frequencies.

    Parameters
    ----------
    dim : int
        Embedding dimension for this axis. Must be even.
    grid : torch.Tensor
        A tensor of positional values (e.g. `HxW` positions) that will be
        flattened to a vector of length `N = grid.numel()`.

    Returns
    -------
    torch.Tensor
        Positional encoding of shape `(N, dim)`, ready to concatenate
        with other dimensions or tokens.
    """
    omega = torch.arange(dim // 2, dtype=torch.float32)
    omega /= dim / 2.0
    omega = 10000**-omega

    sin = torch.sin(grid.reshape(-1, 1) * omega)
    cos = torch.cos(grid.reshape(-1, 1) * omega)
    return torch.cat([sin, cos], axis=1)
