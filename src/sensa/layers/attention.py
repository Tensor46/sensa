from collections import OrderedDict
from collections.abc import Callable
from functools import lru_cache, partial

import torch

from sensa.layers import mask_utils
from sensa.layers.regularizer import RegularizeDP


@lru_cache(maxsize=4)
@torch.no_grad()
def _get_thetas(dim: int, height: int, width: int, frequency: float = 10000.0) -> torch.Tensor:
    """Get thetas for the 2D tensor.

    Args:
        dim (int): embedding dimension
        height (int): height of the image after patching
        width (int): width of the image after patching
        frequency (float): frequency for the RoPE embeddings
    """
    if dim % 2 != 0:
        raise ValueError("Embedding dimension must be even for RoPE")

    # calculate frequency bands
    # each dimension pair gets a different frequency
    freq_bands = torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)
    ifreq = 1.0 / (frequency**freq_bands)  # (dim // 2,)

    # build 2D grid frequencies
    coords_h = torch.arange(height, dtype=torch.float32)
    coords_w = torch.arange(width, dtype=torch.float32)
    freqs_h = torch.einsum("i,j->ij", coords_h, ifreq)  # [height, dim//2]
    freqs_w = torch.einsum("i,j->ij", coords_w, ifreq)  # [width, dim//2]

    # merge to [height, width, dim]
    theta_h = freqs_h.unsqueeze(1).repeat(1, width, 1)
    theta_w = freqs_w.unsqueeze(0).repeat(height, 1, 1)
    theta = torch.cat([theta_h, theta_w], dim=-1)  # [height, width, dim]
    return theta.view(-1, dim)


def add_rope_embeddings(
    tensor: torch.Tensor,
    indices_to_keep: torch.Tensor | None,
    height: int,
    width: int,
    frequency: float = 10000.0,
) -> torch.Tensor:
    """Add 2D RoPE embeddings to the tensor (..., N, D).

    Args:
        tensor (torch.Tensor): input tensor
        indices_to_keep (torch.Tensor | None): indices to keep
        height (int): height of the image after patching
        width (int): width of the image after patching
        frequency (float): frequency for the RoPE embeddings
    """
    theta = _get_thetas(tensor.size(-1), height, width, frequency).to(device=tensor.device, dtype=tensor.dtype)
    if indices_to_keep is not None:
        theta = theta.unsqueeze(0).repeat(tensor.shape[0], 1, 1)
        theta = mask_utils.mask_tensor(theta, indices_to_keep=indices_to_keep)
        if tensor.ndim > 3:
            shape = theta.shape
            for _ in range(tensor.ndim - 3):
                shape = [shape[0], 1, *shape[1:]]
            theta = theta.reshape(*shape)

    x1, x2 = tensor.chunk(2, dim=-1)
    tensor_rotated = torch.cat((-x2, x1), dim=-1)
    return tensor * theta.cos() + tensor_rotated * theta.sin()


class Attention(torch.nn.Module):
    """Attention module with optional RoPE embeddings.

    Args:
        size: (height, width)
        dim: embedding dimension
        num_heads: number of attention heads
        use_rope: whether to use RoPE embeddings
        frequency: frequency for the RoPE embeddings
    """

    def __init__(
        self,
        size: tuple[int, int],
        dim: int,
        num_heads: int,
        use_rope: bool = False,
        frequency: float = 10000.0,
    ):
        super().__init__()
        self.size = size
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.use_rope = use_rope
        self.frequency = frequency
        self.scale = head_dim**-0.5

        # qkv linear
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=False)
        # projection linear
        self.proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        tensor: torch.Tensor,
        indices_to_keep: torch.Tensor | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> torch.Tensor:
        if height is None or width is None:
            height, width = self.size

        b, n, c = tensor.size()
        q, k, v = self.qkv(tensor).reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3).chunk(3, dim=-1)
        if self.use_rope:
            if (extra_tokens := n - height * width) > 0:  # extra tokens
                if indices_to_keep is not None:
                    indices_to_keep = indices_to_keep[..., extra_tokens:] - extra_tokens

                q1, q2 = q[..., :extra_tokens, :], q[..., extra_tokens:, :]
                q = torch.cat([q1, add_rope_embeddings(q2, indices_to_keep, height, width, self.frequency)], dim=-2)
                k1, k2 = k[..., :extra_tokens, :], k[..., extra_tokens:, :]
                k = torch.cat([k1, add_rope_embeddings(k2, indices_to_keep, height, width, self.frequency)], dim=-2)
            else:
                q = add_rope_embeddings(q, indices_to_keep, height, width, self.frequency)
                k = add_rope_embeddings(k, indices_to_keep, height, width, self.frequency)

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            o = attn @ v

        o = o.transpose(1, 2).reshape(b, n, c)
        o = self.proj(o)
        return o


class MLP(torch.nn.Sequential):
    """MLP module with optional activation and dropout.

    Args:
        in_features: input feature dimension
        hidden_features: hidden feature dimension
        out_features: output feature dimension
        act_layer: activation layer (default: nn.GELU)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., torch.nn.Module] = torch.nn.GELU,
    ):
        out_features: int = out_features or in_features
        hidden_features: int = hidden_features or in_features
        super().__init__(
            OrderedDict(
                [
                    ("linear1", torch.nn.Linear(in_features, hidden_features)),
                    ("nlinear", act_layer()),
                    ("linear2", torch.nn.Linear(hidden_features, out_features)),
                ]
            )
        )


class EncoderLayer(torch.nn.Module):
    """Encoder layer with Attention + MLP.

    Args:
        size (tuple[int, int]): (height, width)
        num_heads (int): number of attention heads
        hidden_dim (int): embedding dimension
        mlp_dim (int): hidden dimension
        dropout (float): drop path for the attention and MLP (default: 0.0)
        act_layer (Callable[..., torch.nn.Module]): activation layer (default: nn.GELU)
        norm_layer (Callable[..., torch.nn.Module]): normalization layer (default: nn.LayerNorm)
        use_rope (bool): whether to use RoPE embeddings (default: False)
        frequency (float): frequency for the RoPE embeddings (default: 10000.0)
    """

    def __init__(
        self,
        size: tuple[int, int],
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        act_layer: Callable[..., torch.nn.Module] = torch.nn.GELU,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6),
        use_rope: bool = False,
        frequency: float = 10000.0,
        **kwargs,
    ):
        super().__init__()
        self.p = dropout
        self.attn_norm = norm_layer(hidden_dim)
        self.attn = Attention(size, hidden_dim, num_heads, use_rope, frequency)
        self.mlp_norm = norm_layer(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_dim, None, act_layer)

    def forward(
        self,
        tensor: torch.Tensor,
        indices_to_keep: torch.Tensor | None = None,
        height: int | None = None,
        width: int | None = None,
    ):
        o = self.attn(self.attn_norm(tensor), indices_to_keep, height, width)
        if self.training and self.p > 0:  # drop path
            o = RegularizeDP.apply(o, self.p)
        out = tensor + o

        o = self.mlp(self.mlp_norm(out))
        if self.training and self.p > 0:  # drop path
            o = RegularizeDP.apply(o, self.p)
        out = out + o
        return out
