from collections import OrderedDict
from collections.abc import Callable
from functools import lru_cache, partial

import torch

from sensa.layers import mask_utils
from sensa.layers.base import BaseLayer
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


class Attention(BaseLayer):
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
        extra_tokens: int = 0,
        use_rope: bool = False,
        frequency: float = 10000.0,
    ):
        super().__init__()
        self.size = size
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.extra_tokens = extra_tokens
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
            if self.extra_tokens > 0:  # extra tokens
                if indices_to_keep is not None:
                    indices_to_keep = indices_to_keep[..., self.extra_tokens :] - self.extra_tokens

                q1, q2 = q[..., : self.extra_tokens, :], q[..., self.extra_tokens :, :]
                q = torch.cat([q1, add_rope_embeddings(q2, indices_to_keep, height, width, self.frequency)], dim=-2)
                k1, k2 = k[..., : self.extra_tokens, :], k[..., self.extra_tokens :, :]
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
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        super().__init__(
            OrderedDict(
                [
                    ("linear1", torch.nn.Linear(in_features, hidden_features)),
                    ("nlinear", act_layer()),
                    ("linear2", torch.nn.Linear(hidden_features, out_features)),
                ]
            )
        )


class EncoderLayer(BaseLayer):
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
        extra_tokens: int = 0,
        use_rope: bool = False,
        frequency: float = 10000.0,
        **kwargs,
    ):
        super().__init__()
        self.p = dropout
        self.attn_norm = norm_layer(hidden_dim)
        self.attn = Attention(size, hidden_dim, num_heads, extra_tokens, use_rope, frequency)
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
        out = tensor + (RegularizeDP.apply(o, self.p) if self.training and self.p > 0 else o)  # drop path

        o = self.mlp(self.mlp_norm(out))
        out = out + (RegularizeDP.apply(o, self.p) if self.training and self.p > 0 else o)  # drop path
        return out


class CrossAttention(torch.nn.Module):
    """Cross-attention module.

    Args:
        dim (int): Dimension of the query and key.
        dim_kv (int | None): Dimension of the key and value. If None, it is set to the same as the query.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, dim: int, dim_kv: int | None, num_heads: int):
        super().__init__()
        self.dim = dim
        self.dim_kv = dim_kv or dim
        self.num_heads = num_heads
        self.scale = (self.dim // self.num_heads) ** -0.5
        self.q = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.k = torch.nn.Linear(self.dim_kv, self.dim, bias=False)
        self.v = torch.nn.Linear(self.dim_kv, self.dim, bias=False)
        self.o = torch.nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        b, n, c = q.shape
        q = self.q(q).reshape(b, n, self.num_heads, -1).transpose(1, 2)
        k = self.k(kv).reshape(b, kv.size(1), self.num_heads, -1).transpose(1, 2)
        v = self.v(kv).reshape(b, kv.size(1), self.num_heads, -1).transpose(1, 2)
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            o = attn @ v
        return self.o(o.transpose(1, 2).reshape(b, n, c))


class CrossAttenttionLayer(torch.nn.Module):
    """Cross-attention layer.

    Args:
        dim (int): Dimension of the query and key.
        dim_kv (int | None): Dimension of the key and value. If None, it is set to the same as the query.
        num_heads (int): Number of attention heads.
        dropout (float): Drop path rate (default: 0.0).
        act_layer (Callable[..., torch.nn.Module]): Activation layer (default: nn.GELU)
        norm_layer (Callable[..., torch.nn.Module]): Normalization layer (default: nn.LayerNorm).
    """

    def __init__(
        self,
        dim: int,
        dim_kv: int | None,
        num_heads: int,
        dropout: float = 0.0,
        act_layer: Callable[..., torch.nn.Module] = torch.nn.GELU,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.attn_norm = norm_layer(dim)
        self.cross_attention = CrossAttention(dim, dim_kv, num_heads)
        self.p = dropout

        self.mlp_norm = norm_layer(dim)
        self.mlp = MLP(dim, dim, None, act_layer=act_layer)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # cross attention
        o = self.cross_attention(self.attn_norm(q), kv)
        q = q + (RegularizeDP.apply(o, self.p) if self.training and self.p > 0 else o)  # drop path
        # mlp
        o = self.mlp(self.mlp_norm(q))
        q = q + (RegularizeDP.apply(o, self.p) if self.training and self.p > 0 else o)  # drop path
        return q
