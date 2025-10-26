import logging
from collections.abc import Callable
from functools import partial
from typing import Literal

import einops
import pydantic
import torch
import torchvision as tv

from sensa.layers import mask_utils
from sensa.layers.attention import EncoderLayer
from sensa.layers.dyt import DyT
from sensa.layers.positional_tokens import sincos_2d_positional_encoding
from sensa.layers.rmsnorm import RMSNorm


class Encoder(tv.models.vision_transformer.Encoder):
    """VIT Encoder from torchvision without positional embedding."""

    def __init__(self, size: tuple[int, int], extra_tokens: int, **kwargs):
        super().__init__(seq_length=size[0] * size[1] + extra_tokens, **kwargs)
        self.size = size
        self.extra_tokens = extra_tokens
        self.seq_length, self.hidden_dim = self.pos_embedding.shape[1:]
        del self.pos_embedding
        self.pos_token = torch.nn.Parameter(torch.empty(1, self.seq_length, self.hidden_dim).normal_(std=0.02))
        self.is_sincos_pos_token: bool = False
        self.other_sizes: list[tuple[int, int]] = []

    def extend_sizes(self, size: tuple[int, int]) -> None:
        if not isinstance(size, list | tuple):
            logging.error(f"Encoder: extend_sizes size must be tuple[int, int] - {size}.")
            raise TypeError(f"Encoder: extend_sizes size must be tuple[int, int] - {size}.")
        if not (len(size) == 2 and all(isinstance(val, int) for val in size)):
            logging.error(f"Encoder: extend_sizes size must be tuple[int, int] - {size}.")
            raise TypeError(f"Encoder: extend_sizes size must be tuple[int, int] - {size}.")
        if (self.size[0] * self.size[1]) % (size[0] * size[1]) != 0:
            logging.error(f"Encoder: extend_sizes requires size that are multiple of size - {size}.")
            raise ValueError(f"Encoder: extend_sizes requires size that are multiple of size - {size}.")
        self.other_sizes.append(size)

    def forward(self, tensor: torch.Tensor, indices_to_keep: torch.Tensor | None = None) -> torch.Tensor:
        torch._assert(tensor.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {tensor.shape}")
        slen = tensor.shape[1]
        if slen == self.seq_length:
            o = tensor + self.pos_token
        elif any(slen == (sz[0] * sz[1] + self.extra_tokens) for sz in self.other_sizes):
            index = [slen == (sz[0] * sz[1] + self.extra_tokens) for sz in self.other_sizes].index(True)
            size = self.other_sizes[index]
            o = tensor + self.resize_pos_token(size)
        else:
            torch._assert(False, f"Invalid tensor.shape - {tensor.shape} | {self.other_sizes}.")

        if indices_to_keep is not None:
            o = mask_utils.mask_tensor(o, indices_to_keep=indices_to_keep)
        return self.ln(self.layers(self.dropout(o)))

    def resize_pos_token(self, size: tuple[int, int]) -> torch.Tensor:
        pos_token = self.pos_token[:, slice(self.extra_tokens, self.pos_token.shape[1])]
        pos_token = einops.rearrange(pos_token, "b (h w) c -> b c h w", h=self.size[0], w=self.size[1])
        pos_token = torch.nn.functional.interpolate(
            pos_token,
            size=size,
            mode="bicubic",
            align_corners=True,
            antialias=True,
        )
        pos_token = einops.rearrange(pos_token, "b c h w -> b (h w) c")
        return torch.cat((self.pos_token[:, slice(0, self.extra_tokens)], pos_token), dim=1)

    def use_sincos_pos_token(self, extra_tokens: int, size: tuple[int, int]) -> None:
        del self.pos_token
        self.register_buffer("pos_token", sincos_2d_positional_encoding(self.hidden_dim, extra_tokens, size)[None])
        self.is_sincos_pos_token = True


class Encoder2Config(pydantic.BaseModel):
    """
    Parameters:
        size (tuple[int, int]): (height, width) of the image after patching.
        extra_tokens (int): number of extra tokens (e.g., class token).
        num_layers (int): number of layers.
        num_heads (int): number of heads.
        hidden_dim (int): hidden dimension.
        mlp_dim (int): mlp hidden dimension.
        dropout (float): drop path rate (default: 0.0)
        act_layer (Callable[..., torch.nn.Module]): activation layer (default: torch.nn.GELU).
        norm_layer (Callable[..., torch.nn.Module]): normalization layer (default: torch.nn.LayerNorm).
        pos_token (Literal["learned", "sincos", "rope"]): positional token type (default: "learned").
        frequency (float): frequency for the RoPE embeddings (default: 10000.0).
    """

    size: tuple[int, int]
    extra_tokens: int
    num_layers: int
    num_heads: int
    hidden_dim: int
    mlp_dim: int
    dropout: float = pydantic.Field(..., ge=0.0, lt=1.0)
    act_layer: Callable[..., torch.nn.Module] = torch.nn.GELU
    norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6)
    pos_token: Literal["learned", "sincos", "rope"] = "learned"
    frequency: float = 10000.0

    @pydantic.field_validator("size", mode="before")
    @classmethod
    def validate_size(cls, data: int | tuple[int, int]) -> tuple[int, int]:
        return (data, data) if isinstance(data, int) else data

    @pydantic.field_validator("act_layer", mode="before")
    @classmethod
    def validate_act_layer(cls, data: Callable[..., torch.nn.Module] | str) -> Callable[..., torch.nn.Module]:
        """Validate the activation layer."""
        match data:
            case "gelu":
                data = torch.nn.GELU
            case "mish":
                data = partial(torch.nn.Mish, inplace=True)
            case "relu":
                data = partial(torch.nn.ReLU, inplace=True)
            case "silu":
                data = partial(torch.nn.SiLU, inplace=True)
        return data

    @pydantic.field_validator("norm_layer", mode="before")
    @classmethod
    def validate_norm_layer(cls, data: Callable[..., torch.nn.Module] | str) -> Callable[..., torch.nn.Module]:
        """Validate the normalization layer."""
        match data:
            case "dyt":
                data = DyT
            case "layernorm":
                data = partial(torch.nn.LayerNorm, eps=1e-6)
            case "rmsnorm":
                data = partial(RMSNorm, eps=1e-6)
        return data

    @property
    def seq_length(self) -> int:
        return self.size[0] * self.size[1] + self.extra_tokens


class Encoder2(torch.nn.Module):
    """VIT Encoder with positional embedding (Learned or SinCos or RoPE).

    Args:
        size (tuple[int, int]): (height, width) of the image after patching.
        extra_tokens (int): number of extra tokens (e.g., class token).
        num_layers (int): number of layers.
        num_heads (int): number of heads.
        hidden_dim (int): hidden dimension.
        mlp_dim (int): mlp hidden dimension.
        dropout (float): drop path rate (default: 0.0)
        act_layer (Callable[..., torch.nn.Module]): activation layer (default: torch.nn.GELU).
        norm_layer (Callable[..., torch.nn.Module]): normalization layer (default: torch.nn.LayerNorm).
        pos_token (Literal["learned", "sincos", "rope"]): positional token type (default: "learned").
        frequency (float): frequency for the RoPE embeddings (default: 10000.0).
    """

    def __init__(
        self,
        size: tuple[int, int],
        extra_tokens: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        act_layer: Callable[..., torch.nn.Module] = torch.nn.GELU,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6),
        pos_token: Literal["learned", "sincos", "rope"] = "learned",
        frequency: float = 10000.0,
        **kwargs,
    ):
        super().__init__()
        self.config = Encoder2Config(
            size=size,
            extra_tokens=extra_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            act_layer=act_layer,
            norm_layer=norm_layer,
            pos_token=pos_token,
            frequency=frequency,
        )
        if self.config.pos_token == "learned":
            self.pos_token = torch.nn.Parameter(
                torch.empty(1, self.config.seq_length, self.config.hidden_dim).normal_(std=0.02),
            )
        elif self.config.pos_token == "sincos":
            self.register_buffer(
                "pos_token",
                sincos_2d_positional_encoding(
                    self.config.hidden_dim,
                    self.config.extra_tokens,
                    self.config.size,
                )[None],
            )

        self.layers = torch.nn.ModuleList(
            [
                EncoderLayer(
                    self.config.size,
                    self.config.num_heads,
                    self.config.hidden_dim,
                    self.config.mlp_dim,
                    self.config.dropout,
                    self.config.act_layer,
                    self.config.norm_layer,
                    extra_tokens=self.config.extra_tokens,
                    use_rope=self.config.pos_token == "rope",
                    frequency=self.config.frequency,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.ln = self.config.norm_layer(self.config.hidden_dim)
        self.other_sizes: list[tuple[int, int]] = []

    @classmethod
    def from_config(cls, config: Encoder2Config) -> "Encoder2":
        return cls(**config.model_dump())

    def extend_sizes(self, size: tuple[int, int]) -> None:
        if not isinstance(size, list | tuple):
            logging.error(f"Encoder: extend_sizes size must be tuple[int, int] - {size}.")
            raise TypeError(f"Encoder: extend_sizes size must be tuple[int, int] - {size}.")
        if not (len(size) == 2 and all(isinstance(val, int) for val in size)):
            logging.error(f"Encoder: extend_sizes size must be tuple[int, int] - {size}.")
            raise TypeError(f"Encoder: extend_sizes size must be tuple[int, int] - {size}.")
        self.other_sizes.append(size)

    def forward(self, tensor: torch.Tensor, indices_to_keep: torch.Tensor | None = None) -> torch.Tensor:
        torch._assert(tensor.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {tensor.shape}")
        slen = tensor.shape[1]
        if slen == self.config.seq_length:
            height, width = self.config.size
            o = (tensor + self.pos_token) if hasattr(self, "pos_token") else tensor
        elif any(slen == (sz[0] * sz[1] + self.config.extra_tokens) for sz in self.other_sizes):
            index = [slen == (sz[0] * sz[1] + self.config.extra_tokens) for sz in self.other_sizes].index(True)
            height, width = size = self.other_sizes[index]
            o = (tensor + self.resize_pos_token(size)) if hasattr(self, "pos_token") else tensor
        else:
            torch._assert(False, f"Invalid tensor.shape - {tensor.shape} | {[self.config.size, *self.other_sizes]}.")

        if indices_to_keep is not None:
            o = mask_utils.mask_tensor(o, indices_to_keep=indices_to_keep)
        for layer in self.layers:
            o = layer(o, indices_to_keep=indices_to_keep, height=height, width=width)
        return self.ln(o)

    def resize_pos_token(self, size: tuple[int, int]) -> torch.Tensor:
        pos_token = self.pos_token[:, slice(self.config.extra_tokens, self.pos_token.shape[1])]
        pos_token = einops.rearrange(pos_token, "b (h w) c -> b c h w", h=self.config.size[0], w=self.config.size[1])
        pos_token = torch.nn.functional.interpolate(
            pos_token,
            size=size,
            mode="bicubic",
            align_corners=True,
            antialias=True,
        )
        pos_token = einops.rearrange(pos_token, "b c h w -> b (h w) c")
        return torch.cat((self.pos_token[:, slice(0, self.config.extra_tokens)], pos_token), dim=1)
