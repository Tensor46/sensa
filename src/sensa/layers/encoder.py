import logging

import einops
import torch
import torchvision as tv

from sensa.layers import mask_utils
from sensa.layers.positional_tokens import sincos_2d_positional_encoding


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
        self.register_buffer("pos_token", sincos_2d_positional_encoding(self.hidden_dim, extra_tokens, size))
        self.is_sincos_pos_token = True
