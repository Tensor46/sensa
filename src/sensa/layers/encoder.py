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

    def forward(self, tensor: torch.Tensor, indices_to_keep: torch.Tensor | None = None) -> torch.Tensor:
        torch._assert(tensor.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {tensor.shape}")
        o = tensor + self.pos_token
        if indices_to_keep is not None:
            o = mask_utils.mask_tensor(o, indices_to_keep=indices_to_keep)
        return self.ln(self.layers(self.dropout(o)))

    def use_sincos_pos_token(self, extra_tokens: int, size: tuple[int, int]) -> None:
        del self.pos_token
        self.register_buffer("pos_token", sincos_2d_positional_encoding(self.hidden_dim, extra_tokens, size))
        self.is_sincos_pos_token = True
