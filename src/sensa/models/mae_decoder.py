from collections.abc import Callable
from functools import partial

import torch

from sensa.layers.encoder import Encoder
from sensa.models.base import BaseModel
from sensa.models.registry import register_model


@register_model("MAEDecoder")
class MAEDecoder(BaseModel):
    """Masked Autoencoder (MAE) decoder that reconstructs image patches from encoded tokens.

    Args:
        image_size (tuple[int, int]):
            Height and width of the original image in pixels.
        patch_size (int):
            Side length of each square patch.
        channels (int):
            Number of channels in the input image (e.g., 3 for RGB).
        encoder_dim (int):
            Feature length of the encoder's output tokens.
        decoder_dim (int):
            Feature length used inside the decoder.
        mlp_dim (int):
            Hidden dimension of the MLP in each transformer block.
        num_layers (int):
            Number of transformer layers in the decoder.
        num_heads (int):
            Number of attention heads in each decoder layer.
        norm_layer (Callable[..., torch.nn.Module], optional):
            Constructor for the normalization layer. Defaults to `partial(torch.nn.LayerNorm, eps=1e-6)`.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: int,
        channels: int,
        encoder_dim: int,
        decoder_dim: int,
        mlp_dim: int,
        num_layers: int,
        num_heads: int,
        norm_layer: Callable[..., torch.nn.Module] | None = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.stem_size = image_size[0] // patch_size, image_size[1] // patch_size

        # map encoder outputs to decoder dimension if needed
        self.translate = (
            torch.nn.Identity() if encoder_dim == decoder_dim else torch.nn.Linear(encoder_dim, decoder_dim)
        )
        # build the decoder transformer
        self.decoder = Encoder(
            seq_length=self.stem_size[0] * self.stem_size[1],
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=decoder_dim,
            mlp_dim=mlp_dim,
            dropout=0.0,
            attention_dropout=0.0,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6) if norm_layer is None else norm_layer,
        )
        self.decoder.use_sincos_pos_token(extra_tokens=0, size=self.stem_size)
        # projection head to map decoder outputs back to patch pixels
        self.predict = torch.nn.Linear(decoder_dim, patch_size * patch_size * channels)

    def forward(self, x_encoded: torch.Tensor) -> torch.Tensor:
        """Reconstruct image patches from encoded tokens.

        Args:
            x_encoded (torch.Tensor):
                Encoded token tensor of shape (B, N, encoder_dim),
                without extra tokens (e.g., class tokens) and with mask tokens in place.

        Returns:
            torch.Tensor:
                Reconstructed patches of shape (B, N, patch_size*patch_size*channels).
        """
        x_decoded = self.forward_features(x_encoded)["features"]
        x_patches = self.predict(x_decoded)
        return x_patches

    def forward_features(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"features": self.decoder(self.translate(tensor))}
