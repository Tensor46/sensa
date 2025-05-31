import logging
import math
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from itertools import pairwise
from typing import Any, Literal

import torch
import torchvision as tv

from sensa.layers import mask_utils
from sensa.layers.dyt import DyT
from sensa.layers.encoder import Encoder
from sensa.layers.last_pool import LastPool
from sensa.models.base import BaseModel
from sensa.models.registry import register_model


def build_stem(
    in_channels: int = 3,
    out_channels: int = 128,
    patch_size: int = 8,
    first_stride: int = 2,
    last_stride: int = 4,
) -> torch.nn.Sequential:
    """Constructs a convolutional "stem" for patch embedding, using depthwise and
    pointwise convolutions followed by batch normalization.

    An initial Conv2dNormActivation layer (Convolution + BatchNorm + SiLU).
    A sequence of inverted residual blocks and downsampling blocks.

    Parameters:
        in_channels (int, optional):
            Number of channels in the input image. Defaults to 3.
        out_channels (int, optional):
            Number of channels after the stem and the hidden dimensionality of the transformer.
            Must be an integer multiple of 32. Defaults to 128.
        patch_size (int, optional):
            Patch size for the Vision Transformer. Must be divisible by first_stride * last_stride
            and satisfy power-of-two constraints. Defaults to 8.
        first_stride (int, optional):
            Stride for the initial convolution layer. Must be ≥ 1. Defaults to 2.
        last_stride (int, optional):
            Stride for the final downsampling block. Must be ≥ 3. Defaults to 4.

    Returns:
        A torch.nn.Sequential module comprising:
    """

    class DownBlock(torch.nn.Sequential):
        def __init__(self, ic: int, oc: int, stride: int):
            super().__init__(
                OrderedDict(
                    [
                        ("down", torch.nn.Conv2d(ic, ic, stride, stride, groups=ic)),
                        ("grow", torch.nn.Conv2d(ic, oc, 1)),
                        ("norm", torch.nn.BatchNorm2d(oc)),
                    ]
                )
            )

    norm_layer = torch.nn.BatchNorm2d
    if not isinstance(out_channels, int):
        logging.error(f"VIT: out_channels must be an integer and multiple of 32 - {out_channels}.")
        raise TypeError(f"out_channels must be an integer and multiple of 32 - {out_channels}.")
    if out_channels % 32 != 0:
        logging.error(f"VIT: out_channels must be an integer and multiple of 32 - {out_channels}.")
        raise ValueError(f"out_channels must be an integer and multiple of 32 - {out_channels}.")

    if not (isinstance(patch_size, int) or (isinstance(patch_size, float) and patch_size.is_integer())):
        logging.error(f"VIT: patch_size must be an integer - {patch_size}.")
        raise TypeError(f"patch_size must be an integer - {patch_size}.")
    if not (
        patch_size % (first_stride * last_stride) == 0
        and math.log2(patch_size / first_stride / last_stride).is_integer()
        and math.log2(patch_size / first_stride / last_stride) >= 0
    ):
        s = first_stride * last_stride
        logging.error(f"VIT: patch_size must be [>={s}] and [first_stride * last_stride * powers-of-2] - {patch_size}.")
        raise ValueError(f"patch_size must be [>={s}] and [first_stride * last_stride * powers-of-2] - {patch_size}.")

    if not isinstance(first_stride, int):
        logging.error(f"VIT: first_stride must be an integer >= 1 - {first_stride}.")
        raise TypeError(f"first_stride must be an integer >= 1 - {first_stride}.")
    if not isinstance(last_stride, int):
        logging.error(f"VIT: last_stride must be an integer >= 3 - {last_stride}.")
        raise TypeError(f"last_stride must be an integer >= 3 - {last_stride}.")
    if last_stride < 3:
        logging.error(f"VIT: last_stride must be an integer >= 3 - {last_stride}.")
        raise ValueError(f"last_stride must be an integer >= 3 - {last_stride}.")

    # stem and inverted residual config
    defaults = {"use_se": False, "activation": "HS", "stride": 1, "dilation": 1, "width_mult": 1.0}
    IRC = partial(tv.models.mobilenetv3.InvertedResidualConfig, **defaults)
    IRL = tv.models.mobilenetv3.InvertedResidual
    channels = [out_channels // (2**i) for i in range(2 + int(math.log2(patch_size / first_stride / last_stride)))]
    channels = channels[::-1]

    # build stem
    stem = torch.nn.Sequential()
    stem.add_module(
        "initial",
        tv.ops.Conv2dNormActivation(
            in_channels,
            channels[0],
            kernel_size=(first_stride + 3) if ((first_stride + 3) & 1) else (first_stride + 2),
            stride=first_stride,
            norm_layer=norm_layer,
            activation_layer=torch.nn.SiLU,
        ),
    )
    for i, (ic, oc) in enumerate(pairwise(channels)):
        stage = torch.nn.Sequential()
        for j in range(i + 1):
            stage.add_module(f"ir_{j}", IRL(IRC(ic, 3, ic, ic), norm_layer=norm_layer))
        stride = 2 if oc != out_channels else last_stride
        stage.add_module("down", DownBlock(ic, oc, stride))
        stem.add_module(f"stage_{i}", stage)

    return stem


@register_model("VIT")
class VIT(BaseModel):
    """A Vision Transformer with a convolutional stem, transformer encoder layers,
    optional masking for pretraining, and various pooling strategies.

    Args:
        image_size (int or tuple[int, int]):
            Height and width of the input image. Must be divisible by patch_size.
        patch_size (int):
            Size of each patch extracted by the stem.
        num_layers (int):
            Number of transformer encoder layers.
        num_heads (int):
            Number of attention heads per layer.
        hidden_dim (int):
            Dimensionality of the token embeddings (i.e., output channels of the stem).
        mlp_dim (int):
            Dimensionality of the MLP in each transformer block.
        mask_ratio (float, optional):
            Fraction of tokens to mask during training (for masked autoencoding). Defaults to 0.0.
        num_classes (int, optional):
            Number of classes for classification head. Defaults to 1000 (e.g., ImageNet).
        in_channels (int, optional):
            Number of input channels. Defaults to 3 (RGB).
        first_stride (int, optional):
            Stride for the stem's initial convolution. Defaults to 2.
        last_pool (Literal[...], optional):
            Pooling strategy after the encoder. One of:
                "avg": global average pooling
                "full": flatten all tokens
                "half": flatten half of the tokens after maxpool
                "token": use a learnable [CLS] token
                None: no pooling
                Defaults to "token".
        last_stride (int, optional):
            Stride for the stem's final downsampling block. Defaults to 4.
        norm_layer (Callable[..., nn.Module] | str, optional):
            Normalization layer for encoder. Defaults to LayerNorm(eps=1e-6).
        use_sincos_pos_token (bool, optional):
            Whether to use fixed sinusoidal positional embeddings. Defaults to False.
    """

    def __init__(
        self,
        image_size: int | tuple[int, int],
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        mask_ratio: float = 0.0,
        num_classes: int = 1000,
        in_channels: int = 3,
        first_stride: int = 2,
        last_pool: Literal["avg", "full", "half", "token", None] = "token",
        last_stride: int = 4,
        norm_layer: Callable[..., torch.nn.Module] | str | None = None,
        use_sincos_pos_token: bool = False,
    ):
        super().__init__()
        self.image_size = torch.nn.modules.utils._pair(image_size)
        self.patch_size = patch_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            logging.error(f"VIT: image_size must be divisble by patch_size - {image_size}-{patch_size}.")
            raise ValueError(f"VIT: image_size must be divisble by patch_size - {image_size}-{patch_size}.")

        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.mask_ratio = mask_ratio
        self.num_classes = num_classes
        self.last_pool = last_pool
        if norm_layer is None:
            norm_layer = partial(torch.nn.LayerNorm, eps=1e-6)
        if isinstance(norm_layer, str):
            if norm_layer == "layernorm":
                norm_layer = partial(torch.nn.LayerNorm, eps=1e-6)
            elif norm_layer == "dyt":
                norm_layer = DyT
            else:
                logging.error(f"VIT: norm_layer must be layernorm | dyt when string - {norm_layer}.")
                raise ValueError(f"VIT: norm_layer must be layernorm | dyt when string - {norm_layer}.")

        self.stem = build_stem(
            in_channels=in_channels,
            out_channels=hidden_dim,
            patch_size=patch_size,
            first_stride=first_stride,
            last_stride=last_stride,
        )
        self.stem_size = self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size
        seq_length = self.stem_size[0] * self.stem_size[1]
        if last_pool == "token":
            self.class_token = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim))
            seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=0.0,
            attention_dropout=0.0,
            norm_layer=norm_layer,
        )
        if use_sincos_pos_token:
            self.encoder.use_sincos_pos_token(extra_tokens=int(last_pool == "token"), size=self.stem_size)
        self.seq_length = seq_length

        if self.mask_ratio > 0:
            self.mask_token = torch.nn.Parameter(torch.zeros(hidden_dim))

        if self.last_pool is not None:
            self.pool = LastPool(pool=self.last_pool, size=self.stem_size)
            if isinstance(self.num_classes, int) and self.num_classes:
                if self.last_pool in ("avg", "token"):
                    self.head = torch.nn.Linear(self.hidden_dim, self.num_classes)
                if self.last_pool == "full":
                    self.head = torch.nn.Linear(self.hidden_dim * self.seq_length, self.num_classes)
                if self.last_pool == "half":
                    h, w = self.pool.size_after_pool
                    self.head = torch.nn.Linear(self.hidden_dim * h * w, self.num_classes)

                torch.nn.init.normal_(self.head.weight, std=0.01)
                torch.nn.init.zeros_(self.head.bias)
        self._initialize()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Runs `forward_features`, applies pooling and classification head."""
        o = self.forward_features(tensor)["features"]
        if hasattr(self, "pool"):
            o = self.pool(o)
        if hasattr(self, "head"):
            o = self.head(o)
        return o

    def forward_features(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """Applies stem, optional class token concat, random masking (training only), and transformer encoding.

        Returns a dict with keys:
            features:
                encoded token features
            indices_to_keep, indices_to_mask, indices_to_restore:
                mask bookkeeping tensors (when mask_ratio>0 and training).
        """
        o = self._process_input(tensor)
        if hasattr(self, "class_token"):
            batch_class_token = self.class_token.expand(o.shape[0], -1, -1)
            o = torch.cat([batch_class_token, o], dim=1)

        indices_to_keep, indices_to_mask, indices_to_restore = None, None, None
        if self.mask_ratio > 0 and self.training:
            indices_to_keep, indices_to_mask, indices_to_restore = mask_utils.random_mask_indices(
                o,
                ratio=self.mask_ratio,
                num_cls_tokens=self.class_token.size(1) if hasattr(self, "class_token") else 0,
            )

        o = self.encoder(o, indices_to_keep=indices_to_keep)
        if self.mask_ratio > 0 and self.training:
            o = mask_utils.unmask_tensor(o, indices_to_restore, self.mask_token)
        return {
            "features": o,
            "indices_to_keep": indices_to_keep,
            "indices_to_mask": indices_to_mask,
            "indices_to_restore": indices_to_restore,
        }

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        """Validates input dimensions. Applies stem and reshapes output to (batch, seq_length, hidden_dim)."""
        n, _, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size[0], f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size[1], f"Wrong image width! Expected {self.image_size} but got {w}!")
        x = self.stem(x)
        x = x.reshape(n, self.hidden_dim, (h // p) * (w // p)).permute(0, 2, 1)
        return x

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Returns parameter groups for optimizer.
        Separating stem, transformer blocks, normalization, positional embeddings, and head.
        """
        groups = []
        groups += self._param_groups(self.stem)
        for i in range(0, len(self.encoder.layers), 2):
            groups += self._param_groups(self.encoder.layers[i : i + 2])
        self._param_groups(self.encoder.ln, groups=groups[-2:])
        if isinstance(self.encoder.pos_token, torch.nn.Parameter) and self.encoder.pos_token.requires_grad:
            groups[-1]["params"].append(self.encoder.pos_token)
        if (
            hasattr(self, "class_token")
            and isinstance(self.class_token, torch.nn.Parameter)
            and self.class_token.requires_grad
        ):
            groups[-1]["params"].append(self.class_token)
        if (
            hasattr(self, "mask_token")
            and isinstance(self.mask_token, torch.nn.Parameter)
            and self.mask_token.requires_grad
        ):
            groups[-1]["params"].append(self.mask_token)
        if hasattr(self, "head"):
            groups += self._param_groups(self.head)
        return groups
