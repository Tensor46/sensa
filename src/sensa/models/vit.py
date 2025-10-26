import logging
import math
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from itertools import pairwise
from typing import Any, Literal

import pydantic
import torch
import torchvision as tv

from sensa.layers import mask_utils
from sensa.layers.dyt import DyT2D
from sensa.layers.encoder import Encoder2, Encoder2Config
from sensa.layers.last_pool import LastPool
from sensa.layers.rmsnorm import RMSNorm2D
from sensa.models.base import BaseModel
from sensa.models.registry import register_model


class StemConfig(pydantic.BaseModel):
    """
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
        act_layer (Callable[..., torch.nn.Module], optional):
            Activation layer for the stem blocks. Defaults to torch.nn.SiLU.
        norm_layer (Callable[..., torch.nn.Module] | str, optional):
            Normalization layer for the stem blocks. Defaults to torch.nn.BatchNorm2d.
    """

    in_channels: int = 3
    out_channels: int = 128
    patch_size: int = 8
    first_stride: int = 2
    last_stride: int = 4
    act_layer: Callable[..., torch.nn.Module] = torch.nn.SiLU
    norm_layer: Callable[..., torch.nn.Module] | None = torch.nn.BatchNorm2d

    @pydantic.field_validator("out_channels", mode="after")
    @classmethod
    def validate_name_length(cls, value: int) -> int:
        if value % 32 != 0:
            logging.error(f"VIT: out_channels must be an integer and multiple of 32 - {value}.")
            raise ValueError(f"out_channels must be an integer and multiple of 32 - {value}.")
        return value

    @pydantic.field_validator("act_layer", mode="before")
    @classmethod
    def validate_act_layer(cls, data: Callable[..., torch.nn.Module] | str) -> Callable[..., torch.nn.Module]:
        """Validate the activation layer."""
        match data:
            case "gelu":
                data = torch.nn.GELU
            case "mish":
                data = torch.nn.Mish
            case "relu":
                data = torch.nn.ReLU
            case "silu":
                data = torch.nn.SiLU
        return data

    @pydantic.field_validator("norm_layer", mode="before")
    @classmethod
    def validate_norm_layer(cls, data: Callable[..., torch.nn.Module] | str) -> Callable[..., torch.nn.Module]:
        """Validate the normalization layer."""
        match data:
            case "batchnorm":
                data = torch.nn.BatchNorm2d
            case "dyt":
                data = DyT2D
            case "instancenorm":
                data = torch.nn.InstanceNorm2d
            case "layernorm":
                data = tv.models.convnext.LayerNorm2d
            case "rmsnorm":
                data = RMSNorm2D
        return data


def build_stem(stem_config: StemConfig) -> torch.nn.Sequential:
    """Constructs a convolutional "stem" for patch embedding, using depthwise and
    pointwise convolutions followed by batch normalization.

    An initial Conv2dNormActivation layer (Convolution + BatchNorm + SiLU).
    A sequence of inverted residual blocks and downsampling blocks.

    Parameters:
        stem_config (StemConfig):
            Stem configuration.

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
                        ("norm", stem_config.norm_layer(oc)),
                    ]
                )
            )

    if not (
        stem_config.patch_size % (stem_config.first_stride * stem_config.last_stride) == 0
        and math.log2(stem_config.patch_size / stem_config.first_stride / stem_config.last_stride).is_integer()
        and math.log2(stem_config.patch_size / stem_config.first_stride / stem_config.last_stride) >= 0
    ):
        s = stem_config.first_stride * stem_config.last_stride
        msg = f"VIT: patch_size must be [>={s}] and [first_stride * last_stride * powers-of-2] "
        msg += f"- {stem_config.patch_size}."
        logging.error(msg)
        raise ValueError(msg)

    if not isinstance(stem_config.first_stride, int):
        logging.error(f"VIT: first_stride must be an integer >= 1 - {stem_config.first_stride}.")
        raise TypeError(f"first_stride must be an integer >= 1 - {stem_config.first_stride}.")
    if not isinstance(stem_config.last_stride, int):
        logging.error(f"VIT: last_stride must be an integer >= 3 - {stem_config.last_stride}.")
        raise TypeError(f"last_stride must be an integer >= 3 - {stem_config.last_stride}.")
    if stem_config.last_stride < 3:
        logging.error(f"VIT: last_stride must be an integer >= 3 - {stem_config.last_stride}.")
        raise ValueError(f"last_stride must be an integer >= 3 - {stem_config.last_stride}.")

    # stem and inverted residual config
    defaults = {"use_se": False, "activation": "HS", "stride": 1, "dilation": 1, "width_mult": 1.0}
    IRC = partial(tv.models.mobilenetv3.InvertedResidualConfig, **defaults)
    IRL = tv.models.mobilenetv3.InvertedResidual
    channels = [
        stem_config.out_channels // (2**i)
        for i in range(2 + int(math.log2(stem_config.patch_size / stem_config.first_stride / stem_config.last_stride)))
    ]
    channels = channels[::-1]

    # build stem
    stem = torch.nn.Sequential()
    stem.add_module(
        "initial",
        tv.ops.Conv2dNormActivation(
            stem_config.in_channels,
            channels[0],
            kernel_size=(stem_config.first_stride + 3)
            if ((stem_config.first_stride + 3) & 1)
            else (stem_config.first_stride + 2),
            stride=stem_config.first_stride,
            norm_layer=stem_config.norm_layer,
            activation_layer=torch.nn.SiLU,
        ),
    )
    for i, (ic, oc) in enumerate(pairwise(channels)):
        stage = torch.nn.Sequential()
        for j in range(i + 1):
            stage.add_module(f"ir_{j}", IRL(IRC(ic, 3, ic, ic), norm_layer=stem_config.norm_layer))
        stride = 2 if oc != stem_config.out_channels else stem_config.last_stride
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
        stem_config (StemConfig | dict[str, Any]):
            Stem configuration.
        encoder_config (Encoder2Config | dict[str, Any]):
            Encoder configuration.
        mask_ratio (float, optional):
            Fraction of tokens to mask during training (for masked autoencoding). Defaults to 0.0.
        num_labels (int, optional):
            Number of labels for classification head. Defaults to 1000 (e.g., ImageNet).
        last_pool (Literal["avg", "full", "half", "token", None], optional):
            Pooling strategy after the encoder. One of:
                "avg": global average pooling
                "full": flatten all tokens
                "half": flatten half of the tokens after maxpool
                "token": use a learnable [CLS] token
                None: no pooling
                Defaults to "token".
    """

    def __init__(
        self,
        image_size: int | tuple[int, int],
        stem_config: StemConfig | dict[str, Any],
        encoder_config: Encoder2Config | dict[str, Any],
        mask_ratio: float = 0.0,
        num_labels: int | None = 1000,
        last_pool: Literal["avg", "full", "half", "token", None] = "token",
    ):
        super().__init__()
        if stem_config["out_channels"] != encoder_config["hidden_dim"]:
            raise ValueError(
                f"VIT: stem_config.out_channels must be equal to encoder_config.hidden_dim "
                f"- {stem_config['out_channels']}-{encoder_config['hidden_dim']}."
            )
        stem_config = StemConfig(**stem_config)
        self.image_size = torch.nn.modules.utils._pair(image_size)
        self.patch_size = stem_config.patch_size
        if image_size[0] % stem_config.patch_size != 0 or image_size[1] % stem_config.patch_size != 0:
            logging.error(f"VIT: image_size must be divisble by patch_size - {image_size}-{stem_config.patch_size}.")
            raise ValueError(f"VIT: image_size must be divisble by patch_size - {image_size}-{stem_config.patch_size}.")

        self.mask_ratio = mask_ratio
        self.num_labels = num_labels
        self.last_pool = last_pool
        self.stem = build_stem(stem_config)
        self.stem_size = self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size
        extra_tokens: int = 0
        if last_pool == "token":
            self.class_token = torch.nn.Parameter(torch.zeros(1, 1, encoder_config["hidden_dim"]))
            extra_tokens += 1
        encoder_config["size"] = self.stem_size
        encoder_config["extra_tokens"] = extra_tokens
        self.encoder_config = Encoder2Config(**encoder_config)

        self.encoder = Encoder2.from_config(self.encoder_config)
        self.seq_length = self.encoder_config.seq_length

        if self.mask_ratio > 0:
            self.mask_token = torch.nn.Parameter(torch.zeros(self.encoder_config.hidden_dim))

        if self.last_pool is not None:
            self.pool = LastPool(pool=self.last_pool, size=self.stem_size)
            if isinstance(self.num_labels, int) and self.num_labels:
                if self.last_pool in ("avg", "token"):
                    self.head = torch.nn.Linear(self.encoder_config.hidden_dim, self.num_labels)
                if self.last_pool == "full":
                    self.head = torch.nn.Linear(self.encoder_config.hidden_dim * self.seq_length, self.num_labels)
                if self.last_pool == "half":
                    h, w = self.pool.size_after_pool
                    self.head = torch.nn.Linear(self.encoder_config.hidden_dim * h * w, self.num_labels)

                torch.nn.init.normal_(self.head.weight, std=0.01)
                torch.nn.init.zeros_(self.head.bias)
        self._initialize()

    def extend_sizes(self, size: tuple[int, int]) -> None:
        if not isinstance(size, list | tuple):
            logging.error(f"VIT: extend_sizes size must be tuple[int, int] - {size}.")
            raise TypeError(f"VIT: extend_sizes size must be tuple[int, int] - {size}.")
        if not (len(size) == 2 and all(isinstance(val, int) for val in size)):
            logging.error(f"VIT: extend_sizes size must be tuple[int, int] - {size}.")
            raise TypeError(f"VIT: extend_sizes size must be tuple[int, int] - {size}.")
        if size[0] % self.patch_size != 0 or size[1] % self.patch_size != 0:
            logging.error(f"VIT: extend_sizes requires size that are multiple of patch_size - {size}.")
            raise ValueError(f"VIT: extend_sizes requires size that are multiple of patch_size - {size}.")
        self.encoder.extend_sizes((size[0] // self.patch_size, size[1] // self.patch_size))

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
        x = self.stem(x)
        x = x.reshape(n, self.encoder_config.hidden_dim, (h // p) * (w // p)).permute(0, 2, 1)
        return x

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Returns parameter groups for optimizer.
        Separating stem, transformer blocks, normalization, positional embeddings, and head.
        """
        groups = []
        groups += self._param_groups(self.stem)
        for i in range(0, len(self.encoder.layers), 2):
            groups += self._param_groups(self.encoder.layers[slice(i, i + 2)])
        self._param_groups(self.encoder.ln, groups=groups[-2:])
        if (
            hasattr(self.encoder, "pos_token")
            and isinstance(self.encoder.pos_token, torch.nn.Parameter)
            and self.encoder.pos_token.requires_grad
        ):
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
