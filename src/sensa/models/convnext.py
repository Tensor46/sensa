from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import pydantic
import torch
import torch.nn as nn
import torchvision as tv

from sensa.layers.last_pool import LastPool4D
from sensa.layers.regularized_residual import RegularizedResidual
from sensa.models.base import BaseModel
from sensa.models.registry import register_model
from sensa.params.layer import ActivationParams, Norm2dParams


class BlockParams(ActivationParams, Norm2dParams):
    """ConvNeXt block parameters.

    Parameters:
        channels (int):
            Number of channels in the input image.
        kernel_size (int):
            Kernel size for the convolution layer. Default is 7.
        groups (int):
            Number of groups for the convolution layer.
        expand (float):
            Expansion factor for the convolution layer.
        act_layer (Callable[..., torch.nn.Module], optional):
            Activation layer for the stem blocks. Defaults "gelu".
        norm_layer (Callable[..., torch.nn.Module] | str, optional):
            Normalization layer for the stem blocks. Defaults "layernorm".
        p_regularize_dp (float):
            Regularization parameter for the dot product. Defaults to 0.0.
        p_regularize_ja (float):
            Regularization parameter for the Jacobian. Defaults to 0.0.
    """

    channels: int
    kernel_size: int
    groups: int
    bias: bool
    expand: float
    act_layer: Callable[..., torch.nn.Module]
    norm_layer: Callable[..., torch.nn.Module]
    p_regularize_dp: float
    p_regularize_ja: float

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs.setdefault("kernel_size", 7)
        kwargs.setdefault("groups", kwargs["channels"])
        kwargs.setdefault("bias", True)
        kwargs.setdefault("expand", 4.0)
        kwargs.setdefault("act_layer", "gelu")
        kwargs.setdefault("norm_layer", "layernorm")
        kwargs.setdefault("p_regularize_dp", 0.0)
        kwargs.setdefault("p_regularize_ja", 0.0)
        return kwargs


class Block(RegularizedResidual):
    def __init__(self, params: BlockParams):
        cnxn = partial(torch.nn.Conv2d, padding=params.kernel_size // 2, groups=params.groups, bias=params.bias)
        c1x1 = partial(torch.nn.Conv2d, kernel_size=1, bias=params.bias)
        super().__init__(
            residue=None,
            non_residue=torch.nn.Sequential(
                OrderedDict(
                    [
                        ("convnxn", cnxn(params.channels, params.channels, params.kernel_size)),
                        ("norm", params.norm_layer(params.channels)),
                        ("linear1", c1x1(params.channels, int(params.expand * params.channels))),
                        ("nlinear", params.act_layer()),
                        ("linear2", c1x1(int(params.expand * params.channels), params.channels)),
                    ]
                ),
            ),
            p_regularize_dp=params.p_regularize_dp,
            p_regularize_ja=params.p_regularize_ja,
            add_non_residue_scaler=True,
        )

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "Block":
        return cls(BlockParams(**kwargs))


class Params(ActivationParams, Norm2dParams):
    """ConvNeXt parameters.

    Parameters:
        in_channels (int):
            Number of channels in the input image.
        first_stride (int):
            Stride for the stem block.
        blocks_per_stage (tuple[int, ...]):
            Number of blocks in each stage.
        channels_per_stage (tuple[int, ...]):
            Number of channels in each stage.
        kernel_size_per_stage (tuple[int, ...]):
            Kernel size for each stage. Default is (7, ...).
        expand (float):
            Expansion factor for the convolution layer.
        act_layer (Callable[..., torch.nn.Module], optional):
            Activation layer for the stem blocks. Defaults "gelu".
        norm_layer (Callable[..., torch.nn.Module] | str, optional):
            Normalization layer for the stem blocks. Defaults "layernorm".
        p_regularize_dp (float):
            Regularization parameter for the dot product. Defaults to 0.1.
        num_labels (int, optional):
            Number of labels for classification head. Defaults to 1000 (e.g., ImageNet).
        last_pool (Literal["avg", "full", "half", None], optional):
            Pooling strategy after the encoder. One of:
                "avg": global average pooling
                "full": flatten all tokens
                "half": flatten half of the tokens after maxpool
                None: no pooling
                Defaults to "avg".
    """

    in_channels: int
    first_stride: int
    blocks_per_stage: tuple[int, ...]
    channels_per_stage: tuple[int, ...]
    kernel_size_per_stage: tuple[int, ...]
    expand: float
    act_layer: Callable[..., torch.nn.Module]
    norm_layer: Callable[..., torch.nn.Module]
    p_regularize_dp: float
    num_labels: int | None = None
    last_pool: Literal["avg", "full", "half", None] = "avg"

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs.setdefault("in_channels", 3)
        kwargs.setdefault("first_stride", 4)
        if "blocks_per_stage" not in kwargs and "channels_per_stage" not in kwargs:
            kwargs.setdefault("blocks_per_stage", (3, 3, 9, 3))
            kwargs.setdefault("channels_per_stage", (96, 192, 384, 768))
        kwargs.setdefault("kernel_size_per_stage", (7,) * len(kwargs["blocks_per_stage"]))
        if len(kwargs["blocks_per_stage"]) != len(kwargs["channels_per_stage"]):
            raise ValueError("ConvNeXtConfig: len(blocks_per_stage) must be same as len(channels_per_stage).")
        if len(kwargs["blocks_per_stage"]) != len(kwargs["kernel_size_per_stage"]):
            raise ValueError("ConvNeXtConfig: len(blocks_per_stage) must be same as len(kernel_size_per_stage).")

        kwargs.setdefault("expand", 4.0)
        kwargs.setdefault("act_layer", "gelu")
        kwargs.setdefault("norm_layer", "layernorm")
        kwargs.setdefault("p_regularize_dp", 0.1)
        kwargs.setdefault("num_labels", None)
        kwargs.setdefault("last_pool", "avg")
        return kwargs

    def get_block_params_per_stage(self, stage: int) -> tuple[BlockParams, ...]:
        if stage >= len(self.blocks_per_stage):
            raise ValueError(f"Params: Stage {stage} is out of range.")

        total_blocks = sum(self.blocks_per_stage)
        start = sum(self.blocks_per_stage[:stage])
        end = start + self.blocks_per_stage[stage]
        dps = torch.linspace(0, self.p_regularize_dp, total_blocks)[start:end]
        return tuple(
            BlockParams(
                channels=self.channels_per_stage[stage],
                kernel_size=self.kernel_size_per_stage[stage],
                groups=self.channels_per_stage[stage],
                expand=self.expand,
                act_layer=self.act_layer,
                norm_layer=self.norm_layer,
                p_regularize_dp=dp.item(),
                p_regularize_ja=0.0,
            )
            for dp in dps
        )


@register_model("ConvNeXt")
class ConvNeXt(BaseModel):
    def __init__(self, params: Params) -> None:
        super().__init__()
        self.params = params
        self.stem = tv.ops.Conv2dNormActivation(
            params.in_channels,
            params.channels_per_stage[0],
            kernel_size=params.first_stride if params.first_stride >= 4 else 5,
            stride=params.first_stride,
            padding=0 if params.first_stride >= 4 else 2,
            norm_layer=params.norm_layer,
            activation_layer=None,
            bias=True,
        )
        self.stages = torch.nn.ModuleDict()
        ic: int = params.channels_per_stage[0]
        for i, oc in enumerate(params.channels_per_stage):
            stage = []
            if ic != oc:
                stage.append(("norm", params.norm_layer(ic)))
                stage.append(("conv", nn.Conv2d(ic, oc, 2, 2)))

            for j, bp in enumerate(self.params.get_block_params_per_stage(i)):
                stage.append((f"cnxt{j + 1}", Block(bp)))

            self.stages[f"stage_{i + 1}"] = torch.nn.Sequential(OrderedDict(stage))
            ic = oc
        if self.params.last_pool is not None:
            self.pool = torch.nn.Sequential(
                OrderedDict(
                    ([("norm", self.params.norm_layer(oc))] if self.params.last_pool != "avg" else [])
                    + [("pool", LastPool4D(pool=self.params.last_pool))]
                    + ([("norm", torch.nn.LayerNorm(oc))] if self.params.last_pool == "avg" else [])
                )
            )

            if self.params.num_labels is not None:
                self.head = torch.nn.LazyLinear(self.params.num_labels)

    @classmethod
    def from_kwargs(cls, **kwargs):
        return cls(Params(**kwargs))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(tensor)
        o = list(features.values())[-1]
        if hasattr(self, "pool"):
            o = self.pool(o)
            if hasattr(self, "head"):
                o = self.head(o)
        return o

    def forward_features(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        o = self.stem(tensor)
        features = {}
        for name, net in self.stages.items():
            features[name] = o = net(o)
        return features

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Returns parameter groups for optimizer.
        Separating stem, transformer blocks, normalization, positional embeddings, and head.
        """
        groups = []
        groups += self._param_groups(self.stem)
        for blocks in self.stages.values():
            groups += self._param_groups(blocks)
        if hasattr(self, "pool"):
            groups += self._param_groups(self.pool)
        if hasattr(self, "head"):
            groups[-1]["params"] += self._param_groups(self.head)["params"]
        return groups
