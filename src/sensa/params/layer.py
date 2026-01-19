import inspect
from collections.abc import Callable

import pydantic
import torch
import torchvision as tv


class ActivationParams(pydantic.BaseModel):
    """Activation parameters.

    Parameters:
        act_layer (Callable[..., torch.nn.Module], optional):
            Activation layer for the stem blocks.
    """

    act_layer: Callable[..., torch.nn.Module]

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

    @pydantic.field_serializer("act_layer", when_used="json")
    def serialize_act_layer(self, value: Callable[..., torch.nn.Module]):
        """Serialize act_layer for JSON."""
        if inspect.isclass(value) and issubclass(value, torch.nn.Module):
            return value.__name__
        return str(value)


class NormParams(pydantic.BaseModel):
    """Normalization parameters.

    Parameters:
        norm_layer (Callable[..., torch.nn.Module] | str, optional):
            Normalization layer for the stem blocks. Defaults "layernorm".
    """

    norm_layer: Callable[..., torch.nn.Module]

    @pydantic.field_validator("norm_layer", mode="before")
    @classmethod
    def validate_norm_layer(cls, data: Callable[..., torch.nn.Module] | str) -> Callable[..., torch.nn.Module]:
        """Validate the normalization layer."""
        match data:
            case "batchnorm":
                data = torch.nn.BatchNorm1d
            case "dyt":
                from sensa.layers.dyt import DyT

                data = DyT
            case "instancenorm":
                data = torch.nn.InstanceNorm1d
            case "layernorm":
                data = torch.nn.LayerNorm
            case "rmsnorm":
                from sensa.layers.rmsnorm import RMSNorm

                data = RMSNorm
        return data

    @pydantic.field_serializer("norm_layer", when_used="json")
    def serialize_norm_layer(self, value: Callable[..., torch.nn.Module]):
        """Serialize norm_layer for JSON."""
        if inspect.isclass(value) and issubclass(value, torch.nn.Module):
            return value.__name__
        return str(value)


class Norm2dParams(pydantic.BaseModel):
    """2D Normalization parameters.

    Parameters:
        norm_layer (Callable[..., torch.nn.Module] | str):
            Normalization layer for the stem blocks. Defaults "layernorm".
    """

    norm_layer: Callable[..., torch.nn.Module]

    @pydantic.field_validator("norm_layer", mode="before")
    @classmethod
    def validate_norm_layer(cls, data: Callable[..., torch.nn.Module] | str) -> Callable[..., torch.nn.Module]:
        """Validate the normalization layer."""
        match data:
            case "batchnorm":
                data = torch.nn.BatchNorm2d
            case "dyt":
                from sensa.layers.dyt import DyT2D

                data = DyT2D
            case "instancenorm":
                data = torch.nn.InstanceNorm2d
            case "layernorm":
                data = tv.models.convnext.LayerNorm2d
            case "rmsnorm":
                from sensa.layers.rmsnorm import RMSNorm2D

                data = RMSNorm2D
        return data

    @pydantic.field_serializer("norm_layer", when_used="json")
    def serialize_norm_layer(self, value: Callable[..., torch.nn.Module]):
        """Serialize norm_layer for JSON."""
        if inspect.isclass(value) and issubclass(value, torch.nn.Module):
            return value.__name__
        return str(value)
