import inspect
from collections.abc import Callable

import pydantic
import torch


class ActivationConfig(pydantic.BaseModel):
    """Activation configuration.

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
