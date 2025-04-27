from typing import Any, Literal

import pydantic

from sensa.params.base import BaseParams


class OptimParams(BaseParams):
    """Parameters for optimizer configuration.

    Fields
    ------
        name (Literal["adamw","sgd"]):
            Optimizer algorithm.
        lr (float):
            Initial learning rate.
        lr_end (float):
            Final learning rate (defaults to `lr` if unset).
        lr_max (float | None, default=None):
            Maximum scaled LR given batch size and GPUs.
        lr_min_scaled (float, default=1.0):
            Minimum relative scale factor for the first optimizer parameter group's learning rate.
            During `update_lr`, each group's LR is geometrically interpolated between the base LR
            and `base_lr * lr_min_scaled`, ensuring the first group's LR = `base_lr * lr_min_scaled`
            and last group's LR = `base_lr`. A value of 1.0 results in no decay across groups.
        weight_decay (float, default=5e-4):
            Weight decay factor.
        warmup (float, default=0.1):
            Warmup fraction of total epochs.
        kwargs (dict[str, Any]):
            Additional optimizer-specific settings.
    """

    name: Literal["adamw", "sgd"]
    lr: float
    lr_end: float
    lr_max: float | None = None
    lr_min_scaled: float = 1.0
    weight_decay: float = 5e-4
    warmup: float = 0.1

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "name" in data and isinstance(data["name"], str):
            data["name"] = data["name"].lower()
        if "lr" in data:
            data.setdefault("lr_end", data["lr"])
            data.setdefault("lr_max", None)
        return data

    @pydantic.model_validator(mode="after")
    @classmethod
    def validate_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "name" in data and data["name"] == "sgd":
            data["kwargs"].setdefault("momentum", 0.9)
        return data
