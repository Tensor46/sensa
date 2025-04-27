import logging
import pathlib
from typing import Annotated, Any

import pydantic

from sensa import assets


class BaseParams(pydantic.BaseModel, extra="ignore"):
    """BaseParams: unknown fields are collected into `kwargs`, and you can load from YAML."""

    kwargs: Annotated[dict[str, Any], pydantic.Field(default={})]

    @classmethod
    def from_yaml(cls, path_to_yaml: pathlib.Path):
        """Load parameters from a YAML file into a BaseParams instance.

        Args:
            path_to_yaml (pathlib.Path):
                Path to the YAML file containing configuration data.

        Returns:
            BaseParams:
                A new instance populated with values from the YAML file.
        """
        if not isinstance(path_to_yaml, pathlib.Path):
            logging.error(f"{cls.__name__}: path_to_yaml must be pathlib.Path - {path_to_yaml}")
            raise TypeError(f"{cls.__name__}: path_to_yaml must be pathlib.Path - {path_to_yaml}")

        return cls(**assets.load_yaml(path_to_yaml))

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_kwargs(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Move any unrecognized keys into `kwargs`."""
        for key in list(data.keys()):
            if key not in cls.model_fields:
                if "kwargs" not in data:
                    data["kwargs"] = {}
                data["kwargs"][key] = data[key]
        return data
