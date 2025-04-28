import logging
import pathlib
from typing import Annotated, Any

import pydantic

from sensa.params.base import BaseParams


class LossParams(BaseParams):
    """Parameters for constructing a pytorch loss function.

    Fields
    ------
        name : str
            The expected class name of the loss. Used for sanity-checking
            when loading from YAML or instantiating dynamically.
        skip_freeze_prefixes : list[str]
            List of parameter-name prefixes that should remain trainable (i.e.,
            be exempt from automatic freezing). Defaults to an empty list.
        path_to_checkpoint : pathlib.Path | None
            Optional filesystem path to a pretrained checkpoint. If provided,
            must point to an existing file; otherwise a FileNotFoundError is raised.
    """

    name: str
    skip_freeze_prefixes: Annotated[list[str], pydantic.Field(default=[])]
    path_to_checkpoint: pathlib.Path | None = None

    @pydantic.field_validator("path_to_checkpoint", mode="before")
    @classmethod
    def validate_path(cls, data: Any) -> tuple[pathlib.Path]:
        """Validate the `path_to_checkpoint` field."""
        if isinstance(data, str):
            data = pathlib.Path(data)
        if isinstance(data, pathlib.Path) and not data.is_file():
            data = data.resolve()
            logging.error(f"{cls.__name__}: path_to_checkpoint does not exist - {data}.")
            raise FileNotFoundError(f"{cls.__name__}: path_to_checkpoint does not exist - {data}.")
        return data
