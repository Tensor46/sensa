import logging
import pathlib
from typing import Annotated, Any, Literal

import pydantic

from sensa.params.base import BaseParams


class DataParams(BaseParams):
    """Parameters for constructing an image dataset.

    Fields
    ------
        path (tuple[pathlib.Path]):
            One or more directories, each containing subfolders per class label.
        size (tuple[int, int]):
            Desired output image size as (height, width).
        mode (Literal["RGB", "L"], default="RGB"):
            Color mode: "RGB" for color, "L" for grayscale.
        interpolation (Literal[2, 3], default=2):
            Resize method: 2 = BILINEAR, 3 = BICUBIC.
        is_test (bool, default=False):
            If True, use test-time transforms; otherwise use training transforms.
        kwargs (dict[str, Any]):
            All other parameters are captured here.
    """

    path: tuple[pathlib.Path]
    size: tuple[int, int]
    mode: Annotated[Literal["RGB", "L"], pydantic.Field(default="RGB")]
    interpolation: Annotated[Literal[2, 3], pydantic.Field(default=2)]
    is_test: Annotated[bool, pydantic.Field(default=False)]

    @pydantic.field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, data: Any) -> tuple[pathlib.Path]:
        if isinstance(data, pathlib.Path | str):
            data = [data]
        if isinstance(data, list | tuple):
            data = tuple(pathlib.Path(x) if isinstance(x, str) else x for x in data)
        if isinstance(data, list | tuple) and not all(isinstance(x, pathlib.Path) and x.is_dir() for x in data):
            logging.error(f"{cls.__name__}: invalid path {data}")
            raise NotADirectoryError(f"DataSpecs.path :: invalid path {data}")
        if isinstance(data, list | tuple):  # full path
            data = tuple(x.resolve() for x in data)
        return data
