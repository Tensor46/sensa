import pathlib
from abc import ABC, abstractmethod
from typing import Any

import pydantic
import torch
from PIL import Image as ImPIL
from torchvision.transforms import v2 as tv2

from sensa.data.read_images import read_label_per_folder
from sensa.params.data import DataParams


class BaseImageFolder(ABC):
    """Dataset base class for folder-per-class image collections.

    Expects a directory structure like:
        root/
            class1/
                img1.png
                img2.jpg
            class2/
                img1.png
                ...
    """

    __params__: pydantic.BaseModel = DataParams

    def __init__(self, **kwargs) -> None:
        """Initialize dataset params, build sample database, and set transforms.

        Args:
            kwargs: Either a `__params__` instance or keyword args for DataParams.
        """
        self.dbase: list[tuple[pathlib.Path, int]] = []
        self.num_labels: int = 0

        # allow passing an existing DataParams instance
        if len(kwargs) == 1 and "params" in kwargs and isinstance(kwargs["params"], self.__params__):
            kwargs = kwargs["params"].model_dump()
        self.params = self.__params__(**self.default_specs(**kwargs))
        # populate the internal database and label count
        for path in self.params.path:
            self.add_to_dbase(path)
        # choose transforms based on test/train mode
        self._transforms = self.default_transforms_test() if self.params.is_test else self.default_transforms()

    def add_to_dbase(self, path: pathlib.Path) -> None:
        """Add samples to dbase given path."""
        self.dbase += read_label_per_folder(
            path=path,
            start_label_id_at=self.num_labels,
            stop_label_id_at=self.params.kwargs.get("stop_label_id_at", None),
            max_samples_per_label=self.params.kwargs.get("max_samples_per_label", None),
        )
        self.num_labels = (self.dbase[-1][-1] + 1) if len(self.dbase) else 0

    def __len__(self) -> int:
        return len(self.dbase)

    def __getitem__(self, index: int) -> tuple[Any]:
        file_name, label = self.dbase[index % len(self)]
        return (*self.process(file_name), label)

    def default_specs(self, **kwargs) -> dict[str, Any]:
        """Hook to adjust or fill default params before model creation."""
        return kwargs

    def read_image(self, file_name: pathlib.Path) -> ImPIL.Image:
        """Open an image file and convert to the configured color mode."""
        return ImPIL.open(file_name).convert(self.params.mode)

    def to_tensor(self, image: ImPIL.Image) -> torch.Tensor:
        """Convert a PIL image to a uint8 tensor."""
        return tv2.functional.to_dtype(tv2.functional.to_image(image), torch.uint8)

    def to_pil(self, tensor: torch.Tensor) -> ImPIL.Image:
        """Convert a tensor back to a PIL image."""
        return tv2.functional.to_pil_image(tensor)

    @abstractmethod
    def default_transforms(self) -> tv2.Compose:
        """Define and return training transforms."""
        ...

    @abstractmethod
    def default_transforms_test(self) -> tv2.Compose:
        """Define and return test transforms."""
        ...

    @property
    def transforms(self) -> tv2.Compose:
        """Get the active transform pipeline."""
        return self._transforms

    def process(self, file_name: Any) -> tuple[Any]:
        """Load an image file, apply transforms, and return a tensor tuple.

        Args:
            file_name (pathlib.Path): Path to image file.

        Returns:
            tuple[torch.Tensor]: Transformed image tensor.
        """
        tensor = self.to_tensor(self.read_image(file_name))
        return (self.transforms(tensor),)
