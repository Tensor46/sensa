import torch
from torchvision.transforms import v2 as tv2

from sensa.data.base import BaseImageFolder


class Dataset(BaseImageFolder):
    """Dataset for imagenet."""

    def default_transforms(self) -> tv2.Compose:
        """Build and return the training transform pipeline."""
        return tv2.Compose(
            [
                torch.jit.script(
                    tv2.RandomResizedCrop(
                        size=self.params.size,
                        interpolation=self.params.interpolation,
                        antialias=True,
                    )
                ),
                torch.jit.script(tv2.RandomHorizontalFlip(p=0.5)),
                tv2.ToDtype(torch.float32, scale=True),
            ]
        )

    def default_transforms_test(self) -> tv2.Compose:
        """Build and return the test/eval transform pipeline."""
        return tv2.Compose(
            [
                tv2.Resize(
                    size=(self.params.size[0] + 32, self.params.size[1] + 32),
                    interpolation=self.params.interpolation,
                    antialias=True,
                ),
                tv2.CenterCrop(size=self.params.size),
                tv2.ToDtype(torch.float32, scale=True),
            ]
        )
