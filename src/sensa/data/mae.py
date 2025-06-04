import torch
from torchvision.transforms import v2 as tv2

from sensa.data.base import BaseImageFolder


class Dataset(BaseImageFolder):
    """Dataset for training Masked Autoencoders (MAE)."""

    def default_transforms(self) -> tv2.Compose:
        """Build and return the training transform pipeline."""
        return tv2.Compose(
            [
                torch.jit.script(
                    tv2.RandomResizedCrop(
                        size=self.params.size,
                        scale=(0.25, 1.0),
                        ratio=(3 / 4, 4 / 3),
                        interpolation=self.params.interpolation,
                        antialias=True,
                    )
                ),
                torch.jit.script(
                    tv2.RandomApply(
                        torch.nn.ModuleList([tv2.ColorJitter(0.1, 0.1, 0.1, 0.05)]),
                        p=0.5,
                    ),
                ),
                torch.jit.script(tv2.RandomHorizontalFlip(p=0.5)),
                torch.jit.script(tv2.RandomVerticalFlip(p=0.2)),
                tv2.ToDtype(torch.float32, scale=True),
            ]
        )

    def default_transforms_test(self) -> tv2.Compose:
        """Build and return the test/eval transform pipeline."""
        return tv2.Compose(
            [
                tv2.CenterCrop(size=self.params.size),
                tv2.ToDtype(torch.float32, scale=True),
            ]
        )
