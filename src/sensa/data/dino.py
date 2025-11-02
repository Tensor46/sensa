from typing import Any

import torch
from torchvision.transforms import v2 as tv2

from sensa.data.base import BaseImageFolder


class Dataset(BaseImageFolder):
    """This is an adapted version of the "DINO" (https://github.com/facebookresearch/dino/)."""

    def default_specs(self, **kwargs) -> dict[str, Any]:
        """Hook to adjust or fill default params before model creation."""
        kwargs.setdefault("kwargs", {})
        # global transforms
        if "scale" not in kwargs and "scale" not in kwargs["kwargs"]:
            kwargs.setdefault("scale", (0.4, 1.0))

        # local transforms
        if "num_local_transforms" not in kwargs and "num_local_transforms" not in kwargs["kwargs"]:
            kwargs.setdefault("num_local_transforms", 6)
        if "size_local" not in kwargs and "size_local" not in kwargs["kwargs"]:
            kwargs.setdefault("size_local", (96, 96))
        if "scale_local" not in kwargs and "scale_local" not in kwargs["kwargs"]:
            kwargs.setdefault("scale_local", (0.05, 0.4))

        return kwargs

    def default_transforms(self) -> tv2.Compose | list[tv2.Compose]:
        """Define and return training transforms."""
        # common transforms
        common_transforms = [
            torch.jit.script(tv2.RandomHorizontalFlip(p=0.5)),
            torch.jit.script(tv2.RandomApply(torch.nn.ModuleList([tv2.ColorJitter(0.4, 0.4, 0.2, 0.1)]), p=0.8)),
            torch.jit.script(tv2.RandomGrayscale(p=0.2)),
        ]
        # gaussian blur
        gblur = tv2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.6))
        # global random resized crop
        grrc = torch.jit.script(
            tv2.RandomResizedCrop(
                size=self.params.size,
                scale=self.params.kwargs["scale"],
                ratio=(3 / 4, 4 / 3),
                interpolation=self.params.interpolation,
                antialias=True,
            ),
        )
        # global transforms a
        global_transforms_a = tv2.Compose(
            [
                grrc,
                *common_transforms,
                torch.jit.script(gblur),
                tv2.ToDtype(torch.float32, scale=True),
            ]
        )
        # global transforms b
        global_transforms_b = tv2.Compose(
            [
                grrc,
                *common_transforms,
                torch.jit.script(tv2.RandomApply(torch.nn.ModuleList([gblur]), p=0.1)),
                torch.jit.script(tv2.RandomSolarize(threshold=128.0, p=0.2)),
                tv2.ToDtype(torch.float32, scale=True),
            ]
        )
        # local transforms
        local_transforms = tv2.Compose(
            [
                torch.jit.script(
                    tv2.RandomResizedCrop(
                        size=self.params.kwargs["size_local"],
                        scale=self.params.kwargs["scale_local"],
                        ratio=(3 / 4, 4 / 3),
                        interpolation=self.params.interpolation,
                        antialias=True,
                    ),
                ),
                *common_transforms,
                torch.jit.script(tv2.RandomApply(torch.nn.ModuleList([gblur]), p=0.5)),
                tv2.ToDtype(torch.float32, scale=True),
            ]
        )
        # all transforms
        num_local_transforms = self.params.kwargs["num_local_transforms"]
        return [global_transforms_a, global_transforms_b, *([local_transforms] * num_local_transforms)]

    def default_transforms_validation(self) -> tv2.Compose:
        """Build and return the validation transform pipeline."""
        return tv2.Compose(
            [
                tv2.CenterCrop(size=self.params.size),
                tv2.ToDtype(torch.float32, scale=True),
            ]
        )
