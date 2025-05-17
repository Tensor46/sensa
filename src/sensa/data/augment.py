from typing import ClassVar

import torch
import torchvision as tv
import torchvision.transforms.v2 as TT
from torch import Tensor


class RandomHorizontalFlip(TT.RandomHorizontalFlip):
    """Randomly horizontally flips the given image and its target annotations with a specified probability.

    This transform extends `torchvision.transforms.v2.RandomHorizontalFlip` by additionally
    updating bounding boxes, segmentation masks, and keypoints in the target dictionary.

    Args:
        p (float): Probability of performing the horizontal flip. Default is 0.5.

    Attributes:
        keypoints_flip_indices list[int]: Optional index mapping to reorder keypoints
            when flipping (e.g., to swap left/right keypoints). If length is zero, no reordering is applied.
    """

    # default index mapping for COCO-style keypoints (swap left/right)
    keypoints_flip_indices: ClassVar[list[int]] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

    def forward(
        self,
        image: Tensor,
        target: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        """Apply a random horizontal flip to the image and update target annotations.

        Args:
            image (Tensor | tv.tv_tensors.Image): Input image tensor of shape (C, H, W).
            target (dict[str, Tensor] | None): Optional annotations dict. Supported keys:
                - "boxes" (tv.tv_tensors.BoundingBoxes): Tensor[N, 4], bounding boxes in corner format.
                - "masks" (tv.tv_tensors.Mask): Tensor[N, H, W], binary segmentation masks.
                - "keypoints" (torch.Tensor): Tensor[N, K, 2] or Tensor[N, K, 3] for COCO (x, y, visibility).
        """
        if torch.rand(1) < self.p:
            image = TT.functional.hflip(image)
            if target is not None:
                w = image.shape[-1]

                # flip bounding boxes (x_min, y_min, x_max, y_max)
                if "boxes" in target:
                    torch._assert(
                        isinstance(target["boxes"], tv.tv_tensors.BoundingBoxes),
                        f"{self.__class__.__name__}: boxes must be tv.tv_tensors.BoundingBoxes",
                    )
                    # swap and invert x-coordinates
                    target["boxes"][:, [0, 2]] = w - target["boxes"][:, [2, 0]]

                # flip masks along width dimension
                if "masks" in target:
                    torch._assert(
                        isinstance(target["masks"], tv.tv_tensors.Mask),
                        f"{self.__class__.__name__}: boxes must be tv.tv_tensors.Mask",
                    )
                    target["masks"] = target["masks"].flip(-1)

                # flip keypoints: reorder, invert x, zero-out invisible
                if "keypoints" in target:
                    torch._assert(
                        isinstance(target["keypoints"], Tensor),
                        f"{self.__class__.__name__}: keypoints must be torch.Tensor",
                    )
                    keypoints = target["keypoints"]
                    # reorder according to flip indices, if provided
                    if len(self.keypoints_flip_indices):
                        torch._assert(
                            len(self.keypoints_flip_indices) == keypoints.shape[-2],
                            f"{self.__class__.__name__}: keypoints.shape[-2] != len(self.keypoints_flip_indices)",
                        )
                        keypoints = keypoints[:, self.keypoints_flip_indices]
                    # invert x-coordinate
                    keypoints[..., 0] = w - keypoints[..., 0]
                    # support for COCO-style (x, y, v), zero out coordinates where visibility==0
                    if keypoints.shape[-1] == 3:
                        keypoints[keypoints[..., 2] == 0] = 0
                    target["keypoints"] = keypoints

        return image, target


class Rotate90s(torch.nn.Module):
    """Randomly rotates the given image and its target annotations by one of the specified
    90-degree multiples (±90°, 180°) with a given probability.

    Rotates both the image and its associated bounding boxes, segmentation masks, and keypoints.

    Args:
        p (float): Probability of applying the rotation. Must be in [0.0, 1.0].
    """

    def __init__(self, p: float):
        """Initialize the Rotate90s module."""
        super().__init__()
        self.p = p
        # Default set of rotation angles
        self.angles = (90, 180, 270)

    @property
    def angles(self) -> tuple[int]:
        """Get the current tuple of allowed rotation angles."""
        return self._angles

    @angles.setter
    def angles(self, value: tuple[int] | list[int]) -> None:
        """Set the allowed rotation angles.

        Args:
            value (tuple[int] | list[int]): Rotation angles to allow.
        """
        if isinstance(value, list):
            value = tuple(value)
        if not isinstance(value, tuple):
            raise TypeError(f"{self.__class__.__name__}: angles must be tuple of 90/180/270/-90.")
        if any(not isinstance(val, int) for val in value):
            raise TypeError(f"{self.__class__.__name__}: angles must be tuple of 90/180/270/-90.")
        if any(val not in (-90, 90, 180, 270) for val in value):
            raise ValueError(f"{self.__class__.__name__}: angles must be tuple of 90/180/270/-90.")
        self._angles = value

    def forward(
        self,
        image: Tensor,
        target: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        """Apply a random 90-degree rotation to the image and update target annotations.

        Args:
            image (Tensor): Input image tensor of shape (C, H, W).
            target (dict[str, Tensor] | None): Optional annotations dict. Supported keys:
                - "boxes": Bounding boxes (tv.tv_tensors.BoundingBoxes) in XYXY format.
                - "masks": Segmentation masks (tv.tv_tensors.Mask).
                - "keypoints": Tensor of shape (N, K, 2) or (N, K, 3) where keypoints[..., 2] is the visibility.
        """
        if torch.rand(1) < self.p:
            # choose a random angle from the allowed set
            angle = self.angles[torch.randint(0, len(self.angles), (1,)).item()]
            *_, h, w = image.shape
            # rotate the image (may expand canvas if needed)
            image = TT.functional.rotate(image, angle, expand=True)

            # update annotations if provided
            if target is not None:
                # rotate bounding boxes
                if "boxes" in target:
                    target["boxes"] = TT.functional.rotate(target["boxes"], angle, expand=True)
                # rotate segmentation masks
                if "masks" in target:
                    target["masks"] = TT.functional.rotate(target["masks"], angle, expand=True)
                # rotate keypoints by mapping to and from bounding boxes
                if "keypoints" in target:
                    keypoints = target["keypoints"][..., :2]
                    keypoints_as_box = torch.cat((keypoints, keypoints), -1).reshape(-1, 4)
                    n, ps, _ = keypoints.shape
                    boxes = tv.tv_tensors.BoundingBoxes(keypoints_as_box, format="XYXY", canvas_size=(h, w))
                    boxes = TT.functional.rotate(boxes, angle, expand=True)
                    target["keypoints"][..., :2] = boxes.data[:, :2].reshape(n, ps, -1)
                    # support for COCO-style (x, y, v), zero out coordinates where visibility==0
                    if target["keypoints"].shape[-1] == 3:
                        target["keypoints"][target["keypoints"][..., 2] == 0] = 0

        return image, target
