import pytest
import torch
from torchvision import tv_tensors

import sensa


@pytest.fixture
def sample_inputs():
    image = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    # one bounding box
    boxes = tv_tensors.BoundingBoxes(torch.Tensor([[0, 0, 2, 1]]), format="XYXY", canvas_size=(2, 3))
    # one mask
    masks = tv_tensors.Mask(torch.Tensor([[[0, 1, 2], [3, 4, 5]]]))
    # two keypoints, in COCO (x, y, v) format
    keypoints = torch.Tensor([[[0.0, 0.0, 1.0], [2.0, 1.0, 0.0]]])
    target = {"boxes": boxes, "masks": masks, "keypoints": keypoints}
    return image, target


def test_random_horizontal_flip_all_annotations(sample_inputs):
    image, target = sample_inputs

    sensa.data.augment.RandomHorizontalFlip.keypoints_flip_indices = [1, 0]
    hflip = sensa.data.augment.RandomHorizontalFlip(p=1.0)

    flipped_image, flipped_target = hflip(image.clone(), {k: v.clone() for k, v in target.items()})

    # image should be reversed in the width dimension
    assert torch.equal(flipped_image, image.flip(-1))

    # x coords inverted: new_x_min = W - old_x_max, new_x_max = W - old_x_min
    expected_boxes = target["boxes"].data.clone()
    expected_boxes[:, [0, 2]] = image.shape[-1] - expected_boxes[:, [2, 0]]
    assert torch.equal(flipped_target["boxes"].data, expected_boxes)

    # mask is same as image
    assert torch.equal(flipped_target["masks"].data, target["masks"].data.flip(-1))

    # keypoints
    keypoints = target["keypoints"]  # shape (1,2,3)
    # apply the same idx-swap used in the class:
    swapped_keypoints = keypoints[:, hflip.keypoints_flip_indices]
    swapped_keypoints[..., 0] = image.shape[-1] - swapped_keypoints[..., 0]
    # zero-out where v==0
    swapped_keypoints[swapped_keypoints[..., 2] == 0] = 0.0
    assert torch.equal(flipped_target["keypoints"], swapped_keypoints)
