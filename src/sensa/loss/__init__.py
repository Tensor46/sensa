__all__ = [
    "_LOSS_REGISTRY",
    "BaseLoss",
    "CrossEntropyLoss",
    "CrossEntropyWithTargetMining",
    "DinoLoss",
    "MSELoss",
    "build_loss",
    "register_loss",
]


from sensa.loss.base import BaseLoss
from sensa.loss.cross_entropy import CrossEntropyLoss, CrossEntropyWithTargetMining
from sensa.loss.dino_loss import DinoLoss
from sensa.loss.mse_loss import MSELoss
from sensa.loss.registry import _LOSS_REGISTRY, build_loss, register_loss
