import abc
import logging
import pathlib
from typing import Any

import einops
import lightning as L
import torch
from torchvision.transforms import v2 as tv2

from sensa.data.base import BaseImageFolder
from sensa.models.base import BaseModel
from sensa.params.base import BaseParams
from sensa.params.data import DataParams
from sensa.params.trainer import TrainerParams
from sensa.trainer import scheduler
from sensa.utils.param_grouping import base_param_grouping, merge_param_groups


class BaseLightningVision(L.LightningModule, abc.ABC):
    """Abstract LightningModule for vision-based training workflows.

    Handles hyperparameters, data loading, optimizer configuration,
    and learning-rate scheduling according to TrainerParams.

    Subclasses must implement:
      - @property name: a descriptive model name.
    """

    class Params(BaseParams):
        """Configuration container for BaseVision.

        Fields:
            data (DataParams | None): Parameters for the training dataset.
            data_test (DataParams | None): Parameters for the validation/test dataset.
            trainer (TrainerParams): Training loop and optimizer parameters.
        """

        data: DataParams | None = None
        data_test: DataParams | None = None
        trainer: TrainerParams

    __dataset__ = BaseImageFolder
    __params__ = Params

    def __init__(self, **kwargs):
        """Initialize model, save hyperparameters, build datasets,
        auto-scale learning rate, and compute total iterations.

        Accepts either a single `params=Params(...)` argument or
        keyword args for Params fields.
        """
        super().__init__()
        self.save_hyperparameters()
        if len(kwargs) == 1 and "params" in kwargs and isinstance(kwargs["params"], self.__params__):
            self.params = kwargs["params"]
        else:
            self.params = self.__params__(**kwargs)

        if self.params.data is not None:
            logging.info(f"{self.__class__.__name__}: loading training data.")
            self.data = self.__dataset__(params=self.params.data)
        if self.params.data_test is not None:
            logging.info(f"{self.__class__.__name__}: loading test data.")
            self.data_test = self.__dataset__(params=self.params.data_test)
        self.auto_scale_lr()
        self.set_iterations()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    def auto_scale_lr(self) -> None:
        """Scale the base learning rate (`params.trainer.optimizer.lr`) by:
          - number of GPUs, and
          - capped by `lr_max` if specified.

        Also updates `self.lr_end` accordingly.
        """
        # set learning rate
        self.lr = self.params.trainer.optimizer.lr
        self.lr_end = self.params.trainer.optimizer.lr_end
        self.batch_size = self.params.trainer.batch_size_per_gpu
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.batch_size *= torch.cuda.device_count()
            scale = torch.cuda.device_count()
            if (lr_max := self.params.trainer.optimizer.lr_max) is not None:
                scale = min(scale, lr_max / self.lr)
            self.lr *= scale
            self.lr_end = self.params.trainer.optimizer.lr_end * scale

    def get_param_groups(self) -> list[dict[str, Any]]:
        """See `sensa.utils.param_grouping`."""
        groups = [{"params": []}, {"params": []}]
        for child in self.children():
            if isinstance(child, BaseModel):
                groups = merge_param_groups(groups, child.param_groups)

            else:
                base_param_grouping(child, groups=groups[-2:])

        return groups

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Instantiate optimizer based on TrainerParams.

        Supports 'sgd' and 'adamw'. Ensures zero weight_decay
        on parameters in odd groups.

        Returns:
            torch.optim.Optimizer
        """
        params = self.params.trainer.optimizer
        if params.name == "sgd":
            optimizer = torch.optim.SGD(
                params=self.get_param_groups(),
                lr=float(self.lr),
                weight_decay=float(params.weight_decay),
                **params.kwargs,
            )

        elif params.name == "adamw":
            optimizer = torch.optim.AdamW(
                params=self.get_param_groups(),
                lr=float(self.lr),
                weight_decay=float(params.weight_decay),
                **params.kwargs,
            )

        else:
            raise ValueError("Invalid optimizer.")

        if len(optimizer.param_groups) > 1:
            for i in range(1, len(optimizer.param_groups), 2):
                optimizer.param_groups[i]["weight_decay"] = 0.0
        return optimizer

    def calculate_lr(self) -> float:
        """Compute the current learning rate following:
          - a linear warmup for the first `warmup * iterations` steps,
          - then a cosine decay down to `lr_end`.
          - Constant lr is used if `iterations` is not set.

        Returns:
            float: current learning rate.
        """
        # constant lr
        if not (hasattr(self, "iterations") and isinstance(self.iterations, int) and self.iterations > 0):
            return self.lr

        iwarmup = int(self.params.trainer.optimizer.warmup * self.iterations)
        if self.global_step < iwarmup:  # linear lr
            start = min(self.lr * 1e-8, self.lr_end * 1e-2)
            return scheduler.fn_linear(start, self.lr, self.global_step, iwarmup)
        # cosine lr
        return scheduler.fn_cosine(self.lr, self.lr_end, self.global_step - iwarmup, self.iterations - iwarmup)

    def update_lr(self) -> None:
        """Update each optimizer parameter group's learning rate with a scaled schedule.

        The base learning rate is obtained from calculate_lr(). If multiple parameter groups exist,
        a per-group decay factor is applied so that the last group retains the full rate, and
        the preceding groups are geometrically scaled down toward lr_min_scaled.

        The i-th group's lr is set to:
            lr_base * (decay ** (num_groups - 1 - i))

        This ensures the first group's lr = lr_base * lr_min_scaled.
        """
        opt = self.optimizers()
        num_groups = len(opt.param_groups)
        lr_min_scaled = self.params.trainer.optimizer.lr_min_scaled
        # per-group decay exponent
        decay = lr_min_scaled ** (1 / max(1, num_groups - 1))
        base_lr = self.calculate_lr()

        for i, group in enumerate(opt.param_groups):
            group["lr"] = base_lr * (decay ** (num_groups - 1 - i))

    def set_iterations(self) -> None:
        """Calculate `self.iterations` = total optimization steps."""
        batch_size = self.params.trainer.batch_size_per_gpu
        self.iterations = self.params.trainer.epochs * (len(self.data) // batch_size)
        self.iterations = self.iterations // self.params.trainer.accumulate_grad_batches
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.iterations = self.iterations // torch.cuda.device_count()

    def save_as_image(self, tensor: torch.Tensor, cols: int = 1) -> None:
        """Normalize a batch of image tensors and save as a grid image."""
        tensor -= tensor.amin((1, 2, 3), True)
        tensor /= tensor.amax((1, 2, 3), True)
        tensor = einops.rearrange(tensor, "(row col) c h w -> c (row h) (col w)", col=cols)

        tensor = tensor.cpu()
        pathlib.Path("./checkpoints").mkdir(exist_ok=True)
        tv2.functional.to_pil_image(tensor).save(f"./checkpoints/{self.name}.png")
