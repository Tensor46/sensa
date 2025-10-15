import torch
import torchmetrics as tm

from sensa.data.imagenet import Dataset
from sensa.loss.registry import build_loss
from sensa.models.registry import build_model
from sensa.params.data import DataParams
from sensa.params.loss import LossParams
from sensa.params.model import ModelParams
from sensa.trainer.base import BaseLightningVision


class ClassifierWithOutValidation(BaseLightningVision):
    """LightningModule for training an image classifier without validation.

    This module:
      - Instantiates a backbone model via `ModelParams`.
      - If `mode=="linear"`, freezes all but specified backbone parameters.
      - Defines a cross-entropy loss and top-1 accuracy metric.
      - Implements `training_step` to compute loss, update LR, and log metrics.
      - Provides `train_dataloader` based on `DataParams`.
    """

    class Params(BaseLightningVision.Params):
        """Parameter schema for ClassifierWithOutValidation."""

        data: DataParams
        backbone: ModelParams
        loss: LossParams

    __dataset__ = Dataset
    __params__ = Params

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = build_model(self.params.backbone)
        if self.params.backbone.mode == "linear":
            self.backbone.freeze_parameters(skip_freeze_prefixes=self.params.backbone.skip_freeze_prefixes)
        self.criteria = build_loss(self.params.loss)

        # metrics
        self.accuracy = tm.Accuracy(task="multiclass", num_classes=self.data.num_labels, top_k=1)
        self.accuracy_val = tm.Accuracy(task="multiclass", num_classes=self.data.num_labels, top_k=1)

    @property
    def name(self) -> str:
        return f"classifier_{self.params.mode}_{self.params.backbone.name}"

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone network."""
        return self.backbone(tensor)

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute one training iteration.

        - Updates learning rate scheduler.
        - Computes cross-entropy loss.
        - Logs loss, learning rate, and top-1 accuracy.
        """
        # update learning rate
        self.update_lr()
        # unpack batch
        images, target = batch
        predictions = self(images)
        output = self.criteria(predictions, target)
        if isinstance(output, dict):
            loss = output["loss"]
            predictions = output.get("predictions", predictions)
            if predictions.shape[1] < self.data.num_labels:
                with torch.no_grad():
                    b = predictions.shape[0]
                    n = self.data.num_labels - predictions.shape[1]
                    zeros = torch.zeros(b, n, device=predictions.device, dtype=predictions.dtype)
                    predictions = torch.cat([predictions, zeros], dim=1)

            target = output.get("target", target)
        elif isinstance(output, torch.Tensor):
            loss = output
        else:
            raise ValueError(f"Invalid sensa.loss.* output type: {type(output)}")

        # logging
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        self.accuracy.update(predictions, target)
        self.log("top1", self.accuracy, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Build and return the training DataLoader."""
        self.set_iterations()
        return torch.utils.data.DataLoader(
            self.data,
            batch_size=self.params.trainer.batch_size_per_gpu,
            shuffle=True,
            num_workers=self.params.trainer.workers_per_gpu,
            pin_memory=True,
        )


class Classifier(ClassifierWithOutValidation):
    """Extends ClassifierWithOutValidation by adding validation logic."""

    class Params(ClassifierWithOutValidation.Params):
        data_test: DataParams

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute one validation iteration."""
        # unpack batch
        images, target = batch
        predictions = self(images)
        loss = self.criteria(predictions, target)

        # logging
        self.accuracy_val.update(predictions, target)
        self.log("top1_val", self.accuracy_val, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Build and return the validation DataLoader."""
        self.set_iterations()
        return torch.utils.data.DataLoader(
            self.data_test,
            batch_size=self.params.trainer.batch_size_per_gpu // 2,
            shuffle=False,
            num_workers=self.params.trainer.workers_per_gpu,
            pin_memory=True,
        )
