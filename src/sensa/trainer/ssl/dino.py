import copy

import torch

from sensa.data.dino import Dataset
from sensa.loss.registry import build_loss
from sensa.models.registry import build_model
from sensa.params.data import DataParams
from sensa.params.loss import LossParams
from sensa.params.model import ModelParams
from sensa.trainer.base import BaseLightningVision


class DINO(BaseLightningVision):
    """DINO Lightning module.

    Attributes
    ----------
    __dataset__ : Type[Dataset]
        The dataset class used for loading MAE data samples.
    __params__ : Type[Params]
        The hyperparameter schema, including data and model configurations.
    steps_to_log_images : int
        Number of optimizer steps between saving image reconstructions.
    encoder : torch.nn.Module
        The Vision Transformer encoder built from `params.encoder`.
    head : torch.nn.Module
        The DINO head built from `params.head`.
    criteria : torch.nn.Module
        DINO loss function.
    """

    class Params(BaseLightningVision.Params):
        """Hyperparameters for DINO.

        Parameters
        ----------
        data : DataParams
            Configuration for data loading (paths, transforms, etc.).
        encoder : ModelParams
            Configuration for the encoder model (architecture, patch size, etc.).
        head : ModelParams
            Configuration for the head model (architecture, etc.).
        loss : LossParams
            Configuration for the loss.
        """

        data: DataParams
        encoder: ModelParams
        head: ModelParams
        loss: LossParams

    __dataset__ = Dataset
    __params__ = Params
    steps_to_log_images: int = 500

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = build_model(self.params.encoder)
        self.encoder.extend_sizes(self.params.data.kwargs["size_local"])
        self.head = build_model(self.params.head)

        # teacher model
        self.encoder_teacher = copy.deepcopy(self.encoder)
        self.head_teacher = build_model(self.params.head)
        for param in self.encoder_teacher.parameters():
            param.requires_grad = False
        for param in self.head_teacher.parameters():
            param.requires_grad = False

        self.criteria = build_loss(self.params.loss)
        # cosine scheduler
        from sensa.trainer.scheduler import fn_cosine

        self.scheduler = fn_cosine

    @property
    def name(self) -> str:
        return f"dino_{self.params.encoder.name}"

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder."""
        return self.head(self.encoder(tensor))

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training iteration.

        1. Updates learning rate.

        Parameters
        ----------
        batch : list[torch.Tensor]
            Batch data; expected format [images, ...].
        batch_idx : int
            Index of this batch within the current epoch.

        Returns
        -------
        torch.Tensor
            Scalar loss for backpropagation.
        """
        # update learning rate
        self.update_lr()
        if next(self.encoder.parameters()).grad is None:
            # momentum = self.scheduler(self.current_epoch, 10, 0.996, 0.996)
            momentum = 0.996
            self.ema_update(self.encoder, self.encoder_teacher, momemtum=momentum)
            self.ema_update(self.head, self.head_teacher, momemtum=momentum)

        # unpack batch
        *images, _ = batch
        teacher_out = self.head_teacher(self.encoder_teacher(torch.cat(images[:2], dim=0)))
        student_out = self.head(
            torch.cat(
                (
                    self.encoder(torch.cat(images[:2], dim=0)),
                    self.encoder(torch.cat(images[2:], dim=0)),
                )
            )
        )
        loss = self.criteria(student_out, teacher_out, epoch=self.current_epoch)
        # logging
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return loss

    def on_after_backward(self):
        self.head.cancel_last_layer_gradients(epoch=self.current_epoch)

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
