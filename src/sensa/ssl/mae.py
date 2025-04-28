import einops
import torch

from sensa.base import BaseLightningVision
from sensa.data.mae import Dataset
from sensa.layers.mask_utils import mask_tensor, unmask_tensor
from sensa.loss.registry import build_loss
from sensa.models.registry import build_model
from sensa.params.data import DataParams
from sensa.params.loss import LossParams
from sensa.params.model import ModelParams


class MAE(BaseLightningVision):
    """Masked Autoencoder (MAE) Lightning module.

    This class implements the ViT-based masked autoencoder. It
    wraps an encoder and decoder, applies random masking to input patches,
    computes reconstruction loss, and logs both scalar metrics and periodic
    visualizations of original vs. reconstructed images.

    Attributes
    ----------
    __dataset__ : Type[Dataset]
        The dataset class used for loading MAE data samples.
    __params__ : Type[Params]
        The hyperparameter schema, including data and model configurations.
    steps_to_log_images : int
        Number of optimizer steps between saving image reconstructions.
    encoder : torch.nn.Module
        The Vision Transformer encoder built from `params.mae_encoder`.
    decoder : torch.nn.Module
        The Vision Transformer decoder built from `params.mae_decoder`.
    criteria : torch.nn.Module
        Reconstruction loss function (MSE).
    """

    class Params(BaseLightningVision.Params):
        """Hyperparameters for MAE.

        Parameters
        ----------
        data : DataParams
            Configuration for data loading (paths, transforms, etc.).
        mae_encoder : ModelParams
            Configuration for the encoder model (architecture, patch size, etc.).
        mae_decoder : ModelParams
            Configuration for the decoder model (architecture, patch size, etc.).
        loss : LossParams
            Configuration for the loss.
        """

        data: DataParams
        mae_encoder: ModelParams
        mae_decoder: ModelParams
        loss: LossParams

    __dataset__ = Dataset
    __params__ = Params
    steps_to_log_images: int = 500

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = build_model(self.params.mae_encoder)
        self.decoder = build_model(self.params.mae_decoder)
        self.criteria = build_loss(self.params.loss)

    @property
    def name(self) -> str:
        return f"mae_{self.params.mae_encoder.name}"

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder."""
        return self.encoder(tensor)

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training iteration.

        1. Updates learning rate.
        2. Applies patch masking on encoder inputs (masking is built into VIT).
        3. Runs encoder + decoder to reconstruct masked patches.
        4. Computes MSE reconstruction loss (with a small auxiliary term).
        5. Logs loss and learning rate.
        6. Every `steps_to_log_images` steps, saves a grid of original,
           masked, and reconstructed images.

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
        # unpack batch
        images, *_ = batch
        # encode with random patch masking
        out = self.encoder.forward_features(images)
        o_encoded = out["features"]
        indices_to_keep = out["indices_to_keep"]
        indices_to_mask = out["indices_to_mask"]
        indices_to_restore = out["indices_to_restore"]

        if hasattr(self.encoder, "class_token"):
            # drop cls token and adjust indices accordingly
            o_encoded = o_encoded[:, 1:, :]
            indices_to_keep = indices_to_keep[:, 1:] - 1
            indices_to_mask = indices_to_mask - 1
            indices_to_restore = indices_to_restore[:, 1:] - 1

        # decode to reconstruct all patches
        rimages = self.decoder(o_encoded)
        # prepare original patches for loss
        ps = self.params.mae_encoder.kwargs["patch_size"]
        oimages = einops.rearrange(
            (images - 0.5) / 0.25,
            "b c (h ph) (w pw) -> b (h w) (ph pw c)",
            ph=self.params.mae_encoder.kwargs["patch_size"],
            pw=self.params.mae_encoder.kwargs["patch_size"],
        )
        # mask and select only the reconstructed patches
        rpatches = mask_tensor(rimages, indices_to_mask)
        opatches = mask_tensor(oimages, indices_to_mask)

        # compute reconstruction loss + auxiliary full-image term
        loss = self.criteria(rpatches, opatches) + 0.01 * self.criteria(rimages, oimages)

        # periodic visualization of reconstructions
        if self.steps_to_log_images > 0 and self.global_step % self.steps_to_log_images == 0:
            n: int = 8  # number of examples to visualize
            osubset = oimages[:n]
            osubset_masked = unmask_tensor(mask_tensor(osubset, indices_to_keep[:n]), indices_to_restore[:n], None)
            rsubset = rimages[:n]
            canvas = einops.rearrange(
                torch.stack((osubset, osubset_masked, rsubset), dim=1),
                "b a (h w) (ph pw c) -> (b a) c (h ph) (w pw)",
                ph=ps,
                pw=ps,
                h=self.params.mae_encoder.kwargs["image_size"][0] // ps,
                w=self.params.mae_encoder.kwargs["image_size"][0] // ps,
            )
            self.save_as_image(canvas, cols=6)

        # logging
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
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
