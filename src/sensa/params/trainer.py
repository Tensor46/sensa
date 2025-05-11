import torch

from sensa.params.base import BaseParams
from sensa.params.optim import OptimParams


class TrainerParams(BaseParams):
    """Training configuration parameters.

    Fields
    ------
        batch_size (int, default=None):
            Total batch size (accumulate_grad_batches * batch_size_per_gpu * num_gpus).
        batch_size_per_gpu (int):
            Batch size for each GPU.
        epochs (int):
            Total number of training epochs.
        logger_frequency (int, default=20):
            Log metrics every N steps.
        mixed_precision (bool, default=True):
            Use BFLOAT16 mixed precision when True.
        optimizer (OptimParams):
            Optimizer settings (see OptimParams).
        seed (int, default=46):
            Random seed for reproducibility.
        workers_per_gpu (int, default=4):
            Number of dataloader worker processes per GPU.
    """

    batch_size: int | None = None
    batch_size_per_gpu: int
    epochs: int
    logger_frequency: int = 20
    mixed_precision: bool = True
    optimizer: OptimParams
    seed: int = 46
    workers_per_gpu: int = 4

    @property
    def accumulate_grad_batches(self) -> int:
        """Number of batches to accumulate gradients before stepping."""
        if isinstance(self.batch_size, int) and self.batch_size > self.batch_size_per_gpu:
            return max(1, self.batch_size // (self.batch_size_per_gpu * self.gpus))
        return 1

    @property
    def gpus(self) -> int:
        return torch.cuda.device_count() if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 1
