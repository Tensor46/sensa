from sensa.params.base import BaseParams
from sensa.params.optim import OptimParams


class TrainerParams(BaseParams):
    """Training configuration parameters.

    Fields
    ------
        accumulate_grad_batches (int, default=1):
            Number of batches to accumulate gradients before stepping.
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

    accumulate_grad_batches: int = 1
    batch_size_per_gpu: int
    epochs: int
    logger_frequency: int = 20
    mixed_precision: bool = True
    optimizer: OptimParams
    seed: int = 46
    workers_per_gpu: int = 4
