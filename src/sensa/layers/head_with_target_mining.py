__all__ = ["HeadWithTargetMining", "mine_targets_by_distance"]

from typing import Literal

import torch
import torch.nn.functional as F


@torch.no_grad()
def mine_targets_by_distance(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    keep_ratio: float = 0.5,
    batch_size: int = 50_000,
    distance_type: Literal["cosine", "euclidean"] = "cosine",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a subset of label indices (“subsample”) based on similarity or distance between
    feature vectors and weight vectors, using either cosine similarity or Euclidean distance.

    Args:
        tensor: (B, D) batch of feature vectors.
        weight: (C, D) class/label weight matrix; one row per label.
        target: (B,) ground-truth label indices in [0, C).
        keep_ratio: fraction of classes to keep (0 <= r <= 1) (a minimum of 1 label is kept).
        batch_size: chunk size when scanning classes to limit memory.
        distance_type: "cosine" or "euclidean".
            - "cosine" uses max cosine similarity (higher = closer)
            - "euclidean" uses min Euclidean distance (lower = closer)

    Returns:
        new_target: (B,) targets remapped into the subsampled label index space.
        subsample: (K,) kept global label indices in ascending order.
    """
    if keep_ratio <= 0.0 or keep_ratio >= 1.0:
        return target, torch.arange(weight.shape[0], device=target.device, dtype=target.dtype)

    if tensor.ndim != 2:
        raise ValueError(f"mine_targets_by_distance: tensor must be 2D, got {tensor.ndim}D")
    if weight.ndim != 2:
        raise ValueError(f"mine_targets_by_distance: weight must be 2D, got {weight.ndim}D")
    if target.ndim != 1:
        raise ValueError(f"mine_targets_by_distance: target must be 1D, got {target.ndim}D")

    if tensor.shape[1] != weight.shape[1]:
        raise ValueError(
            "mine_targets_by_distance: tensor and weight must have "
            f"the same number of features, got {tensor.shape[1]} and {weight.shape[1]}"
        )
    if tensor.shape[0] != target.shape[0]:
        raise ValueError(
            "mine_targets_by_distance: tensor and target must have "
            f"the same number of samples, got {tensor.shape[0]} and {target.shape[0]}"
        )
    if distance_type not in {"cosine", "euclidean"}:
        raise ValueError(f"mine_targets_by_distance: Invalid distance_type: {distance_type}")

    C: int = weight.shape[0]
    device: torch.device = tensor.device
    scores = torch.zeros(C, device=device, dtype=tensor.dtype)
    eps = torch.finfo(tensor.dtype).eps if tensor.is_floating_point() else 0.0

    for i in range(0, C, batch_size):
        j = min(C, i + batch_size)
        weight_chunk = weight[i:j]  # (chunk, D)

        if distance_type == "cosine":
            # Cosine similarity: higher is better
            #   Tensor and weight normalization inside the for loop is redundant
            #   but takes less memory when compared to the outer normalization.
            cosine = F.linear(
                F.normalize(tensor, dim=1, eps=eps),
                F.normalize(weight_chunk, dim=1, eps=eps),
            )  # (B, chunk)
            chunk_score, _ = cosine.max(dim=0)
        else:
            # Euclidean distance: lower is better → invert sign so higher is better for consistency
            # (B, D) vs (chunk, D)
            diff = tensor.unsqueeze(1) - weight_chunk.unsqueeze(0)
            dist = diff.pow(2).sum(dim=2).add(eps).sqrt()  # (B, chunk)
            chunk_score, _ = (-dist).max(dim=0)  # negative distance (so larger = closer)

        scores[i:j] = chunk_score.to(scores.dtype)

    # Ensure ground-truth classes are always kept
    gt_unique = target.unique()
    scores.scatter_(0, gt_unique, 2.0)

    # Optional distributed reduction
    if torch.cuda.is_available() and torch.distributed.is_available():
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(scores, op=torch.distributed.ReduceOp.MAX)

    # Determine how many classes to keep
    K = max(1, int(scores.numel() * keep_ratio))
    _, topk_idx = torch.topk(scores, k=K, largest=True, sorted=False)
    subsample = torch.sort(topk_idx).values  # (K,)

    # Map original targets into new label index space
    new_target = (target.view(-1, 1) == subsample.view(1, -1)).float().argmax(dim=1)
    return new_target, subsample


class HeadWithTargetMining(torch.nn.Module):
    """Head with target mining.

    Args:
        dim: dimension of the input features.
        num_classes: number of classes.
        keep_ratio: fraction of classes to keep (0 <= r <= 1) (a minimum of 1 label is kept).
        batch_size: chunk size when scanning classes to limit memory.
        distance_type: "cosine" or "euclidean".
            - "cosine" uses max cosine similarity (higher = closer)
            - "euclidean" uses min Euclidean distance (lower = closer)
        normalize: whether to normalize the input features and weight vectors.

    Returns:
        output: (B, num_classes) output tensor.
        target_new: (B,) targets remapped into the subsampled label index space.
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        keep_ratio: float = 0.5,
        batch_size: int = 50_000,
        distance_type: Literal["cosine", "euclidean"] = "cosine",
        normalize: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.weight = torch.nn.Parameter(torch.randn(num_classes, dim))
        torch.nn.init.xavier_normal_(self.weight)
        self.kwargs_mining = {
            "keep_ratio": keep_ratio,
            "batch_size": batch_size,
            "distance_type": distance_type,
        }
        self.normalize = normalize

    def forward(self, tensor: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_new, subsample = mine_targets_by_distance(tensor, self.weight, target, **self.kwargs_mining)
        weight = self.weight[subsample]
        if self.normalize:
            tensor = F.normalize(tensor, dim=-1)
            weight = F.normalize(weight, dim=-1)

        return F.linear(tensor, weight), target_new

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_classes={self.num_classes}, "
            f"keep_ratio={self.kwargs_mining['keep_ratio']}, "
            f"batch_size={self.kwargs_mining['batch_size']}"
        )
