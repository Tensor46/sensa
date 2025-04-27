import torch


def random_mask_indices(x: torch.Tensor, ratio: float, num_cls_tokens: int = 0) -> tuple[torch.Tensor, ...]:
    """Generate indices for random masking and restoration.

    Args:
        x (torch.Tensor): Input of shape (B, N, C).
        ratio (float): Fraction of tokens to mask (0 ≤ ratio ≤ 1).
        num_cls_tokens (int): Number of leading tokens to always keep.

    Returns:
        indices_to_keep (torch.Tensor): Shape (B, n_keep), indices of tokens kept.
        indices_to_mask (torch.Tensor): Shape (B, N - n_keep), indices of tokens masked.
        indices_to_restore (torch.Tensor): Shape (B, N), inverse permutation to restore order.
    """
    b, n, _ = x.shape
    n_keep = int(max(1, n * (1 - ratio)))
    rand = torch.rand(b, n, device=x.device)
    rand[:, :num_cls_tokens] = 0
    indices_shuffled = torch.argsort(rand, dim=1)
    indices_to_restore = torch.argsort(indices_shuffled, dim=1)
    indices_to_keep = indices_shuffled[:, :n_keep]
    indices_to_mask = indices_shuffled[:, n_keep:]
    return indices_to_keep, indices_to_mask, indices_to_restore


def mask_tensor(x: torch.Tensor, indices_to_keep: torch.Tensor) -> torch.Tensor:
    """Gather only the indices_to_keep tokens from a tensor.

    Args:
        x (torch.Tensor): Input of shape (B, N, C).
        indices_to_keep (torch.Tensor): Shape (B, n_keep), indices to keep.

    Returns:
        torch.Tensor: Masked tensor of shape (B, n_keep, C).
    """
    *_, c = x.shape
    x_masked = torch.gather(x, dim=1, index=indices_to_keep[..., None].repeat(1, 1, c))
    return x_masked


def unmask_tensor(x_masked: torch.Tensor, indices_to_restore: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    """Restore a masked tensor to its original sequence length.

    Args:
        x_masked (torch.Tensor): Masked tensor of shape (B, n_keep, C).
        indices_to_restore (torch.Tensor): Shape (B, N), restore permutation.
        mask_token (torch.Tensor | None): (C,) token to insert for masked positions; if None, uses zeros.

    Returns:
        torch.Tensor: Unmasked tensor of shape (B, N, C).
    """
    b, n_keep, c = x_masked.shape
    n = indices_to_restore.size(1)
    mask_token = torch.zeros_like(x_masked[0, 0]) if mask_token is None else mask_token.reshape(1, 1, -1)
    mask_token = mask_token.repeat(b, n - n_keep, 1)
    x = torch.cat([x_masked[:, :, :], mask_token], dim=1)
    return torch.gather(x, dim=1, index=indices_to_restore.unsqueeze(-1).repeat(1, 1, c))
