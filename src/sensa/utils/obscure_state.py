import getpass
import hashlib
import logging
import pathlib

import torch


def get_password(path_to_password: pathlib.Path | None = None) -> bytes:
    """Prompt for or load a password and return its SHA-256 digest.
    If `path_to_password` is None, prompts the user (no echo) to enter a password.
    Otherwise, reads the password from the specified file.

    Args:
        path_to_password (Optional[pathlib.Path]):
            Path to a text file containing the password, or None to prompt.

    Returns:
        bytes: 32-byte SHA-256 hash of the entered or loaded password.
    """
    try:
        if path_to_password is None:
            pwd = getpass.getpass("Enter your password: ")
            if not pwd:
                raise ValueError("No password entered")

        else:
            if not isinstance(path_to_password, pathlib.Path):
                raise TypeError("get_password: file_name must be pathlib.Path | None.")
            if not path_to_password.is_file():
                raise FileNotFoundError(f"get_password: {path_to_password} does not exist.")
            with open(path_to_password) as txt:
                pwd = txt.read()

    except (KeyboardInterrupt, EOFError):
        logging.exception("Operation cancelled.")
        raise SystemExit(1)  # noqa: B904

    return hashlib.sha256(pwd.encode("utf-8")).digest()


def shuffle_tensor(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Shuffle the 0th dimension of a tensor deterministically.

    Args:
        x (torch.Tensor): Tensor to shuffle along dim 0.
        seed (int): Seed for reproducible permutation.

    Returns:
        torch.Tensor: New tensor with rows permuted.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(x.size(0), generator=gen, device=x.device)
    return x[perm]


def unshuffle_tensor(x_shuffled: torch.Tensor, seed: int):
    """Reverse a deterministic shuffle along the 0th dimension.

    Args:
        x_shuffled (torch.Tensor): Tensor that was shuffled using `shuffle_tensor`.
        seed (int): Seed originally used to generate the shuffle.

    Returns:
        torch.Tensor: Tensor restored to its original ordering.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(x_shuffled.size(0), generator=gen, device=x_shuffled.device)
    return x_shuffled[perm.argsort()]


def shuffle_and_save(
    path_to_password: pathlib.Path | None,
    state_dict: dict[str, torch.Tensor],
    precision: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Flatten, quantize, and deterministically shuffle a model's state_dict.

    Reads or prompts for a password to derive 4 independent seeds, then:
      1. Flattens each tensor in `state_dict` into a 1D tensor.
      2. Clamps values to the finite range of `precision`.
      3. Converts all data to CPU at the given dtype.
      4. Concatenates into one long 1D tensor.
      5. Applies four rounds of shuffle using password-derived seeds.

    Args:
        path_to_password (pathlib.Path | None):
            Path to a file containing the password, or None to prompt the user.
        state_dict (dict[str, torch.Tensor]):
            Mapping of parameter names to tensors (e.g., model.state_dict()).
        precision (torch.dtype, optional):
            Target dtype for quantization and shuffling (default: torch.float16).

    Returns:
        torch.Tensor:
            A single 1D tensor containing the shuffled, precision-converted state.

    Disclosure: This code merely obscures model weights by shuffling them with
    a password-derived seed and does not provide true cryptographic protection.
    It is not a substitute for standard encryption schemes—use it at your own risk.
    """
    info = torch.finfo(precision)
    pbytes: bytes = get_password(path_to_password)

    # flatten, clamp, and cast each tensor + concatenate
    tensor = torch.cat(
        [p.reshape(-1).clamp(info.min, info.max).to("cpu", dtype=precision) for p in state_dict.values()]
    )

    # shuffle with each 8-byte segment of the password hash
    for i in range(0, 32, 8):
        seed = int.from_bytes(pbytes[i : i + 8], "big")
        tensor = shuffle_tensor(tensor, seed)
    return tensor


def unshuffle_and_load(
    path_to_password: pathlib.Path | None,
    state_dict: dict[str, torch.Tensor],
    weights: pathlib.Path | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Reverse a multi-round deterministic shuffle and load values into a state_dict.

    This function:
      1. Derives the same password-based seeds used for shuffling.
      2. Loads the flat weights tensor from disk or accepts it directly.
      3. Applies the inverse shuffle in reverse order of the rounds.
      4. Splits the restored 1D tensor according to each entry's original size.
      5. Copies each slice back into the corresponding tensor in `state_dict`.

    Args:
        path_to_password (pathlib.Path | None):
            Path to a file with the password, or None to prompt the user.
        state_dict (Dict[str, torch.Tensor]):
            A template mapping of parameter names to tensors (e.g., model.state_dict())
            whose shapes will be used to split and reshape the weights.
        weights (pathlib.Path | torch.Tensor):
            Either a path to a saved weights file (loaded via `torch.load`) or
            a flat 1D `torch.Tensor` of shuffled weights.

    Returns:
        Dict[str, torch.Tensor]:
            The same `state_dict` dict, but with each tensor's filled by the unshuffled
            weights, matching original shapes.
    """
    pbytes: bytes = get_password(path_to_password)

    # load or verify the flat weights tensor
    if isinstance(weights, pathlib.Path):
        weights = torch.load(weights, map_location=torch.device("cpu"), weight_only=True)
    if not isinstance(weights, torch.Tensor):
        raise ValueError("unshuffle_and_load_to_state_dict: weights must be torch.Tensor.")

    # inverse-shuffle in reverse round order
    for i in reversed(range(0, 32, 8)):
        seed = int.from_bytes(pbytes[i : i + 8], "big")
        weights = unshuffle_tensor(weights, seed)

    # partition and copy back into state_dict
    idx = 0
    for param in state_dict.values():
        numel = param.numel()
        chunk = weights[idx] if param.ndim == 0 else weights[idx : idx + numel]
        param.data.copy_(chunk.reshape(param.shape).type_as(param))
        idx += numel
    return state_dict
