import shutil

import torch

import sensa

from . import utils


def dummy_state_dict() -> dict[str, torch.Tensor]:
    return {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "bias": torch.tensor(22 / 7, dtype=torch.float32),
    }


def test_shuffle_and_unshuffle_roundtrip():
    utils.PATH_TO_TMP.mkdir(exist_ok=True)
    state = dummy_state_dict()
    path_to_password = utils.PATH_TO_TMP / "pwd.txt"
    path_to_password.write_text("password")

    # shuffle & flatten
    shuffled = sensa.utils.obscure_state.shuffle_and_save(path_to_password, state, precision=torch.float32)
    # should no longer match the simple concatenation
    flat_original = torch.cat([p.reshape(-1) for p in state.values()])
    assert not torch.allclose(shuffled, flat_original)
    # unshuffle back into a fresh state_dict template
    restored = sensa.utils.obscure_state.unshuffle_and_load(path_to_password, dummy_state_dict(), shuffled)
    # verify that every tensor was restored exactly
    for key in state:
        assert torch.allclose(restored[key], state[key]), f"Mismatch in '{key}'"

    shutil.rmtree(utils.PATH_TO_TMP, ignore_errors=True)


def test_file_io_unshuffle():
    utils.PATH_TO_TMP.mkdir(exist_ok=True)
    state = dummy_state_dict()
    path_to_password = utils.PATH_TO_TMP / "pwd.txt"
    path_to_password.write_text("password")

    # shuffle and save to file
    shuffled = sensa.utils.obscure_state.shuffle_and_save(path_to_password, state, precision=torch.float32)
    torch.save(shuffled, utils.PATH_TO_TMP / "weights.pt")
    # unshuffle by providing the file path
    restored = sensa.utils.obscure_state.unshuffle_and_load(path_to_password, dummy_state_dict(), shuffled)

    for key in state:
        assert torch.allclose(restored[key], state[key]), f"File-based restore failed for '{key}'"

    shutil.rmtree(utils.PATH_TO_TMP, ignore_errors=True)
