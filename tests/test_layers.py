import pytest
import torch

import sensa


def test_dyt():
    r"""Test sensa.layers.DyT."""
    module = sensa.layers.DyT(16)
    assert module(torch.randn(1, 32, 16)).shape == (1, 32, 16)
    del module


def test_encoder():
    r"""Test sensa.layers.Encoder."""
    module = sensa.layers.Encoder(
        size=(8, 6),
        extra_tokens=0,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        mlp_dim=128,
        dropout=0.0,
        attention_dropout=0.0,
    )
    assert module(torch.randn(1, 48, 64)).shape == (1, 48, 64)

    # other size are not possible
    with pytest.raises(AssertionError) as _:
        module(torch.randn(1, 12, 64))
    # added additional size
    module.extend_sizes((4, 3))
    assert module(torch.randn(1, 12, 64)).shape == (1, 12, 64)


def test_last_pool():
    r"""Test sensa.layers.LastPool."""
    module = sensa.layers.LastPool(pool="avg", size=None)
    assert module(torch.randn(1, 32, 16)).shape == (1, 16)
    del module

    module = sensa.layers.LastPool(pool="full", size=None)
    assert module(torch.randn(1, 32, 16)).shape == (1, 32 * 16)
    del module

    with pytest.raises(TypeError) as _:
        sensa.layers.LastPool(pool="half", size=None)
    with pytest.raises(TypeError) as _:
        sensa.layers.LastPool(pool="half", size=(1.0, 1.0))
    with pytest.raises(ValueError) as _:
        sensa.layers.LastPool(pool="half", size=(4, 8, 2))

    module = sensa.layers.LastPool(pool="half", size=(4, 8))
    assert module(torch.randn(1, 32, 16)).shape == (1, 4 * 8 * 16 // 4)
    del module

    module = sensa.layers.LastPool(pool="token", size=None)
    assert module(torch.randn(1, 32, 16)).shape == (1, 16)
    del module


def test_random_mask_indices_shapes_and_inverse():
    b, n, c = 4, 8, 6
    x = torch.randn(b, n, c)
    ratio = 0.4
    num_cls = 2

    keep_idx, mask_idx, restore_idx = sensa.layers.mask_utils.random_mask_indices(x, ratio, num_cls_tokens=num_cls)
    n_keep = int(max(1, n * (1 - ratio)))
    assert keep_idx.shape == (b, n_keep)
    assert mask_idx.shape == (b, n - n_keep)
    assert restore_idx.shape == (b, n)
    for i in range(num_cls):
        assert torch.allclose(keep_idx[:, i], torch.zeros(n_keep, dtype=keep_idx.dtype) + i)

    # check that restore_idx is indeed the inverse permutation of concatenated keep+mask
    concatenated = torch.cat([keep_idx, mask_idx], dim=1)
    recovered = torch.gather(concatenated, 1, restore_idx)
    target = torch.arange(n, device=x.device).unsqueeze(0).expand(b, -1)
    assert torch.equal(recovered, target)


def test_mask_and_unmask_roundtrip_default_token():
    b, n, c = 2, 4, 6
    x = torch.arange(b * n * c, dtype=torch.float32).reshape(b, n, c)
    ratio = 0.3

    keep_idx, _, restore_idx = sensa.layers.mask_utils.random_mask_indices(x, ratio)
    x_masked = sensa.layers.mask_utils.mask_tensor(x, keep_idx)
    x_restored = sensa.layers.mask_utils.unmask_tensor(x_masked, restore_idx, mask_token=None)

    # check shapes
    assert x_masked.shape == (b, keep_idx.size(1), c)
    assert x_restored.shape == x.shape

    # verify kept positions match original, masked positions are zero
    zeros = torch.zeros(c)
    for b_ in range(b):
        kept = set(keep_idx[b_].tolist())
        for i in range(n):
            if i in kept:
                assert torch.allclose(x_restored[b_, i], x[b_, i])
            else:
                assert torch.allclose(x_restored[b_, i], zeros)
