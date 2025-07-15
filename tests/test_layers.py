import pytest
import torch

import sensa


def test_attention():
    """Test sensa.layers.attention.Attention module for shape correctness and forward pass."""
    batch_size = 2
    size = h, w = 4, 6
    embed_dim = 32
    num_heads = 4

    tensor = torch.randn(batch_size, h * w, embed_dim)

    # instantiate attention module
    attn = sensa.layers.attention.Attention(size=size, dim=embed_dim, num_heads=num_heads)
    assert attn(tensor).shape == tensor.shape

    # test with rope embedding
    attn = sensa.layers.attention.Attention(size=size, dim=embed_dim, num_heads=num_heads, use_rope=True)
    assert attn(tensor).shape == tensor.shape

    # test with masked tensor
    keep_idx, *_ = sensa.layers.mask_utils.random_mask_indices(tensor, 0.2)
    tensor_masked = sensa.layers.mask_utils.mask_tensor(tensor, keep_idx)
    assert attn(tensor_masked, indices_to_keep=keep_idx).shape == tensor_masked.shape


def test_cross_attention():
    """Test sensa.layers.attention.CrossAttention module for shape correctness and forward pass."""
    batch_size = 2
    h, w = 4, 6
    h_kv, w_kv = 5, 7
    embed_dim = 32
    num_heads = 4

    q = torch.randn(batch_size, h * w, embed_dim)
    kv = torch.randn(batch_size, h_kv * w_kv, embed_dim)

    cross_attention = sensa.layers.attention.CrossAttention(dim=embed_dim, dim_kv=embed_dim, num_heads=num_heads)
    assert cross_attention(q, kv).shape == q.shape


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


def test_encoder2():
    r"""Test sensa.layers.Encoder2."""
    for pos_token in ("learned", "sincos", "rope"):
        module = sensa.layers.Encoder2(
            size=(8, 6),
            extra_tokens=0,
            num_layers=2,
            num_heads=2,
            hidden_dim=64,
            mlp_dim=128,
            dropout=0.0,
            pos_token=pos_token,
        )
        assert module(torch.randn(1, 48, 64)).shape == (1, 48, 64)

    # other size are not possible
    with pytest.raises(AssertionError) as _:
        module(torch.randn(1, 12, 64))
    # added additional size
    module.extend_sizes((4, 3))
    assert module(torch.randn(1, 12, 64)).shape == (1, 12, 64)


def test_encoder2_with_dyt_norm():
    r"""Test sensa.layers.Encoder2 with DyT normalization."""
    module = sensa.layers.Encoder2(
        size=(8, 6),
        extra_tokens=0,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        mlp_dim=128,
        dropout=0.0,
        pos_token="learned",
        norm_layer=sensa.layers.DyT,
    )
    assert module(torch.randn(1, 48, 64)).shape == (1, 48, 64)

    # other size are not possible
    import pytest

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
    r"""Test sensa.layers.mask_utils.random_mask_indices."""
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
    r"""Test sensa.layers.mask_utils.mask_tensor and sensa.layers.mask_utils.unmask_tensor."""
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
