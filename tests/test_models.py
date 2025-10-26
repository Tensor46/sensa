import pytest
import torch

import sensa
import sensa.models.base


def test_registry_not_a_base_model():
    with pytest.raises(TypeError) as _:

        @sensa.models.register_model("Test")
        class Test(torch.nn.Linear):
            pass

    assert "Test" not in sensa.models._MODEL_REGISTRY


def test_registry_name_mismatch():
    with pytest.raises(ValueError) as _:

        @sensa.models.register_model("NotTest")
        class Test(sensa.models.base.BaseModel):
            pass

    assert "NotTest" not in sensa.models._MODEL_REGISTRY


def test_registry_cannot_use_existing_name():
    with pytest.raises(ValueError) as _:

        @sensa.models.register_model("VIT")
        class VIT(sensa.models.base.BaseModel):
            pass

    assert sensa.models._MODEL_REGISTRY["VIT"] == sensa.models.VIT


def test_vit_features():
    model = sensa.models.VIT(
        image_size=(128, 128),
        stem_config={
            "in_channels": 3,
            "out_channels": 128,
            "patch_size": 8,
            "first_stride": 2,
            "last_stride": 4,
            "act_layer": "silu",
            "norm_layer": "batchnorm",
        },
        encoder_config={
            "size": None,
            "extra_tokens": None,
            "num_layers": 2,
            "num_heads": 4,
            "hidden_dim": 128,
            "mlp_dim": 256,
            "dropout": 0.0,
            "act_layer": "gelu",
            "norm_layer": "layernorm",
            "pos_token": "sincos",
        },
        mask_ratio=0.0,
        num_labels=None,
        last_pool=None,
    )
    output = model(torch.randn(1, 3, 128, 128))
    assert output.shape[-1] == model.encoder_config.hidden_dim, f"output shape must be {output.shape}"
    assert output.shape[-2] == model.encoder_config.seq_length, f"output shape must be {output.shape}"
    del model


def test_vit_features_with_pool():
    model = sensa.models.VIT(
        image_size=(128, 128),
        stem_config={
            "in_channels": 3,
            "out_channels": 128,
            "patch_size": 8,
            "first_stride": 2,
            "last_stride": 4,
            "act_layer": "silu",
            "norm_layer": "batchnorm",
        },
        encoder_config={
            "size": None,
            "extra_tokens": None,
            "num_layers": 2,
            "num_heads": 4,
            "hidden_dim": 128,
            "mlp_dim": 256,
            "dropout": 0.0,
            "act_layer": "gelu",
            "norm_layer": "layernorm",
            "pos_token": "rope",
        },
        mask_ratio=0.0,
        num_labels=None,
        last_pool="half",
    )
    output = model(torch.randn(1, 3, 128, 128))
    size = model.encoder_config.hidden_dim * (model.stem_size[0] // 2) * (model.stem_size[1] // 2)
    assert output.shape[-1] == size, f"output shape must be {output.shape}"
    del model
