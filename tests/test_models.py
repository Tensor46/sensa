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
        patch_size=8,
        num_layers=2,
        num_heads=4,
        hidden_dim=128,
        mlp_dim=256,
        mask_ratio=0.0,
        num_classes=None,
        in_channels=3,
        last_pool=None,
        use_sincos_pos_token=True,
    )
    output = model(torch.randn(1, 3, 128, 128))
    assert output.shape[-1] == model.hidden_dim, f"output shape must be {output.shape}"
    assert output.shape[-2] == model.seq_length, f"output shape must be {output.shape}"
    del model


def test_vit_features_with_pool():
    model = sensa.models.VIT(
        image_size=(128, 128),
        patch_size=8,
        num_layers=2,
        num_heads=4,
        hidden_dim=128,
        mlp_dim=256,
        mask_ratio=0.0,
        num_classes=None,
        in_channels=3,
        last_pool="half",
        use_sincos_pos_token=True,
    )
    output = model(torch.randn(1, 3, 128, 128))
    size = model.hidden_dim * (model.stem_size[0] // 2) * (model.stem_size[1] // 2)
    assert output.shape[-1] == size, f"output shape must be {output.shape}"
    del model
