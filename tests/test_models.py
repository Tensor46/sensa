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


def test_registry_use_existing_name():
    with pytest.raises(ValueError) as _:

        @sensa.models.register_model("VIT")
        class VIT(sensa.models.base.BaseModel):
            pass

    assert sensa.models._MODEL_REGISTRY["VIT"] == sensa.models.VIT
