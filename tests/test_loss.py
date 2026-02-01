import torch

import sensa


def test_cross_entropy_with_target_mining():
    """Test CrossEntropyWithTargetMining with default parameters."""
    batch_size = 4
    dim = 32
    num_labels = 10

    # Default keep_ratio=0.5, distance_mining="cosine", distance_prediction="dot"
    loss_fn = sensa.loss.CrossEntropyWithTargetMining(dim=dim, num_labels=num_labels)
    assert loss_fn.head.kwargs_mining["keep_ratio"] == 0.5
    assert loss_fn.head.kwargs_mining["distance_mining"] == "cosine"
    assert loss_fn.head.distance_prediction == "dot"

    tensor = torch.randn(batch_size, dim, requires_grad=True)
    target = torch.randint(0, num_labels, (batch_size,))
    result = loss_fn(tensor, target)

    # result must a dictionary with expected keys
    assert isinstance(result, dict)
    assert "loss" in result
    assert "predictions" in result
    assert "target" in result

    # check shapes
    assert result["loss"].shape == ()
    # predictions shape depends on target mining (keep_ratio), so it may be less than num_labels
    assert result["predictions"].shape[0] == batch_size
    assert result["predictions"].shape[1] <= num_labels
    assert result["target"].shape == (batch_size,)

    assert result["loss"].item() > 0  # loss must be positive

    # backward pass
    result["loss"].backward()
    # gradients must exist
    assert tensor.grad is not None
    assert not torch.allclose(tensor.grad, torch.zeros_like(tensor.grad))


def test_cosface():
    """Test CosFace with default parameters."""
    batch_size = 4
    dim = 32
    num_labels = 10

    # Default m=0.3, s=32
    loss_fn = sensa.loss.CosFace(dim=dim, num_labels=num_labels)
    assert loss_fn.m == 0.3
    assert loss_fn.s == 32.0

    tensor = torch.randn(batch_size, dim, requires_grad=True)
    target = torch.randint(0, num_labels, (batch_size,))
    result = loss_fn(tensor, target)

    # result must a dictionary with expected keys
    assert isinstance(result, dict)
    assert "loss" in result
    assert "predictions" in result
    assert "target" in result

    # check shapes
    assert result["loss"].shape == ()
    # predictions shape depends on target mining (keep_ratio), so it may be less than num_labels
    assert result["predictions"].shape[0] == batch_size
    assert result["predictions"].shape[1] <= num_labels
    assert result["target"].shape == (batch_size,)

    assert result["loss"].item() > 0  # loss must be positive

    # backward pass
    result["loss"].backward()
    # gradients must exist
    assert tensor.grad is not None
    assert not torch.allclose(tensor.grad, torch.zeros_like(tensor.grad))

    # head must be configured with cosine distance
    assert loss_fn.head.kwargs_mining["distance_mining"] == "cosine"
    assert loss_fn.head.distance_prediction == "cosine"


def test_sphereface2():
    """Test SphereFace2 with default parameters."""
    batch_size = 4
    dim = 32
    num_labels = 10

    # Default lam=0.7, m=0.4, r=40, t=3.0
    loss_fn = sensa.loss.SphereFace2(dim=dim, num_labels=num_labels)
    assert loss_fn.lam == 0.7
    assert loss_fn.m == 0.4
    assert loss_fn.r == 40.0
    assert loss_fn.t == 3.0

    tensor = torch.randn(batch_size, dim, requires_grad=True)
    target = torch.randint(0, num_labels, (batch_size,))
    result = loss_fn(tensor, target)

    # result must a dictionary with expected keys
    assert isinstance(result, dict)
    assert "loss" in result
    assert "predictions" in result
    assert "target" in result

    # check shapes
    assert result["loss"].shape == ()
    # predictions shape depends on target mining (keep_ratio), so it may be less than num_labels
    assert result["predictions"].shape[0] == batch_size
    assert result["predictions"].shape[1] <= num_labels
    assert result["target"].shape == (batch_size,)

    assert result["loss"].item() > 0  # loss must be positive

    # backward pass
    result["loss"].backward()
    # gradients must exist
    assert tensor.grad is not None
    assert not torch.allclose(tensor.grad, torch.zeros_like(tensor.grad))

    # head must be configured with cosine distance
    assert loss_fn.head.kwargs_mining["distance_mining"] == "cosine"
    assert loss_fn.head.distance_prediction == "cosine"

    # bias parameter must exist
    assert hasattr(loss_fn, "bias")
    assert isinstance(loss_fn.bias, torch.nn.Parameter)
