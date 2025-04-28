import pathlib

from sensa.loss.base import BaseLoss


_LOSS_REGISTRY: dict[str, type] = {}


def register_loss(name: str):
    """Decorator to register a class under a given name."""

    def decorator(cls: type):
        if not issubclass(cls, BaseLoss):
            raise TypeError("LOSS_REGISTRY: Loss must inherit sensa.loss.BaseLoss")
        if name != cls.__name__:
            raise ValueError(f"LOSS_REGISTRY: cls.__name__ != name ({cls.__name__} != {name}).")
        if name in _LOSS_REGISTRY:
            raise ValueError(f"LOSS_REGISTRY: {name} exists.")

        _LOSS_REGISTRY[name] = cls
        return cls

    return decorator


def build_loss(params):
    import logging

    from sensa.params.loss import LossParams

    if not isinstance(params, LossParams):
        logging.error("build_loss: params must be sensa.params.LossParams")
        raise TypeError("build_loss: params must be sensa.params.LossParams")

    if params.name in _LOSS_REGISTRY:
        pt_loss = _LOSS_REGISTRY[params.name](**params.kwargs)
        if isinstance(params.path_to_checkpoint, pathlib.Path):
            if params.path_to_checkpoint.name.startswith("mae_"):
                pt_loss.load_from_mae(params.path_to_checkpoint)

        return pt_loss

    else:
        logging.error(f"build_loss: {params.name} loss is not available.")
        raise ValueError(f"build_loss: {params.name} loss is not available.")
