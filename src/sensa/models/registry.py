import pathlib

from sensa.models.base import BaseModel


_MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Decorator to register a class under a given name."""

    def decorator(cls: type):
        if not issubclass(cls, BaseModel):
            raise TypeError("MODEL_REGISTRY: Model must inherit sensa.models.BaseModel")
        if name != cls.__name__:
            raise ValueError(f"MODEL_REGISTRY: cls.__name__ != name ({cls.__name__} != {name}).")
        if name in _MODEL_REGISTRY:
            raise ValueError(f"MODEL_REGISTRY: {name} exists.")

        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(params):
    import logging

    from sensa.params.model import ModelParams

    if not isinstance(params, ModelParams):
        logging.error("build_model: params must be sensa.params.ModelParams")
        raise TypeError("build_model: params must be sensa.params.ModelParams")

    if params.name in _MODEL_REGISTRY:
        pt_model = _MODEL_REGISTRY[params.name](**params.kwargs)
        if isinstance(params.path_to_checkpoint, pathlib.Path):
            if params.path_to_checkpoint.name.startswith("mae_"):
                pt_model.load_from_mae(params.path_to_checkpoint)

        return pt_model

    else:
        logging.error(f"build_model: {params.name} model is not available.")
        raise ValueError(f"build_model: {params.name} model is not available.")
