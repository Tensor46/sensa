import abc
import logging
import pathlib
from typing import Any

import torch

from sensa.assets import load_yaml
from sensa.params.model import ModelParams
from sensa.utils.load import filter_state_dict


class BaseModel(torch.nn.Module, abc.ABC):
    """Abstract base for all sensa vision models.

    Subclasses must implement:
      - `forward(...)` to compute outputs,
      - `forward_features(...)` to return intermediate representations.
    """

    @classmethod
    def build_from_yaml(cls, path_to_yaml: pathlib.Path, tag: str) -> "BaseModel":
        """Initializes from a YAML file.

        Parameters
        ----------
        path_to_yaml : pathlib.Path or str
            Path to a YAML file containing a dictionary of model configs.
        tag : str
            The key under which this model's parameters are stored in the YAML.

        Returns
        -------
        BaseModel
            An instance of `cls`, constructed with `ModelParams(**config)`.
        """
        if isinstance(path_to_yaml, str):
            path_to_yaml = pathlib.Path(path_to_yaml)
        if not path_to_yaml.is_file():
            raise FileNotFoundError(f"{cls.__name__}: {path_to_yaml} does not exist.")

        info: dict[str, Any] = load_yaml(path_to_yaml)
        if tag not in info:
            raise ValueError(f"{cls.__name__}: tag ({tag}) does not exist in {path_to_yaml}.")

        mparams = ModelParams(**info[tag])
        if cls.__name__ != mparams.name:
            raise ValueError(f"{cls.__name__}: ModelParams.name in yaml (under {tag}) does not match with cls.")
        return cls(**mparams.kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Compute the model's output given input arguments."""
        pass

    @abc.abstractmethod
    def forward_features(self, *args, **kwargs) -> Any:
        """Compute intermediate feature representations."""
        pass

    def freeze_parameters(self, skip_freeze_prefixes: list[str]) -> None:
        """Disable gradient updates for all parameters except those with specified name prefixes."""
        for name, params in self.named_parameters():
            if len(skip_freeze_prefixes) and any(name.startswith(tag) for tag in skip_freeze_prefixes):
                continue
            params.requires_grad_(False)

    def unfreeze_parameters(self) -> None:
        """Re-enable gradient updates for all parameters."""
        for params in self.parameters():
            params.requires_grad_(True)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Partition this model's trainable parameters into two (minimum) groups:

          - Group 0 | even: parameters that should have weight decay.
          - Group 1 | odd:  parameters that should not have weight decay. (e.g., biases, normalizations, tokens).

        Returns
        -------
        list of dict
            Two dictionaries, each with key `"params"` mapping to a list of Tensors.
        """
        return self._param_groups(self)

    @staticmethod
    def _param_groups(model: torch.nn.Module, *, groups: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Helper to split `model` parameters for optimizer weight-decay settings.

        Parameters
        ----------
        model : torch.nn.Module
            The model whose parameters to inspect.
        groups : list of dict, optional
            An existing list to append to; if `None`, two new empty groups are created.

        Returns
        -------
        list of dict
            `[{"params": [...]}, {"params": [...]}]`. The first group holds
            weight-decay parameters; the second holds parameters exempt from decay.
        """
        groups = [{"params": []}, {"params": []}] if groups is None else groups
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            no_wdecay = "token" in name or p.ndim <= 1
            groups[int(no_wdecay)]["params"].append(p)
        return groups

    def _initialize(self) -> None:
        """Initialize Conv2d and Linear layers with Xavier-normal weights, and zero out their biases (if present)."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d | torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def load_from_mae(self, path: pathlib.Path) -> None:
        """Load and filter encoder weights from a Masked Autoencoder checkpoint."""
        logging.info(f"{self.__class__.__name__}: load {path}")
        state_dict = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
        state_dict = filter_state_dict(state_dict, "encoder.")
        logging.warning(self.load_state_dict(state_dict, strict=False))
