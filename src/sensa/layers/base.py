import abc
import logging
from typing import Any

import torch


class BaseLayer(torch.nn.Module, abc.ABC):
    """Base class for all layers."""

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def assert_once(self, condition: bool, message: str) -> None:
        """Assert a condition once."""
        if not hasattr(self, "_asserted"):
            self._asserted: dict[str, bool] = {}
        if not self._asserted.get(message, False):
            self._asserted[message] = True
            torch._assert(condition, message)

    def warn_once(self, message: str) -> None:
        """Warn once."""
        if not hasattr(self, "_warned"):
            self._warned: dict[str, bool] = {}
        if not self._warned.get(message, False):
            self._warned[message] = True
            logging.warning(message)
