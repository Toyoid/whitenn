from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .autodiff import Grad
from .params import ParamError, ParamStore
from .values import Value


class OptimizerError(Exception):
    pass


class Optimizer:
    def apply(self, grad: Grad, params: ParamStore) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class SGD(Optimizer):
    lr: float

    def apply(self, grad: Grad, params: ParamStore) -> None:
        if self.lr <= 0:
            raise OptimizerError("SGD learning rate must be positive")
        for name, grad_value in grad.grads.items():
            try:
                param = params.get(name)
            except ParamError as exc:
                raise OptimizerError(str(exc)) from exc
            if not param.trainable:
                continue
            self._apply_one(param, grad_value)

    def _apply_one(self, param, grad_value: Value) -> None:
        param_array = param.value.as_array()
        grad_array = grad_value.as_array()
        if param_array.shape != grad_array.shape:
            raise OptimizerError(
                f"Gradient shape mismatch for '{param.name}': "
                f"{param_array.shape} vs {grad_array.shape}"
            )
        updated = param_array - self.lr * grad_array
        param.value = Value(np.array(updated))
