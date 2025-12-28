from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .values import Value


class ParamError(Exception):
    pass


InitFn = Callable[[Tuple[int, ...], np.random.Generator], np.ndarray | float]


@dataclass
class Param:
    name: str
    shape: Tuple[int, ...]
    value: Value
    trainable: bool = True


def init_zeros(shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray | float:
    if not shape:
        return 0.0
    return np.zeros(shape, dtype=float)


def init_normal(
    shape: Tuple[int, ...], rng: np.random.Generator, mean: float = 0.0, std: float = 1.0
) -> np.ndarray | float:
    if not shape:
        return float(rng.normal(loc=mean, scale=std))
    return rng.normal(loc=mean, scale=std, size=shape)


# parse param decl → ParamStore.add_param(...) → NumPy array created → wrapped in Value → stored by name.
class ParamStore:
    def __init__(self, rng: Optional[np.random.Generator] = None, seed: Optional[int] = None) -> None:
        self._params: Dict[str, Param] = {}
        if rng is not None and seed is not None:
            raise ParamError("Pass either rng or seed, not both")
        self._rng = rng or np.random.default_rng(seed)

    def add_param(
        self,
        name: str,
        shape: Optional[Sequence[int]],
        init: InitFn,
        trainable: bool = True,
    ) -> Param:
        if name in self._params:
            raise ParamError(f"Param '{name}' already exists")
        shape_tuple = _shape_tuple(shape)
        data = init(shape_tuple, self._rng)
        param = Param(name=name, shape=shape_tuple, value=Value(data), trainable=trainable)
        self._params[name] = param
        return param

    def get(self, name: str) -> Param:
        try:
            return self._params[name]
        except KeyError as exc:
            raise ParamError(f"Unknown param '{name}'") from exc

    def items(self) -> Iterable[tuple[str, Param]]:
        return self._params.items()

    def resolve_init(self, name: str, args: Sequence[float]) -> InitFn:
        if name == "zeros":
            return init_zeros
        if name == "normal":
            mean = args[0] if len(args) > 0 else 0.0
            std = args[1] if len(args) > 1 else 1.0

            def _fn(shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray | float:
                return init_normal(shape, rng, mean=mean, std=std)

            return _fn
        raise ParamError(f"Unknown initializer '{name}'")


def _shape_tuple(shape: Optional[Sequence[int]]) -> Tuple[int, ...]:
    if not shape:
        return ()
    return tuple(int(dim) for dim in shape)
