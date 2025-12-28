from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np


@dataclass(frozen=True)
class Value:
    data: Any

    @property
    def shape(self) -> Tuple[int, ...]:
        if isinstance(self.data, np.ndarray):
            return self.data.shape
        return np.array(self.data).shape

    def as_array(self) -> np.ndarray:
        if isinstance(self.data, np.ndarray):
            return self.data
        return np.array(self.data)
