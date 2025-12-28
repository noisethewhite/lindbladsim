from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias, Self
from functools import cached_property


F64Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class Params(object):
    nodes:        int
    t_start:      float
    t_horizon:    float
    time_samples: int
    gamma:        float
    eta:          float
    epsilon:      F64Array
    v_pot:        F64Array

    @cached_property
    def times(self: Self) -> F64Array:
        return np.linspace(self.t_start, self.t_horizon, self.time_samples)
