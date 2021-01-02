import numpy as np
from gym.spaces import Box
from typing import List


class EveBox(Box):
    """
    Extend :class:`gym.spaces.Box` to Eve environment.

    A (possibly unbounded) box in R^n. Specifically, a Box represents the
    Cartesian product of n closed intervals. Each interval has the form of one
    of [a, b], (-oo, b], [a, oo), or (-oo, oo).

    There are two common use cases:

    * Identical bound for each dimension::
        >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(3, 4)

    * Independent bound for each dimension::
        >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        Box(2,)
    """
    def __init__(self,
                 low,
                 high,
                 static_obs_num: int = 0,
                 dynamic_obs_num: int = 0,
                 neurons: int = 0,
                 states: int = 0,
                 eve_shape: List[int] = None,
                 shape: List[int] = None,
                 dtype=np.float32):
        super().__init__(low, high, shape, dtype)
        self.static_obs_num = static_obs_num
        self.dynamic_obs_num = dynamic_obs_num
        self.neurons = neurons
        self.states = states
        self.eve_shape = eve_shape