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

    def _sample(self):
        """
        Generates a single random sample inside of the Box.

        In creating a sample of the box, each coordinate is sampled according to
        the form of the interval:

        * [a, b] : uniform distribution
        * [a, oo) : shifted exponential distribution
        * (-oo, b] : shifted negative exponential distribution
        * (-oo, oo) : normal distribution
        """
        high = self.high if self.dtype.kind == 'f' \
                else self.high.astype('int64') + 1

        sample = np.empty(self.shape)

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        sample[unbounded] = self.np_random.normal(
            size=unbounded[unbounded].shape)

        sample[low_bounded] = self.np_random.exponential(
            size=low_bounded[low_bounded].shape) + self.low[low_bounded]

        sample[upp_bounded] = -self.np_random.exponential(
            size=upp_bounded[upp_bounded].shape) + self.high[upp_bounded]

        sample[bounded] = self.np_random.uniform(low=self.low[bounded],
                                                 high=high[bounded],
                                                 size=bounded[bounded].shape)
        if self.dtype.kind == 'i':
            sample = np.floor(sample)

        return sample.astype(self.dtype)

    def sample(self):
        # Stack neurons times of actions
        action = [self._sample() for _ in range(self.neurons)]
        return np.stack(action, axis=0)