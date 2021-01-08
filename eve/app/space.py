from collections import OrderedDict

import numpy as np
from gym import logger
from gym.spaces import Space


class EveSpace(Space):
    """Wraper gym.Space to EveSpace.

    Some operations will be different if accept an EveSpace.
    """
    pass


class EveBox(EveSpace):
    """
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
                 shape=None,
                 max_neurons: int = None,
                 max_states: int = None,
                 dtype=np.float32):
        assert dtype is not None, 'dtype must be explicitly provided. '
        self.dtype = np.dtype(dtype)
        self.max_neurons = max_neurons
        self.max_states = max_states

        # determine shape if it isn't provided directly
        if shape is not None:
            shape = tuple(shape)
            assert np.isscalar(
                low
            ) or low.shape == shape, "low.shape doesn't match provided shape"
            assert np.isscalar(
                high
            ) or high.shape == shape, "high.shape doesn't match provided shape"
        elif not np.isscalar(low):
            shape = low.shape
            assert np.isscalar(
                high
            ) or high.shape == shape, "high.shape doesn't match low.shape"
        elif not np.isscalar(high):
            shape = high.shape
            assert np.isscalar(
                low
            ) or low.shape == shape, "low.shape doesn't match high.shape"
        else:
            raise ValueError(
                "shape must be provided or inferred from the shapes of low or high"
            )

        if np.isscalar(low):
            low = np.full(shape, low, dtype=dtype)

        if np.isscalar(high):
            high = np.full(shape, high, dtype=dtype)

        self.shape = shape
        self.low = low
        self.high = high

        def _get_precision(dtype):
            if np.issubdtype(dtype, np.floating):
                return np.finfo(dtype).precision
            else:
                return np.inf

        low_precision = _get_precision(self.low.dtype)
        high_precision = _get_precision(self.high.dtype)
        dtype_precision = _get_precision(self.dtype)
        if min(low_precision, high_precision) > dtype_precision:
            logger.warn("Box bound precision lowered by casting to {}".format(
                self.dtype))
        self.low = self.low.astype(self.dtype)
        self.high = self.high.astype(self.dtype)

        # Boolean arrays which indicate the interval type for each coordinate
        self.bounded_below = -np.inf < self.low
        self.bounded_above = np.inf > self.high

        super(EveBox, self).__init__(self.shape, self.dtype)

    def is_bounded(self, manner="both"):
        below = np.all(self.bounded_below)
        above = np.all(self.bounded_above)
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {'below', 'above', 'both'}")

    def sample(self):
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

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        return x.shape == self.shape and np.all(x >= self.low) and np.all(
            x <= self.high)

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "Box({}, {}, {}, {})".format(self.low.min(), self.high.max(),
                                            self.shape, self.dtype)

    def __eq__(self, other):
        return isinstance(
            other, EveBox) and (self.shape == other.shape) and np.allclose(
                self.low, other.low) and np.allclose(self.high, other.high)


class EveDict(EveSpace):
    """
    A dictionary of simpler spaces.

    Example usage:
    self.observation_space = spaces.Dict({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})

    Example usage [nested]:

        >>> self.nested_observation_space = spaces.Dict({
        >>>        'sensors':  spaces.Dict({
        >>>            'position': spaces.Box(low=-100, high=100, shape=(3,)),
        >>>            'velocity': spaces.Box(low=-1, high=1, shape=(3,)),
        >>>            'front_cam': spaces.Tuple((
        >>>                spaces.Box(low=0, high=1, shape=(10, 10, 3)),
        >>>                spaces.Box(low=0, high=1, shape=(10, 10, 3))
        >>>            )),
        >>>            'rear_cam': spaces.Box(low=0, high=1, shape=(10, 10, 3)),
        >>>        }),
        >>>        'ext_controller': spaces.MultiDiscrete((5, 2, 2)),
        >>>        'inner_state':spaces.Dict({
        >>>            'charge': spaces.Discrete(100),
        >>>            'system_checks': spaces.MultiBinary(10),
        >>>            'job_status': spaces.Dict({
        >>>                'task': spaces.Discrete(5),
        >>>                'progress': spaces.Box(low=0, high=100, shape=()),
        >>>            })
        >>>        })
        >>>    })
    
    """
    def __init__(self, spaces=None, **spaces_kwargs):
        assert (spaces is None) or (
            not spaces_kwargs
        ), 'Use either Dict(spaces=dict(...)) or Dict(foo=x, bar=z)'
        if spaces is None:
            spaces = spaces_kwargs
        if isinstance(spaces, dict) and not isinstance(spaces, OrderedDict):
            spaces = OrderedDict(sorted(list(spaces.items())))
        if isinstance(spaces, list):
            spaces = OrderedDict(spaces)
        self.spaces = spaces
        for space in spaces.values():
            assert isinstance(
                space,
                Space), 'Values of the dict should be instances of gym.Space'
        # None for shape and dtype, since it'll require special handling
        super(EveDict, self).__init__(None, None)

    def seed(self, seed=None):
        [space.seed(seed) for space in self.spaces.values()]

    def sample(self):
        return OrderedDict([(k, space.sample())
                            for k, space in self.spaces.items()])

    def contains(self, x):
        if not isinstance(x, dict) or len(x) != len(self.spaces):
            return False
        for k, space in self.spaces.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True

    def __getitem__(self, key):
        return self.spaces[key]

    def __iter__(self):
        for key in self.spaces:
            yield key

    def __repr__(self):
        return "EveDict(" + ", ".join(
            [str(k) + ":" + str(s) for k, s in self.spaces.items()]) + ")"

    def to_jsonable(self, sample_n):
        # serialize as dict-repr of vectors
        return {
            key: space.to_jsonable([sample[key] for sample in sample_n])
            for key, space in self.spaces.items()
        }

    def from_jsonable(self, sample_n):
        dict_of_list = {}
        for key, space in self.spaces.items():
            dict_of_list[key] = space.from_jsonable(sample_n[key])
        ret = []
        for i, _ in enumerate(dict_of_list[key]):
            entry = {}
            for key, value in dict_of_list.items():
                entry[key] = value[i]
            ret.append(entry)
        return ret

    def __eq__(self, other):
        return isinstance(other, EveDict) and self.spaces == other.spaces


class EveDiscrete(EveSpace):
    r"""A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`. 

    Example::

        >>> EveDiscrete(2)

    """
    def __init__(self, n, max_neurons: int = None, max_states: int = None):
        assert n >= 0
        self.n = n
        self.max_neurons = max_neurons
        self.max_states = max_states
        super(EveDiscrete, self).__init__((), np.int64)

    def sample(self):
        return self.np_random.randint(self.n)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (
                x.dtype.char in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    def __repr__(self):
        return "EveDiscrete(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, EveDiscrete) and self.n == other.n


class EveMultiBinary(EveSpace):
    '''
    An n-shape binary space. 

    The argument to MultiBinary defines n, which could be a number or a `list` of numbers.

    Example Usage:

    >> self.observation_space = spaces.MultiBinary(5)

    >> self.observation_space.sample()

        array([0,1,0,1,0], dtype =int8)

    >> self.observation_space = spaces.MultiBinary([3,2])

    >> self.observation_space.sample()

        array([[0, 0],
               [0, 1],   
               [1, 1]], dtype=int8)

    '''
    def __init__(self, n, max_neurons: int = None, max_states: int = None):
        self.n = n
        self.max_neurons = max_neurons
        self.max_states = max_states
        if type(n) in [tuple, list, np.ndarray]:
            input_n = n
        else:
            input_n = (n, )
        super(EveMultiBinary, self).__init__(input_n, np.int8)

    def sample(self):
        return self.np_random.randint(low=0,
                                      high=2,
                                      size=self.n,
                                      dtype=self.dtype)

    def contains(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = np.array(x)  # Promote list to array for contains check
        if self.shape != x.shape:
            return False
        return ((x == 0) | (x == 1)).all()

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self):
        return "EveMultiBinary({})".format(self.n)

    def __eq__(self, other):
        return isinstance(other, EveMultiBinary) and self.n == other.n


class EveMultiDiscrete(EveSpace):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different number of actions in eachs
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of positive integers specifying number of actions for each discrete action space

    Note: Some environment wrappers assume a value of 0 always represents the NOOP action.

    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:

        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1

    - Can be initialized as

        MultiDiscrete([ 5, 2, 2 ])

    """
    def __init__(self, nvec, max_neurons: int = None, max_states: int = None):
        """
        nvec: vector of counts of each categorical variable
        """
        assert (np.array(nvec) > 0).all(), 'nvec (counts) have to be positive'
        self.nvec = np.asarray(nvec, dtype=np.int64)

        self.max_neurons = max_neurons
        self.max_states = max_states

        super(EveMultiDiscrete, self).__init__(self.nvec.shape, np.int64)

    def sample(self):
        return (self.np_random.random_sample(self.nvec.shape) *
                self.nvec).astype(self.dtype)

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return x.shape == self.shape and (0 <= x).all() and (x <
                                                             self.nvec).all()

    def to_jsonable(self, sample_n):
        return [sample.tolist() for sample in sample_n]

    def from_jsonable(self, sample_n):
        return np.array(sample_n)

    def __repr__(self):
        return "EveMultiDiscrete({})".format(self.nvec)

    def __eq__(self, other):
        return isinstance(other,
                          EveMultiDiscrete) and np.all(self.nvec == other.nvec)


class EveTuple(EveSpace):
    """
    A tuple (i.e., product) of simpler spaces

    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
    """
    def __init__(self, spaces):
        self.spaces = spaces
        for space in spaces:
            assert isinstance(
                space,
                Space), "Elements of the tuple must be instances of gym.Space"
        super(EveTuple, self).__init__(None, None)

    def seed(self, seed=None):
        [space.seed(seed) for space in self.spaces]

    def sample(self):
        return tuple([space.sample() for space in self.spaces])

    def contains(self, x):
        if isinstance(x, list):
            x = tuple(x)  # Promote list to tuple for contains check
        return isinstance(x, tuple) and len(x) == len(self.spaces) and all(
            space.contains(part) for (space, part) in zip(self.spaces, x))

    def __repr__(self):
        return "EveTuple(" + ", ".join([str(s) for s in self.spaces]) + ")"

    def to_jsonable(self, sample_n):
        # serialize as list-repr of tuple of vectors
        return [
            space.to_jsonable([sample[i] for sample in sample_n])
            for i, space in enumerate(self.spaces)
        ]

    def from_jsonable(self, sample_n):
        return [
            sample for sample in zip(*[
                space.from_jsonable(sample_n[i])
                for i, space in enumerate(self.spaces)
            ])
        ]

    def __getitem__(self, index):
        return self.spaces[index]

    def __len__(self):
        return len(self.spaces)

    def __eq__(self, other):
        return isinstance(other, EveTuple) and self.spaces == other.spaces


def flatdim(space):
    """Return the number of dimensions a flattened equivalent of this space
    would have.

    Accepts a space and returns an integer. Raises ``NotImplementedError`` if
    the space is not defined in ``gym.spaces``.
    """
    if isinstance(space, EveBox):
        return int(np.prod(space.shape))
    elif isinstance(space, EveDiscrete):
        return int(space.n)
    elif isinstance(space, EveTuple):
        return int(sum([flatdim(s) for s in space.spaces]))
    elif isinstance(space, EveDict):
        return int(sum([flatdim(s) for s in space.spaces.values()]))
    elif isinstance(space, EveMultiBinary):
        return int(space.n)
    elif isinstance(space, EveMultiDiscrete):
        return int(np.prod(space.shape))
    else:
        raise NotImplementedError


def flatten(space, x):
    """Flatten a data point from a space.

    This is useful when e.g. points from spaces must be passed to a neural
    network, which only understands flat arrays of floats.

    Accepts a space and a point from that space. Always returns a 1D array.
    Raises ``NotImplementedError`` if the space is not defined in
    ``gym.spaces``.
    """
    if isinstance(space, EveBox):
        return np.asarray(x, dtype=space.dtype).flatten()
    elif isinstance(space, EveDiscrete):
        onehot = np.zeros(space.n, dtype=space.dtype)
        onehot[x] = 1
        return onehot
    elif isinstance(space, EveTuple):
        return np.concatenate(
            [flatten(s, x_part) for x_part, s in zip(x, space.spaces)])
    elif isinstance(space, EveDict):
        return np.concatenate(
            [flatten(s, x[key]) for key, s in space.spaces.items()])
    elif isinstance(space, EveMultiBinary):
        return np.asarray(x, dtype=space.dtype).flatten()
    elif isinstance(space, EveMultiDiscrete):
        return np.asarray(x, dtype=space.dtype).flatten()
    else:
        raise NotImplementedError


def unflatten(space, x):
    """Unflatten a data point from a space.

    This reverses the transformation applied by ``flatten()``. You must ensure
    that the ``space`` argument is the same as for the ``flatten()`` call.

    Accepts a space and a flattened point. Returns a point with a structure
    that matches the space. Raises ``NotImplementedError`` if the space is not
    defined in ``gym.spaces``.
    """
    if isinstance(space, EveBox):
        return np.asarray(x, dtype=space.dtype).reshape(space.shape)
    elif isinstance(space, EveDiscrete):
        return int(np.nonzero(x)[0][0])
    elif isinstance(space, EveTuple):
        dims = [flatdim(s) for s in space.spaces]
        list_flattened = np.split(x, np.cumsum(dims)[:-1])
        list_unflattened = [
            unflatten(s, flattened)
            for flattened, s in zip(list_flattened, space.spaces)
        ]
        return tuple(list_unflattened)
    elif isinstance(space, EveDict):
        dims = [flatdim(s) for s in space.spaces.values()]
        list_flattened = np.split(x, np.cumsum(dims)[:-1])
        list_unflattened = [
            (key, unflatten(s, flattened))
            for flattened, (key,
                            s) in zip(list_flattened, space.spaces.items())
        ]
        return OrderedDict(list_unflattened)
    elif isinstance(space, EveMultiBinary):
        return np.asarray(x, dtype=space.dtype).reshape(space.shape)
    elif isinstance(space, EveMultiDiscrete):
        return np.asarray(x, dtype=space.dtype).reshape(space.shape)
    else:
        raise NotImplementedError


def flatten_space(space):
    """Flatten a space into a single ``Box``.

    This is equivalent to ``flatten()``, but operates on the space itself. The
    result always is a `Box` with flat boundaries. The box has exactly
    ``flatdim(space)`` dimensions. Flattening a sample of the original space
    has the same effect as taking a sample of the flattenend space.

    Raises ``NotImplementedError`` if the space is not defined in
    ``gym.spaces``.

    Example::

        >>> box = Box(0.0, 1.0, shape=(3, 4, 5))
        >>> box
        Box(3, 4, 5)
        >>> flatten_space(box)
        Box(60,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True

    Example that flattens a discrete space::

        >>> discrete = Discrete(5)
        >>> flatten_space(discrete)
        Box(5,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True

    Example that recursively flattens a dict::

        >>> space = Dict({"position": Discrete(2),
        ...               "velocity": Box(0, 1, shape=(2, 2))})
        >>> flatten_space(space)
        Box(6,)
        >>> flatten(space, space.sample()) in flatten_space(space)
        True
    """
    if isinstance(space, EveBox):
        return EveBox(space.low.flatten(),
                      space.high.flatten(),
                      dtype=space.dtype)
    if isinstance(space, EveDiscrete):
        return EveBox(low=0, high=1, shape=(space.n, ), dtype=space.dtype)
    if isinstance(space, EveTuple):
        space = [flatten_space(s) for s in space.spaces]
        return EveBox(low=np.concatenate([s.low for s in space]),
                      high=np.concatenate([s.high for s in space]),
                      dtype=np.result_type(*[s.dtype for s in space]))
    if isinstance(space, EveDict):
        space = [flatten_space(s) for s in space.spaces.values()]
        return EveBox(low=np.concatenate([s.low for s in space]),
                      high=np.concatenate([s.high for s in space]),
                      dtype=np.result_type(*[s.dtype for s in space]))
    if isinstance(space, EveMultiBinary):
        return EveBox(low=0, high=1, shape=(space.n, ), dtype=space.dtype)
    if isinstance(space, EveMultiDiscrete):
        return EveBox(low=np.zeros_like(space.nvec),
                      high=space.nvec,
                      dtype=space.dtype)
    raise NotImplementedError
