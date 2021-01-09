#          _     _          _      _                 _   _        _             _
#         /\ \  /\ \    _ / /\    /\ \              /\_\/\_\ _   _\ \          /\ \
#        /  \ \ \ \ \  /_/ / /   /  \ \            / / / / //\_\/\__ \         \ \ \
#       / /\ \ \ \ \ \ \___\/   / /\ \ \          /\ \/ \ \/ / / /_ \_\        /\ \_\
#      / / /\ \_\/ / /  \ \ \  / / /\ \_\ ____   /  \____\__/ / / /\/_/       / /\/_/
#     / /_/_ \/_/\ \ \   \_\ \/ /_/_ \/_/\____/\/ /\/________/ / /           / / /
#    / /____/\    \ \ \  / / / /____/\  \/____\/ / /\/_// / / / /           / / /
#   / /\____\/     \ \ \/ / / /\____\/        / / /    / / / / / ____      / / /
#  / / /______      \ \ \/ / / /______       / / /    / / / /_/_/ ___/\___/ / /__
# / / /_______\      \ \  / / /_______\      \/_/    / / /_______/\__\/\__\/_/___\
# \/__________/       \_\/\/__________/              \/_/\_______\/   \/_________/

from abc import abstractmethod
from typing import Callable, List, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from torch.nn import Parameter

import eve.core.quantize_fn
from eve.core.eve import Eve
from eve.core.state import State

# pylint: disable=no-member
# pylint: disable=access-member-before-definition


class Quan(Eve):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def quantization_forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def non_quantization_forward(self, x: Tensor) -> Tensor:
        return x

    def forward(self, x: Tensor) -> Tensor:
        if self.quantization:
            return self.quantization_forward(x)
        else:
            return self.non_quantization_forward(x)


class Quantizer(Quan):
    """The base quantizer to hold different quantization function.

    Args: 
        state: the object contains necessary arguments.
        bits: the bits/init bits of current layer.
        quantize_fn: the quantization function used.
            If str, we will lookup the available quantization function, others 
            we will use it directly.
        range_tracker: the method to record val_min and val_max.
            `average_tracker` and `glocal_tracker` is supported.
            if callable function provided, we will use it directly.
        average_tracker_momentum: if use average tracker, the momentum of the tracker.
        upgrade_bits: if ``True``, the bits can be changed all the time by upgrader.
        neuron_wise: if ``False``, the bits will be shared among each neurons.
            NOTE: if ``True``, the observation returned is [neurons, states] formot.
            if ``Flase``, the observation returned is [states] formot.
        asymmetric: asymmetric quantization or not.
        signed_quantization: if ``True``, the quantization range can be negative.
        learnable_alpha: if the alpha should be learnable. in some quantization
            functions, the alpha value can be updated by gradient. if the alpha
            can be learnabled, we will not update it by tracker information.
        state_list: a list contains ["name", fn] to calculate states.
            if fn is None, we will try to load the build-in methods.
            refer ``eve.core.state`` to see the supported function.
        upgrade_fn: the function to update eve parameter, e.i. bits in this module.
            if None, a directly replace operation will be used. 
        kwargs: the extra arguments to quantization function.

    .. note::

        all the eve parameters, which needed upgrader is treated as neuron wise.
        if learnable_alpha is enabled, the range tracker will be disabled to 
        avoid the conflict of grad update of alpha.
    """
    def __init__(
        self,
        state: State,
        bits: int = 8,
        quantize_fn: str = "Round",
        range_tracker: str = "average_tracker",
        average_tracker_momentum: float = 0.1,
        upgrade_bits: bool = False,
        neuron_wise: bool = False,
        asymmetric: bool = False,
        signed_quantization: bool = False,
        learnable_alpha: bool = False,
        state_list: List = None,
        upgrade_fn: Callable = None,
        **kwargs,
    ):
        super().__init__()

        self.state = state
        self.kwargs = kwargs
        self.asymmetric = asymmetric
        self.signed_quantization = signed_quantization
        self.learnable_alpha = learnable_alpha
        self.max_bits = bits
        self.neuron_wise = neuron_wise

        self.state.set_neuron_wise(self.neuron_wise)

        # bit eve is in range [0, 1]
        bits_eve = self.state._align_dims(
            th.Tensor([1.0] * (self.state.neurons if self.neuron_wise else 1)))
        self.register_eve_parameter("bits_eve",
                                    Parameter(bits_eve, upgrade_bits))

        # register function for bits_eve
        if upgrade_fn is None:

            def upgrade_fn(param, action=None):
                # action is always in [0, 1]
                # if action is not None, take action
                if action is not None:
                    param.zero_().add_(action)
                else:
                    pass

        self.register_upgrade_fn(self.bits_eve, upgrade_fn)

        # register the state observation needed for bits_eve.
        if state_list is not None:
            for name, fn in state_list:
                self.state.register_state_fn(name, fn)

        # used for tracker
        self.register_buffer("min_val", None, persistent=False)
        self.register_buffer("max_val", None, persistent=False)

        # alpha
        self.alpha = nn.Parameter(
            self.state._align_dims(th.Tensor([1.0] * self.state.neurons)),
            learnable_alpha)

        # register forward pre hook to do range tracker.
        self.average_tracker_momentum = average_tracker_momentum
        if isinstance(range_tracker, str):
            range_tracker = getattr(self, range_tracker)
        self.register_forward_pre_hook(range_tracker)

        # get quantization function used in this layer
        if isinstance(quantize_fn, str):
            try:
                quantize_fn = getattr(eve.core.quantize_fn, quantize_fn)
            except Exception as e:
                raise NotImplementedError(
                    f"only the following function supported: {eve.core.quantize_fn.__all__}"
                    f"Got: {quantize_fn}"
                    f"Raise: {e}")
        self.quantize_fn = quantize_fn

    @property
    def states(self):
        return len(self.state.state_list())

    def obs(self, input, output):
        if not self.bits_eve.requires_grad:
            return None
        else:
            return self.state.gather_states(input, output)

    @property
    def bits(self):
        return th.floor(self.bits_eve * self.max_bits)

    @property
    def positive(self):
        if self.signed_quantization:
            return (2**(self.bits - 1) - 1) * (self.bits > 0)
        else:
            return (2**(self.bits) - 1) * (self.bits > 0)

    @property
    def negative(self):
        if self.signed_quantization:
            return -(2**(self.bits - 1)) * (self.bits > 0)
        else:
            return th.zeros_like(self.bits)

    @th.no_grad()
    def asymmetric_quantization_param(self):
        quantized_range = self.positive - self.negative
        float_range = self.max_val - self.min_val
        self.alpha.data.zero_().add_(
            float_range * quantized_range /
            (quantized_range * quantized_range + 1e-8))
        self.zero_point = th.round(self.min_val * self.alpha /
                                   (self.alpha * self.alpha + 1e-8))

    @th.no_grad()
    def symmetric_quantization_param(self):
        quantized_range = th.max(th.abs(self.positive), th.abs(self.negative))
        float_range = th.max(th.abs(self.min_val), th.abs(self.max_val))

        # in bits == 0, the quantized_range equals zero too.
        # we multiply quantized_range both in Numerator and Denominator to keep the
        # final alpha equals to 0
        self.alpha.data.zero_().add_(
            float_range * quantized_range /
            (quantized_range * quantized_range + 1e-8))
        self.zero_point = th.zeros_like(self.alpha)

    @staticmethod
    @th.no_grad()
    def average_tracker(cls, input: Tensor):
        # input is tuple, take the first element.
        input = input[0]
        # flatten input
        input = input.transpose(0, 1).flatten(start_dim=1)  # [neurons, -1]

        min_val = cls.state._align_dims(input.min(1, keepdim=False)[0])
        max_val = cls.state._align_dims(input.max(1, keepdim=False)[0])

        if cls.learnable_alpha:
            if cls.asymmetric:
                cls.zero_point = th.round(min_val * cls.alpha)
            else:
                cls.zero_point = th.zeros_like(cls.alpha)
            return

        if cls.min_val is None and cls.max_val is None:
            cls.min_val = min_val
            cls.max_val = max_val
        else:
            cls.min_val.mul_(1 - cls.average_tracker_momentum).add_(
                min_val, alpha=cls.average_tracker_momentum)
            cls.max_val.mul_(1 - cls.average_tracker_momentum).add_(
                max_val, alpha=cls.average_tracker_momentum)

        # update parameters
        if cls.asymmetric:
            cls.asymmetric_quantization_param()
        else:
            cls.symmetric_quantization_param()

    @staticmethod
    @th.no_grad()
    def global_tracker(cls, input: Tensor):
        # flatten input
        input = input.transpose(0, 1).flatten()

        min_val = cls.state._align_dims(input.min(1, keepdim=False)[0])
        max_val = cls.state._align_dims(input.max(1, keepdim=False)[0])

        if cls.learnable_alpha:
            if cls.asymmetric:
                cls.zero_point = th.round(min_val * cls.alpha)
            else:
                cls.zero_point = th.zeros_like(cls.alpha)
            return

        if cls.min_val is None:
            cls.min_val = min_val
            cls.max_val = max_val
        else:
            min_val = th.min(min_val, cls.min_val)
            max_val = th.max(max_val, cls.max_val)

            cls.min_val.zero_().add_(min_val)
            cls.max_val.zero_().add_(max_val)

        # update parameters
        if cls.asymmetric:
            cls.asymmetric_quantization_param()
        else:
            cls.symmetric_quantization_param()

    def quantization_forward(self, x: Tensor) -> Tensor:
        return self.quantize_fn.apply(x, self.alpha, self.zero_point,
                                      self.positive, self.negative,
                                      **self.kwargs)
