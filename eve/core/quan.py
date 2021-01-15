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
from typing import Callable, List, Union, Tuple

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
    # state to record infomation.
    state: State

    def __init__(self):
        super().__init__()

    def _reset(self, set_to_none: bool = False):
        super()._reset(set_to_none=set_to_none)
        if hasattr(self, "state") and isinstance(self.state, State):
            self.state.reset(set_to_none=set_to_none)

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
            can be learnable, we will not update it by tracker information.
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
        learnable_alpha: bool = None,
        upgrade_fn: Callable = None,
        **kwargs,
    ):
        super().__init__()

        self.state = state

        # set neuron wise for state which support neuron wise mode
        self.state.set_neuron_wise(neuron_wise)

        self.kwargs = kwargs
        self.asymmetric = asymmetric
        self.signed_quantization = signed_quantization

        # set default learnable values
        if learnable_alpha is None and isinstance(quantize_fn, str) and quantize_fn in ["Lsq", "Llsq"]:
            learnable_alpha = True
        elif learnable_alpha is None:
            learnable_alpha = False

        self.learnable_alpha = learnable_alpha
        self.max_bits = bits
        self.neuron_wise = neuron_wise

        self.state.set_neuron_wise(self.neuron_wise)

        # bit eve is in range [0, 1]
        bits_eve = th.Tensor(
            [1.0] * (self.state.neurons if self.neuron_wise else 1))
        bits_eve = bits_eve.reshape(self.state.align_dims)
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

        # used for tracker
        self.register_buffer("min_val", None, persistent=False)
        self.register_buffer("max_val", None, persistent=False)

        # alpha
        alpha = th.Tensor(
            [1.0] * (self.state.neurons if self.neuron_wise else 1))
        alpha = alpha.reshape(self.state.align_dims)
        self.alpha = nn.Parameter(alpha, learnable_alpha)

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
        self.quantize_fn = quantize_fn(**self.kwargs)

    @property
    def states(self):
        return len(self.state.local_statistics)

    def obs(self, input: th.Tensor, output: th.Tensor) -> th.Tensor:
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
    def average_tracker(cls, input: Tuple):
        # input is tuple, take the first element.
        input = input[0]

        # min_val = cls.state._align_dims(input.min(1, keepdim=False)[0])
        # max_val = cls.state._align_dims(input.max(1, keepdim=False)[0])
        min_val = input
        max_val = input
        dims = {"data": cls.state.data_reduce_dims,
                "param": cls.state.param_reduce_dims}[cls.state.apply_on]

        for dim in dims:
            min_val = min_val.min(dim, keepdim=True)[0]
            max_val = max_val.max(dim, keepdim=True)[0]
        min_val = min_val.reshape(cls.state.align_dims)
        max_val = max_val.reshape(cls.state.align_dims)

        if cls.learnable_alpha:
            if cls.asymmetric:
                with th.no_grad():
                    cls.zero_point = th.round(cls.min_val * cls.alpha)
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
    def global_tracker(cls, input: Tuple):
        # input is tuple, take the first element.
        input = input[0]

        min_val = input
        max_val = input
        dims = {"data": cls.state.data_reduce_dims,
                "param": cls.state.param_reduce_dims}[cls.state.apply_on]
        for dim in dims:
            min_val = min_val.min(dim, keepdim=True)[0]
            max_val = max_val.max(dim, keepdim=True)[0]
        min_val = min_val.reshape(cls.state.align_dims)
        max_val = max_val.reshape(cls.state.align_dims)

        if cls.learnable_alpha:
            if cls.asymmetric:
                with th.no_grad():
                    cls.zero_point = th.round(cls.min_val * cls.alpha)
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
        return self.quantize_fn(x, self.alpha, self.zero_point,
                                self.positive, self.negative)


class INQuantizer(Quantizer):
    """The implementation of Incremental Network Quantization
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.state.apply_on == "param", "INQuantizer can only be applied on param!"

        mask_eve = Parameter(th.ones_like(self.state.filter_weight))
        self.register_eve_parameter("mask_eve", mask_eve)

        # register upgrade fn
        @th.no_grad()
        def upgrade_fn(param, action=None):
            """In this case, the action is quan_ratio!!!"""
            # rename the arguments here for a better understanding
            quan_ratio = action
            assert 0 <= quan_ratio <= 1.0, "quan_ratio must be in [0, 1]"
            mask_eve = param
            # get remaining weights (mask as 1)
            remaining_weights = self.state.filter_weight * mask_eve
            n_elements = self.state.filter_weight.numel()
            n_remaining_elements = mask_eve.sum()
            # Number of elements to be quantizated in this iteration
            # No.Remaining Elements - No.Require.Remaining Elements
            n_quan = int(n_remaining_elements - n_elements *
                         (1 - quan_ratio))

            # higher than quantization point will be quantized
            quan_point = th.topk(
                th.abs(remaining_weights).view(-1), n_quan)[0][-1]
            quantized_idx = th.abs(remaining_weights) >= quan_point

            # NOTE: Quantize weight here, not in forward pass!
            quantized_weights = remaining_weights * quantized_idx.float()

            # quantized weight
            quantized_weights = self.quantize_fn(
                quantized_weights, self.alpha, self.zero_point, self.positive, self.negative)

            filter_weight = self.state.filter_weight.clone()
            # set quantized weight to filter weight, in-place operation!
            self.state.filter_weight[quantized_idx].zero_()
            self.state.filter_weight.add_(quantized_weights)

            # set new mask, in-place operation!
            mask_eve[quantized_idx].zero_()

        # register to upgrade_fn
        self.register_upgrade_fn(self.mask_eve, upgrade_fn)

    def quantization_forward(self, x: Tensor) -> Tensor:
        # NOTE: DO NOT QUAN HERE, BUT AT THE UPGRADE FN OF mask_eve
        # quan_x = slef.quantize_fn(
        #     x, self.alpha, self.zero_point, self.positive, self.negative)

        # mask out grad
        if self.training:
            x.register_hook(lambda grad: grad * self.mask_eve)

        return x


class QILQuantizer(Quantizer):
    """The implementation of "Learning to Quantize Deep Networks by Optimizing Quantization Intervals with Task Loss".
    """

    def __init__(self, *args, **kwargs):
        if args[0].apply_on == "data":
            range_tracker = self.data_range_tracker
        elif args[0].apply_on == "param":
            range_tracker = self.param_range_tracker
        else:
            raise TypeError
        kwargs["range_tracker"] = range_tracker
        super().__init__(*args, **kwargs)

        # re-register max_val and min_val to parameters, which need grad
        del self.max_val
        del self.min_val

        max_val = th.Tensor(
            [1.0] * (self.state.neurons if self.neuron_wise else 1))
        max_val = max_val.reshape(self.state.align_dims)
        self.max_val = nn.Parameter(max_val)
        min_val = th.Tensor(
            [0.0] * (self.state.neurons if self.neuron_wise else 1))
        min_val = min_val.reshape(self.state.align_dims)
        self.min_val = nn.Parameter(min_val)
        if self.state.apply_on == "param":
            # only param needed gamma
            gamma = th.Tensor(
                [1.0] * (self.state.neurons if self.neuron_wise else 1))
            gamma = gamma.reshape(self.state.align_dims)
            self.gamma = nn.Parameter(gamma)

    @staticmethod
    def data_range_tracker(cls, input: Tuple):
        input = input[0]
        assert th.all(cls.max_val > cls.min_val)

        c_w = 0.5 * (cls.min_val + cls.max_val)  # 0.5
        d_w = 0.5 * (cls.max_val - cls.min_val)  # 0.5
        assert d_w > 0
        alpha_w = 0.5 / d_w  # 1
        beta_w = - 0.5 * c_w / d_w + 0.5  # 0

        interval_act = input * (th.abs(input) > cls.min_val).type_as(input) * \
            (th.abs(input) < cls.max_val).type_as(input)

        transformed_act = (th.abs(input) > cls.max_val).type_as(
            input) + alpha_w * th.abs(interval_act) + beta_w

        if cls.learnable_alpha:
            if cls.asymmetric:
                with th.no_grad():
                    cls.zero_point = th.round(cls.min_val * cls.alpha)
            else:
                cls.zero_point = th.zeros_like(cls.alpha)
            return

        # update parameters
        if cls.asymmetric:
            cls.asymmetric_quantization_param()
        else:
            cls.symmetric_quantization_param()

        return transformed_act

    @staticmethod
    def param_range_tracker(cls, input: Tuple):
        input = input[0]
        assert th.all(cls.max_val > cls.min_val)

        c_w = 0.5 * (cls.max_val + cls.min_val)  # 0.5
        d_w = 0.5 * (cls.max_val - cls.min_val)  # 0.5
        assert d_w > 0
        alpha_w = 0.5 / d_w  # 1
        beta_w = - 0.5 * c_w / d_w + 0.5  # 0

        interval_weight = input * (th.abs(input) > cls.min_val).type_as(
            input) * (th.abs(input) < cls.max_val).type_as(input)

        transformed_weight = th.sign(input) * (th.abs(input) > cls.max_val).type_as(input) + th.pow(
            alpha_w * th.abs(interval_weight) + beta_w,  cls.gamma) * th.sign(interval_weight)

        if cls.learnable_alpha:
            if cls.asymmetric:
                with th.no_grad():
                    cls.zero_point = th.round(cls.min_val * cls.alpha)
            else:
                cls.zero_point = th.zeros_like(cls.alpha)
            return

        # update parameters
        if cls.asymmetric:
            cls.asymmetric_quantization_param()
        else:
            cls.symmetric_quantization_param()

        return transformed_weight
