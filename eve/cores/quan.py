from abc import abstractmethod
from collections import namedtuple
from typing import List, OrderedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function, Variable
from torch.nn import Parameter

from eve.cores.eve import Eve
from eve.cores.state import State
from eve.cores.utils import _align_dims


# pylint: disable=no-member
# pylint: disable=access-member-before-definition
class Quan(Eve):
    r"""Implements a Quan of Quantization Neural Network.

    Args:
        state (State): the object to compute static and dynamic states of this layer.
        max_bit_width (int): the max bit width of current layer. if bit width is not
            upgradable, the bit_width is the initial bit width of current layer.
            if bit width is upgradable, the bit_width is the max bits allowed.
        requires_upgrade (bool): if ``True``, the bit width will be added to eve
            parameters, which can be upgraded. Default: ``False``.
    
    .. note::

        The bit_width is vital to quantization neural networks, you must make it
        carefully. At the same time, take care of bit_width == 1 or 
        bit_width == 0. If bit_width == 1, any value large than 0 will output
        1, others 0. If bit_width == 0, all values output 0.
        bit_width == 1 can be concated with :class:`Node` to generate spiking 
        signals.
        
    """
    neurons: int  # the number of neurons in this layer
    filter_type: str  # the type of kernels in previous layers
    requires_upgrade: bool  # whether add bit_width to eve parameters.

    def __init__(self,
                 state: State,
                 max_bit_width: int = 8,
                 requires_upgrade: bool = False,
                 **kwargs):
        super(Quan, self).__init__()
        self.state = state
        self.neurons = state.neurons
        self.filter_type = state.filter_type
        self.kwargs = kwargs
        self.requires_upgrade = requires_upgrade

        # register bit_width
        bit_width = torch.Tensor([max_bit_width] * self.neurons)
        bit_width = _align_dims(self.filter_type, bit_width)

        self.register_eve_parameter(
            "bit_width_eve",
            Parameter(bit_width, requires_grad=requires_upgrade))

        def upgrade_fn(param, action=None, obs=None):
            if action is not None:
                # ensure y in [0, 1]
                action = torch.clamp(action, min=0.0, max=1.0)
                # convert to specified bit
                new_bit_width = torch.floor(action * (max_bit_width + 1))
                param.zero_().add_(new_bit_width)
            else:
                pass

        self.register_upgrade_fn(self.bit_width_eve, upgrade_fn)

        # register parameter used in optimizer optimization.
        # this parameter will be ignored in some cases, such as ste quantization
        # method, however, we still define it here.
        alpha = 1.0 / (2**self.bit_width_eve - 1 + 1e-8)
        alpha = _align_dims(self.filter_type, alpha)
        self.alpha = Parameter(alpha, requires_grad=True)

    def obs(self):
        if not self.requires_upgrade:
            return None
        else:
            return torch.stack([
                self.state.k,
                self.state.feat_out,
                self.state.feat_in,
                self.state.kernel_size,
                self.state.stride,
                self.state.param_num,
                self.state.l1_norm,
                self.state.kl_div,
            ],
                               dim=-1)

    def _reset(self, set_to_none: bool = False) -> None:
        """Resets current layer's hidden state to None.
        """
        super()._reset(set_to_none=True)

    @abstractmethod
    def quan(self, x: Tensor) -> Tensor:
        """Implements different quan in subclass.
        """
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        quan = self.quan(x)

        # compute static obs
        self.state.kl_div_fn(x, quan)

        return quan


def quantize(input: Tensor,
             alpha: Tensor,
             zero_point: Tensor,
             positive: Tensor,
             negative: Tensor,
             eps=1e-8) -> Tensor:
    output = torch.round(1. / (alpha + eps) * input - zero_point)

    output = torch.where(output < negative, negative, output)
    output = torch.where(output > positive, positive, output)
    return output


def dequantize(input: Tensor, alpha: Tensor, zero_point: Tensor) -> Tensor:
    return (input - zero_point) * alpha


class ste(Function):
    """SurrogateFunction 
    """
    @staticmethod
    def forward(ctx, x: Tensor, alpha: Tensor, bit_width: int) -> Tensor:
        quan_x = quantize(x, alpha, torch.zeros_like(alpha), 2**bit_width - 1,
                          torch.zeros_like(alpha))

        dequan_x = dequantize(quan_x, alpha, torch.zeros_like(alpha))

        return dequan_x

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> List[Union[Tensor, None]]:
        return grad_output, None, None


class lsq(Function):
    """
    .. note::
        
        LEARNED STEP SIZE QUANTIZATION: https://quanoview.readthedocs.io/en/latest/_raw/LSQ.html
    """
    @staticmethod
    def forward(ctx, x: Tensor, alpha: Tensor, bit_width: Tensor) -> Tensor:
        if torch.any(alpha < 0):
            raise ValueError("alpha must be positive")

        quan_x = quantize(x, alpha, torch.zeros_like(alpha), 2**bit_width - 1,
                          torch.zeros_like(alpha))

        dequan_x = dequantize(quan_x, alpha, torch.zeros_like(alpha))

        ctx.save_for_backward(x, quan_x, alpha, bit_width)

        return dequan_x

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> List[Union[Tensor, None]]:
        x, quan_x, alpha, bit_width = ctx.saved_tensors

        positive = 1**bit_width - 1
        g = (torch.ones_like(positive) *
             (positive > 0).float()) / (torch.sqrt(x.numel() * positive) +
                                        1e-8)

        lower = (quan_x < 0.0).float()
        upper = (quan_x > positive).float()
        middle = torch.ones_like(upper) - upper - lower

        grad_alpha = ((lower * 0.0 + upper * positive + middle *
                       (quan_x - x / alpha)) * grad_output * g)
        for i, dims in enumerate(alpha.shape):
            if dims == 1:
                grad_alpha = grad_alpha.sum(dim=i, keepdim=True)

        grad_x = middle * grad_output

        # HACK: alpha must be positive.
        return grad_x, grad_alpha, None


def ln_error(x: Tensor, alpha: Tensor, bit_width: Tensor,
             regular: str) -> Tensor:
    quan_x = quantize(x, alpha, torch.zeros_like(alpha), 2**bit_width - 1,
                      torch.zeros_like(alpha))
    dequan_x = dequantize(quan_x, alpha, torch.zeros_like(alpha))

    if regular == "l2":
        error = (x - dequan_x)**2
    else:
        error = (x - dequan_x).abs()

    for i, dims in enumerate(alpha.shape):
        if dims == 1:
            error = error.mean(dim=i, keepdim=True)
    return error


def update_running_alpha(error: Tensor, lower_error: Tensor,
                         upper_error: Tensor) -> List[Tensor]:
    a1 = error - lower_error
    a2 = upper_error - error

    g1 = a1 >= 0
    g2 = a2 > 0
    g3 = a1 + a2 >= 0
    """
        g1  g2  g3  res
        0   0   0   big
        0   0   1   big
        0   1   0   keep
        0   1   1   keep
        1   0   0   big
        1   0   1   small
        1   1   0   small
        1   1   1   small
    """
    b = ((g1 == 0) * (g2 == 0) == 1) + ((g1 * (g2 == 0) * (g3 == 0)) > 0) > 0
    s = (((g1 * g2) > 0) + ((g1 * (g2 == 0) * g3) > 0)) > 0
    return b, s


class llsq(Function):
    """
    .. note::
        
        Linear Symmetric Quantization of Neural Networks for Low-precision Integer Hardware: https://openreview.net/forum?id=H1lBj2VFPS

    """
    @staticmethod
    def forward(ctx, x: Tensor, alpha: Tensor, bit_width: Tensor,
                regular: str) -> Tensor:
        if torch.any(alpha < 0):
            raise ValueError("alpha must be positive")

        quan_x = quantize(x, alpha, torch.zeros_like(alpha), 2**bit_width - 1,
                          torch.zeros_like(alpha))
        dequan_x = dequantize(quan_x, alpha, torch.zeros_like(alpha))

        ctx.save_for_backward(x, quan_x, alpha, bit_width)
        ctx.others = (regular, )

        return dequan_x

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> List[Union[Tensor, None]]:
        x, quan_x, alpha, bit_width = ctx.saved_tensors
        regular, = ctx.others

        positive = 2**bit_width - 1
        lower = (quan_x < 0.0).float()
        upper = (quan_x > positive).float()
        middle = torch.ones_like(upper) - upper - lower

        grad_output = grad_output * middle

        error = ln_error(x, alpha, bit_width, regular)
        lower_error = ln_error(x / 2, alpha, bit_width, regular)
        upper_error = ln_error(x * 2, alpha, bit_width, regular)

        b, s = update_running_alpha(error, lower_error, upper_error)

        grad_alpha = torch.zeros_like(alpha)
        grad_alpha = torch.where(b, -(alpha**2), grad_alpha) + torch.where(
            s, alpha**2, grad_alpha)
        # HACK: alpha must be positive.
        return grad_output, grad_alpha, None, None


class SteQuan(Quan):
    """SteQuan.
    
    In SteQuan, the initial alpha value set as 1 / (x << bit_width - 1).
    """
    def quan(self, x: Tensor) -> Tensor:
        # if bit width is changed by upgrader, alpha should be updated.
        def update_alpha():
            alpha = 1.0 / (2**self.bit_width_eve - 1 + 1e-8)
            alpha = _align_dims(self.filter_type, alpha)
            self.alpha.data.zero_().add_(alpha.to(self.alpha))

        if self.requires_upgrade:  # only happens in requires_upgrade.
            update_alpha()

        return ste().apply(x, self.alpha, self.bit_width_eve)


class LsqQuan(Quan):
    """LsqQuan.

    In LsqQuan, the initial alpha value is directly reset to 1.0.
    """
    def __init__(self, *args, **kwargs):
        super(LsqQuan, self).__init__(*args, **kwargs)
        self.alpha.data.fill_(1.0)

    def quan(self, x: Tensor) -> Tensor:
        return lsq().apply(x, self.alpha, self.bit_width_eve)


class LlsqQuan(Quan):
    """LlsqQuan

    In LlsqQuan, the initial alpha value is directly reset to 1.0.
    """
    def __init__(self, *args, **kwargs):
        super(LlsqQuan, self).__init__(*args, **kwargs)
        self.alpha.data.fill_(1.0)

    def quan(self, x: Tensor) -> Tensor:
        return llsq().apply(x, self.alpha, self.bit_width_eve,
                            self.kwargs.get("regular", "l2"))
