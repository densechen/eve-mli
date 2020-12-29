from abc import abstractmethod
from collections import namedtuple
from typing import List, OrderedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function, Variable
from torch.nn import Parameter

from eve.cores.eve import Eve, EveParameter
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
        requires_upgrading (bool): if ``True``, the bit width will be added to eve
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
    requires_upgrading: bool  # whether add bit_width to eve parameters.

    def __init__(self,
                 state: State,
                 max_bit_width: int = 8,
                 requires_upgrading: bool = False,
                 **kwargs):
        super(Quan, self).__init__()
        self.state = state
        self.neurons = state.neurons
        self.filter_type = state.filter_type
        self.kwargs = kwargs
        self.requires_upgrading = requires_upgrading

        # It is not necessary to register buffer any more.
        # # register static_obs as buffer, not hidden states
        # # NOTE: buffer will be saved along with parameters, but hidden states
        # # will not. hidden states will be cleared while calling :meth:`reset()`,
        # # but buffer will not.
        # for n, v in zip(static_obs._fields, static_obs):
        #     v = torch.Tensor([v] * self.neurons).view(self.neurons, -1)
        #     # NOTE: the buffer should be started as `static_obs_`, otherwise will
        #     # not be recognized as a static observation.
        #     # NOTE: the shape of static_obs must in [neurons, 1], even though
        #     # you have various static obs, you should register them one by one.
        #     # but not register them together as [neurons, n].
        #     self.register_buffer(f"static_obs_{n}", v, persistent=False)
        #     # the static_obs keep the same every time, it is not necessary to
        #     # save along with parametes. so, set persistent = False

        # it is not necessary to register dynamic observation states to hidden state.
        # zero_obs will clear them.
        # # register dynamic_obs as hidden state, not buffer.
        # # differnt with static_obs, which keeps unchanged, dynamic_obs
        # # is calculated online and will be changed all the time.
        # # we need to clear it in differnt batches. so, a hidden states is
        # # more suitable.
        # # NOTE: the dynamic_obs should begin with `dynamic_obs_`, otherwise
        # # they will not fecthed as dynamic_obs, but common hidden_states.
        # # once you register a hidden state with None, this attribute will
        # # have all the properties of hidden_states, even it reassigned a new
        # # value.
        # self.register_hidden_state("dynamic_obs_kl_div", None)

        # register bit_width
        bit_width = torch.Tensor([max_bit_width] * self.neurons)
        bit_width = _align_dims(self.filter_type, bit_width)

        # using the requires_upgrading operation to control this, not buffer.
        # if requires_upgrading:
        #     self.bit_width = EveParameter(bit_width, requires_upgrading=True)

        #     def upgrade_fn(x, y=None, z=None):
        #         if y is not None:
        #             # y is in [0, 1], we convert it to [0, .., max_bit_width]
        #             new_bit_width = torch.floor(y * (max_bit_width + 1))
        #             x.zero_().add_(new_bit_width)
        #         elif z is not None:
        #             kl_div = z[..., -1]  # neurons times 1
        #             # FIXME: 0.5 may not be a good value.
        #             y = torch.where(kl_div > 0.5, 1, -1)
        #             x.add_(y.view_as(x))
        #         else:
        #             # keep unchanged
        #             pass

        #     self.bit_width.register_upgrade_fn(upgrade_fn)
        # else:
        #     self.register_buffer("bit_width", bit_width)

        self.bit_width = EveParameter(bit_width,
                                      requires_upgrading=requires_upgrading)

        def upgrade_fn(x, y=None, z=None):
            if y is not None:
                # ensure y in [0, 1]
                y = torch.clamp(y, min=0.0, max=1.0)
                # convert to specified bit
                new_bit_width = torch.floor(y * (max_bit_width + 1))
                x.zero_().add_(new_bit_width)
            else:
                pass

        self.bit_width.register_upgrade_fn(upgrade_fn)

        # register parameter used in optimizer optimization.
        # this parameter will be ignored in some cases, such as ste quantization
        # method, however, we still define it here.
        alpha = 1.0 / (2**self.bit_width - 1 + 1e-8)
        alpha = _align_dims(self.filter_type, alpha)
        self.alpha = Parameter(alpha, requires_grad=True)

    # Move to utils
    # def _align_dims(self, x: Tensor) -> Tensor:
    #     """Aligns x to kernel type dimensions.

    #     If kernel type is linear, x will be viewed as [1, 1, -1].
    #     If kernel type is conv2d, x will be viewed as [1, -1, 1, 1].
    #     """
    #     if self.filter_type == nn.Linear:
    #         return x.view(1, 1, -1)
    #     elif self.filter_type == nn.Conv2d:
    #         return x.view(1, -1, 1, 1)
    #     else:
    #         return TypeError("kernel type {} not supported".format(
    #             self.filter_type))

    # @property
    # def static_obs(self):
    #     """all buffer starts with `static_obs_` will be added to this property.
    #     """
    #     static_obs = [
    #         v.view(-1, 1) for k, v in self.named_buffers()
    #         if k.startswith("static_obs_") and v is not None
    #     ]
    #     if len(static_obs):
    #         # NOTE: Must detach it from static observation, otherwise,
    #         # a reset operation applied on all observation will erase all datas
    #         # via a in-place operation.
    #         return torch.cat(static_obs,
    #                          dim=-1).detach().clone()  # [neurons, states]
    #     else:
    #         return None

    # @property
    # def dynamic_obs(self):
    #     """all hidden states starts with `dynamic_obs_` will be added to this property.
    #     """
    #     dynamic_obs = [
    #         v.view(-1, 1) for k, v in self.named_hidden_states()
    #         if k.startswith("dynamic_obs_") and v is not None
    #     ]
    #     if len(dynamic_obs):
    #         return torch.cat(dynamic_obs, dim=-1)  # [neurons, states]
    #     else:
    #         return None

    @staticmethod
    def _attach_obs_to_eve_parameters(cls, input: Tensor,
                                      output: Tensor) -> None:
        r"""Attaches static and dynamic observation states to eve parameters.
        
        This function will be register as a forward hook automatically.
        This function cannot modified both input and output values.
    
        Args:
            input (Tensor): the input of this layer.
            output (Tensor): the result of this layer.

        .. note::

            At spiking neural network, the network will be repeat many times, 
            and the observation states will be changed at every time. It need 
            a simple but effect method to average the observation states over time.
            Here, we adapt an move exp average strategy to the observation,
            which is 
            :math:`\text{obs}_{t} = \text{obs}_{t-1} \times 0.5 + \text{obs}_{t} \times 0.5`
        """
        # # if not eve parameters, we skip this function to speed up.
        # if not self.requires_upgrading:
        #     return
        # if self.static_obs is not None and self.dynamic_obs is not None:
        #     obs = torch.cat([self.static_obs, self.dynamic_obs], dim=-1)
        # elif self.static_obs is not None:
        #     obs = self.static_obs
        # elif self.dynamic_obs is not None:
        #     obs = self.dynamic_obs
        # else:
        #     raise ValueError("Invalid observation states."
        #                      "Got {} and {}.".format(
        #                          torch.typename(self.static_obs),
        #                          torch.typename(self.dynamic_obs)))
        if not cls.requires_upgrading: 
            return
            
        l1_norm = cls.state.l1_norm  # [neurons, ]
        kl_div = cls.state.kl_div(input, output)  # [neurons, ]

        obs = torch.stack([l1_norm, kl_div], dim=-1)  # [neurons, states]

        # NOTE: observation states is special for current layer eve parameters.
        # do not apply to sub-module or other module. so, set resurse=False.
        for k, v in cls.named_eve_parameters(recurse=False):
            # only attach to the eve parameters needed upgrading.
            if v is None or not v.requires_upgrading:
                continue
            elif v.obs is not None and v.obs.shape == obs:
                v.obs.mul_(0.5).add_(obs, alpha=0.5)
            elif v.obs is None:
                v.obs = obs.detach().clone()
            else:
                raise ValueError("Cannot assign {} to {}".format(
                    torch.typename(obs), k))
            # if v.numel() == 1:
            #     # neuron wise mode
            #     if v.obs is not None:
            #         v.obs.mul_(0.5).add_(obs.mean(dim=0, keepdim=True),
            #                              alpha=0.5)
            #     else:
            #         v.obs = obs.mean(dim=0, keepdim=True).detach().clone()
            # elif v.numel() == self.neurons:
            #     # neuron share mode
            #     if v.obs is not None:
            #         v.obs.mul_(0.5).add_(obs, alpha=0.5)
            #     else:
            #         v.obs = obs.detach().clone()
            # else:
            #     raise ValueError(
            #         f"eve parameters {k} with shape {v.shape} is not"
            #         "compatible with observation states with shape {obs.shape}"
            #     )

    # @torch.no_grad()
    # def compute_dynamic_obs(self, x: Tensor, quan: Tensor) -> None:
    #     """Computes the dynamic observation states based on currently data flow.
    #     """
    #     kl_div = F.kl_div(x, quan, reduction="none")

    #     # means
    #     dim = {"linear": [0, 1], "conv2d": [0, 2, 3]}[self.filter_type]

    #     kl_div = kl_div.mean(dim, keepdim=True)
    #     if (self.dynamic_obs_kl_div is not None
    #             and self.dynamic_obs_kl_div.shape == kl_div.shape):
    #         # dynamic_obs_ is not None, means that the Eve use zero_ reset.
    #         # then, if the shape is the same, we assign it as in-place operation.
    #         # the shape may be changed if the batch size is not the same.
    #         self.dynamic_obs_kl_div.add_(kl_div)
    #     else:
    #         # use None reset, or the shape is not the same, just replace it.
    #         self.dynamic_obs_kl_div = kl_div.detach().clone()

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
        return self.quan(x)

    # def forward(self, x: Tensor) -> Tensor:
    #     # ensure x with standard shape, which means [b, c, h, w] for conv2d
    #     # [b, n, c] for linear
    #     # expand_dim = False
    #     # if self.filter_type == "linear" and x.dim() == 2:
    #     #     x = x.unsqueeze(dim=1)
    #     #     expand_dim = True

    #     quan = self.quan(x)

    #     # # if no eve parameters, skip the computing process to speed up.
    #     # if self.requires_upgrading:
    #     #     self.compute_dynamic_obs(x, quan)

    #     # if expand_dim:
    #     #     quan = quan.squeeze(dim=1)
    #     return quan


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
            alpha = 1.0 / (2**self.bit_width - 1 + 1e-8)
            alpha = _align_dims(self.filter_type, alpha)
            self.alpha.data.zero_().add_(alpha.to(self.alpha))

        if self.requires_upgrading:  # only happens in requires_upgrading.
            update_alpha()

        return ste().apply(x, self.alpha, self.bit_width)


class LsqQuan(Quan):
    """LsqQuan.

    In LsqQuan, the initial alpha value is directly reset to 1.0.
    """
    def __init__(self, *args, **kwargs):
        super(LsqQuan, self).__init__(*args, **kwargs)
        self.alpha.data.fill_(1.0)

    def quan(self, x: Tensor) -> Tensor:
        return lsq().apply(x, self.alpha, self.bit_width)


class LlsqQuan(Quan):
    """LlsqQuan

    In LlsqQuan, the initial alpha value is directly reset to 1.0.
    """
    def __init__(self, *args, **kwargs):
        super(LlsqQuan, self).__init__(*args, **kwargs)
        self.alpha.data.fill_(1.0)

    def quan(self, x: Tensor) -> Tensor:
        return llsq().apply(x, self.alpha, self.bit_width,
                            self.kwargs.get("regular", "l2"))
