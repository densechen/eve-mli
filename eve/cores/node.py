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
class Node(Eve):
    r"""Implements a Node of Spiking Neural Network.

    Args:
        state (State): the object to compute static and dynamic states of this layer.
        voltage_threshold (float): the initial voltage threshold. 
            NOTE: the voltage threshold should be large than 0.5. :class:`Quan`
            in bit_width==1 will take a floor operation to generate spiking trains.
            which means even if the neuron fires, the voltage lower then 0.5 will
            not generate a valid spiking signal.
        time_independent (bool): if ``True``, we will detach membrane voltage
            from last time. Default: ``True``.
        requires_upgrading (bool): if ``True``, the voltage_threshold will be added to
            eve parameters, which can be upgraded. Default: ``False``.
    
    .. note::
        
        The voltage_threshold have a greate influence on the fire rate of
        neurons. Ensure a suitable fire rate will make spiking network easier to
        train. 
        Different with many existing tools designed for spiking neural network,
        which returns a spiking signal of 0/1 directly, :class:`Node` returns
        the fired voltage directly and you can deliver it to different :class:`Quan`
        to binary it.
        We have kept this flexible design approach, making it compatible with 
        the modules it has followed.
    """

    neurons: int  # the number of neurons in this layer
    filter_type: str  # the type of kernels in previous layers
    requires_upgrading: bool  # whether add voltage_threshold to eve parameters.

    def __init__(
        self,
        state: State = None,
        voltage_threshold: float = 0.5,
        time_independent: bool = True,
        requires_upgrading: bool = False,
    ):
        super(Node, self).__init__()

        self.neurons = state.neurons
        self.filter_type = state.filter_type
        self.time_independent = time_independent
        self.requires_upgrading = requires_upgrading
        self.state = state

        # It is not necessary to register it as hidden state or buffer again.
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
        # self.register_hidden_state("dynamic_obs_fire_rate", None)

        # register voltage_threshold
        voltage_threshold = torch.Tensor([voltage_threshold] * self.neurons)
        voltage_threshold = _align_dims(self.filter_type, voltage_threshold)

        # if upgradable:
        #     self.voltage_threshold = EveParameter(voltage_threshold,
        #                                           requires_upgrading=True)

        #     def upgrade_fn(x, y=None, z=None):
        #         if y is not None:
        #             # directly take the action
        #             x.zero_().add_(y.view_as(x))
        #         elif z is not None:
        #             # compute the new action based on fire rate
        #             fire_rate = z[..., -1]
        #             # we can control the fire rate via voltage_threshold
        #             # a suitable fire rate is very import for spiking neural network
        #             # it still remains a further work to figure out which fire rate
        #             # is better for spiking neural network, here, we adapt an
        #             # adaptive methods to make the fire rate around 0.5.
        #             y = torch.where(fire_rate > 0.5, 0.9, 1.1)
        #             y = y.view_as(x) * x
        #             x.zero_().add_(y)
        #         else:
        #             # keep x unchanged.
        #             pass

        #     # directly call eve parameters' functions
        #     self.voltage_threshold.register_upgrade_fn(upgrade_fn)
        #     # or
        #     # self.register_upgrade_fn("voltage_threshold", upgrade_fn)
        #     # is also fine.
        # else:
        #     self.register_buffer("voltage_threshold", voltage_threshold)
        self.voltage_threshold = EveParameter(
            voltage_threshold, requires_upgrading=requires_upgrading)

        def upgrade_fn(x, y=None, z=None):
            if y is not None:
                # directly take the action
                x.zero_().add_(y.view_as(x))
            # elif z is not None:
            #     # compute the new action based on fire rate
            #     fire_rate = z[..., -1]
            #     # we can control the fire rate via voltage_threshold
            #     # a suitable fire rate is very import for spiking neural network
            #     # it still remains a further work to figure out which fire rate
            #     # is better for spiking neural network, here, we adapt an
            #     # adaptive methods to make the fire rate around 0.5.
            #     y = torch.where(fire_rate > 0.5, 0.9, 1.1)
            #     y = y.view_as(x) * x
            #     x.zero_().add_(y)
            else:
                # keep x unchanged.
                pass

        # directly call eve parameters' functions
        self.voltage_threshold.register_upgrade_fn(upgrade_fn)
        # or
        # self.register_upgrade_fn("voltage_threshold", upgrade_fn)
        # is also fine.

        # register voltage as hidden state, which will reset every time.
        self.register_hidden_state("voltage", None)

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
        # if not self.upgradable:
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
        fire_rate = cls.state.fire_rate(input, output)  # [neurons, ]
        obs = torch.stack([l1_norm, fire_rate], dim=-1)  # [neurons, states]

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
    # def compute_dynamic_obs(self, x: Tensor, fire: Tensor) -> None:
    #     """Computes the dynamic observation states based on currently data flow.
    #     """
    #     fire_rate = (fire > 0.0).float()

    #     # means
    #     dim = {"linear": [0, 1], "conv2d": [0, 2, 3]}[self.filter_type]

    #     fire_rate = fire_rate.mean(dim, keepdim=True)
    #     if (self.dynamic_obs_fire_rate is not None
    #             and self.dynamic_obs_fire_rate.shape == fire_rate.shape):
    #         # dynamic_obs_ is not None, means that the Eve use zero_ reset.
    #         # then, if the shape is the same, we assign it as in-place operation.
    #         # the shape may be changed if the batch size is not the same.
    #         self.dynamic_obs_fire_rate.add_(fire_rate)
    #     else:
    #         # use None reset, or the shape is not the same, just replace it.
    #         self.dynamic_obs_fire_rate = fire_rate.detach().clone()

    # Move to cores.utils
    # def _align_dims(self, x: Tensor) -> Tensor:
    #     """Aligns x to kernel type dimensions.

    #     If kernel type is linear, x will be viewed as [1, 1, -1].
    #     If kernel type is conv2d, x will be viewed as [1, -1, 1, 1].
    #     """
    #     if self.filter_type == "linear":
    #         return x.view(1, 1, -1)
    #     elif self.filter_type == "conv2d":
    #         return x.view(1, -1, 1, 1)
    #     else:
    #         return TypeError("kernel type {} not supported".format(
    #             self.filter_type))

    def _reset(self, set_to_none: bool = False) -> None:
        """Resets current layer's hidden state to None.
        """
        super()._reset(set_to_none=True)

    @abstractmethod
    def node(self, x: Tensor) -> Tensor:
        """defines different node behavious.

        You should pay attention to spiking and non-spiking mode.
        """
        raise NotImplementedError

    # def forward(self, x: Tensor) -> Tensor:
    #     # ensure x with standard shape, which means [b, c, h, w] for conv2d
    #     # [b, n, c] for linear
    #     if self.spiking:
    #         expand_dim = False
    #         if self.filter_type == "linear" and x.dim() == 2:
    #             x = x.unsqueeze(dim=1)
    #             expand_dim = True

    #         fire = self.node(x)

    #         # if no eve parameters, skip the computing process to speed up.
    #         if self.upgradable:
    #             self.compute_dynamic_obs(x, fire)

    #         if expand_dim:
    #             fire = fire.squeeze(dim=1)
    #         return fire
    #     else:
    #         return x
    def forward(self, x: Tensor) -> Tensor:
        return self.node(x)


def heaviside(x: Tensor) -> Tensor:
    r"""The heaviside function."""
    return torch.ge(x, 0.).to(x)


class if_fire(Function):
    """fire and reset membrane voltage.

    Different with traditional spiking neural network, which fires will emit 1,
    QuanSpike will emit a voltage and deliver it to Quan for further process.
    We can make sure that, when the bit width of Quan equals to 1, QuanSpike
    will act the same as spiking neural network for that any value large than 0 
    will be emitted 1 in the Quan.
    """
    @staticmethod
    def forward(ctx, voltage: Tensor, dv: Tensor,
                threshold: Tensor) -> List[Tensor]:
        # neuron charging
        tmp = voltage + dv

        # fire
        fire = heaviside(tmp - threshold)

        # output
        dv = torch.where(fire == 1, tmp, torch.zeros_like(tmp))
        # reset
        voltage = torch.where(fire == 0, tmp, torch.zeros_like(tmp))

        return voltage, dv

    @staticmethod
    def backward(ctx, voltage_grad: Tensor,
                 dv_grad: Tensor) -> List[Union[Tensor, None]]:
        return voltage_grad, dv_grad, None


class lif_fire(Function):
    """fire and reset membrane voltage.

    Different with traditional spiking neural network, which fires will emit 1,
    QuanSpike will emit a voltage and deliver it to Quan for further process.
    We can make sure that, when the bit width of Quan equals to 1, QuanSpike
    will act the same as spiking neural network for that any value large than 0 
    will be emitted 1 in the Quan.
    """
    @staticmethod
    def forward(ctx, voltage: Tensor, dv: Tensor, threshold: Tensor,
                tau: Tensor) -> Tensor:
        # neuron charging
        tmp = voltage + (dv - voltage) / tau
        ctx.others = (tau, )

        # fire
        fire = heaviside(tmp - threshold)

        # output
        dv = torch.where(fire == 1, tmp, torch.zeros_like(tmp))
        # reset
        voltage = torch.where(fire == 0, tmp, torch.zeros_like(tmp))

        return voltage, dv

    @staticmethod
    def backward(ctx, voltage_grad: Tensor, dv_grad: Tensor):
        tau, = ctx.others
        voltage_grad = (1 - 1 / tau) * voltage_grad
        dv_grad = 1 / tau * dv_grad

        return voltage_grad, dv_grad, None, None


class IfNode(Node):
    """Implementation of IFNode.

    .. note::
        
        In non-spiking mode, IFNode is equal to a ReLU Function.
    """
    def node(self, dv: Tensor) -> Tensor:
        if self.spiking:
            if self.voltage is not None and self.voltage.shape == dv.shape:
                voltage = (self.voltage if not self.time_independent else
                           self.voltage.detach())
            else:
                voltage = torch.zeros_like(dv)

            self.voltage, output = if_fire().apply(voltage, dv,
                                                   self.voltage_threshold)
        else:
            output = F.relu(dv)

        return output


class LifNode(Node):
    """Implementation of LIFNode.

    .. note::

        In non-spiking mode, LIFNode is equal to a LeakyReLU Function.
    """
    def __init__(self, tau: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = tau

    def node(self, dv: Tensor) -> Tensor:
        if self.spiking:
            if self.voltage is not None and self.voltage.shape == dv.shape:
                voltage = (self.voltage if not self.time_independent else
                           self.voltage.detach())
            else:
                voltage = torch.zeros_like(dv)

            self.voltage, output = lif_fire().apply(voltage, dv,
                                                    self.voltage_threshold,
                                                    self.tau)
        else:
            output = F.leaky_relu(dv)
        return output
