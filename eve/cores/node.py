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

        # register voltage_threshold
        voltage_threshold = torch.Tensor([voltage_threshold] * self.neurons)
        voltage_threshold = _align_dims(self.filter_type, voltage_threshold)

        self.voltage_threshold = EveParameter(
            voltage_threshold, requires_upgrading=requires_upgrading)

        def upgrade_fn(x, y=None, z=None):
            if y is not None:
                # directly take the action
                x.zero_().add_(y.view_as(x))
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
