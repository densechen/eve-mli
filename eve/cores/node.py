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
        requires_upgrade (bool): if ``True``, the voltage_threshold will be added to
            eve parameters, which can be upgraded. Default: ``False``.
    
    .. note::
        
        The voltage_threshold have a great influence on the fire rate of
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
    requires_upgrade: bool  # whether add voltage_threshold to eve parameters.

    def __init__(
        self,
        state: State = None,
        voltage_threshold: float = 0.5,
        time_independent: bool = True,
        requires_upgrade: bool = False,
    ):
        super(Node, self).__init__()

        self.neurons = state.neurons
        self.filter_type = state.filter_type
        self.time_independent = time_independent
        self.requires_upgrade = requires_upgrade
        self.state = state

        # register voltage_threshold
        voltage_threshold = torch.Tensor([voltage_threshold] * self.neurons)
        voltage_threshold = _align_dims(self.filter_type, voltage_threshold)

        self.register_eve_parameter(
            "voltage_threshold_eve",
            Parameter(voltage_threshold, requires_grad=requires_upgrade))

        def upgrade_fn(param, action=None, obs=None):
            if action is None:
                # directly take the action
                param.zero_().add_(action.view_as(param))
            else:
                # keep x unchanged.
                pass

        self.register_upgrade_fn(self.voltage_threshold_eve, upgrade_fn)

        # register voltage as hidden state, which will reset every time.
        self.register_hidden_state("voltage_hid", None)

        # register an forward hook to calculate the observation states
        self.register_forward_hook(Node._attach_obs_to_eve_parameters)

    def obs(self):
        if not self.requires_upgrade or not self.spiking:
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
                self.state.fire_rate,
            ],
                               dim=-1)

    def _reset(self, set_to_none: bool = False) -> None:
        """Resets current layer's hidden state to None.
        """
        super()._reset(set_to_none=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.spiking:
            fire = self.spiking_forward(x)
            # compute observation states
            self.state.fire_rate_fn(x, fire)
            return fire
        else:
            return self.non_spiking_forward(x)

def heaviside(x: Tensor) -> Tensor:
    r"""The heaviside function."""
    return torch.ge(x, 0.).to(x)

class if_fire(Function):
    """fire and reset membrane voltage.

    Different with traditional spiking neural network, which fires will emit 1,
    eve will emit a voltage and deliver it to Quan for further process.
    We can make sure that, when the bit width of Quan equals to 1, eve
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
    eve will emit a voltage and deliver it to Quan for further process.
    We can make sure that, when the bit width of Quan equals to 1, eve
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
    def spiking_forward(self, dv: Tensor) -> Tensor:
        if self.voltage_hid is not None and self.voltage_hid.shape == dv.shape:
            voltage = (self.voltage_hid if not self.time_independent else
                       self.voltage_hid.detach())
        else:
            voltage = torch.zeros_like(dv)

        self.voltage_hid, output = if_fire().apply(voltage, dv,
                                                   self.voltage_threshold_eve)
        return output

    def non_spiking_forward(self, x: Tensor) -> Tensor:
        return F.relu(x)

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

    def spiking_forward(self, dv: Tensor) -> Tensor:
        if self.voltage_hid is not None and self.voltage_hid.shape == dv.shape:
            voltage = (self.voltage_hid if not self.time_independent else
                       self.voltage_hid.detach())
        else:
            voltage = torch.zeros_like(dv)

        self.voltage_hid, output = lif_fire().apply(voltage, dv,
                                                    self.voltage_threshold_eve,
                                                    self.tau)
        return output

    def non_spiking_forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x)

