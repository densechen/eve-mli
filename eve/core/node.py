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
from typing import List, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function, Variable
from torch.nn import Parameter

import eve.core.surrogate_fn
from eve.core.eve import Eve
from eve.core.state import State

# pylint: disable=no-member
# pylint: disable=access-member-before-definition


class Node(Eve):
    r"""The base class of different spiking neuron node.
    """

    def __init__(self):
        super().__init__()

    def _reset(self, set_to_none: bool = False) -> None:
        super()._reset(set_to_none=True)
        if hasattr(self, "state") and isinstance(self.state, State):
            self.state.reset(set_to_none=set_to_none)

    @abstractmethod
    def spiking_forward(self, dv: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def non_spiking_forward(self, dv: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, dv: Tensor) -> Tensor:
        if self.spiking:
            return self.spiking_forward(dv)
        else:
            return self.non_spiking_forward(dv)


class IFNode(Node):
    """IFNode function.

    IFNode only acts in spiking mode, in non-spiking mode, it equals to 
    :class:`nn.ReLU`.

    Args: 
        state: the object contains necessary information from previous layer.
        voltage_threshold: the voltage threshold to fire a neuron.
        voltage_reset: the voltage to reset if fires. Set ``None`` to disable it.
        learnbale_threshold: if ``True``, make voltage threshold learnable.
        learnbale_reset: if ``True``, make voltage reset learnable.
        time_dependent: if ``True``, chain the membrane voltage in computing graph.
        neuron_wise: if ``False``, share the voltage threshold and voltage reset
            accross all neurons.
        surrogate_fn (str or Callable): the function used to estimate the 
            gradient for fire operation. if str, we will look up the surrogate
            function from implemented approaches, if fn, we will set it directly.
            the surrogate fn should have the following signatures:

            >>> class surrogate_fn(Function):
            >>>     @staticmethod
            >>>     def forward(cls, )
            NOTE: we will not check the signatures of surrogate fn.
        binary: if ``True``, return the value after surrogate fn (0 ro 1), 
            otherwise, return the fired voltage by (voltage - voltage_reset).
        kwargs: the extra arguments needed for specified surrogate function.

    .. note::

        If binary is ``True``, we will apply the surrogate function on fired 
        neurons' voltage, otherwise, we will DISABLE the surrogate function and
        return (voltage - voltage_reset).
    """

    def __init__(self,
                 state: State,
                 voltage_threshold: float = 1.0,
                 voltage_reset: float = 0.0,
                 learnable_threshold: bool = False,
                 learnable_reset: bool = False,
                 time_dependent: bool = True,
                 neuron_wise: bool = True,
                 surrogate_fn: str = "Sigmoid",
                 binary: bool = True,
                 **kwargs):
        super().__init__()
        self.state = state

        # set neuron wise for state which support neuron wise mode
        self.state.set_neuron_wise(neuron_wise)

        self.neurons = self.state.neurons
        self.neuron_wise = neuron_wise
        self.time_dependent = time_dependent
        self.binary = binary
        self.kwargs = kwargs

        if isinstance(surrogate_fn, str):
            try:
                self.surrogate_fn = getattr(eve.core.surrogate_fn,
                                            surrogate_fn)(**kwargs)
            except Exception as e:
                raise NotImplementedError(
                    f"only the following function supported: {eve.core.surrogate_fn.__all__}"
                    f"Got: {surrogate_fn}"
                    f"Raise: {e}")
        else:
            self.surrogate_fn = surrogate_fn(**kwargs)

        voltage_threshold = th.Tensor(
            [voltage_threshold] * (self.neurons if self.neuron_wise else 1))
        voltage_threshold = voltage_threshold.reshape(self.state.align_dims)

        self.voltage_threshold = nn.Parameter(
            voltage_threshold, requires_grad=learnable_threshold)
        if voltage_reset is not None:
            voltage_reset = th.Tensor(
                [voltage_reset] * (self.neurons if self.neuron_wise else 1))
            voltage_reset = voltage_reset.reshape(self.state.align_dims)
            self.voltage_reset = nn.Parameter(voltage_reset,
                                              requires_grad=learnable_reset)
        else:
            self.register_parameter("voltage_reset", None)

        # register a hidden state
        self.register_eve_buffer("membrane_voltage_eve", None)

    def charging(self, membrane_voltage: Tensor, dv: Tensor) -> Tensor:
        return membrane_voltage + dv

    def spiking_forward(self, dv: Tensor) -> Tensor:
        # charging
        if self.membrane_voltage_eve is not None and self.membrane_voltage_eve.shape == dv.shape:
            membrane_voltage_eve = (self.membrane_voltage_eve
                                    if self.time_dependent else
                                    self.membrane_voltage_eve.detach())
        else:
            membrane_voltage_eve = th.zeros_like(dv)

        membrane_voltage_eve = self.charging(membrane_voltage_eve, dv)

        # select fired neurons
        fired_neurons = (membrane_voltage_eve > self.voltage_threshold).float()

        if self.voltage_reset is None:
            self.membrane_voltage_eve = membrane_voltage_eve - \
                fired_neurons * self.voltage_threshold
        else:
            self.membrane_voltage_eve = (
                1 - fired_neurons
            ) * membrane_voltage_eve + fired_neurons * self.voltage_reset

        if self.binary:
            return self.surrogate_fn(membrane_voltage_eve -
                                     self.voltage_threshold)
        else:
            return membrane_voltage_eve - self.voltage_threshold

    def non_spiking_forward(self, dv: Tensor) -> Tensor:
        return F.relu(dv)


class LIFNode(IFNode):
    """LIFNode function.

    LIFNode only acts in spiking mode, in non-spiking mode, it equals to 
    :class:`nn.LeakyReLU`.

    Args: 
        state: the object contains necessary information from previous layer.
        tau: membrane time constant.
        voltage_threshold: the voltage threshold to fire a neuron. 
            shared among all neurons.
        voltage_reset: the voltage to reset if fires. Set ``None`` to disable it.
        learnbale_threshold: if ``True``, make voltage threshold learnable.
        learnbale_reset: if ``True``, make voltage reset learnable.
        time_dependent: if ``True``, chain the membrane voltage in computing graph.
        neuron_wise: if ``False``, share the voltage threshold and voltage reset
            accross all neurons.
        surrogate_fn (str or Callable): the function used to estimate the 
            gradient for fire operation. if str, we will look up the surrogate
            function from implemented approaches, if fn, we will set it directly.
            the surrogate fn should have the following signatures:

            >>> class surrogate_fn(Function):
            >>>     @staticmethod
            >>>     def forward(cls, )
            NOTE: we will not check the signatures of surrogate fn.
        binary: if ``True``, return the value after surrogate fn (0 ro 1), 
            otherwise, return the fired voltage by (voltage - voltage_reset).
        kwargs: the extra arguments needed for specified surrogate function.

    .. note::

        If binary is ``True``, we will apply the surrogate function on fired 
        neurons' voltage, otherwise, we will DISABLE the surrogate function and
        return (voltage - voltage_reset).
    """

    def __init__(
        self,
        *args,
        tau: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert tau > 0
        self.tau = tau

    def charging(self, membrane_voltage: Tensor, dv: Tensor) -> Tensor:
        if self.voltage_reset is None:
            return membrane_voltage + (dv - membrane_voltage) / self.tau
        else:
            return membrane_voltage + (
                dv - (membrane_voltage - self.voltage_reset)) / self.tau

    def non_spiking_forward(self, dv: Tensor) -> Tensor:
        return F.leaky_relu(dv)
