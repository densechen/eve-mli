import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import abstractmethod

from eve.cores.eve import Eve


# pylint: disable=access-member-before-definition
# pylint: disable=no-member
class Encoder(Eve):
    """Base class of different encoders.

    Args:
        max_timesteps (int): the length of spiking trains.

    .. note:: 
        Only take effect under spiking mode. In non-spiking mode, it returns input directly.
    """
    def __init__(self, max_timesteps: int = 1):
        super().__init__()

        assert max_timesteps >= 1

        self.max_timesteps = max_timesteps

        self.register_hidden_state("raw_input", None)

    def _reset(self, set_to_none: bool = False) -> None:
        """Sets to None.
        """
        for name, _ in self.named_hidden_states():
            self.__setattr__(name, None)

    def forward(self, x: Tensor) -> Tensor:
        """Encode x to spikes.

        The x will only be used at first time forward, until the reset is called.
        """
        if self.spiking:
            return self.encode(x).float()
        else:
            return x

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError("encode is not implemented yet")


class RateEncoder(Encoder):
    """Just return the input as encoding trains.
    """
    def encode(self, x: Tensor) -> Tensor:
        if self.raw_input is None:
            self.raw_input = x
        return self.raw_input


class IntervalEncoder(Encoder):
    def __init__(self, interval_steps, max_timesteps):
        """In this encoder, max_timesteps is only used to make sure the spiking times.
        the total spiking times is max_timesteps // interval_steps.

        Useless in most cases.
        """
        super(IntervalEncoder, self).__init__(max_timesteps)
        self.interval_steps = interval_steps

        if interval_steps > max_timesteps:
            raise ValueError(
                f"interval_steps ({self.interval_steps}) should not be larger than max_timesteps ({self.max_timesteps})."
            )

    def encode(self, x: Tensor) -> Tensor:
        if self.raw_input is None:
            self.index = 0
            self.raw_input = x
        if self.index == 0:
            output = torch.ones_like(self.raw_input)
        else:
            output = torch.zeros_like(self.raw_input)
        self.index += 1
        self.index %= self.interval_steps

        return output


class LatencyEncoder(Encoder):
    def __init__(self, encoder_type: str = "linear", **kwargs):
        super().__init__(**kwargs)

        if encoder_type not in ["linear", "log"]:
            raise ValueError("Unknown encoder type")

        self.encoder_type = encoder_type
        if self.encoder_type == "log":
            self.alpha = math.exp(self.max_timesteps - 1) - 1

    def encode(self, x: Tensor) -> Tensor:
        if self.raw_input is None:
            if self.encoder_type == "log":
                spike_time = (self.max_timesteps - 1 -
                              torch.log(self.alpha * x + 1)).round().long()
            elif self.encoder_type == "linear":
                spike_time = ((self.max_timesteps - 1) *
                              (1 - x)).round().long()

            self.raw_input = F.one_hot(spike_time,
                                       num_classes=self.max_timesteps).bool()

            self.index = 0
        output = self.raw_input[..., self.index]
        self.index += 1
        self.index %= self.max_timesteps

        return output


class PoissonEncoder(Encoder):
    """Widely used in spiking neuronal network.
    """
    def encode(self, x: Tensor) -> Tensor:
        if self.raw_input is None:
            self.raw_input = x
        return torch.rand_like(self.raw_input).le(self.raw_input)
