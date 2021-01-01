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
        timesteps (int): the length of spiking trains.

    .. note:: 
        Only take effect under spiking mode. In non-spiking mode, it returns input directly.
    """
    def __init__(self, timesteps: int = 1):
        super().__init__()

        assert timesteps >= 1

        self.timesteps = timesteps

        self.register_hidden_state("raw_input_hid", None)

    def _reset(self, set_to_none: bool = False) -> None:
        """Sets to None.
        """
        for name, _ in self.named_hidden_states():
            self.__setattr__(name, None)

    def non_spiking_forward(self, x: Tensor) -> Tensor:
        return x


class RateEncoder(Encoder):
    """Just return the input as encoding trains.
    """
    def spiking_forward(self, x: Tensor) -> Tensor:
        if self.raw_input_hid is None:
            self.raw_input_hid = x
        return self.raw_input_hid.float()


class IntervalEncoder(Encoder):
    def __init__(self, interval_steps, timesteps):
        """In this encoder, timesteps is only used to make sure the spiking times.
        the total spiking times is timesteps // interval_steps.

        Useless in most cases.
        """
        super(IntervalEncoder, self).__init__(timesteps)
        self.interval_steps = interval_steps

        if interval_steps > timesteps:
            raise ValueError(
                f"interval_steps ({self.interval_steps}) should not be larger than timesteps ({self.timesteps})."
            )

    def spiking_forward(self, x: Tensor) -> Tensor:
        if self.raw_input_hid is None:
            self.index = 0
            self.raw_input_hid = x
        if self.index == 0:
            output = torch.ones_like(self.raw_input_hid)
        else:
            output = torch.zeros_like(self.raw_input_hid)
        self.index += 1
        self.index %= self.interval_steps

        return output.float()


class LatencyEncoder(Encoder):
    def __init__(self, encoder_type: str = "linear", **kwargs):
        super().__init__(**kwargs)

        if encoder_type not in ["linear", "log"]:
            raise ValueError("Unknown encoder type")

        self.encoder_type = encoder_type
        if self.encoder_type == "log":
            self.alpha = math.exp(self.timesteps - 1) - 1

    def spiking_forward(self, x: Tensor) -> Tensor:
        if self.raw_input_hid is None:
            if self.encoder_type == "log":
                spike_time = (self.timesteps - 1 -
                              torch.log(self.alpha * x + 1)).round().long()
            elif self.encoder_type == "linear":
                spike_time = ((self.timesteps - 1) *
                              (1 - x)).round().long()

            self.raw_input_hid = F.one_hot(
                spike_time, num_classes=self.timesteps).bool()

            self.index = 0
        output = self.raw_input_hid[..., self.index]
        self.index += 1
        self.index %= self.timesteps

        return output.float()


class PoissonEncoder(Encoder):
    """Widely used in spiking neuronal network.
    """
    def spiking_forward(self, x: Tensor) -> Tensor:
        if self.raw_input_hid is None:
            self.raw_input_hid = x
        return torch.rand_like(self.raw_input_hid).le(
            self.raw_input_hid).float()
