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

import math
from abc import abstractmethod

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from torch.nn import Parameter

from eve.core.eve import Eve
from eve.core.quan import Quantizer
from eve.core.state import State

# pylint: disable=no-member
# pylint: disable=access-member-before-definition

r"""
                                     .__  .__ 
  _______  __ ____             _____ |  | |__|
_/ __ \  \/ // __ \   ______  /     \|  | |  |
\  ___/\   /\  ___/  /_____/ |  Y Y  \  |_|  |
 \___  >\_/  \___  >         |__|_|  /____/__|
     \/          \/                \/       

Quan
"""


class QuanModule(Eve):
    def assign_quantizer(self, quantizer: Quantizer):
        self.quantizer = quantizer


class QuanConv2d(nn.Conv2d, QuanModule):
    def forward(self, input):
        quan_weight = self.quantizer(self.weight)

        return F.conv2d(input, quan_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class QuanConvTranspose2d(nn.ConvTranspose2d, QuanModule):
    def forward(self, input):
        quan_weight = self.quantizer(self.weight)
        return F.conv_transpose2d(input, quan_weight, self.bias, self.stride,
                                  self.padding, self.output_padding,
                                  self.groups, self.dilation)


class QuanBNFuseConv2d(nn.Conv2d, QuanModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode="zeros",
                 eps=1e-5,
                 momentum=0.01,
                 **kwargs):
        """
        Args:
            kwargs: the argument used to define quantizer, except state. 
        """
        super(QuanBNFuseConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias, padding_mode)
        self.eps = eps
        self.momentum = momentum
        self.gamma = Parameter(th.Tensor(out_channels))
        self.beta = Parameter(th.Tensor(out_channels))
        self.register_buffer("running_mean", th.zeros(out_channels))
        self.register_buffer("running_var", th.ones(out_channels))
        nn.init.uniform_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, input):
        def reshape_to_activation(input):
            return input.reshape(1, -1, 1, 1)

        def reshape_to_weight(input):
            return input.reshape(-1, 1, 1, 1)

        def reshape_to_bias(input):
            return input.reshape(-1)

        if self.training:
            output = F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

            dims = [dim for dim in range(4) if dim != 1]
            batch_mean = th.mean(output, dim=dims)
            batch_var = th.var(output, dim=dims)
            with th.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(
                    batch_mean * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(
                    batch_var * self.momentum)

            if self.bias is not None:
                bias = reshape_to_bias(
                    self.beta + (self.bias - batch_mean) *
                    (self.gamma / th.sqrt(batch_var + self.eps)))
            else:
                bias = reshape_to_bias(
                    self.beta - batch_mean *
                    (self.gamma / th.sqrt(batch_var + self.eps)))
            weight = self.weight * reshape_to_weight(
                self.gamma / th.sqrt(self.running_var + self.eps))

        else:
            if self.bias is not None:
                bias = reshape_to_bias(
                    self.beta + (self.bias - self.running_mean) *
                    (self.gamma / th.sqrt(self.running_var + self.eps)))
            else:
                bias = reshape_to_bias(
                    self.beta - self.running_mean *
                    (self.gamma / th.sqrt(self.running_var + self.eps)))
            weight = self.weight * reshape_to_weight(
                self.gamma / th.sqrt(self.running_var + self.eps))

        quan_weight = self.quantizer(weight)

        if self.training:
            output = F.conv2d(input, quan_weight, self.bias, self.stride,
                              self.padding, self.dilation,
                              self.groups)  # no bias
            # running ——> batch
            output *= reshape_to_activation(
                th.sqrt(self.running_var + self.eps) /
                th.sqrt(batch_var + self.eps))
            output += reshape_to_activation(bias)
        else:
            output = F.conv2d(input, quan_weight, bias, self.stride,
                              self.padding, self.dilation,
                              self.groups)  # add bias
        return output


class QuanLinear(nn.Linear, QuanModule):
    def forward(self, input):
        quan_weight = self.quantizer(self.weight)
        return F.linear(input, quan_weight, self.bias)


r"""
                                     .__  .__ 
  _______  __ ____             _____ |  | |__|
_/ __ \  \/ // __ \   ______  /     \|  | |  |
\  ___/\   /\  ___/  /_____/ |  Y Y  \  |_|  |
 \___  >\_/  \___  >         |__|_|  /____/__|
     \/          \/                \/       

Spiking
"""


class Dropout(Eve):
    """Dropout layer. 

    The dropout mask will not be changed among a spike trains under spiking mode.
    """

    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("p must be in (0, 1)")

        self.register_eve_buffer("mask_eve", None)
        self.p = p

    def _reset(self, set_to_none: bool = False) -> None:
        """Resets the :attr:`mask` to ``None``.

        Args:
            set_to_none (bool): unused.
        """
        super()._reset(set_to_none=True)

    def drop_mask(self, x: Tensor):
        """Generates a new drop mask.
        """
        if self.mask_eve is None:
            self.mask_eve = F.dropout(th.ones_like(x),
                                      self.p,
                                      training=self.training)

    def spiking_forward(self, x: Tensor) -> Tensor:
        self.drop_mask(x)
        return x * self.mask_eve

    def non_spiking_forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, training=self.training)

    def forward(self, x: Tensor):
        if self.spiking:
            return self.spiking_forward(x)
        else:
            return self.non_spiking_forward(x)


class Dropout2d(Dropout):
    def __init__(self, p=0.2):
        super().__init__(p)

    def drop_mask(self, x: Tensor):
        if self.mask_eve is None:
            self.mask_eve = F.dropout2d(th.ones_like(x),
                                        self.p,
                                        training=self.training)


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

        self.register_eve_buffer("raw_input_eve", None)

    def _reset(self, set_to_none: bool = False) -> None:
        """Sets to None.
        """
        for name, _ in self.named_eve_buffers():
            self.__setattr__(name, None)

    def non_spiking_forward(self, x: Tensor) -> Tensor:
        return x

    @abstractmethod
    def spiking_forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        if self.spiking:
            return self.spiking_forward(x)
        else:
            return self.non_spiking_forward(x)


class RateEncoder(Encoder):
    """Just return the input as encoding trains.
    """

    def spiking_forward(self, x: Tensor) -> Tensor:
        if self.raw_input_eve is None:
            self.raw_input_eve = x
        return self.raw_input_eve.float()


class RateOnceEncoder(Encoder):
    """Just return the input as encoding trains at the first time.
    """

    def spiking_forward(self, x: Tensor) -> Tensor:
        if self.raw_input_eve is None:
            self.raw_input_eve = x
            return self.raw_input_eve.float()
        else:
            return th.zeros_like(self.raw_input_eve).float()


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
        if self.raw_input_eve is None:
            self.index = 0
            self.raw_input_eve = x
        if self.index == 0:
            output = th.ones_like(self.raw_input_eve)
        else:
            output = th.zeros_like(self.raw_input_eve)
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
        if self.raw_input_eve is None:
            if self.encoder_type == "log":
                spike_time = (self.timesteps - 1 -
                              th.log(self.alpha * x + 1)).round().long()
            elif self.encoder_type == "linear":
                spike_time = ((self.timesteps - 1) * (1 - x)).round().long()

            self.raw_input_eve = F.one_hot(spike_time,
                                           num_classes=self.timesteps).bool()

            self.index = 0
        output = self.raw_input_eve[..., self.index]
        self.index += 1
        self.index %= self.timesteps

        return output.float()


class PoissonEncoder(Encoder):
    """Widely used in spiking neuronal network.
    """

    def spiking_forward(self, x: Tensor) -> Tensor:
        if self.raw_input_eve is None:
            self.raw_input_eve = x
        return th.rand_like(self.raw_input_eve).le(self.raw_input_eve).float()


r"""
                                     .__  .__ 
  _______  __ ____             _____ |  | |__|
_/ __ \  \/ // __ \   ______  /     \|  | |  |
\  ___/\   /\  ___/  /_____/ |  Y Y  \  |_|  |
 \___  >\_/  \___  >         |__|_|  /____/__|
     \/          \/                \/       

Pruning  
"""


class fbs_penalty(Function):
    @staticmethod
    def forward(ctx, x: th.Tensor, penalty: float = 1e-8):
        ctx.save_for_backward(x)
        ctx.others = (penalty,)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        penalty, = ctx.others

        penalty_grad = penalty * x.sum()

        return grad_output + penalty_grad, None


class FBSConv2d(nn.Conv2d):
    """The implementation of Dynamic Channel Pruning: Feature Boosting and Suppression.

    Args:
        pruning_ratio: how may channels will be kept.
        saliency_penalty: the penalty of saliency gradient.

    """

    def __init__(self, *args, pruning_ratio: float = 0.5, saliency_penalty: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)

        # define saliency module
        self.saliency_predictor = nn.Linear(
            in_features=self.in_channels, out_features=self.out_channels, bias=True)
        self.pruning_ratio = pruning_ratio
        self.saliency_penalty = saliency_penalty

    def forward(self, x: th.Tensor) -> th.Tensor:
        # subsample input features [B, C, H, W] to [B, C] via l1 norm
        subsample_x = th.abs(x).mean(dim=(2, 3))
        saliency = th.abs(self.saliency_predictor(subsample_x))

        threshold = th.topk(saliency, k=int(
            self.out_channels * self.pruning_ratio), dim=1)[0][:, -1]

        saliency = th.where(saliency < threshold.view(-1, 1),
                            th.zeros_like(saliency), saliency)

        # add gradient penalty to gradient
        saliency = fbs_penalty.apply(saliency, self.saliency_penalty)

        return saliency.unsqueeze(dim=-1).unsqueeze(dim=-1) * super().forward(x)


class FBSLinear(nn.Linear):
    def __init__(self, *args, pruning_ratio: float = 0.5, saliency_penalty: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)

        self.saliency_predictor = nn.Linear(
            in_features=self.in_features, out_features=self.out_features, bias=True)

        self.pruning_ratio = pruning_ratio
        self.saliency_penalty = saliency_penalty

    def forward(self, x: th.Tensor) -> th.Tensor:
        # in `Eve`, the input to linear layer is [batch_size, numel, feature]
        subsample_x = th.abs(x).mean(dim=(1,))
        saliency = th.abs(self.saliency_predictor(subsample_x))

        threshold = th.topk(saliency, k=int(
            self.out_features * self.pruning_ratio), dim=1)[0][:, -1]

        saliency = th.where(saliency < threshold.view(-1, 1),
                            th.zeros_like(saliency), saliency)

        saliency = fbs_penalty.apply(saliency, self.saliency_penalty)

        return saliency.unsqueeze(dim=1) * super().forward(x)
