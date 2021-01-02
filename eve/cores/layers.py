import torch
import torch.nn as nn
import torch.nn.functional as F
from eve.cores.eve import Eve
from torch import Tensor


# pylint: disable=access-member-before-definition
# pylint: disable=no-member
class Dropout(Eve):
    """Dropout layer. 

    The dropout mask will not be changed among a spike trains under spiking mode.
    """
    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("p must be in (0, 1)")

        self.register_hidden_state("mask_hid", None)
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
        if self.mask_hid is None:
            self.mask_hid = F.dropout(torch.ones_like(x),
                                  self.p,
                                  training=self.training)

    def spiking_forward(self, x: Tensor) -> Tensor:
        self.drop_mask(x)
        return x * self.mask_hid

    def non_spiking_forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, training=self.training)


class Dropout2d(Dropout):
    def __init__(self, p=0.2):
        super().__init__(p)

    def drop_mask(self, x: Tensor):
        if self.mask_hid is None:
            self.mask_hid = F.dropout2d(torch.ones_like(x),
                                    self.p,
                                    training=self.training)


class Cell(Eve):
    """A Cell contains a convolution/linear layer with batch normalization.

    In cell, the behavior is the same as nn.Sequential while training, 
    but will automatically fold the parameter of batch normalization layer 
    into convolution or linear layer.

    Linear layer should be converted to Conv1d.

    Args: 
        filter_moduel (Module): currently, only :class:`nn.Conv2d` is supported.
        norm_module (Module): currently, only :class:`nn.BatchNorm2d` is supported.
        act_module (Module): any kind of activation functions.
    
    """
    def __init__(self, filter_module: nn.Module, norm_module: nn.Module,
                 act_module: nn.Module):
        super().__init__()
        if not isinstance(filter_module, nn.Conv2d):
            raise TypeError(
                f"filter moduel must be nn.Conv2d, but got {torch.typename(filter_module)}"
            )
        if not isinstance(norm_module, nn.BatchNorm2d):
            raise TypeError(
                f"norm module must be nn.BatchNorm2d, but got {torch.typename(norm_module)}"
            )

        self.filter_module = filter_module
        self.norm_module = norm_module
        self.act_module = act_module

    def forward(self, x: Tensor):
        if self.training:
            return self.act_module(self.norm_module(self.filter_module(x)))
        else:
            mu = self.norm_module.running_mean
            var = self.norm_module.running_var

            if self.norm_module.affine:
                gamma = self.norm_module.weight
                beta = self.norm_module.bias
            else:
                gamma = torch.ones(self.norm_module.num_features,
                                   device=var.device)
                beta = torch.ones(self.norm_module.num_features,
                                  device=var.device)
            A = gamma.div(torch.sqrt(var + self.norm_module.eps))
            A_expand = A.expand_as(self.filter_module.weight.transpose(
                0, -1)).transpose(0, -1)

            weight_fold = self.filter_module.weight * A_expand
            if self.filter_module.bias is not None:
                bias_fold = (self.filter_module.bias - mu) * A + beta
            else:
                bias_fold = (-mu) * A + beta

            out = F.conv2d(x, weight_fold, bias_fold,
                           self.filter_module.stride,
                           self.filter_module.padding,
                           self.filter_module.dilation,
                           self.filter_module.groups)
            return self.act_module(out)