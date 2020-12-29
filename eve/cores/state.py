"""Contains the Pruning algorithm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# pylint: disable=no-member
class State(object):
    """The class contains various states computing methods.

    We can classify different states into two categories, inner and outter.
    The inner state is computed without the necessary of other input data, 
    and will be decorated with @property.
    The outter one is computed depend on the input data, and a callable
    function is designed to implement it.

    Args:
        module (nn.Module): the module contains conv2d or linear layers.
        If several linear or conv2d layer is contained, we only take the 
        last layer into account. If a BatchNorm layer is followed, 
        we will fold the weight of bn into the weight of linear or filter.

    .. note::

        Do not succeed :class:`nn.Module` or :class:`cores.Eve`, any parameter 
        of this class is a copy of other existing module and only used for 
        computing related object state. The weight of input module cannot be 
        modified in this layer.
        All the state is with shape [neurons,]
    """
    def __init__(self, module: nn.Module):
        super(State, self).__init__()

        # Fetch the last convolution layer or linear layer
        filter_module = None
        norm_module = None
        last_m = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                filter_module = m
            elif isinstance(last_m, nn.Conv2d) and isinstance(
                    m, nn.BatchNorm2d):
                norm_module = m
            elif isinstance(last_m, nn.Linear) and isinstance(
                    m, nn.BatchNorm1d):
                norm_module = m
            last_m = m
        assert filter_module, "{} does not contains any conv2d or linear layer.".format(
            module)

        self.filter_module = filter_module
        self.norm_module = norm_module

    @property
    def neurons(self):
        """return the number of neurons"""
        return self.filter_module.weight.shape[0]

    @property
    def filter_type(self):
        return type(self.filter_module)

    @property
    def filter_weight(self):
        # if self.norm_module is not None, fold the weight to filter_module
        if self.norm_module is not None:
            var = self.norm_module.running_var

            if self.norm_module.affine:
                gamma = self.norm_module.weight
            else:
                gamma = torch.ones(self.norm_module.num_features,
                                   device=var.device)
            A = gamma.div(torch.sqrt(var + self.norm_module.eps))
            A_expand = A.expand_as(self.filter_module.weight.transpose(
                0, -1)).transpose(0, -1)
            weight_fold = self.filter_module.weight * A_expand
            return weight_fold
        else:
            return self.filter_module.weight

    @property
    def filter_bias(self):
        # if self.norm_module is not None, fold the weight to filter_module
        if self.norm_module is not None and self.filter_module.bias is not None:
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
            if self.filter_module.bias is not None:
                bias_fold = (self.filter_module.bias - mu) * A + beta
            else:
                bias_fold = (-mu) * A + beta
            return bias_fold
        else:
            return self.filter_module.bias

    @property
    def l1_norm(self):
        if self.filter_type == nn.Linear:
            # filter shape is [featout, featin]
            weight = self.filter_weight()
            return weight.abs().mean(dim=1)  # [featout]
        elif self.filter_type == nn.Conv2d:
            # filter shape is [outchannel, inchannel, w, h]
            weight = self.filter_weight()
            return weight.abs().mean(dim=(1, 2, 3))  # [featout]
        else:
            raise NotImplementedError

    def kl_div(self, x: Tensor, quan: Tensor):
        dim = {nn.Conv2d: [0, 2, 3], nn.Linear: [0, 1]}[self.filter_type]
        return F.kl_div(x, quan, reduction="none").mean(dim, keepdim=False)

    def fire_rate(self, x: Tensor, fire: Tensor):
        dim = {nn.Conv2d: [0, 2, 3], nn.Linear: [0, 1]}[self.filter_type]
        return (fire > 0.0).float().mean(dim, keepdim=False)
