"""Contains the Pruning algorithm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List
from eve.cores.utils import _align_dims
import copy


# pylint: disable=no-member
# pylint: disable=not-callable
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

    layer_index = 0  # global index of layer
    max_feat_out = 0
    max_feat_in = 0
    max_kernel_size = 0
    max_stride = 0
    max_param_num = 0

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

        # record self layer index
        self.layer_index = copy.deepcopy(State.layer_index)

        # update global layer index
        State.layer_index += 1
        State.max_feat_out = max(State.max_feat_out,
                                 self.filter_module.weight.shape[0])
        State.max_feat_in = max(State.max_feat_in,
                                self.filter_module.weight.shape[1])
        State.max_kernel_size = max(
            State.max_kernel_size, 0 if self.filter_type == nn.Linear else
            self.filter_module.kernel_size[0])
        State.max_stride = max(
            State.max_stride, 0
            if self.filter_type == nn.Linear else self.filter_module.stride[0])
        State.max_param_num = max(State.max_param_num,
                                  self.filter_module.weight.numel())

    @staticmethod
    def reset_global_state():
        """Resets all global states.

        This function will reset all global states of :class:`State`.
        """
        State.layer_index = 0  # global index of layer
        State.max_feat_out = 0
        State.max_feat_in = 0
        State.max_kernel_size = 0
        State.max_stride = 0
        State.max_param_num = 0

    @property
    def neurons(self):
        """return the number of neurons"""
        return self.filter_module.weight.shape[0]

    @property
    def k(self):
        k = torch.tensor([self.layer_index / (self.layer_index + 1e-8)] *
                         self.neurons,
                         device=self.device)
        return k

    @property
    def device(self):
        return self.filter_module.weight.device

    @property
    def feat_out(self):
        feat_out = torch.tensor([self.neurons / (State.max_feat_out + 1e-8)] *
                                self.neurons,
                                device=self.device)
        return feat_out

    @property
    def feat_in(self):
        feat_in = torch.tensor(
            [self.filter_module.weight.shape[1] /
             (State.max_feat_in + 1e-8)] * self.neurons,
            device=self.device)
        return feat_in

    @property
    def param_num(self):
        param_num = torch.tensor(
            [self.filter_module.weight.numel() /
             (State.max_param_num + 1e-8)] * self.neurons,
            device=self.device)
        return param_num

    @property
    def stride(self):
        if self.filter_type == nn.Linear:
            stride = 0.0
        else:
            stride = self.filter_module.stride[0]
        stride = torch.tensor([stride / (State.max_stride + 1e-8)] *
                              self.neurons,
                              device=self.device)
        return stride

    @property
    def kernel_size(self):
        if self.filter_type == nn.Linear:
            kernel_size = 0.0
        else:
            kernel_size = self.filter_module.kernel_size[0]
        kernel_size = torch.tensor(
            [kernel_size / (State.max_kernel_size + 1e-8)] * self.neurons,
            device=self.device)
        return kernel_size

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
            weight = self.filter_weight
            return weight.abs().sum(dim=1)  # [featout]
        elif self.filter_type == nn.Conv2d:
            # filter shape is [outchannel, inchannel, w, h]
            weight = self.filter_weight
            return weight.abs().sum(dim=(1, 2, 3))  # [featout]
        else:
            raise NotImplementedError

    @property
    def kl_div(self):
        return self._kl_div

    def kl_div_fn(self, x: List[Tensor], quan: Tensor):
        dim = {nn.Conv2d: [0, 2, 3], nn.Linear: [0, 1]}[self.filter_type]
        self._kl_div = F.kl_div(x[0], quan,
                                reduction="none").mean(dim, keepdim=False)

    @property
    def fire_rate(self):
        return self._fire_rate

    def fire_rate_fn(self, x: Tensor, fire: Tensor):
        dim = {nn.Conv2d: [0, 2, 3], nn.Linear: [0, 1]}[self.filter_type]
        self._fire_rate = (fire > 0.0).float().mean(dim, keepdim=False)
