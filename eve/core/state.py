from collections import OrderedDict, namedtuple
from inspect import signature
from typing import Callable, List

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from copy import deepcopy
from eve.core.eve import Eve
# pylint: disable=not-callable, no-member


class State(object):
    """Records various state objects.

    Args:
        module: the module contains convolution layer or linear layer.
            NOTE: if the module contains multi-conv or linear layers, only the 
            last layer will take into account.

    .. note::

        We will apply a empirely normalization method for implemented state.
        You shoud reset this class if you want to begin a new record via :meth:`State.reset()`.
    """
    global_idx = 0.0
    global_feat_out = 0.0
    global_feat_in = 0.0
    global_param_num = 0.0
    global_stride = 0.0
    global_kernel_size = 0.0

    def __init__(self, module: Eve):
        super(State, self).__init__()

        # store it if some special state needed.
        self.module = module

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

        # update global values
        State.global_idx += 1
        self.idx = deepcopy(State.global_idx)

        State.global_feat_out += self.neurons
        State.global_feat_in += self.filter_module.weight.shape[1]
        State.global_param_num += self.filter_module.weight.numel()
        State.global_stride += self.filter_module.stride[0]
        State.global_kernel_size += self.filter_module.kernel_size[0]

        self.__fn__ = OrderedDict()

    def _align_dims(self, x):
        if isinstance(self.filter_module, nn.Linear):
            return x.view(1, 1, -1)
        elif isinstance(self.filter_module, nn.Conv2d):
            return x.view(1, -1, 1, 1)
        else:
            raise TypeError(f"{self.filter_module} not defined.")

    @staticmethod
    def reset():
        State.global_idx = 0.0
        State.global_feat_out = 0.0
        State.global_feat_in = 0.0
        State.global_param_num = 0.0
        State.global_stride = 0.0
        State.global_kernel_size = 0.0

    def state_list(self):
        return list(self.__fn__.keys())

    def register_state_fn(self, name: str, fn: Callable = None) -> None:
        """ Registers a function to calcute specified state.

        Args:
            name: the name of states. We obey the named relus that, if fn
                returns state for each neuron, we will add a `_neuron_wise`
                postfix in the name, otherwise, the fn should return a 
                single value.
            fn: the function to calcute the state.

        fn must have the following signature:
            >>> def fn(state, input=None, output=None):
            >>>     ...
            >>>     return state

        .. note::

            If fn is ``None``, we will try to register a build-in method with
            the same name.
        """
        assert name.endswith(
            "_fn"), f"fn name should be ended with `_fn`, but got {name}"
        if fn is None:
            fn = self.__getattribute__(name)

        def foo(cls, input=None, output=None):
            pass

        foo_sig = signature(foo)
        fn_sig = signature(fn)

        assert foo_sig == fn_sig, f"the signature of fn must be {foo_sig}, got {fn_sig}"

        self.__fn__[name] = fn

    @th.no_grad()
    def gather_states(self, input: Tensor = None, output: Tensor = None) -> namedtuple:
        """Run over all states functions and stack all states along the last dim.
        """
        state = namedtuple("state", self.state_list())

        states = state(*[fn(self, input, output)
                         for fn in self.__fn__.values()])
        return th.stack(states, dim=-1)

    @property
    def device(self):
        return self.filter_module.weight.device

    @property
    def neurons(self):
        return len(self.filter_module.weight)

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
                gamma = th.ones(self.norm_module.num_features,
                                device=var.device)
            A = gamma.div(th.sqrt(var + self.norm_module.eps))
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
                gamma = th.ones(self.norm_module.num_features,
                                device=var.device)
                beta = th.ones(self.norm_module.num_features,
                               device=var.device)
            A = gamma.div(th.sqrt(var + self.norm_module.eps))
            if self.filter_module.bias is not None:
                bias_fold = (self.filter_module.bias - mu) * A + beta
            else:
                bias_fold = (-mu) * A + beta
            return bias_fold
        else:
            return self.filter_module.bias

    @staticmethod
    def k_fn(cls, input=None, output=None):
        return th.tensor([cls.idx / State.global_idx] * cls.neurons, device=cls.device)

    @staticmethod
    def feat_out_fn(cls, input=None, output=None):
        return th.tensor([cls.neurons / State.global_feat_out]*cls.neurons, device=cls.device)

    @staticmethod
    def feat_in_fn(cls, input=None, output=None):
        return th.tensor([cls.filter_module.weight.shape[1] / State.global_feat_in] * cls.neurons, device=cls.device)

    @staticmethod
    def param_num_fn(cls, input=None, output=None):
        return th.tensor([cls.filter_module.weight.numel() / State.global_param_num] * cls.neurons, device=cls.device)

    @staticmethod
    def stride_fn(cls, input=None, output=None):
        if cls.filter_type == nn.Linear:
            stride = 0.0
        else:
            stride = cls.filter_module.stride[0]
        return th.tensor([stride / State.global_stride] * cls.neurons, device=cls.device)

    @staticmethod
    def kernel_size_fn(cls, input=None, output=None):
        if cls.filter_type == nn.Linear:
            kernel_size = 0.0
        else:
            kernel_size = cls.filter_module.kernel_size[0]
        return th.tensor([kernel_size * State.global_kernel_size] * cls.neurons, device=cls.device)

    @staticmethod
    def l1_norm_fn(cls, input=None, output=None):
        if cls.filter_type == nn.Linear:
            # filter shape is [featout, featin]
            weight = cls.filter_weight
            return weight.abs().sum(dim=1)  # [featout]
        elif cls.filter_type == nn.Conv2d:
            # filter shape is [outchannel, inchannel, w, h]
            weight = cls.filter_weight
            return weight.abs().sum(dim=(1, 2, 3))  # [featout]
        else:
            raise NotImplementedError

    @staticmethod
    def kl_div_fn(cls, input=None, output=None):
        """input is a list containing all input non-position value.
        """
        dim = {nn.Conv2d: [0, 2, 3], nn.Linear: [0, 1]}[cls.filter_type]
        return F.kl_div(input[0], output,
                        reduction="none").mean(dim, keepdim=False)

    @staticmethod
    def fire_rate_fn(cls, input=None, output=None):
        dim = {nn.Conv2d: [0, 2, 3], nn.Linear: [0, 1]}[cls.filter_type]
        return (output > 0.0).float().mean(dim, keepdim=False)
