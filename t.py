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

from collections import OrderedDict, namedtuple
from copy import deepcopy
from inspect import signature
from typing import Callable, List

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import abstractmethod
from eve.core.eve import Eve
from typing import Any
# pylint: disable=not-callable, no-member


class Statistic(object):
    """The basic class to trace various state, then normalize input value.
    """
    def __init__(self):
        """initialize  statistic."""

    @abstractmethod
    def reset(self):
        """Resets statistic.
        """

    @abstractmethod
    def on_init(self, *args, **kwargs):
        """This function will be called at `State.__init__()`.

        It is useful to count the static infomation which is only calculated once.
        """

    @abstractmethod
    def on_step(self, *args, **kwargs):
        """This function will be called at `State.gather_states()`.

        It is useful to count the dynamic information which is changed all the time.
        """

    @abstractmethod
    def normalize(self, *args, **kwargs):
        """Apply normalization based on statistic infomation.
        """


class StaticStatistic(Statistic):
    def __init__(self, init_value: float = 0.0):
        super().__init__()
        self.value = init_value
    
    def reset(self):
        pass

    def on_init(self, *args, **kwargs):
        self.value += args[0]
    
    def on_step(self, *args, **kwargs):
        pass

    def normalize(self, *args, **kwargs):
        return args[0] / (self.value + 1e-8)


class cnt(StaticStatistic):
    pass


class feat_out(StaticStatistic):
    pass


class feat_in(StaticStatistic):
    pass


class param_num(StaticStatistic):
    pass


class stride(StaticStatistic):
    pass


class kernel_size(StaticStatistic):
    pass


# register global statistic here
__global_statistics__ = {
    "feat_out": feat_out,
    "feat_in": feat_in,
    "cnt": cnt,
    "param_num": param_num
    "stride": stride,
}


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
    _global_statistic = OrderedDict()

    # used to align tensor to fit the input data
    align_dims: List[int]

    # used to reduce tensor to fit neuron wise
    reduce_dims: List[int]

    # neuron wise flag, may be ignored in same cases.
    neuron_wise: bool

    def __init__(self, 
        module: Eve, 
        align_dims: List[int] = None, 
        reduce_dims: List[int] = None):
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

        # set default neuron wise, use `self.set_neuron_wise()` to reset.
        # we do not directly deliver this flag as argument to `__init__`
        # for that not all state need this signal.
        self.neuron_wise = False
        self.__fn__ = OrderedDict()

    def set_neuron_wise(self, mode=True):
        self.neuron_wise = mode

    @staticmethod
    def reset():
        State._global_statistic = OrderedDict()

    def state_list(self):
        return list(self.__fn__.keys())

    def register_state_fn(self, name: str, fn: Callable = None, statistic: Statistic = None) -> None:
        """ Registers a function to calculate specified state.

        Args:
            name: the name of states. We obey the named relus.
            fn: the function to calculate the state.
            statistic: if not None, the state value calculated by 
                fn will be normalized.

        fn must have the following signature:
            >>> def fn(state, input=None, output=None):
            >>>     ...
            >>>     return output

        .. note::

            If fn is ``None``, we try to register a build-in method with
            the same name, and raise Error if failed. If statistic is ``None``,
            we try to register a build-in method with the same name, and DO NOT
            raise Error if failed.
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

        assert name not in self.__fn__, f"fn {name} already registered."
        self.__fn__[name] = fn

        if statistics is not None:
            State._global_statistic[name] = statistics
        else:
            try:
                State._global_statistic[name] = __global_statistics__[name]()
            except KeyError:
                pass

    @th.no_grad()
    def gather_states(self,
                      input: Tensor = None,
                      output: Tensor = None) -> namedtuple:
        """Run over all states functions and stack all states along the last dim.
        """
        if len(self.__fn__) == 0:
            return None
        states = [fn(self, input, output) for fn in self.__fn__.values()]
        states = th.stack(states, dim=-1)

        return states  # [neurons, obs]

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
        return th.tensor([cls.idx / State.global_idx] *
                         (cls.neurons if cls.neuron_wise else 1),
                         device=cls.device)

    @staticmethod
    def feat_out_fn(cls, input=None, output=None):
        return th.tensor([cls.neurons / State.global_feat_out] *
                         (cls.neurons if cls.neuron_wise else 1),
                         device=cls.device)

    @staticmethod
    def feat_in_fn(cls, input=None, output=None):
        return th.tensor(
            [cls.filter_module.weight.shape[1] / State.global_feat_in] *
            (cls.neurons if cls.neuron_wise else 1),
            device=cls.device)

    @staticmethod
    def param_num_fn(cls, input=None, output=None):
        return th.tensor(
            [cls.filter_module.weight.numel() / State.global_param_num] *
            (cls.neurons if cls.neuron_wise else 1),
            device=cls.device)

    @staticmethod
    def stride_fn(cls, input=None, output=None):
        if cls.filter_type == nn.Linear:
            stride = 0.0
        else:
            stride = cls.filter_module.stride[0]
        return th.tensor([stride / State.global_stride] *
                         (cls.neurons if cls.neuron_wise else 1),
                         device=cls.device)

    @staticmethod
    def kernel_size_fn(cls, input=None, output=None):
        if cls.filter_type == nn.Linear:
            kernel_size = 0.0
        else:
            kernel_size = cls.filter_module.kernel_size[0]
        return th.tensor([kernel_size / State.global_kernel_size] *
                         (cls.neurons if cls.neuron_wise else 1),
                         device=cls.device)

    @staticmethod
    def l1_norm_fn(cls, input=None, output=None):
        if cls.filter_type == nn.Linear:
            # filter shape is [featout, featin]
            weight = cls.filter_weight
            l1_norm = weight.abs().mean(dim=1)  # [featout]
        elif cls.filter_type == nn.Conv2d:
            # filter shape is [outchannel, inchannel, w, h]
            weight = cls.filter_weight
            l1_norm = weight.abs().mean(dim=(1, 2, 3))  # [featout]
        else:
            raise NotImplementedError

        return l1_norm if cls.neuron_wise else l1_norm.mean(0, keepdim=True)

    @staticmethod
    def kl_div_fn(cls, input=None, output=None):
        """input is a list containing all input non-position value.
        """
        dim = {nn.Conv2d: [0, 2, 3], nn.Linear: [0, 1]}[cls.filter_type]
        kl_div = F.kl_div(input[0], output,
                          reduction="none").mean(dim, keepdim=False)
        return kl_div if cls.neuron_wise else kl_div.mean(0, keepdim=True)

    @staticmethod
    def fire_rate_fn(cls, input=None, output=None):
        dim = {nn.Conv2d: [0, 2, 3], nn.Linear: [0, 1]}[cls.filter_type]
        fire_rate = (output > 0.0).float().mean(dim, keepdim=False)

        return fire_rate if cls.neuron_wise else fire_rate.mean(0,
                                                                keepdim=True)
