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
from collections import OrderedDict, namedtuple
from copy import deepcopy
from inspect import signature
from typing import Any, Callable, List, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.conv import _ConvNd
from eve.core.eve import Eve


# pylint: disable=not-callable, no-member
class Statistic(object):
    """Basic class to trace differetn variable state.
    """

    def reset(self, set_to_none: bool = False):
        """reset statistic variables.
        """

    def on_init(self, state: "State"):
        """ This function will be called at `State.__init__()`
        """

    def on_step(self, state: "State", input: Tensor, output: Tensor):
        """This function will be called at `State.gather_states()`
        """


class State(object):
    """A special auxiliary module to manage different data and parameter state.

    Args:
        module: the module which used to extra state.
        apply_on: data or param.
        align_dims: use to align a variabl with single dimension to as many
            dimensions as state. if None, we will infer it from filter type.
        param_reduce_dims: use to reduce param to singgle dimension.
            if None, we will infer it from filter type.
        data_reduce_dims: use to reduce data to single dimension.
            if None, we will infer it from filter type.

    Examples:

        >>> tensor = th.randn(4)
        >>> tensor.reshape(align_dims)

        >>> tensor = th.randn(3, 4, 5)
        >>> tensor.mean(data_reduce_dims)
    """

    __global_statistics = []  # type: List[Statistic], class used to instantiate

    local_statistics = []  # type: List[Statistic], class already initialized.

    # use to align a variable with one dimension to as many dimensions as state
    align_dims: List[int]

    # to reduce param to one dimension one.
    param_reduce_dims: List[int]

    # to reduce data to one dimension one.
    data_reduce_dims: List[int]

    neuron_wise: bool

    def __init__(self,
                 module: Eve,
                 apply_on: str = "data",
                 align_dims: List[int] = None,
                 param_reduce_dims: List[int] = None,
                 data_reduce_dims: List[int] = None,
                 neuron_wise: bool = False,
                 statistic: Union[Statistic, str] = None,
                 ):
        super().__init__()

        # store the original model if same special state needed.
        self.module = module
        assert apply_on in ["data", "param"]
        self.apply_on = apply_on

        for m in reversed(list(module.modules())):
            if isinstance(m, _ConvNd):
                filter_module = m
                break
            elif isinstance(m, nn.Linear):
                filter_module = m
                break
        else:
            raise TypeError(
                f"{th.typename(module)} is not a valid type for filter module")

        self.filter_module = filter_module

        self.neuron_wise = neuron_wise

        if align_dims is None:
            if self.filter_type == nn.Linear:
                align_dims = {"data": [1, 1, -1],
                              "param": [-1, 1]}[self.apply_on]
            elif self.filter_type == nn.Conv2d:
                align_dims = {"data": [1, -1, 1, 1],
                              "param": [-1, 1, 1, 1]}[self.apply_on]
        self.align_dims = align_dims

        if param_reduce_dims is None:
            if self.filter_type == nn.Linear:
                param_reduce_dims = [1, ]
            elif self.filter_type == nn.Conv2d:
                param_reduce_dims = [1, 2, 3]
        self.param_reduce_dims = param_reduce_dims

        if data_reduce_dims is None:
            if self.filter_type == nn.Linear:
                data_reduce_dims = [0, 1, ]
            elif self.filter_type == nn.Conv2d:
                data_reduce_dims = [0, 2, 3]
        self.data_reduce_dims = data_reduce_dims

        if statistic is not None:
            for s in statistic:
                if isinstance(s, str):
                    s = __build_in_statistic__[s]
                self.local_statistics.append(s())
        for s in State.__global_statistics:
            self.local_statistics.append(s())

        # setup the init operation
        for s in self.local_statistics:
            s.on_init(self)

    @staticmethod
    def register_global_statistic(statistic: Union[Statistic, str]):
        """Register statistic to share among all states.
        """
        if isinstance(statistic, Statistic):
            State.__global_statistics.append(statistic)
        elif isinstance(statistic, str):
            State.__global_statistics.append(__build_in_statistic__[statistic])

    def set_neuron_wise(self, mode: bool = True):
        self.neuron_wise = mode

    def reset(self, set_to_none: bool = False):
        for s in self.local_statistics:
            s.reset(set_to_none=set_to_none)

    @ th.no_grad()
    def gather_states(
        self, input: Tensor, output: Tensor
    ) -> Tensor:
        """Run over all statistic functions and stack all states along the last dim.
        """
        if len(self.local_statistics) == 0:
            return None

        states = [s.on_step(self, input, output)
                  for s in self.local_statistics]
        states = th.stack(states, dim=-1)

        return states  # [x, obs]

    @ property
    def device(self):
        return self.filter_module.weight.device

    @ property
    def neurons(self):
        return len(self.filter_module.weight)

    @ property
    def filter_type(self):
        return type(self.filter_module)

    @ property
    def filter_weight(self):
        return self.filter_module.weight


class cnt(Statistic):
    __cnt = 0.0

    def on_init(self, state: State):
        self._cnt = cnt.__cnt
        cnt.__cnt += 1

    def on_step(self, state: State, input: Tensor, output: Tensor):
        return th.tensor([self._cnt / cnt.__cnt] * (state.neurons if state.neuron_wise else 1), device=state.device)


class feat_out(Statistic):
    __feat_out = 0.0

    def on_init(self, state: State):
        self._feat_out = state.neurons
        feat_out.__feat_out += self._feat_out

    def on_step(self, state: State, input: Tensor, output: Tensor):
        return th.tensor([self._feat_out / feat_out.__feat_out] * (state.neurons if state.neuron_wise else 1), device=state.device)


class feat_in(Statistic):
    __feat_in = 0.0

    def on_init(self, state: State):
        self._feat_in = state.filter_module.weight.shape[1]
        feat_in.__feat_in += self._feat_in

    def on_step(self, state: State, input: Tensor, output: Tensor):
        return th.tensor([self._feat_in / feat_in.__feat_in] * (state.neurons if state.neuron_wise else 1), device=state.device)


class param_num(Statistic):
    __param_num = 0.0

    def on_init(self, state: State):
        self._param_num = state.filter_module.numel()
        param_num.__param_num += self._param_num

    def on_step(self, state: State, input: Tensor, output: Tensor):
        return th.tensor([self._param_num / param_num.__param_num] * (state.neurons if state.neuron_wise else 1), device=state.device)


class stride(Statistic):
    __stride = 0.0

    def on_init(self, state: State):
        if state.filter_type == nn.Linear:
            self._stride = 0.0
        elif state.filter_type == _ConvNd:
            self._stride = state.filter_module.stride[0]
        else:
            raise NotImplementedError

        stride.__stride += self._stride

    def on_step(self, state: State, input: Tensor, output: Tensor):
        return th.tensor([self._stride / stride.__stride] * (state.neurons if state.neuron_wise else 1), device=state.device)


class kernel_size(Statistic):
    __kernel_size = 0.0

    def on_init(self, state: State):
        if state.filter_type == nn.Linear:
            self._kernel_size = 0
        elif state.filter_type == _ConvNd:
            self._kernel_size = state.filter_module.kernel_size[0]
        else:
            raise NotImplementedError

        kernel_size.__kernel_size += self._kernel_size

    def on_step(self, state: State, input: Tensor, output: Tensor):
        return th.tensor([self._kernel_size / kernel_size.__kernel_size] * (state.neurons if state.neuron_wise else 1), device=state.device)


class l1_norm(Statistic):
    def on_step(self, state: State, input: Tensor, output: Tensor):
        weight = state.filter_weight

        _l1_norm = weight.abs().mean(dim=state.param_reduce_dims)
        return _l1_norm if state.neuron_wise else _l1_norm.mean(0, keepdim=True)


class kl_div(Statistic):
    def on_step(self, state: State, input: Tensor, output: Tensor):
        _kl_div = F.kl_div(input[0], output, reduction="none").mean(
            state.data_reduce_dims, keepdim=False)
        return _kl_div if state.neuron_wise else _kl_div.mean(0, keepdim=True)


class fire_rate(Statistic):
    def on_step(self, state: State, input: Tensor, output: Tensor):
        _fire_rate = (output > 0.0).float().mean(
            state.data_reduce_dims, keepdim=False)

        return _fire_rate if state.neuron_wise else _fire_rate.mean(0, keepdim=True)


__build_in_statistic__ = {
    "cnt": cnt,
    "feat_out": feat_out,
    "feat_in": feat_in,
    "param_num": param_num,
    "stride": stride,
    "kernel_size": kernel_size,
    "l1_norm": l1_norm,
    "kl_div": kl_div,
    "fire_rate": fire_rate,
}
